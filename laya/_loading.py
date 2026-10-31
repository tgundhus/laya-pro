"""Checkpoint loading without a second resident copy of compatible CPU weights."""
import inspect
import os

import torch


def _compatible_state(model, weights):
    # assign arrived after the supported torch 2.0 floor. Inspect before loading: retrying
    # on TypeError would also hide errors from a model's own loader or hooks.
    if getattr(model.load_state_dict, "__func__", None) is not torch.nn.Module.load_state_dict:
        return None
    registration = torch.nn.modules.module
    if (getattr(registration, "_global_parameter_registration_hooks", None)
            or getattr(registration, "_global_buffer_registration_hooks", None)):
        return None
    try:
        if "assign" not in inspect.signature(model.load_state_dict).parameters:
            return None
    except (TypeError, ValueError):
        return None

    state = {}
    model_storage, checkpoint_storage = set(), set()
    for prefix, module in model.named_modules(remove_duplicate=False):
        if (getattr(module._load_from_state_dict, "__func__", None) is not torch.nn.Module._load_from_state_dict
                or type(module).__setattr__ is not torch.nn.Module.__setattr__
                or module._load_state_dict_pre_hooks or module._load_state_dict_post_hooks
                or type(module).get_extra_state is not torch.nn.Module.get_extra_state):
            return None
        for name, target in list(module._parameters.items()) + list(module._buffers.items()):
            if target is None:
                continue
            if target.device.type != "cpu" or target.layout != torch.strided or target.is_quantized:
                return None
            storage = target.untyped_storage()
            if storage.data_ptr() in model_storage:
                return None
            model_storage.add(storage.data_ptr())
            # Non-persistent encoder buffers (including rotary frequencies) stay untouched,
            # but must not alias a parameter/buffer whose storage assignment replaces.
            if name in module._non_persistent_buffers_set:
                continue
            # A tied parameter/buffer, view, tensor subclass or attached parameter metadata
            # relies on copy semantics. Keep the established loader for those cases.
            if (type(target) not in (torch.Tensor, torch.nn.Parameter) or target.__dict__
                    or not target.is_contiguous() or target.storage_offset()
                    or target.is_conj() or target.is_neg() or not target.is_leaf or target.grad is not None
                    or target._backward_hooks or getattr(target, "_post_accumulate_grad_hooks", None)
                    or any(dim is not None for dim in getattr(target, "names", ()))):
                return None
            if storage.nbytes() != target.numel() * target.element_size() or not target.numel():
                return None
            key = prefix + "." + name if prefix else name
            state[key] = target

    if not state or state.keys() != weights.keys() or getattr(weights, "_metadata", None):
        return None
    for name, target in state.items():
        source = weights[name]
        if (type(source) is not torch.Tensor or source.__dict__ or source.requires_grad or source.device != target.device
                or source.layout != target.layout
                or source.shape != target.shape or source.stride() != target.stride()
                or source.storage_offset() or source.is_quantized or source.is_conj() or source.is_neg()
                or any(dim is not None for dim in getattr(source, "names", ()))
                or (not isinstance(target, torch.nn.Parameter) and source.requires_grad != target.requires_grad)):
            return None
        storage = source.untyped_storage()
        if (storage.nbytes() != source.numel() * source.element_size()
                or storage.data_ptr() in checkpoint_storage):
            return None
        checkpoint_storage.add(storage.data_ptr())
    return state


def _checkpoint_identity(path):
    stat = os.stat(path)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _check_identity(path, identity):
    try:
        unchanged = _checkpoint_identity(path) == identity
    except OSError as error:
        raise RuntimeError("Checkpoint changed while loading %r; retry with a stable checkpoint." % path) from error
    if not unchanged:
        raise RuntimeError("Checkpoint changed while loading %r; retry with a stable checkpoint." % path)


def _load_weights(model, weights, path=None, identity=None):
    """Strictly load a fresh model; return ``assign``, ``stream`` or ``copy``.

    Compatible same-dtype tensors adopt checkpoint storage. Mixed floating dtypes copy one
    tensor at a time, closing each file mapping before reading the next. Custom loading
    semantics and unsupported tensor properties keep the established strict copy loader.
    Passing a path/identity transfers ownership of the weights dict, which streaming clears.
    """
    state = _compatible_state(model, weights)
    if state is not None and all(weights[name].dtype == target.dtype for name, target in state.items()):
        model.load_state_dict(weights, strict=True, assign=True)
        return "assign"

    floating = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if (state is not None and isinstance(weights, dict) and path is not None and identity is not None
            and all(weights[name].dtype == target.dtype
                    or (weights[name].dtype in floating and target.dtype in floating)
                    for name, target in state.items())):
        from safetensors import safe_open

        _check_identity(path, identity)
        dtypes = {name: tensor.dtype for name, tensor in weights.items()}
        # No tensor from the initial full-file map may survive into the bounded reads.
        dict.clear(weights)
        for name, target in state.items():
            _check_identity(path, identity)
            try:
                with safe_open(path, framework="pt", device="cpu") as checkpoint:
                    source = checkpoint.get_tensor(name)
                    try:
                        if source.shape != target.shape or source.dtype != dtypes[name]:
                            raise RuntimeError("Checkpoint tensor %r changed while loading %r." % (name, path))
                        with torch.no_grad():
                            target.copy_(source)
                    finally:
                        del source
                del checkpoint
            finally:
                _check_identity(path, identity)
        return "stream"

    model.load_state_dict(weights, strict=True)
    return "copy"
