"""Checkpoint metadata and tokenizer loading shared by the local runtimes."""
import json
import os
import tempfile
import threading
import warnings


def _fix_tokenizer_config(path: str):
    """Ensure tokenizer_config.json can be loaded across all transformers versions."""
    cfg_file = os.path.join(path, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        return
    try:
        with open(cfg_file) as f:
            tcfg = json.load(f)
        changed = False
        if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            changed = True
        # Checkpoints built on the mmBERT/Gemma tokenizer store extra_special_tokens as a list;
        # transformers expects a mapping and raises "'list' object has no attribute 'keys'",
        # which makes AutoTokenizer -- and so the whole model -- fail to load.
        extra = tcfg.get("extra_special_tokens")
        if isinstance(extra, list):
            tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
            changed = True
        if changed:
            # HuggingFace snapshots are symlinks into a shared blob store, so writing through the
            # link would truncate a file shared with other revisions and processes, race
            # concurrent loads, and desync the hub's cache metadata. Write to a temporary file in
            # the same directory and `os.replace` it into place: the snapshot entry becomes a
            # regular file and is swapped atomically, so a concurrent load never sees a missing
            # or half-written config.
            cfg_dir = os.path.dirname(cfg_file)
            fd, tmp_file = tempfile.mkstemp(dir=cfg_dir, prefix=".tokenizer_config.", suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(tcfg, f, indent=2)
                # mkstemp creates the file 0600; keep the mode the cache file had so a shared
                # cache stays readable to the same users as before.
                try:
                    os.chmod(tmp_file, os.stat(cfg_file).st_mode & 0o777)
                except OSError:
                    pass
                os.replace(tmp_file, cfg_file)
            except BaseException:
                try:
                    os.unlink(tmp_file)
                except OSError:
                    pass
                raise
    except Exception as e:
        # Do not swallow this silently: if the patch did not apply, AutoTokenizer may fail later
        # with a confusing error and no hint that the config was the cause.
        warnings.warn(
            "laya: could not patch %s (%s); the tokenizer may fail to load with this "
            "transformers version." % (cfg_file, e),
            RuntimeWarning, stacklevel=2)


def _tokenizer_files(tok_dir: str):
    with os.scandir(tok_dir) as entries:
        files = []
        for entry in entries:
            if entry.is_file():
                stat = entry.stat()
                files.append((entry.name, stat.st_size, stat.st_mtime_ns, stat.st_ino))
    return tuple(sorted(files))


def _checkpoint_signature(model_dir: str, artifact: str, expected=None):
    """Load-time identity for an unversioned local checkpoint, without reading its weights again."""
    if expected is None:
        raw = os.environ.get("LAYA_SHA256_DIGESTS", "").strip()
        expected = json.loads(raw) if raw else {}
    expected = expected or {}
    files = [("rl_agent_config.json", os.path.join(model_dir, "rl_agent_config.json")),
             (os.path.basename(artifact), artifact)]
    encoder_config = os.path.join(model_dir, "encoder", "config.json")
    if os.path.isfile(encoder_config):
        files.append(("encoder/config.json", encoder_config))
    tok_dir = os.path.join(model_dir, "tokenizer")
    if os.path.isdir(tok_dir):
        files.extend(("tokenizer/" + name, os.path.join(tok_dir, name)) for name, *_ in _tokenizer_files(tok_dir))
    signature = []
    for name, path in files:
        digest = expected.get(name)
        if path == artifact and not name.endswith(".safetensors"):
            digest = expected.get("onnx", expected.get("onnx_path"))
            if digest is None and os.path.realpath(path) == os.path.realpath(os.path.join(model_dir, name)):
                digest = expected.get(name)
        if digest is not None:
            signature.append([name, "sha256", str(digest).strip().lower()])
        else:
            stat = os.stat(path)
            signature.append([name, "stat", stat.st_size, stat.st_mtime_ns, stat.st_ino])
    return signature


class _NativeTokenizer:
    """The inference tokenizer surface, backed by a checkpoint's complete tokenizer.json.

    Loading is local-only: a missing JSON or required special-token metadata is an error,
    rather than a reason to substitute another tokenizer or download an encoder.
    """

    def __init__(self, path: str):
        from tokenizers import AddedToken, Tokenizer

        tokenizer_file = os.path.join(path, "tokenizer.json")
        if not os.path.isfile(tokenizer_file):
            raise FileNotFoundError("native ONNX tokenizer requires %r" % tokenizer_file)
        metadata = {}
        for name in ("tokenizer_config.json", "special_tokens_map.json"):
            if name == "special_tokens_map.json" and "added_tokens_decoder" in metadata:
                continue
            filename = os.path.join(path, name)
            if os.path.isfile(filename):
                with open(filename, encoding="utf-8") as f:
                    metadata.update(json.load(f))
        self._tokenizer = Tokenizer.from_file(tokenizer_file)
        self._lock = threading.RLock()
        self.model_max_length = metadata.get("model_max_length")
        self.truncation_side = metadata.get("truncation_side", "right")
        if self.truncation_side not in ("left", "right"):
            raise ValueError("native ONNX tokenizer truncation_side must be 'left' or 'right'")
        self.clean_up_tokenization_spaces = metadata.get("clean_up_tokenization_spaces", False)
        self._clean_bpe = metadata.get(
            "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output", False)
        saved_tokens = {v.get("content"): v for v in metadata.get("added_tokens_decoder", {}).values()
                        if isinstance(v, dict)}
        backend_tokens = self._tokenizer.get_added_tokens_decoder()
        required = ("cls_token", "sep_token", "mask_token", "pad_token")
        optional = [name for name in ("unk_token", "bos_token", "eos_token") if metadata.get(name) is not None]
        for name in required + tuple(optional):
            raw = metadata.get(name)
            token = raw.get("content") if isinstance(raw, dict) else raw
            token_id = self._tokenizer.token_to_id(token) if isinstance(token, str) else None
            if token_id is None:
                raise ValueError("native ONNX tokenizer requires %s metadata naming a token in tokenizer.json" % name)
            raw = saved_tokens.get(token, raw)
            if token not in saved_tokens and token_id in backend_tokens:
                special = backend_tokens[token_id]
            else:
                flags = ({k: raw[k] for k in ("single_word", "lstrip", "rstrip", "normalized") if k in raw}
                         if isinstance(raw, dict) else {})
                special = AddedToken(token, special=True, **flags)
            self._tokenizer.add_special_tokens([special])
            setattr(self, name, token)
            setattr(self, name + "_id", token_id)

    def __call__(self, text: str, *, add_special_tokens: bool = True,
                 truncation: bool = False, max_length=None):
        # Tokenizers mutates padding/truncation settings even for a read-style encode. Preserve
        # the serialized settings across calls, and guard the entire mutation/encode/restore.
        with self._lock:
            prior_truncation = self._tokenizer.truncation
            prior_padding = self._tokenizer.padding
            try:
                self._tokenizer.no_padding()
                self._tokenizer.no_truncation()
                if truncation:
                    limit = max_length if max_length is not None else self.model_max_length
                    if limit is None:
                        raise ValueError("native ONNX tokenizer truncation requires max_length or model_max_length")
                    self._tokenizer.enable_truncation(max_length=limit, direction=self.truncation_side)
                encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
                return {"input_ids": encoded.ids, "attention_mask": encoded.attention_mask}
            finally:
                if prior_truncation is None:
                    self._tokenizer.no_truncation()
                else:
                    self._tokenizer.enable_truncation(**prior_truncation)
                if prior_padding is None:
                    self._tokenizer.no_padding()
                else:
                    self._tokenizer.enable_padding(**prior_padding)

    def decode(self, ids, *, skip_special_tokens: bool = False,
               clean_up_tokenization_spaces=None) -> str:
        with self._lock:
            text = self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        clean = (self.clean_up_tokenization_spaces if clean_up_tokenization_spaces is None
                 else clean_up_tokenization_spaces)
        if clean and (type(self._tokenizer.model).__name__ != "BPE" or self._clean_bpe):
            # Match PreTrainedTokenizerFast's optional English punctuation/contraction cleanup.
            for before, after in ((" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","),
                                  (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"),
                                  (" 've", "'ve"), (" 're", "'re")):
                text = text.replace(before, after)
        return text
