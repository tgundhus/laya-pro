"""Deployment presets use local, frozen budgets without loading checkpoints."""
import builtins
import json
import os

import pytest

import laya
import laya.router
import laya.serve as serve
from laya.performance import resolve_performance_mode


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("LAYA_"):
            monkeypatch.delenv(name)


class StubRouter:
    def __init__(self, **options):
        self.options = options
        config = resolve_performance_mode(options.get("performance_mode"))
        self.performance_config = config
        self.performance_mode = config["mode"] if config else None
        self.max_loaded = options.get("max_loaded", config["max_loaded"] if config else 2)
        self.loaded = []
        self.preloads = []
        self.batch_calls = []

    def preload(self, names=None):
        self.preloads.append(names)
        self.loaded = list(names) if names is not None else list(laya.router.DEFAULT_MODELS)
        self.max_loaded = max(self.max_loaded, len(self.loaded))

    def predict_batch(self, requests, **kwargs):
        self.batch_calls.append(kwargs)
        return [{"model": "fake", "answers": {}, "usage": {}} for _ in requests]


@pytest.fixture
def stub_builder(monkeypatch):
    monkeypatch.setattr(laya.router, "Router", StubRouter)
    monkeypatch.setattr(laya, "Router", StubRouter)
    monkeypatch.setattr(serve, "_apply_thread_limit", lambda: None)
    monkeypatch.setattr(serve, "_configure_decision_cache", lambda router: None)
    return StubRouter


@pytest.mark.parametrize("level, names", [(1, []), (2, []), (3, []),
                                         (4, ["english", "multilingual"]),
                                         (5, list(laya.router.DEFAULT_MODELS))])
def test_http_profiles_bound_automatic_preloads(stub_builder, level, names):
    before = dict(os.environ)
    router = serve.build_router(level)
    assert router.options["performance_mode"] == level
    assert router.options["preload"] is False
    assert router.loaded == names
    assert len(router.preloads) == bool(names)
    assert router.performance_config["max_loaded"] == router.max_loaded
    assert dict(os.environ) == before


def test_default_builder_retains_old_options_and_all_preload(stub_builder, monkeypatch):
    router = serve.build_router()
    assert router.options == {"device": None, "auto_task_detection": False}
    assert router.preloads == [None]
    assert router._laya_server_performance_config is None
    monkeypatch.setenv("LAYA_PERFORMANCE_MODE", "1")
    router = serve.build_router("default")
    assert router.preloads == [None]
    assert router.options["performance_mode"] == "default"


def test_explicit_env_overrides_profile_and_reports_expanded_preload(stub_builder, monkeypatch):
    for name, value in {"LAYA_THREADS": "7", "LAYA_MAX_LOADED": "1", "LAYA_PRELOAD": "1",
                        "LAYA_MAX_BATCH_TOKENS": "2048", "LAYA_MAX_CONCURRENT": "3"}.items():
        monkeypatch.setenv(name, value)
    router = serve.build_router(1)
    config = router.performance_config
    assert router.preloads == [None]
    assert config["max_loaded"] == len(laya.router.DEFAULT_MODELS)
    assert (config["threads"], config["max_batch_tokens"], config["max_concurrent"]) == (7, 2048, 3)
    assert config["overrides"] == {"LAYA_THREADS": 7, "LAYA_MAX_LOADED": 1, "LAYA_PRELOAD": True,
                                   "LAYA_MAX_BATCH_TOKENS": 2048, "LAYA_MAX_CONCURRENT": 3}


def test_explicit_models_and_lazy_override_keep_existing_semantics(stub_builder, monkeypatch):
    monkeypatch.setenv("LAYA_MODELS", "ml, typed")
    assert serve.build_router(4).preloads == [["ml", "typed"]]
    monkeypatch.setenv("LAYA_PRELOAD", "0")
    assert serve.build_router(5).preloads == []


def test_http_budget_is_frozen_and_omitted_batch_size_uses_profile(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    router = StubRouter(performance_mode=3)
    monkeypatch.setenv("LAYA_MAX_BATCH_TOKENS", "2048")
    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "2")
    app = serve.create_app(router=router, performance_mode=3)
    monkeypatch.setenv("LAYA_MAX_BATCH_TOKENS", "131072")
    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "16")
    request = {"states": ["hello"] * 8, "questions": {"ok": {"type": "noul"}}}
    with TestClient(app) as client:
        assert client.post("/v1/systemone/batch", json=request).status_code == 200
        assert router.batch_calls[-1]["batch_size"] == 4
        request["questions"]["other"] = {"type": "noul"}
        assert client.post("/v1/systemone/batch", json=request).status_code == 200
        assert router.batch_calls[-1]["batch_size"] == 2
        health = client.get("/health").json()
        assert health["performance_mode"] == 3
        assert health["performance_config"]["max_batch_tokens"] == 2048
        assert health["performance_config"]["max_concurrent"] == 2


def test_explicit_batch_size_can_override_profile_but_stays_inside_budget(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    router = StubRouter(performance_mode=1)
    with TestClient(serve.create_app(router=router, performance_mode=1)) as client:
        request = {"states": ["hello"] * 32, "questions": {"ok": {"type": "noul"}}, "batch_size": 8}
        assert client.post("/v1/systemone/batch", json=request).status_code == 200
        assert router.batch_calls[-1]["batch_size"] == 8
        request["batch_size"] = 32
        assert client.post("/v1/systemone/batch", json=request).status_code == 200
        assert router.batch_calls[-1]["batch_size"] == 16


@pytest.mark.parametrize("router_mode, app_mode", [("default", 1), (2, 1), (2, "default")])
def test_injected_router_rejects_a_conflicting_app_mode_before_creating_pool(monkeypatch, router_mode, app_mode):
    pytest.importorskip("fastapi")
    import concurrent.futures

    monkeypatch.setattr(laya.router, "_apply_torch_threads", lambda config: None)
    router = laya.router.Router(performance_mode=router_mode, preload=False)

    def unexpected_pool(*args, **kwargs):
        raise AssertionError("a rejected profile must not create a worker pool")

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", unexpected_pool)
    with pytest.raises(ValueError, match="must match the injected Router's configured mode"):
        serve.create_app(router=router, performance_mode=app_mode)


@pytest.mark.parametrize("app_mode", [None, "2"])
def test_injected_router_keeps_runtime_profile_and_captures_http_overrides(monkeypatch, app_mode):
    pytest.importorskip("fastapi")
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient

    thread_configurations = []
    monkeypatch.setattr(laya.router, "_apply_torch_threads",
                        lambda config: thread_configurations.append(config["threads"]))
    router = laya.router.Router(performance_mode=2, max_loaded=3, preload=False)
    original_config = dict(router.performance_config)
    for name, value in {"LAYA_PERFORMANCE_MODE": "5", "LAYA_THREADS": "11", "LAYA_MAX_LOADED": "1",
                        "LAYA_PRELOAD": "1", "LAYA_MAX_BATCH_TOKENS": "512", "LAYA_MAX_CONCURRENT": "1"}.items():
        monkeypatch.setenv(name, value)
    calls = []
    started, release = threading.Event(), threading.Event()
    block = []

    def predict_batch(requests, **kwargs):
        calls.append(kwargs)
        if block:
            started.set()
            assert release.wait(5), "test must release the admitted request"
        return [{"model": "fake", "answers": {}, "usage": {}} for _ in requests]

    monkeypatch.setattr(router, "predict_batch", predict_batch)
    app = serve.create_app(router=router, performance_mode=app_mode)
    monkeypatch.setenv("LAYA_MAX_BATCH_TOKENS", "131072")
    monkeypatch.setenv("LAYA_MAX_CONCURRENT", "8")
    request = {"states": ["hello"] * 2, "questions": {"ok": {"type": "noul"}}}
    with TestClient(app) as client:
        assert client.post("/v1/systemone/batch", json=request).status_code == 200
        assert calls[-1]["batch_size"] == 1
        config = client.get("/health").json()["performance_config"]
        assert (config["mode"], config["threads"], config["max_loaded"], config["preload"]) == (
            2, original_config["threads"], 3, False)
        assert (config["max_batch_tokens"], config["max_concurrent"]) == (512, 1)
        assert config["overrides"] == {"LAYA_MAX_BATCH_TOKENS": 512, "LAYA_MAX_CONCURRENT": 1}
        block.append(True)
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.post, "/v1/systemone/batch", json=request)
            try:
                assert started.wait(3), "the first request must start inference"
                assert client.post("/v1/systemone/batch", json=request).status_code == 503
            finally:
                release.set()
            assert first.result(timeout=3).status_code == 200
    assert router.performance_config == original_config
    assert thread_configurations == [original_config["threads"]]


def test_default_injected_router_ignores_a_later_environment_mode(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(laya.router, "_apply_torch_threads", lambda config: None)
    router = laya.router.Router(performance_mode="default")
    monkeypatch.setenv("LAYA_PERFORMANCE_MODE", "1")
    with TestClient(serve.create_app(router=router)) as client:
        health = client.get("/health").json()
        assert health["performance_mode"] is None
        assert health["performance_config"] is None


def test_mcp_preloads_once_and_retains_legacy_default(stub_builder, monkeypatch):
    pytest.importorskip("mcp")
    import laya.mcp.server as server

    monkeypatch.setattr(server, "_ROUTER", None)
    monkeypatch.setattr(server, "_PERFORMANCE_MODE", None)
    monkeypatch.setattr(server, "_apply_thread_limit", lambda: None)
    monkeypatch.setattr(server, "_configure_decision_cache", lambda router: None)
    router = server._ensure_router()
    assert router.preloads == [["english", "multilingual"]]
    monkeypatch.setattr(server, "_ROUTER", None)
    monkeypatch.setenv("LAYA_PERFORMANCE_MODE", "4")
    router = server._ensure_router()
    assert router.options["preload"] is False
    assert router.preloads == [["english", "multilingual"]]
    assert server._ensure_router() is router
    assert len(router.preloads) == 1


def test_mcp_status_before_router_reports_profile_without_torch(monkeypatch):
    pytest.importorskip("mcp")
    import laya.mcp.server as server

    monkeypatch.setattr(server, "_ROUTER", None)
    monkeypatch.setattr(server, "_PERFORMANCE_MODE", "2")
    monkeypatch.setenv("LAYA_PERFORMANCE_MODE", "5")
    original_import = builtins.__import__

    def no_backend(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "transformers"):
            raise AssertionError("status must not import a backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backend)
    status = json.loads(server.laya_status_tool())
    assert status["performance_mode"] == 2
    assert status["performance_config"]["preload"] is False
    assert status["router_ready"] is False
    assert status["router_preload"] is False


def test_remote_profile_comes_from_http_server_and_never_imports_torch(monkeypatch):
    from laya.mcp.remote import RemoteRouter
    from laya.mcp.tools import laya_status

    monkeypatch.setenv("LAYA_PERFORMANCE_MODE", "5")
    monkeypatch.setenv("LAYA_THREADS", "16")
    original_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name.split(".")[0] == "torch":
            raise AssertionError("remote mode must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    router = RemoteRouter("http://localhost:8000", performance_mode=4)
    monkeypatch.setattr(router, "health", lambda: {"loaded": [], "performance_mode": 1,
                                                  "performance_config": {"mode": 1, "threads": 1}})
    status = laya_status(router=router)
    assert router.performance_mode is None
    assert status["performance_mode"] == 1
    assert status["performance_config"]["threads"] == 1


def test_server_cli_flag_precedes_environment_and_preserves_environment(monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")
    monkeypatch.setenv("LAYA_PERFORMANCE_MODE", "5")
    calls = []
    monkeypatch.setattr(serve, "create_app", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: None)
    serve.main(["--performance-mode", "default"])
    assert calls == [{"performance_mode": "default"}]
    assert os.environ["LAYA_PERFORMANCE_MODE"] == "5"
    with pytest.raises(SystemExit) as error:
        serve.main(["--performance-mode", "0"])
    assert error.value.code == 2


def test_mcp_cli_lazy_profile_does_not_inherit_legacy_startup_preload(monkeypatch):
    pytest.importorskip("mcp")
    import laya.mcp.server as server

    calls = []
    monkeypatch.setattr(server, "_ROUTER", None)
    monkeypatch.setattr(server, "_PERFORMANCE_MODE", None)
    monkeypatch.setattr(server, "_ensure_router", lambda: calls.append("build"))
    monkeypatch.setattr(server.server, "run", lambda: calls.append("run"))
    monkeypatch.setattr(server, "_close_decision_cache", lambda router: None)
    monkeypatch.setenv("LAYA_PERFORMANCE_MODE", "5")
    server.main(["--performance-mode", "1"])
    assert calls == ["run"]
    assert server._PERFORMANCE_MODE == "1"
    assert os.environ["LAYA_PERFORMANCE_MODE"] == "5"
