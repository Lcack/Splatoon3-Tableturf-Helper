"""Reload a stopped analysis core while the Tk/capture/controller stay alive."""
from __future__ import annotations

import ast
import hashlib
import importlib
from pathlib import Path
import sys


UI_REFRESH_PATCH_API = 1
TERMINAL_UI_MODULE = "tableturf_rebuild.application.task_terminal_ui"
TERMINAL_REFRESH_METHODS = (
    "_refresh_frame", "_start_task_worker", "_run_idle_tick",
    "_collect_task_result", "_run_battle_debug_tick", "_publish_diagnostics",
    "_wait_for_task_worker", "_update_battle_debug_live",
)
_MISSING = object()


def _compile_terminal_refresh_methods(module, source: str, path: Path, bindings=None):
    """Compile only an explicit method whitelist; never execute UI startup."""
    parsed = ast.parse(source, filename=str(path))
    terminal = next((node for node in parsed.body
                     if isinstance(node, ast.ClassDef) and node.name == "TaskTerminalApp"), None)
    if terminal is None:
        raise RuntimeError("terminal refresh patch has no TaskTerminalApp class")
    methods = {node.name: node for node in terminal.body if isinstance(node, ast.FunctionDef)}
    missing = set(TERMINAL_REFRESH_METHODS) - methods.keys()
    if missing:
        raise RuntimeError("terminal refresh patch is missing methods: " + ", ".join(sorted(missing)))
    selected = [methods[name] for name in TERMINAL_REFRESH_METHODS]
    if any(node.decorator_list for node in selected):
        raise RuntimeError("terminal refresh patch methods cannot have decorators")
    if any(not isinstance(value, ast.Constant)
           for node in selected for value in (*node.args.defaults, *node.args.kw_defaults)
           if value is not None):
        raise RuntimeError("terminal refresh patch method defaults must be literals")
    staged = ast.parse("from __future__ import annotations\nclass _TerminalRefreshPatch:\n    pass\n")
    staged.body[1].body = selected
    namespace = dict(module.__dict__)
    namespace.update(bindings or {})
    exec(compile(ast.fix_missing_locations(staged), str(path), "exec"), namespace)
    replacement = namespace["_TerminalRefreshPatch"]
    return {name: replacement.__dict__[name] for name in TERMINAL_REFRESH_METHODS}


def patch_loaded_terminal_ui(*, app, bindings=None, source: str | None = None):
    """Patch the existing class only after its paused reload drained analysis.

    Returns a rollback token. This creates no app/window/controller/thread and
    leaves the existing Tk callback registrations and non-whitelisted methods.
    """
    module = sys.modules.get(TERMINAL_UI_MODULE)
    if module is None:
        return None
    terminal = getattr(module, "TaskTerminalApp", None)
    if terminal is None or not isinstance(app, terminal):
        raise RuntimeError("terminal refresh patch requires the existing app")
    thread = getattr(app, "_task_thread", None)
    pending = getattr(app, "_pending_task_result", None)
    status = getattr(getattr(app, "runner", None), "status", None)
    status = getattr(status, "value", status)
    if (getattr(app, "_reload_context", None) is None
        or getattr(app, "battle_debug_active", False)
        or status not in {"IDLE", "PAUSED", "COMPLETED"}
        or (thread is not None and thread.is_alive()) or pending is not None):
        raise RuntimeError("terminal refresh patch requires paused and drained analysis")
    path = Path(__file__).with_name("task_terminal_ui.py")
    source = source if source is not None else path.read_text(encoding="utf-8-sig")
    methods = _compile_terminal_refresh_methods(module, source, path, bindings)
    previous = {name: terminal.__dict__.get(name, _MISSING) for name in methods}
    previous["_refresh_patch_version"] = terminal.__dict__.get("_refresh_patch_version", _MISSING)
    token = (terminal, previous)
    try:
        for name, method in methods.items():
            setattr(terminal, name, method)
        terminal._refresh_patch_version = hashlib.sha256(source.encode()).hexdigest()[:12]
    except Exception:
        rollback_terminal_refresh_patch(token)
        raise
    return token


def rollback_terminal_refresh_patch(token):
    if token is None:
        return
    terminal, previous = token
    for name, method in previous.items():
        if method is _MISSING:
            if name in terminal.__dict__:
                delattr(terminal, name)
        else:
            setattr(terminal, name, method)


def bootstrap_terminal_refresh():
    """Allow an already-cached older reload_core to load this UI patch.

    Called by the fresh runtime factory after the older reload_core returned.
    Outside the existing app's paused reload callback this is a no-op.
    """
    module = sys.modules.get(TERMINAL_UI_MODULE)
    terminal = getattr(module, "TaskTerminalApp", None) if module else None
    if terminal is None:
        return None
    frame = sys._getframe(1)
    app = None
    while frame is not None:
        if frame.f_code.co_name == "reload_core":
            return None  # Never reload support during its module traversal.
        candidate = frame.f_locals.get("self")
        if frame.f_code.co_name == "reload_analysis_core" and isinstance(candidate, terminal):
            app = candidate
            break
        frame = frame.f_back
    if app is None:
        return None
    token = patch_loaded_terminal_ui(app=app)
    # Existing Tk commands may still be bound to the earlier reload callback.
    # Capture progress from its old runner now; the patched collector restores
    # it after that callback has installed the replacement runner.
    runner_class = getattr(module, "TaskRunner", None)
    old_runner = getattr(app, "runner", None)
    if (getattr(old_runner, "task", None) is not None
        and hasattr(runner_class, "progress_checkpoint")):
        app._pending_progress_restore = (old_runner, runner_class.progress_checkpoint(old_runner))
    return token


def reload_core() -> tuple[dict, dict]:
    root = Path(__file__).resolve().parents[1]
    paths = {}
    for folder in ("vision", "planning"):
        for path in (root / folder).glob("*.py"):
            if path.stem.endswith("_ui"):
                continue
            parts = list(path.relative_to(root.parent).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            paths[".".join(parts)] = path
    for name in ("state_machine", "battle_vision", "battle_executor", "task_runner", "live_diagnostics"):
        paths[f"tableturf_rebuild.application.{name}"] = root / "application" / f"{name}.py"
    sources = {name: path.read_text(encoding="utf-8-sig") for name, path in paths.items()}
    # Compile every source before changing any live module. Build import order
    # so users of GameState/RegionRole receive the same newly created enum.
    codes = {name: compile(source, str(paths[name]), "exec") for name, source in sources.items()}
    dependencies = {
        name: {node.module for node in ast.walk(ast.parse(source))
               if isinstance(node, ast.ImportFrom) and node.module in paths}
        for name, source in sources.items()
    }
    ordered = []
    remaining = set(paths)
    while remaining:
        ready = sorted(name for name in remaining if not dependencies[name] & remaining)
        if not ready:
            raise RuntimeError("core reload dependency cycle: " + ", ".join(sorted(remaining)))
        ordered.extend(ready)
        remaining.difference_update(ready)
    modules = {name: importlib.import_module(name) for name in ordered}
    previous = {name: dict(module.__dict__) for name, module in modules.items()}
    try:
        for name in ordered:
            # Executing freshly compiled code also avoids a same-second .pyc
            # timestamp hiding edits during rapid debugging.
            exec(codes[name], modules[name].__dict__)
        bindings = {
            "GameState": modules["tableturf_rebuild.application.state_machine"].GameState,
            "BattleVisionRuntime": modules["tableturf_rebuild.application.battle_vision"].BattleVisionRuntime,
            "VisionMctsBattleExecutor": modules["tableturf_rebuild.application.battle_executor"].VisionMctsBattleExecutor,
            "TaskRunner": modules["tableturf_rebuild.application.task_runner"].TaskRunner,
            "TaskStatus": modules["tableturf_rebuild.application.task_runner"].TaskStatus,
            "observation_details": modules["tableturf_rebuild.application.live_diagnostics"].observation_details,
            "LiveDiagnostics": modules["tableturf_rebuild.application.live_diagnostics"].LiveDiagnostics,
        }
        debug = sys.modules.get("tableturf_rebuild.application.task_debug_ui")
        if debug is not None:
            debug.GameState = bindings["GameState"]
        versions = {name: hashlib.sha256(source.encode()).hexdigest()[:12] for name, source in sources.items()}
        return bindings, versions
    except Exception:
        for name, namespace in previous.items():
            modules[name].__dict__.clear()
            modules[name].__dict__.update(namespace)
        raise
