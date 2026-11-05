# Modified by the MotionDisco authors from ShinkaEvolve (Sakana AI, Apache-2.0); see NOTICE.
"""Packaged Hydra configs for installed Shinka CLI usage."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

from omegaconf import OmegaConf


@contextmanager
def config_root() -> Iterator[Path]:
    """Yield a filesystem path to packaged Hydra configs."""
    with as_file(files(__name__)) as path:
        yield Path(path)


def _py_resolver(target: str, *args):
    """OmegaConf resolver: ``${py:module.path.fn[,arg1,arg2]}``.

    Imports ``module.path``, looks up the attribute ``fn``, calls it
    with the trailing args (all passed as strings — cast inside the
    target if needed), and returns the result. OmegaConf's grammar
    splits resolver args on commas (not colons), so nested
    interpolations like ``${py:fn,${task.scene},${task.description}}``
    feed each interpolated value as a separate positional arg.

    Used by task configs that want to compute fields like
    ``task_sys_msg`` at config-resolve time from a Python function,
    instead of carrying a literal string in YAML. The resolver itself
    is generic and knows nothing about specific examples.
    """
    module_path, _, func_name = target.rpartition(".")
    if not module_path:
        raise ValueError(
            f"${{py:...}} resolver expects a dotted path like "
            f"'module.path.func', got {target!r}"
        )
    mod = importlib.import_module(module_path)
    fn = getattr(mod, func_name)
    result = fn(*args)
    # OmegaConf accepts only primitives or its own containers from a
    # resolver; wrap lists / dicts (e.g. a list of prompt variants).
    if isinstance(result, (list, dict)):
        return OmegaConf.create(result)
    return result


def register_resolvers() -> None:
    """Register Shinka's custom OmegaConf resolvers (idempotent).

    Must run before Hydra's ``@hydra.main`` composes any config that
    uses ``${py:...}``. ``shinka/launch_hydra.py`` calls this at
    module-load time so the standard launcher path is covered.
    Embedded users invoking Hydra directly should call it themselves.
    """
    if not OmegaConf.has_resolver("py"):
        OmegaConf.register_new_resolver("py", _py_resolver)
