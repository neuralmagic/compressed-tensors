from typing import Iterable, Tuple
from collections.abc import Generator

import re
import torch
import logging

_LOGGER: logging.Logger = logging.getLogger(__name__)


__all__ = ["match_named_modules", "is_match"]


def match_named_modules(
    model: torch.nn.Module,
    targets: Iterable[str] = tuple(),
    ignore: Iterable[str] = tuple(),
    warn_on_fail: bool = True
) -> Generator[Tuple[str, torch.nn.Module], None, None]:
    unmatched_targets = set(targets)
    for name, module in model.named_modules():
        for target in targets:
            if is_match(name, module, target):
                unmatched_targets.remove(target)
        
                if not any(is_match(name, module, ign) for ign in ignore):
                    yield name, module

    if warn_on_fail:
        for target in unmatched_targets:
            _LOGGER.warning(
                f"Could not match `{target}` in instance of {model.__class__.__name__}"
            )

def is_match(name: str, module: torch.nn.Module, target: str) -> bool:
    return _match_name(name, target) or _match_class(module, target)


def _match_name(name: str, target: str) -> bool:
    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name)
    else:
        return target == name


def _match_class(module: torch.nn.Module, target: str) -> bool:
    """
    Will never match against a regex pattern since `:` is not allowed in class names
    
    """
    return any(
        issubclass(cls, torch.nn.Module) and cls.__name__ == target
        for cls in module.__class__.__mro__
    )
