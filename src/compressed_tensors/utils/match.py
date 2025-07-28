# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from collections.abc import Generator
from typing import Callable, Iterable, Tuple

import torch
from compressed_tensors.utils.internal import InternalModule


_LOGGER: logging.Logger = logging.getLogger(__name__)


__all__ = [
    "match_named_modules",
    "match_named_parameters",
    "match_modules_set",
    "is_match",
]


def match_named_modules(
    model: torch.nn.Module,
    targets: Iterable[str] | None,
    ignore: Iterable[str] | None = None,
    warn_on_fail: bool = False,
    warn_on_unmatched_ignores: bool = False,
    return_matched_targets: bool = False,
    preprocess_name: Callable[[str], str] = lambda x: x,
) -> Generator[Tuple[str, torch.nn.Module]]:
    """
    Yields names and modules which match `targets` but do not match `ignore`.
    Values are returned in order of `model.named_modules()`

    :param model: model containing submodules to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    :param warn_on_fail: if True, warns if any targets do not match any modules in model
    :return: generator of module names and modules
    """
    ignore = ignore or []
    targets = targets or []

    unmatched_targets = set(targets)
    unmatched_ignores = set(ignore)

    # Order targets by type: exact name match, regex name match, class name match
    targets = sorted(targets, key=lambda x: ("re:" in x, x))
    for name, module in model.named_modules():
        if isinstance(module, InternalModule):
            continue

        # preprocess the module name and module
        name = preprocess_name(name)

        ignore_matched = False
        for ign in ignore:
            if is_match(name, module, ign):
                unmatched_ignores -= {ign}
                ignore_matched = True
                break
        if ignore_matched:
            continue

        matched_targets = []
        # Check for name matches first (exact then regex)
        for target in targets:
            if _match_name(name, target):
                unmatched_targets -= {target}
                matched_targets.append(target)
                if not return_matched_targets:
                    break

        if not return_matched_targets and matched_targets:
            # Don't need to check other targets, one match is enough
            yield name, module
            continue

        # Check for class matches
        for target in targets:
            if _match_class(module, target):
                unmatched_targets -= {target}
                matched_targets.append(target)
                if not return_matched_targets:
                    break

        if matched_targets:
            if return_matched_targets:
                yield name, module, matched_targets
            else:
                yield name, module

    if warn_on_fail:
        for target in unmatched_targets:
            _LOGGER.warning(
                f"Could not match `{target}` in instance of {model.__class__.__name__}"
            )

    if warn_on_unmatched_ignores:
        for ign in unmatched_ignores:
            _LOGGER.warning(
                f"Unmatched ignore targets: {unmatched_ignores}, in instance of {model.__class__.__name__}"
            )


def match_named_parameters(
    model: torch.nn.Module,
    targets: Iterable[str],
    ignore: Iterable[str] = tuple(),
    warn_on_fail: bool = False,
) -> Generator[Tuple[str, torch.nn.Module, torch.nn.Parameter]]:
    """
    Yields parameters which match `targets` but do not match `ignore`.
    Values are returned in order of `model.named_modules()`

    :param model: model containing params to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    :param warn_on_fail: if True, warns if any targets do not match any params in model
    :return: generator of fully-qualified param names, parent modules, and params
    """
    unmatched_targets = set(targets)
    for module_name, module in model.named_modules():
        if isinstance(module, InternalModule):
            continue

        for param_name, param in module.named_parameters(recurse=False):
            param_fqn = f"{module_name}.{param_name}"
            for target in targets:
                if _match_name(param_fqn, target):
                    unmatched_targets -= {target}

                    if not any(_match_name(param_fqn, ign) for ign in ignore):
                        yield param_fqn, module, param

    if warn_on_fail:
        for target in unmatched_targets:
            _LOGGER.warning(
                f"Could not match `{target}` in instance of {model.__class__.__name__}"
            )


def match_modules_set(
    model: torch.nn.Module,
    targets: Iterable[str],
    ignore: Iterable[str] = tuple(),
) -> Generator[Iterable[torch.nn.Module]]:
    """
    Yields modules grouped with the same order and size as `targets`.
    Values are returned in order of `model.named_modules()`

    For example, the following targets would yield module belonging to the following layers:
    ```python3
    match_modules_set(model, ["q_proj", "k_proj", "v_proj"]) == (
        (
            `model.layers.0.self_attn.q_proj`,
            `model.layers.0.self_attn.k_proj`,
            `model.layers.0.self_attn.v_proj`,
        ),
        (
            `model.layers.1.self_attn.q_proj`,
            `model.layers.1.self_attn.k_proj`,
            `model.layers.1.self_attn.v_proj`,
        ),
        ...
        (
            `model.layers.32.self_attn.q_proj`,
            `model.layers.32.self_attn.k_proj`,
            `model.layers.32.self_attn.v_proj`,
        ),
    )
    ```

    This can be used to match layers to their corresponding downstream counterparts.
    For example, matching layer norms to their subsequent linear layers
    ```python3
    for norm, q, k, v in match_modules_set(model, (norm_tgt, q_tgt, k_tgt, v_tgt)):
        fuse_norm_linears(norm, [q, k, v])

    :param model: model containing modules to match against
    :param targets: target strings, potentially containing "re:" prefixes
    :param ignore: targets to ignore, potentially containing "re:" prefixes
    """
    matches = dict.fromkeys(targets, None)
    for name, module in model.named_modules():
        # match until we get a full set
        for target in targets:
            if is_match(name, module, target) and not any(
                is_match(name, module, ign) for ign in ignore
            ):
                if matches[target] is not None:
                    raise ValueError(f"Matched a {target} twice before completing set")
                matches[target] = module

        # once we have a full set, yield and reset
        if targets and all((matches[target] is not None for target in targets)):
            yield [matches[target] for target in targets]  # ensure correct ordering
            matches = dict.fromkeys(targets, None)

    # check that none are left over
    unmatched_keys = [match for match, value in matches.items() if value is not None]
    if len(unmatched_keys):
        raise ValueError(f"Unable to match targets into set: {unmatched_keys}")


def is_match(name: str, module: torch.nn.Module, target: str) -> bool:
    """
    Returns true if either module name or module parent classes match against target
    and the module is not an internal module
    """
    return not isinstance(module, InternalModule) and (
        _match_name(name, target) or _match_class(module, target)
    )


def _match_name(name: str, target: str) -> bool:
    """
    Returns true if target string begins with "re:" and
    regex matches or if target string exactly matches name
    """
    if target.startswith("re:"):
        return re.match(target.removeprefix("re:"), name) is not None
    else:
        return target == name


def _match_class(module: torch.nn.Module, target: str) -> bool:
    """
    Returns true if any torch parent class names match the target string exactly
    """
    # will never match against a regex pattern since `:` is not allowed in class names
    return any(
        issubclass(cls, torch.nn.Module) and cls.__name__ == target
        for cls in module.__class__.__mro__
    )
