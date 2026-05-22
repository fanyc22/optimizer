from __future__ import annotations

import re
from typing import Literal

from codesign_optimizer.models.hardware import ComponentLibrary, LinkTypeSpec


LinkScope = Literal["intra", "inter"]


def link_type_allowed_for_scope(
    library: ComponentLibrary,
    link_type: str | None,
    scope: LinkScope,
) -> bool:
    if link_type is None:
        return True
    spec = library.link_types.get(link_type)
    if spec is None:
        return False
    return link_spec_allowed_for_scope(spec, scope)


def link_spec_allowed_for_scope(spec: LinkTypeSpec, scope: LinkScope) -> bool:
    level = link_level_number(spec)
    if level is None:
        return False
    if scope == "inter":
        return level >= 4
    return level <= 3


def link_types_for_scope(library: ComponentLibrary, scope: LinkScope) -> list[str]:
    return [
        name
        for name, spec in library.link_types.items()
        if link_spec_allowed_for_scope(spec, scope)
    ]


def ordered_link_types_for_scope(library: ComponentLibrary, scope: LinkScope) -> list[str]:
    return sorted(
        link_types_for_scope(library, scope),
        key=lambda name: (
            library.link_types[name].bandwidth_gbps,
            -library.link_types[name].latency_ns,
            library.link_types[name].cost_unit,
            name,
        ),
    )


def link_level_number(spec: LinkTypeSpec) -> int | None:
    if not spec.level:
        return None
    match = re.fullmatch(r"[Ll](\d+)", str(spec.level).strip())
    if not match:
        return None
    return int(match.group(1))


def default_level_for_scope(scope: LinkScope) -> str:
    return "L4" if scope == "inter" else "L3"


def scope_label(scope: LinkScope) -> str:
    return "inter-rack" if scope == "inter" else "intra-rack"
