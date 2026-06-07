import json
from pathlib import Path

import pytest

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import chromosome_from_template
from codesign_optimizer.optimizer.exporter import HardwareTopologyExporter
from codesign_optimizer.optimizer.repair import CandidateRepairer
from codesign_optimizer.optimizer.search_space import SearchSpace
from codesign_optimizer.optimizer.tgrl import (
    apply_graph_edit_action,
    enumerate_graph_edit_actions,
    GraphEditAction,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _library() -> ComponentLibrary:
    return ComponentLibrary.model_validate(_load_json(ROOT / "examples" / "component_catalog_host_templates.json"))


def _space() -> SearchSpace:
    return SearchSpace.model_validate(_load_json(ROOT / "examples" / "search_space_host_template_tgrl.json"))


def test_host_template_search_space_parses_and_legacy_slots_still_parse() -> None:
    space = _space()
    assert space.mutation.search_granularity == "host"
    assert {template.template_id for template in space.host_templates} == {
        "pcie_2cpu_4gpu_host",
        "nvlink_2cpu_8gpu_host",
        "cpu_2socket_host",
    }
    assert space.host_template_map()["pcie_2cpu_4gpu_host"].rack_units == 4
    assert space.host_template_map()["nvlink_2cpu_8gpu_host"].rack_units == 8

    legacy = SearchSpace.model_validate(
        {
            "templates": [
                {
                    "name": "legacy",
                    "racks": [
                        {
                            "rack_id": "rack0",
                            "role": "compute",
                            "max_slots": 1,
                            "slots": [{"slot_id": "gpu0", "node_type": "GPU"}],
                            "switch_count": 1,
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                        }
                    ],
                    "inter_rack": "none",
                }
            ]
        }
    )
    assert legacy.templates[0].racks[0].slots[0].slot_id == "gpu0"
    assert legacy.mutation.search_granularity == "slot"

    oversized = _load_json(ROOT / "examples" / "search_space_host_template_tgrl.json")
    oversized.pop("limits")
    oversized["templates"][0]["racks"][0]["limits"].pop("max_rack_units")
    oversized["host_templates"][0]["rack_units"] = 41
    with pytest.raises(ValueError, match="host rack_units exceed limit"):
        SearchSpace.model_validate(oversized)


def test_exporter_lowers_hosts_to_l2_groups_and_links() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0], host_templates=space.host_template_map())
    exported = HardwareTopologyExporter(library).export(chromosome)
    topology = exported.hardware_topology

    group_ids = {group["id"]: group for group in topology["hierarchy"]["groups"]}
    assert group_ids["rack0_host0"]["level"] == "L2"
    assert group_ids["rack0_host2"]["parent"] == "rack0"
    assert group_ids["rack0"]["attrs"]["rack_units"] == 7
    assert group_ids["rack0_host0"]["attrs"]["rack_units"] == 4
    assert exported.rank_count == 8
    assert topology["rank_map"][0]["node_id"] == "rack0_host0_cpu0"

    rank_node = next(node for node in topology["nodes"] if node["id"] == "rack0_host0_gpu0")
    assert rank_node["parent"] == "rack0_host0"
    assert rank_node["level"] == "L2"
    assert rank_node["attrs"]["host_template_id"] == "pcie_2cpu_4gpu_host"

    host_link = next(link for link in topology["links"] if link["id"] == "rack0_host0_gpu0_to_rack0_host0_sw0")
    rack_link = next(link for link in topology["links"] if link["id"] == "rack0_host0_sw0_to_rack0_sw0")
    assert host_link["level"] == "L2"
    assert host_link["domain"] == "host:rack0/host0"
    assert rack_link["level"] == "L3"
    assert rack_link["domain"] == "rack:rack0"

    repair = CandidateRepairer(library, space).repair_and_validate(chromosome)
    assert repair.feasible, repair.messages


def test_tgrl_host_mode_only_enumerates_and_applies_host_actions() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0], host_templates=space.host_template_map())
    actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)

    action_types = {action.action_type for action in actions}
    assert action_types <= {
        "add_host_to_bay",
        "remove_host_from_bay",
        "replace_host_template",
        "change_intra_rack_topology",
        "change_inter_rack_topology",
        "add_rack_from_template",
        "remove_rack",
    }
    assert any(action.action_type == "add_rack_from_template" and action.target == "pcie_compute_leaf" for action in actions)
    assert any(action.action_type == "change_intra_rack_topology" and action.target == "ring" for action in actions)
    assert any(
        action.action_type == "change_intra_rack_topology" and action.target == "fully_connected"
        for action in actions
    )
    assert not any(action.action_type == "add_node_to_slot" for action in actions)
    assert not any(action.action_type == "replace_node_type" for action in actions)
    assert any(action.action_type == "add_host_to_bay" and action.resource == "host1" for action in actions)
    assert any(action.action_type == "remove_host_from_bay" and action.resource == "host0" for action in actions)
    assert any(action.action_type == "replace_host_template" and action.resource == "host0" for action in actions)

    add_host = next(
        action
        for action in actions
        if action.action_type == "add_host_to_bay"
        and action.resource == "host1"
        and action.target == "nvlink_2cpu_8gpu_host"
    )
    updated = apply_graph_edit_action(chromosome, add_host, search_space=space)
    rack = updated.racks[0]
    host1 = next(host for host in rack.hosts if host.host_id == "host1")
    assert host1.template_id == "nvlink_2cpu_8gpu_host"
    assert len(host1.occupied_slots) == 10

    exported = HardwareTopologyExporter(library).export(updated)
    assert exported.rank_count == 18

    change_intra = next(
        action
        for action in actions
        if action.action_type == "change_intra_rack_topology" and action.target == "ring"
    )
    topology_updated = apply_graph_edit_action(chromosome, change_intra, search_space=space)
    assert topology_updated.racks[0].intra_rack_topology == "ring"
    repair = CandidateRepairer(library, space).repair_and_validate(topology_updated)
    assert repair.feasible, repair.messages


def test_tgrl_host_mode_filters_host_actions_by_rack_units() -> None:
    library = _library()
    payload = _load_json(ROOT / "examples" / "search_space_host_template_tgrl.json")
    payload["templates"][0]["racks"][0]["limits"]["max_rack_units"] = 14
    space = SearchSpace.model_validate(payload)
    chromosome = chromosome_from_template(space.templates[0], host_templates=space.host_template_map())

    actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)

    assert not any(
        action.action_type == "add_host_to_bay"
        and action.resource == "host1"
        and action.target == "nvlink_2cpu_8gpu_host"
        for action in actions
    )
    assert any(
        action.action_type == "add_host_to_bay"
        and action.resource == "host1"
        and action.target == "pcie_2cpu_4gpu_host"
        for action in actions
    )

    forced = apply_graph_edit_action(
        chromosome,
        GraphEditAction(
            "add_host_to_bay",
            rack_id="rack0",
            resource="host1",
            target="nvlink_2cpu_8gpu_host",
        ),
        search_space=space,
    )
    repair = CandidateRepairer(library, space).repair_and_validate(forced)
    assert not repair.feasible
    assert any("rack units exceed limit" in message for message in repair.messages)


def test_host_rack_unit_capacity_replaces_rank_slot_limit() -> None:
    library = _library()
    payload = _load_json(ROOT / "examples" / "search_space_host_template_tgrl.json")
    payload["templates"][0]["racks"][0]["max_slots"] = 8
    payload["templates"][0]["racks"][0]["limits"]["max_slots"] = 8
    space = SearchSpace.model_validate(payload)
    chromosome = chromosome_from_template(space.templates[0], host_templates=space.host_template_map())

    action = next(
        action
        for action in enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)
        if action.action_type == "add_host_to_bay"
        and action.resource == "host1"
        and action.target == "nvlink_2cpu_8gpu_host"
    )
    updated = apply_graph_edit_action(chromosome, action, search_space=space)

    assert len(updated.racks[0].occupied_slots) == 18
    repair = CandidateRepairer(library, space).repair_and_validate(updated)
    assert repair.feasible, repair.messages


def test_tgrl_host_mode_can_add_and_remove_dynamic_rack_from_template() -> None:
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0], host_templates=space.host_template_map())
    add_rack = next(
        action
        for action in enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)
        if action.action_type == "add_rack_from_template" and action.target == "pcie_compute_leaf"
    )

    added = apply_graph_edit_action(chromosome, add_rack, search_space=space)
    dynamic_rack = next(rack for rack in added.racks if rack.dynamic)
    assert dynamic_rack.rack_id.startswith("dyn-pcie-compute-leaf-")
    assert dynamic_rack.hosts[0].template_id == "pcie_2cpu_4gpu_host"
    assert added.inter_rack == "ring"

    exported = HardwareTopologyExporter(library).export(added)
    rack_groups = {group["id"] for group in exported.hardware_topology["hierarchy"]["groups"] if group["level"] == "L3"}
    assert dynamic_rack.rack_id in rack_groups
    assert exported.rank_count == 14

    remove_actions = enumerate_graph_edit_actions(added, component_library=library, search_space=space)
    remove_rack = next(
        action
        for action in remove_actions
        if action.action_type == "remove_rack" and action.rack_id == dynamic_rack.rack_id
    )
    removed = apply_graph_edit_action(added, remove_rack, search_space=space)
    assert all(not rack.dynamic for rack in removed.racks)
