import json
from pathlib import Path

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import chromosome_from_template
from codesign_optimizer.optimizer.exporter import HardwareTopologyExporter
from codesign_optimizer.optimizer.repair import CandidateRepairer
from codesign_optimizer.optimizer.search_space import SearchSpace
from codesign_optimizer.optimizer.tgrl import GraphEditAction, apply_graph_edit_action, enumerate_graph_edit_actions


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_enterprise_h100_3tier_example_is_feasible_and_cost_calibrated() -> None:
    library = ComponentLibrary.model_validate(
        _load_json(ROOT / "examples" / "component_catalog_enterprise.json")
    )
    space = SearchSpace.model_validate(
        _load_json(ROOT / "examples" / "search_space_enterprise.json")
    )
    chromosome = chromosome_from_template(space.templates[0], host_templates=space.host_template_map())

    exported = HardwareTopologyExporter(library).export(chromosome)
    repair = CandidateRepairer(library, space).repair_and_validate(chromosome)

    assert repair.feasible, repair.messages
    assert exported.rank_count == 80
    assert 9_500_000 <= exported.proposal.total_estimated_cost() <= 10_500_000

    groups = {group["id"]: group for group in exported.hardware_topology["hierarchy"]["groups"]}
    assert groups["cluster0"]["level"] == "L4"
    assert groups["rack0"]["level"] == "L3"
    assert groups["rack0"]["attrs"]["rack_units"] == 25
    assert groups["rack0_host0"]["level"] == "L2"
    assert groups["rack0_host0"]["attrs"]["rack_units"] == 6

    action_types = {
        action.action_type
        for action in enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)
    }
    assert {"add_rack_from_template", "add_host_to_bay", "remove_host_from_bay", "replace_host_template"} <= action_types


def test_enterprise_h100_3tier_host_options_have_perf_cost_gradient() -> None:
    library = ComponentLibrary.model_validate(
        _load_json(ROOT / "examples" / "component_catalog_enterprise.json")
    )
    space = SearchSpace.model_validate(
        _load_json(ROOT / "examples" / "search_space_enterprise.json")
    )
    chromosome = chromosome_from_template(space.templates[0], host_templates=space.host_template_map())

    template_ids = {template.template_id for template in space.host_templates}
    assert {
        "xe9680_hgx_h200_8gpu_host",
        "xe9680_hgx_h100_8gpu_host",
        "pcie_h100_4gpu_4u_host",
        "l40s_8gpu_inference_4u_host",
        "l40s_4gpu_inference_2u_host",
        "l4_8gpu_inference_2u_host",
        "l4_4gpu_edge_inference_1u_host",
        "dual_xeon_cpu_2u_host",
    } <= template_ids

    actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)
    action_targets = {
        action.target
        for action in actions
        if action.action_type in {"add_host_to_bay", "replace_host_template"}
    }
    assert template_ids <= action_targets

    replacement_costs: dict[str, float] = {}
    for template_id in template_ids:
        replaced = apply_graph_edit_action(
            chromosome,
            GraphEditAction(
                "replace_host_template",
                rack_id="rack0",
                resource="host0",
                target=template_id,
            ),
            search_space=space,
        )
        repair = CandidateRepairer(library, space).repair_and_validate(replaced)
        assert repair.feasible, repair.messages
        replacement_costs[template_id] = HardwareTopologyExporter(library).export(replaced).proposal.total_estimated_cost()

    assert replacement_costs["xe9680_hgx_h200_8gpu_host"] > replacement_costs["xe9680_hgx_h100_8gpu_host"]
    assert replacement_costs["xe9680_hgx_h100_8gpu_host"] > replacement_costs["pcie_h100_4gpu_4u_host"]
    assert replacement_costs["pcie_h100_4gpu_4u_host"] > replacement_costs["l40s_8gpu_inference_4u_host"]
    assert replacement_costs["l40s_8gpu_inference_4u_host"] > replacement_costs["l4_8gpu_inference_2u_host"]


def test_enterprise_h100_3tier_has_multiple_4gpu_host_choices() -> None:
    library = ComponentLibrary.model_validate(
        _load_json(ROOT / "examples" / "component_catalog_enterprise.json")
    )
    space = SearchSpace.model_validate(
        _load_json(ROOT / "examples" / "search_space_enterprise.json")
    )
    chromosome = chromosome_from_template(space.templates[0], host_templates=space.host_template_map())

    four_gpu_templates = {
        template.template_id: template
        for template in space.host_templates
        if sum(1 for slot in template.slots if slot.node_type and slot.node_type.startswith("GPU_")) == 4
    }
    assert {
        "pcie_h100_4gpu_4u_host",
        "l40s_4gpu_inference_2u_host",
        "l4_4gpu_edge_inference_1u_host",
    } <= set(four_gpu_templates)

    actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)
    action_targets = {
        action.target
        for action in actions
        if action.action_type in {"add_host_to_bay", "replace_host_template"}
    }
    assert set(four_gpu_templates) <= action_targets

    replacement_costs: dict[str, float] = {}
    replacement_units: dict[str, float] = {}
    for template_id, template in four_gpu_templates.items():
        replaced = apply_graph_edit_action(
            chromosome,
            GraphEditAction(
                "replace_host_template",
                rack_id="rack0",
                resource="host0",
                target=template_id,
            ),
            search_space=space,
        )
        repair = CandidateRepairer(library, space).repair_and_validate(replaced)
        assert repair.feasible, repair.messages
        replacement_costs[template_id] = HardwareTopologyExporter(library).export(replaced).proposal.total_estimated_cost()
        replacement_units[template_id] = template.rack_units

    assert replacement_units["pcie_h100_4gpu_4u_host"] == 4
    assert replacement_units["l40s_4gpu_inference_2u_host"] == 2
    assert replacement_units["l4_4gpu_edge_inference_1u_host"] == 1
    assert replacement_costs["pcie_h100_4gpu_4u_host"] > replacement_costs["l40s_4gpu_inference_2u_host"]
    assert replacement_costs["l40s_4gpu_inference_2u_host"] > replacement_costs["l4_4gpu_edge_inference_1u_host"]


def test_enterprise_h100_2rack_search_space_keeps_exploration_headroom() -> None:
    library = ComponentLibrary.model_validate(
        _load_json(ROOT / "examples" / "component_catalog_enterprise.json")
    )
    space = SearchSpace.model_validate(
        _load_json(ROOT / "examples" / "search_space_enterprise.json")
    )
    chromosome = chromosome_from_template(space.templates[0], host_templates=space.host_template_map())

    exported = HardwareTopologyExporter(library).export(chromosome)
    repair = CandidateRepairer(library, space).repair_and_validate(chromosome)

    assert repair.feasible, repair.messages
    assert len(chromosome.racks) == 2
    assert all(len(rack.occupied_hosts) == 4 for rack in chromosome.racks)
    assert exported.rank_count == 80
    assert 9_500_000 <= exported.proposal.total_estimated_cost() <= 10_500_000
    assert exported.proposal.total_estimated_cost() < space.limits.max_total_cost

    groups = {group["id"]: group for group in exported.hardware_topology["hierarchy"]["groups"]}
    assert groups["rack0"]["attrs"]["rack_units"] == 25
    assert groups["rack1"]["attrs"]["rack_units"] == 25

    actions = enumerate_graph_edit_actions(chromosome, component_library=library, search_space=space)
    assert any(action.action_type == "add_rack_from_template" for action in actions)
    assert any(
        action.action_type == "change_intra_rack_topology"
        and action.rack_id == "rack0"
        and action.target == "ring"
        for action in actions
    )
    assert any(
        action.action_type == "change_inter_rack_topology"
        and action.target == "fully_connected"
        for action in actions
    )
    assert any(
        action.action_type == "add_host_to_bay"
        and action.resource == "host4"
        and action.target == "l40s_4gpu_inference_2u_host"
        for action in actions
    )
    assert any(
        action.action_type == "replace_host_template"
        and action.resource == "host0"
        and action.target == "xe9680_hgx_h200_8gpu_host"
        for action in actions
    )

    add_rack = next(action for action in actions if action.action_type == "add_rack_from_template")
    expanded = apply_graph_edit_action(chromosome, add_rack, search_space=space)
    expanded_repair = CandidateRepairer(library, space).repair_and_validate(expanded)
    assert expanded_repair.feasible, expanded_repair.messages

    rack_topology_action = next(
        action
        for action in actions
        if action.action_type == "change_intra_rack_topology"
        and action.rack_id == "rack0"
        and action.target == "ring"
    )
    rack_topology_changed = apply_graph_edit_action(chromosome, rack_topology_action, search_space=space)
    assert rack_topology_changed.racks[0].intra_rack_topology == "ring"
    rack_topology_repair = CandidateRepairer(library, space).repair_and_validate(rack_topology_changed)
    assert rack_topology_repair.feasible, rack_topology_repair.messages

    inter_topology_action = next(
        action
        for action in actions
        if action.action_type == "change_inter_rack_topology"
        and action.target == "fully_connected"
    )
    inter_topology_changed = apply_graph_edit_action(chromosome, inter_topology_action, search_space=space)
    assert inter_topology_changed.inter_rack == "fully_connected"
    inter_topology_repair = CandidateRepairer(library, space).repair_and_validate(inter_topology_changed)
    assert inter_topology_repair.feasible, inter_topology_repair.messages
