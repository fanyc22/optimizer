import json
from pathlib import Path

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import chromosome_from_template
from codesign_optimizer.optimizer.exporter import HardwareTopologyExporter
from codesign_optimizer.optimizer.repair import CandidateRepairer
from codesign_optimizer.optimizer.search_space import SearchSpace
from codesign_optimizer.optimizer.tgrl import enumerate_graph_edit_actions


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_enterprise_h100_3tier_example_is_feasible_and_cost_calibrated() -> None:
    library = ComponentLibrary.model_validate(
        _load_json(ROOT / "examples" / "component_catalog_enterprise_h100_3tier.json")
    )
    space = SearchSpace.model_validate(
        _load_json(ROOT / "examples" / "search_space_enterprise_h100_3tier_tgrl.json")
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
