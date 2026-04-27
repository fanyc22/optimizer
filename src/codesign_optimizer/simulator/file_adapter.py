from __future__ import annotations

from pathlib import Path

from codesign_optimizer.io.jsonc import dump_json, load_jsonc
from codesign_optimizer.models.feedback import SimulationFeedback
from codesign_optimizer.models.hardware import HardwareProposal
from codesign_optimizer.models.workload import WorkloadSpec
from codesign_optimizer.simulator.interface import SimulatorClient


class FileBackedSimulatorClient(SimulatorClient):
    """
    File-oriented adapter for environments where simulator exchange is done via JSON files.

    Current behavior:
    - Writes proposal to `proposal_out_path`
    - Reads feedback from a pre-generated `feedback_in_path`
    """

    def __init__(self, proposal_out_path: Path, feedback_in_path: Path) -> None:
        self._proposal_out_path = proposal_out_path
        self._feedback_in_path = feedback_in_path

    def run(self, proposal: HardwareProposal, workload: WorkloadSpec) -> SimulationFeedback:
        payload = proposal.to_dict()
        payload["requested_workload"] = workload.model_dump(mode="json")
        dump_json(self._proposal_out_path, payload)

        feedback_payload = load_jsonc(self._feedback_in_path)
        return SimulationFeedback.model_validate(feedback_payload)
