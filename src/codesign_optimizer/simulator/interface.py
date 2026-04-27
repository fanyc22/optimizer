from __future__ import annotations

from abc import ABC, abstractmethod

from codesign_optimizer.models.feedback import SimulationFeedback
from codesign_optimizer.models.hardware import HardwareProposal
from codesign_optimizer.models.workload import WorkloadSpec


class SimulatorClient(ABC):
    @abstractmethod
    def run(self, proposal: HardwareProposal, workload: WorkloadSpec) -> SimulationFeedback:
        """
        Submit proposal + workload to simulator and return feedback.
        """
