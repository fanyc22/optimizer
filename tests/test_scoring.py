import pytest

from codesign_optimizer.optimizer.scoring import budget_wall_pressure, weighted_score_from_objectives
from codesign_optimizer.optimizer.search_space import SearchLimits, SearchObjectiveWeights


def test_budget_wall_pressure_uses_quadratic_wall_after_knee() -> None:
    weights = SearchObjectiveWeights()

    assert budget_wall_pressure(700.0, 1000.0, weights=weights) == pytest.approx(0.0007)
    assert budget_wall_pressure(800.0, 1000.0, weights=weights) == pytest.approx(0.0008)
    assert budget_wall_pressure(900.0, 1000.0, weights=weights) == pytest.approx(0.2509)
    assert budget_wall_pressure(1000.0, 1000.0, weights=weights) == pytest.approx(1.001)


def test_weighted_score_uses_budget_limits_for_cost_and_power_wall() -> None:
    weights = SearchObjectiveWeights()
    limits = SearchLimits(max_total_cost=1000.0, max_peak_power_watts=100.0)
    objectives = (10_000.0, 900.0, 90.0, 0.0, 0.0, 0.0)

    score = weighted_score_from_objectives(
        objectives,
        weights=weights,
        limits=limits,
        feasible=True,
        penalty=0.0,
    )

    expected = 1.0 + weights.cost * 0.2509 + weights.power * 0.2509
    assert score == pytest.approx(expected)
