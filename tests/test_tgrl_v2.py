import json
import math
from pathlib import Path
import tempfile
import threading
import time

import pytest

torch = pytest.importorskip("torch")

from codesign_optimizer.models.hardware import ComponentLibrary
from codesign_optimizer.optimizer.chromosome import chromosome_from_template
from codesign_optimizer.optimizer.exporter import HardwareTopologyExporter
from codesign_optimizer.optimizer.feedback_parser import ParsedPipelineFeedback, parse_pipeline_feedback
from codesign_optimizer.optimizer.repair import CandidateRepairer
from codesign_optimizer.optimizer.scoring import budget_wall_pressure
from codesign_optimizer.optimizer.search_space import SearchSpace
from codesign_optimizer.optimizer.tgrl import TGRLConfig, build_masked_actions
from codesign_optimizer.optimizer.tgrl_v2 import ppo as ppo_module
from codesign_optimizer.optimizer.tgrl_v2.model import (
    TensorObservation,
    TGRLGNNPolicy,
    batch_tensor_observations,
    policy_distribution,
    policy_logits_from_batched_tensor,
    tensorize_observation,
)
from codesign_optimizer.optimizer.tgrl_v2.observation import (
    ACTION_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphObservationBuilder,
)
from codesign_optimizer.optimizer.tgrl_v2.ppo import PPOConfig, PPOTransition, attach_gae, ppo_update
from codesign_optimizer.optimizer.tgrl_v2.trainer import (
    TGRLPPOConfig,
    TGRLPPOTrainer,
    _estimated_finite_candidate_count,
    _write_svg_lines,
)
from codesign_optimizer.optimizer.workload_suite import (
    WorkloadRunFeedback,
    WorkloadSuite,
    WorkloadSuiteBaseline,
    aggregate_multi_workload_feedback,
)


class FakePipeline:
    def __init__(self, *, remote_queue: int = 0, delay_s: float = 0.01) -> None:
        self.calls = 0
        self.remote_queue = remote_queue
        self.delay_s = delay_s
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def run(
        self,
        *,
        topology_path: Path,
        workload_path: Path,
        out_dir: Path,
        workload_rank_parallel: bool | None = None,
        workload_kind: str | None = None,
        llm_use_all_gpus: bool | None = None,
    ) -> ParsedPipelineFeedback:
        out_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            call_id = self.calls
        try:
            time.sleep(self.delay_s)
            makespan = max(100, 1000 - call_id * 20)
            summary = {
                "case_name": topology_path.stem,
                "success": True,
                "inputs": {"workload": str(workload_path)},
                "simulator": {"finished_count": 1, "expected_finished_count": 1},
            }
            stdout = f"""
            [x] [statistics] [info] sys[0], Wall time: {makespan}
            [x] [statistics] [info] sys[0], Average compute utilization: 90.000%
            [x] [statistics] [info] sys[0], Remote mem provider queue time: {self.remote_queue}
            [x] [network] [info] Network top congested link rank=1 id=rack0_sw0_to_rack1_sw0 src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns=900000 transmissions=2 max_queue_depth=3 utilization=0.800000
            """
            return parse_pipeline_feedback(summary=summary, simulator_stdout=stdout)
        finally:
            with self._lock:
                self.active -= 1


def _library() -> ComponentLibrary:
    return ComponentLibrary.model_validate(
        {
            "node_types": {
                "GPU": {"role": "gpu", "peak_tflops": 80, "memory_bw_gbps": 1800, "tdp_watts": 700, "cost_unit": 20000},
                "CPU": {"role": "cpu", "peak_tflops": 6, "memory_bw_gbps": 220, "tdp_watts": 350, "cost_unit": 6000},
                "MEM": {"role": "memory_pool", "capacity_gb": 1024, "memory_bw_gbps": 320, "tdp_watts": 250, "cost_unit": 12000},
                "SW": {"role": "switch", "radix": 64, "tdp_watts": 180, "cost_unit": 8000},
            },
            "link_types": {
                "FAST": {
                    "bandwidth_gbps": 100,
                    "latency_ns": 100,
                    "protocol": "NVLink",
                    "level": "L3",
                    "cost_unit": 1000,
                },
                "CXL": {
                    "bandwidth_gbps": 64,
                    "latency_ns": 250,
                    "protocol": "CXL",
                    "level": "L3",
                    "cost_unit": 300,
                },
                "OPTICAL": {"bandwidth_gbps": 400, "latency_ns": 800, "protocol": "Optical", "level": "L4", "cost_unit": 3000},
            },
        }
    )


def _space() -> SearchSpace:
    return SearchSpace.model_validate(
        {
            "seed": 31,
            "templates": [
                {
                    "name": "v2_small",
                    "racks": [
                        {
                            "rack_id": "rack0",
                            "role": "hybrid",
                            "max_slots": 4,
                            "slots": [
                                {"slot_id": "slot0", "node_type": "GPU"},
                                {"slot_id": "slot1", "node_type": "CPU"},
                                {"slot_id": "slot2"},
                                {"slot_id": "slot3"},
                            ],
                            "memory_pool_count": 1,
                            "switch_count": 1,
                            "memory_pool_type": "MEM",
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "memory_link_type": "CXL",
                            "limits": {"max_slots": 4, "max_memory_pool_count": 2, "max_switch_count": 1},
                        },
                        {
                            "rack_id": "latent-mem",
                            "role": "memory",
                            "optional": True,
                            "active": False,
                            "max_slots": 0,
                            "slots": [],
                            "memory_pool_type": "MEM",
                            "switch_type": "SW",
                            "intra_rack_topology": "switch",
                            "intra_rack_link_type": "FAST",
                            "memory_link_type": "CXL",
                            "limits": {"max_slots": 0, "max_memory_pool_count": 2, "max_switch_count": 1},
                        },
                    ],
                    "inter_rack": "ring",
                    "inter_rack_link_type": "OPTICAL",
                }
            ],
            "mutation": {
                "max_intra_rack_link_qty": 3,
                "max_inter_rack_link_qty": 3,
            },
            "limits": {"max_total_cost": 300000, "max_peak_power_watts": 20000, "max_rack_power_watts": 10000, "max_rack_units": 42},
        }
    )


def _observation():
    library = _library()
    space = _space()
    chromosome = chromosome_from_template(space.templates[0])
    repairer = CandidateRepairer(library, space)
    repair = repairer.repair_and_validate(chromosome)
    feedback = FakePipeline(remote_queue=2_000_000).run(
        topology_path=Path("topology.json"),
        workload_path=Path("workload.json"),
        out_dir=Path(tempfile.gettempdir()) / "tgrl_v2_fake",
    )
    masked = build_masked_actions(
        chromosome,
        component_library=library,
        search_space=space,
        repairer=repairer,
        exporter=HardwareTopologyExporter(library),
        feedback=feedback,
        current_repair=repair,
        policy=None,
        config=TGRLConfig(),
    )
    return GraphObservationBuilder(library, space).build(
        chromosome=repair.chromosome,
        repair=repair,
        feedback=feedback,
        masked_actions=masked,
        current_score=10.0,
        best_score=10.0,
        update=0,
        step=0,
        total_updates=1,
        rollout_steps=1,
    )


def _feedback_with_makespan(makespan: float, *, workload: str) -> ParsedPipelineFeedback:
    summary = {
        "case_name": "fake",
        "success": True,
        "inputs": {"workload": workload},
        "simulator": {"finished_count": 1, "expected_finished_count": 1},
    }
    stdout = f"""
    [x] [statistics] [info] sys[0], Wall time: {makespan}
    [x] [statistics] [info] sys[0], Average compute utilization: 90.000%
    [x] [network] [info] Network top congested link rank=1 id=rack0_sw0_to_rack1_sw0 src_device=0 dst_device=1 level=L4 domain=cluster:cluster0 stats_domain=cluster:cluster0 technology=optical route_class= bytes=4096 busy_time_ns=80 queue_delay_ns=900000 transmissions=2 max_queue_depth=3 utilization=0.800000
    """
    return parse_pipeline_feedback(summary=summary, simulator_stdout=stdout)


def _suite_feedback(*, speedup_a: float, speedup_b: float):
    suite = WorkloadSuite.model_validate(
        {
            "name": "mixed",
            "workloads": [
                {"name": "a", "path": "a.json", "weight": 0.5},
                {"name": "b", "path": "b.json", "weight": 0.5},
            ],
        }
    )
    baseline = WorkloadSuiteBaseline(suite_name="mixed", makespans_us={"a": 1000.0, "b": 1000.0})
    return aggregate_multi_workload_feedback(
        suite,
        [
            WorkloadRunFeedback(
                name="a",
                path=Path("a.json"),
                weight=0.5,
                workload_rank_parallel=False,
                out_dir=Path("/tmp/a"),
                feedback=_feedback_with_makespan(int(round(1000.0 / speedup_a)), workload="a"),
                speedup=speedup_a,
            ),
            WorkloadRunFeedback(
                name="b",
                path=Path("b.json"),
                weight=0.5,
                workload_rank_parallel=False,
                out_dir=Path("/tmp/b"),
                feedback=_feedback_with_makespan(int(round(1000.0 / speedup_b)), workload="b"),
                speedup=speedup_b,
            ),
        ],
        baseline,
    )


def test_finite_candidate_count_skips_host_granularity() -> None:
    fixture = Path(__file__).resolve().parents[1] / "examples" / "search_space_host_template.json"
    space = SearchSpace.model_validate(json.loads(fixture.read_text()))

    assert space.mutation.search_granularity == "host"
    assert (
        _estimated_finite_candidate_count(
            space,
            freeze_topology=False,
            allow_empty_slots=True,
        )
        is None
    )


def test_observation_builder_shapes() -> None:
    observation = _observation()

    assert observation.node_features
    assert len(observation.node_features[0]) == NODE_FEATURE_DIM
    assert len(observation.edge_features[0]) == EDGE_FEATURE_DIM
    assert len(observation.global_features) == GLOBAL_FEATURE_DIM
    assert len(observation.action_features[0]) == ACTION_FEATURE_DIM
    assert len(observation.action_features) == len(observation.masked_actions)
    assert any(action.action.action_type == "activate_optional_rack" for action in observation.masked_actions)


def test_tgrl_v2_reward_uses_workload_normalized_log_improvement(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    trainer = TGRLPPOTrainer(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=FakePipeline(delay_s=0.0),
        workload_path=workload,
        out_dir=tmp_path / "reward",
        updates=1,
        rollout_steps=1,
        env_count=1,
        config=TGRLPPOConfig(
            best_improvement_bonus=0.0,
            duplicate_penalty=0.0,
            device="cpu",
        ),
    )

    baseline = _suite_feedback(speedup_a=1.0, speedup_b=1.0)
    improved = _suite_feedback(speedup_a=2.0, speedup_b=1.0)
    worsened = _suite_feedback(speedup_a=0.5, speedup_b=1.0)

    reward = trainer._reward(
        previous_score=1.0,
        new_score=0.9,
        feasible=True,
        signature="improved",
        previous_suite_feedback=baseline,
        new_suite_feedback=improved,
    )
    negative_reward = trainer._reward(
        previous_score=1.0,
        new_score=1.1,
        feasible=True,
        signature="worsened",
        previous_suite_feedback=baseline,
        new_suite_feedback=worsened,
    )

    assert pytest.approx(reward) == math.log(improved.geomean_speedup / baseline.geomean_speedup)
    assert pytest.approx(negative_reward) == math.log(worsened.geomean_speedup / baseline.geomean_speedup)


def test_model_forward_distribution_and_ppo_update() -> None:
    device = torch.device("cpu")
    observation = _observation()
    model = TGRLGNNPolicy().to(device)
    dist, logits, value, _tensor_observation = policy_distribution(
        model,
        observation,
        device=device,
        heuristic_weight=1.0,
    )

    assert logits.shape[0] == len(observation.masked_actions)
    assert value.ndim == 0
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value)
    assert torch.allclose(dist.probs.sum(), torch.tensor(1.0), atol=1e-6)

    before = [param.detach().clone() for param in model.parameters()]
    transition = PPOTransition(
        observation=observation,
        action_index=0,
        old_logprob=float(dist.log_prob(torch.tensor(0)).item()),
        value=float(value.item()),
        reward=1.0,
        done=False,
        candidate_signature="sig",
        episode_env=0,
        rollout_step=0,
    )
    transitions = attach_gae([[transition]], gamma=0.95, gae_lambda=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = ppo_update(
        model=model,
        optimizer=optimizer,
        transitions=transitions,
        config=PPOConfig(ppo_epochs=1, minibatch_size=1),
        device=device,
        rng=__import__("random").Random(1),
    )

    assert metrics["updates"] > 0
    assert any(not torch.allclose(old, new) for old, new in zip(before, model.parameters(), strict=True))


def test_batched_model_forward_matches_single_forward_and_keeps_checkpoint_keys() -> None:
    device = torch.device("cpu")
    observation = tensorize_observation(_observation(), device)
    small_observation = TensorObservation(
        node_features=torch.stack(
            [
                torch.arange(NODE_FEATURE_DIM, dtype=torch.float32, device=device) / 100.0,
                torch.arange(NODE_FEATURE_DIM, dtype=torch.float32, device=device).flip(0) / 100.0,
                torch.ones(NODE_FEATURE_DIM, dtype=torch.float32, device=device) * 0.25,
            ],
            dim=0,
        ),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long, device=device),
        edge_features=torch.stack(
            [
                torch.arange(EDGE_FEATURE_DIM, dtype=torch.float32, device=device) / 10.0,
                torch.ones(EDGE_FEATURE_DIM, dtype=torch.float32, device=device) * 0.5,
                torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float32, device=device),
            ],
            dim=0,
        ),
        global_features=torch.arange(GLOBAL_FEATURE_DIM, dtype=torch.float32, device=device) / 50.0,
        action_features=torch.stack(
            [
                torch.arange(ACTION_FEATURE_DIM, dtype=torch.float32, device=device) / 100.0,
                torch.ones(ACTION_FEATURE_DIM, dtype=torch.float32, device=device) * 0.1,
            ],
            dim=0,
        ),
        action_target_indices=torch.tensor([0, 2], dtype=torch.long, device=device),
        heuristic_logits=torch.tensor([0.25, -0.5], dtype=torch.float32, device=device),
    )
    model = TGRLGNNPolicy().to(device)
    checkpoint_keys = tuple(model.state_dict().keys())

    single_logits_a, single_value_a = model(observation)
    single_logits_a = single_logits_a + observation.heuristic_logits
    single_logits_b, single_value_b = model(small_observation)
    single_logits_b = single_logits_b + small_observation.heuristic_logits
    batched = batch_tensor_observations([observation, small_observation])
    batched_logits, batched_values = policy_logits_from_batched_tensor(
        model,
        batched,
        heuristic_weight=1.0,
    )

    split = int(observation.action_features.shape[0])
    torch.testing.assert_close(batched_logits[:split], single_logits_a, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched_logits[split:], single_logits_b, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched_values[0], single_value_a, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batched_values[1], single_value_b, rtol=1e-5, atol=1e-6)
    assert tuple(model.state_dict().keys()) == checkpoint_keys
    reloaded = TGRLGNNPolicy().to(device)
    reloaded.load_state_dict(model.state_dict(), strict=True)


def test_ppo_update_tensorizes_each_transition_once(monkeypatch: pytest.MonkeyPatch) -> None:
    device = torch.device("cpu")
    observation = _observation()
    model = TGRLGNNPolicy().to(device)
    dist, _logits, value, _tensor_observation = policy_distribution(
        model,
        observation,
        device=device,
        heuristic_weight=1.0,
    )
    transitions = attach_gae(
        [
            [
                PPOTransition(
                    observation=observation,
                    action_index=idx,
                    old_logprob=float(dist.log_prob(torch.tensor(idx)).item()),
                    value=float(value.item()),
                    reward=1.0 + idx,
                    done=False,
                    candidate_signature=f"sig-{idx}",
                    episode_env=0,
                    rollout_step=idx,
                )
                for idx in range(2)
            ]
        ],
        gamma=0.95,
        gae_lambda=0.9,
    )

    calls = 0
    original = ppo_module.tensorize_observation

    def counted_tensorize(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ppo_module, "tensorize_observation", counted_tensorize)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ppo_update(
        model=model,
        optimizer=optimizer,
        transitions=transitions,
        config=PPOConfig(ppo_epochs=3, minibatch_size=1),
        device=device,
        rng=__import__("random").Random(1),
    )

    assert calls == len(transitions)


def test_ppo_update_uses_one_batched_forward_per_minibatch() -> None:
    class CountingPolicy(TGRLGNNPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.batched_calls = 0

        def forward_batched(self, observation):
            self.batched_calls += 1
            return super().forward_batched(observation)

    device = torch.device("cpu")
    observation = _observation()
    model = CountingPolicy().to(device)
    dist, _logits, value, _tensor_observation = policy_distribution(
        model,
        observation,
        device=device,
        heuristic_weight=1.0,
    )
    transitions = attach_gae(
        [
            [
                PPOTransition(
                    observation=observation,
                    action_index=idx,
                    old_logprob=float(dist.log_prob(torch.tensor(idx)).item()),
                    value=float(value.item()),
                    reward=1.0 + idx,
                    done=False,
                    candidate_signature=f"sig-{idx}",
                    episode_env=0,
                    rollout_step=idx,
                )
                for idx in range(2)
            ]
        ],
        gamma=0.95,
        gae_lambda=0.9,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ppo_update(
        model=model,
        optimizer=optimizer,
        transitions=transitions,
        config=PPOConfig(ppo_epochs=3, minibatch_size=2),
        device=device,
        rng=__import__("random").Random(1),
    )

    assert model.batched_calls == 3


def test_tgrl_v2_trainer_smoke_and_checkpoint_resume(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    out_dir = tmp_path / "v2"
    trainer = TGRLPPOTrainer(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=FakePipeline(remote_queue=2_000_000),
        workload_path=workload,
        out_dir=out_dir,
        updates=1,
        rollout_steps=2,
        env_count=2,
        config=TGRLPPOConfig(ppo_epochs=1, minibatch_size=2, device="cpu"),
    )
    result = trainer.run()

    assert result.best.feasible
    assert (out_dir / "tgrl_summary.json").exists()
    assert (out_dir / "trajectory.jsonl").exists()
    assert (out_dir / "checkpoints" / "policy_latest.pt").exists()
    assert (out_dir / "best_hardware_topology.json").exists()
    assert (out_dir / "curves" / "candidate_scores.csv").exists()
    assert (out_dir / "curves" / "update_scores.json").exists()
    assert (out_dir / "curves" / "ppo_metrics.csv").exists()
    assert (out_dir / "curves" / "score_curve.svg").exists()
    assert (out_dir / "curves" / "ppo_loss_curve.svg").exists()

    resume = out_dir / "checkpoints" / "policy_latest.pt"
    checkpoint_before_resume = torch.load(resume, map_location="cpu", weights_only=False)
    update0_scores = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((out_dir / "update_000").glob("env_*/initial/score.json"))
        + sorted((out_dir / "update_000").glob("env_*/step_*/score.json"))
    ]
    update0_best = min(update0_scores, key=lambda item: item["weighted_score"])
    assert checkpoint_before_resume["rollout_seed_chromosome"] == update0_best["chromosome"]

    trainer_resume = TGRLPPOTrainer(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=FakePipeline(),
        workload_path=workload,
        out_dir=out_dir,
        updates=1,
        rollout_steps=1,
        env_count=1,
        config=TGRLPPOConfig(ppo_epochs=1, minibatch_size=1, device="cpu", resume=resume),
    )
    resumed = trainer_resume.run()

    assert resumed.best.feasible
    assert (out_dir / "update_001").exists()
    update1_initial = json.loads(
        (out_dir / "update_001" / "env_000" / "initial" / "score.json").read_text(encoding="utf-8")
    )
    assert update1_initial["chromosome"] == checkpoint_before_resume["rollout_seed_chromosome"]
    latest = torch.load(out_dir / "checkpoints" / "policy_latest.pt", map_location="cpu", weights_only=False)
    assert latest["update"] == 1
    assert "rng_state" in latest
    assert "seen_signatures" in latest
    assert latest["seed_archive"]
    assert latest["rollout_seed_chromosome"] is not None
    update_rows = json.loads((out_dir / "curves" / "update_scores.json").read_text(encoding="utf-8"))["rows"]
    metric_rows = json.loads((out_dir / "curves" / "ppo_metrics.json").read_text(encoding="utf-8"))["rows"]
    candidate_rows = json.loads((out_dir / "curves" / "candidate_scores.json").read_text(encoding="utf-8"))["rows"]
    assert [row["update"] for row in update_rows] == [0, 1]
    assert [row["update"] for row in metric_rows] == [0, 1]
    assert max(row["update"] for row in candidate_rows) == 1


def test_tgrl_v2_diversifies_parallel_env_initial_seeds(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    trainer = TGRLPPOTrainer(
        component_library=_library(),
        search_space=_space(),
        pipeline_client=FakePipeline(delay_s=0.0),
        workload_path=workload,
        out_dir=tmp_path / "diverse",
        updates=1,
        rollout_steps=1,
        env_count=4,
        config=TGRLPPOConfig(
            ppo_epochs=1,
            minibatch_size=1,
            device="cpu",
            seed_diversity_steps=1,
            seed_diversity_attempts=8,
        ),
    )

    envs = trainer._initialize_envs(0)

    signatures = {env.initial_evaluation.chromosome.signature() for env in envs}
    assert len(envs) == 4
    assert len(signatures) > 1


def test_tgrl_v2_masks_seen_action_candidates(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    workload.write_text("{}", encoding="utf-8")
    library = _library()
    space = _space()
    trainer = TGRLPPOTrainer(
        component_library=library,
        search_space=space,
        pipeline_client=FakePipeline(delay_s=0.0),
        workload_path=workload,
        out_dir=tmp_path / "mask",
        updates=1,
        rollout_steps=1,
        env_count=1,
        config=TGRLPPOConfig(
            ppo_epochs=1,
            minibatch_size=1,
            device="cpu",
            mask_seen_actions=True,
            restart_on_stall=False,
        ),
    )
    env = trainer._initialize_envs(0)[0]
    repairer = CandidateRepairer(library, space)
    repair = repairer.repair_and_validate(env.chromosome)
    actions = build_masked_actions(
        env.chromosome,
        component_library=library,
        search_space=space,
        repairer=repairer,
        exporter=HardwareTopologyExporter(library),
        feedback=env.last_feedback,
        current_repair=repair,
        policy=None,
        config=TGRLConfig(),
    )
    assert len(actions) > 1
    expected_signature = actions[-1].chromosome.signature()
    for action in actions[:-1]:
        trainer._seen_signatures.add(action.chromosome.signature())

    selections = trainer._select_actions(
        [env],
        update=0,
        update_position=0,
        step=0,
        best_score=env.previous_score,
    )

    assert len(selections) == 1
    _env, selected, transition = selections[0]
    assert selected.chromosome.signature() == expected_signature
    assert len(transition.observation.masked_actions) == 1


def test_tgrl_v2_trainer_runs_workload_suite(tmp_path: Path) -> None:
    workload_a = tmp_path / "a.json"
    workload_b = tmp_path / "b.json"
    workload_a.write_text("{}", encoding="utf-8")
    workload_b.write_text("{}", encoding="utf-8")
    suite = WorkloadSuite.model_validate(
        {
            "name": "mixed",
            "workload_concurrency": 2,
            "workloads": [
                {"name": "a", "path": str(workload_a), "weight": 0.7},
                {"name": "b", "path": str(workload_b), "weight": 0.3},
            ],
        }
    )
    out_dir = tmp_path / "suite_v2"
    out_dir.mkdir()
    (out_dir / "baseline_suite.json").write_text(
        json.dumps({"suite_name": "mixed", "makespans_us": {"a": 1_000_000_000.0, "b": 1_000_000_000.0}}),
        encoding="utf-8",
    )
    fake_pipeline = FakePipeline(remote_queue=1_000_000, delay_s=0.01)
    space = _space()
    trainer = TGRLPPOTrainer(
        component_library=_library(),
        search_space=space,
        pipeline_client=fake_pipeline,
        workload_path=None,
        workload_suite=suite,
        out_dir=out_dir,
        updates=1,
        rollout_steps=1,
        env_count=1,
        config=TGRLPPOConfig(ppo_epochs=1, minibatch_size=1, device="cpu"),
    )

    result = trainer.run()

    assert result.best.feasible
    assert fake_pipeline.calls >= 2
    assert fake_pipeline.max_active >= 2
    assert (out_dir / "baseline_suite.json").exists()
    baseline = json.loads((out_dir / "baseline_suite.json").read_text(encoding="utf-8"))
    assert baseline["makespans_us"]["a"] != 1_000_000_000.0
    assert baseline["makespans_us"]["b"] != 1_000_000_000.0
    assert any((out_dir / "update_000").glob("env_*/step_*/suite_feedback.json"))
    summary = json.loads((out_dir / "tgrl_summary.json").read_text(encoding="utf-8"))
    assert summary["workload_suite"]["name"] == "mixed"
    assert {row["workload"] for row in summary["per_workload_single_task_scores"]} == {"a", "b"}
    score_row = summary["per_workload_single_task_scores"][0]
    weights = space.objective_weights
    expected_single_task_score = (
        weights.makespan * (score_row["makespan_us"] / 10_000.0)
        + weights.cost
        * budget_wall_pressure(score_row["estimated_cost"], space.limits.max_total_cost, weights=weights)
        + weights.power
        * budget_wall_pressure(
            score_row["estimated_power_watts"],
            space.limits.max_peak_power_watts,
            weights=weights,
        )
        + weights.max_link_utilization * score_row["max_link_utilization"]
        + weights.max_queue_delay * (score_row["max_queue_delay_ns"] / 1_000_000.0)
        + weights.remote_memory_contention * (score_row["remote_memory_contention_ns"] / 1_000_000.0)
    )
    assert pytest.approx(score_row["single_task_score"]) == expected_single_task_score
    rows = json.loads((out_dir / "curves" / "candidate_scores.json").read_text(encoding="utf-8"))["rows"]
    assert "geomean_speedup" in rows[0]
    assert "min_speedup" in rows[0]
    workload_rows = json.loads((out_dir / "curves" / "workload_scores.json").read_text(encoding="utf-8"))["rows"]
    assert {row["workload"] for row in workload_rows} == {"a", "b"}
    assert "normalized_score" in workload_rows[0]
    assert "single_task_score" in workload_rows[0]
    assert "weighted_log_score" in workload_rows[0]
    assert (out_dir / "curves" / "workload_scores.csv").exists()


def test_svg_curve_uses_robust_y_axis_for_outliers(tmp_path: Path) -> None:
    path = tmp_path / "curve.svg"

    _write_svg_lines(
        path,
        title="Outlier Curve",
        x_label="update",
        y_label="score",
        series=[
            ("score", [(0.0, 100.0), (1.0, 110.0), (2.0, 120.0), (3.0, 1_000_000.0)]),
            ("best", [(0.0, 95.0), (1.0, 96.0), (2.0, 97.0), (3.0, 98.0)]),
        ],
    )

    svg = path.read_text(encoding="utf-8")

    assert 'data-clipped-points="1"' in svg
    assert 'data-y-max="1000000' not in svg
    assert "clipPath" in svg
    assert svg.count("<polyline ") == 2
    assert "outlier point(s) clipped" in svg


def test_svg_curve_breaks_line_when_x_resets(tmp_path: Path) -> None:
    path = tmp_path / "curve.svg"

    _write_svg_lines(
        path,
        title="Restarted Curve",
        x_label="update",
        y_label="score",
        series=[("score", [(0.0, 100.0), (1.0, 110.0), (0.0, 105.0), (1.0, 115.0)])],
    )

    svg = path.read_text(encoding="utf-8")

    assert svg.count("<polyline ") == 2
