#!/usr/bin/env python3
"""Extract per-update TG-RL best topologies and plot their metric trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "update",
    "best_env",
    "best_step",
    "best_score",
    "best_action",
    "topology_path",
    "extracted_topology_path",
    "node_count",
    "link_count",
    "link_count_per_node",
    "average_link_bandwidth_gbps",
    "cpu_count",
    "gpu_count",
    "cpu_total_tflops",
    "gpu_total_tflops",
    "cpu_gpu_compute_ratio",
    "plotted",
    "skipped_reason",
]


@dataclass
class MetricRow:
    update: int
    best_env: int
    best_step: int
    best_score: float
    best_action: str
    topology_path: Path
    extracted_topology_path: Path | None
    node_count: int
    link_count: int
    link_count_per_node: float
    average_link_bandwidth_gbps: float
    cpu_count: int
    gpu_count: int
    cpu_total_tflops: float
    gpu_total_tflops: float
    cpu_gpu_compute_ratio: float
    plotted: bool = False
    skipped_reason: str = ""

    @property
    def point(self) -> tuple[float, float, float]:
        return (
            self.link_count_per_node,
            self.average_link_bandwidth_gbps,
            self.cpu_gpu_compute_ratio,
        )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _csv_float(value: float) -> str:
    return f"{value:.12g}" if _finite(value) else ""


def _extract_link_bandwidth_gbps(link: dict[str, Any]) -> float | None:
    for container in (link, link.get("attrs", {})):
        if not isinstance(container, dict):
            continue
        for key in (
            "bandwidth_gbps",
            "bandwidth_Gbps",
            "bandwidth",
            "bw_gbps",
            "bw",
        ):
            value = _to_float(container.get(key))
            if value is not None:
                return value
    return None


def _node_compute_items(node: dict[str, Any]) -> list[dict[str, Any]]:
    capabilities = node.get("capabilities", {})
    compute = capabilities.get("compute", []) if isinstance(capabilities, dict) else []
    if isinstance(compute, dict):
        compute_items = compute.values()
    elif isinstance(compute, list):
        compute_items = compute
    else:
        compute_items = []
    return [item for item in compute_items if isinstance(item, dict)]


def _classify_compute_item(item: dict[str, Any], node_type: str) -> set[str]:
    kinds: set[str] = set()
    for value in (item.get("kind"), item.get("id"), item.get("type")):
        item_text = str(value or "").lower()
        if "cpu" in item_text:
            kinds.add("cpu")
        if "gpu" in item_text:
            kinds.add("gpu")
    node_type_text = node_type.upper()
    if "CPU" in node_type_text:
        kinds.add("cpu")
    if "GPU" in node_type_text:
        kinds.add("gpu")
    return kinds


def _node_compute_kinds(node: dict[str, Any]) -> set[str]:
    attrs = node.get("attrs", {})
    node_type = str(attrs.get("node_type", "") if isinstance(attrs, dict) else "")
    kinds: set[str] = set()
    for item in _node_compute_items(node):
        kinds.update(_classify_compute_item(item, node_type))
    if not kinds:
        kinds.update(_classify_compute_item({}, node_type))
    return kinds


def _node_compute_tflops(node: dict[str, Any]) -> tuple[float, float]:
    attrs = node.get("attrs", {})
    node_type = str(attrs.get("node_type", "") if isinstance(attrs, dict) else "")
    cpu_total = 0.0
    gpu_total = 0.0
    for item in _node_compute_items(node):
        peak_tflops = None
        for key in ("peak_tflops", "tflops", "peak_tfLOPS"):
            peak_tflops = _to_float(item.get(key))
            if peak_tflops is not None:
                break
        if peak_tflops is None:
            continue
        kinds = _classify_compute_item(item, node_type)
        if "cpu" in kinds:
            cpu_total += peak_tflops
        if "gpu" in kinds:
            gpu_total += peak_tflops
    return cpu_total, gpu_total


def _topology_metrics(topology: dict[str, Any]) -> tuple[int, int, float, float, int, int, float, float, float]:
    nodes = topology.get("nodes", [])
    links = topology.get("links", [])
    if not isinstance(nodes, list):
        nodes = []
    if not isinstance(links, list):
        links = []

    node_count = len(nodes)
    link_count = len(links)
    link_count_per_node = link_count / node_count if node_count else math.nan

    bandwidths = [
        value
        for link in links
        if isinstance(link, dict)
        for value in [_extract_link_bandwidth_gbps(link)]
        if value is not None
    ]
    average_bandwidth = sum(bandwidths) / len(bandwidths) if bandwidths else math.nan

    cpu_count = 0
    gpu_count = 0
    cpu_total_tflops = 0.0
    gpu_total_tflops = 0.0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        kinds = _node_compute_kinds(node)
        cpu_count += int("cpu" in kinds)
        gpu_count += int("gpu" in kinds)
        node_cpu_tflops, node_gpu_tflops = _node_compute_tflops(node)
        cpu_total_tflops += node_cpu_tflops
        gpu_total_tflops += node_gpu_tflops
    cpu_gpu_compute_ratio = cpu_total_tflops / gpu_total_tflops if gpu_total_tflops else math.nan
    return (
        node_count,
        link_count,
        link_count_per_node,
        average_bandwidth,
        cpu_count,
        gpu_count,
        cpu_total_tflops,
        gpu_total_tflops,
        cpu_gpu_compute_ratio,
    )


def _best_topology_path(artifact_dir: Path, update: int, env: int, step: int) -> Path:
    leaf = "initial" if step < 0 else f"step_{step:03d}"
    return artifact_dir / f"update_{update:03d}" / f"env_{env:03d}" / leaf / "hardware_topology.json"


def _load_update_score_rows(artifact_dir: Path) -> list[dict[str, Any]]:
    path = artifact_dir / "curves" / "update_scores.json"
    payload = _load_json(path)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a rows list")
    return sorted(rows, key=lambda row: int(row["update"]))


def _copy_topology(source: Path, out_dir: Path, update: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"update_{update:03d}_best_hardware_topology.json"
    shutil.copyfile(source, destination)
    return destination


def collect_metrics(
    artifact_dir: Path,
    *,
    topology_out_dir: Path | None,
) -> list[MetricRow]:
    metric_rows: list[MetricRow] = []
    for score_row in _load_update_score_rows(artifact_dir):
        update = int(score_row["update"])
        best_env = int(score_row["best_env"])
        best_step = int(score_row["best_step"])
        topology_path = _best_topology_path(artifact_dir, update, best_env, best_step)
        if not topology_path.exists():
            raise FileNotFoundError(
                f"best topology for update {update} not found: {topology_path}"
            )

        topology = _load_json(topology_path)
        (
            node_count,
            link_count,
            link_count_per_node,
            average_bandwidth,
            cpu_count,
            gpu_count,
            cpu_total_tflops,
            gpu_total_tflops,
            cpu_gpu_compute_ratio,
        ) = _topology_metrics(topology)
        extracted_path = (
            _copy_topology(topology_path, topology_out_dir, update)
            if topology_out_dir is not None
            else None
        )
        metric_rows.append(
            MetricRow(
                update=update,
                best_env=best_env,
                best_step=best_step,
                best_score=float(score_row.get("best_score", math.nan)),
                best_action=str(score_row.get("best_action", "")),
                topology_path=topology_path,
                extracted_topology_path=extracted_path,
                node_count=node_count,
                link_count=link_count,
                link_count_per_node=link_count_per_node,
                average_link_bandwidth_gbps=average_bandwidth,
                cpu_count=cpu_count,
                gpu_count=gpu_count,
                cpu_total_tflops=cpu_total_tflops,
                gpu_total_tflops=gpu_total_tflops,
                cpu_gpu_compute_ratio=cpu_gpu_compute_ratio,
            )
        )
    return metric_rows


def _same_point(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    *,
    tol: float,
) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(left, right))


def mark_plotted_rows(rows: list[MetricRow], *, dedupe_tol: float) -> list[MetricRow]:
    plotted: list[MetricRow] = []
    previous_point: tuple[float, float, float] | None = None
    for row in rows:
        if not all(_finite(value) for value in row.point):
            row.plotted = False
            row.skipped_reason = "non_finite_point"
            continue
        if previous_point is not None and _same_point(previous_point, row.point, tol=dedupe_tol):
            row.plotted = False
            row.skipped_reason = "same_as_previous_point"
            continue
        row.plotted = True
        row.skipped_reason = ""
        previous_point = row.point
        plotted.append(row)
    return plotted


def _adaptive_axis_range(
    values: list[float],
    *,
    projection_side: str,
    data_pad_fraction: float = 0.08,
    projection_gap_fraction: float = 0.12,
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("cannot compute axis range for an empty value list")

    data_min = min(values)
    data_max = max(values)
    data_span = data_max - data_min
    if data_span <= 0:
        reference = max(abs(data_min), abs(data_max), 1.0)
        half_span = reference * 0.05
        data_min -= half_span
        data_max += half_span
        data_span = data_max - data_min

    data_pad = data_span * data_pad_fraction
    projection_gap = data_span * projection_gap_fraction
    lower = data_min - data_pad
    upper = data_max + data_pad

    if projection_side == "low":
        projection_coord = lower - projection_gap
        lower = projection_coord
    elif projection_side == "high":
        projection_coord = upper + projection_gap
        upper = projection_coord
    else:
        raise ValueError(f"unsupported projection_side: {projection_side}")

    return lower, upper, projection_coord


def write_csv(rows: list[MetricRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "update": row.update,
                    "best_env": row.best_env,
                    "best_step": row.best_step,
                    "best_score": _csv_float(row.best_score),
                    "best_action": row.best_action,
                    "topology_path": str(row.topology_path),
                    "extracted_topology_path": (
                        str(row.extracted_topology_path)
                        if row.extracted_topology_path is not None
                        else ""
                    ),
                    "node_count": row.node_count,
                    "link_count": row.link_count,
                    "link_count_per_node": _csv_float(row.link_count_per_node),
                    "average_link_bandwidth_gbps": _csv_float(row.average_link_bandwidth_gbps),
                    "cpu_count": row.cpu_count,
                    "gpu_count": row.gpu_count,
                    "cpu_total_tflops": _csv_float(row.cpu_total_tflops),
                    "gpu_total_tflops": _csv_float(row.gpu_total_tflops),
                    "cpu_gpu_compute_ratio": _csv_float(row.cpu_gpu_compute_ratio),
                    "plotted": int(row.plotted),
                    "skipped_reason": row.skipped_reason,
                }
            )


def plot_trajectory(rows: list[MetricRow], out_path: Path, *, title: str, dpi: int) -> None:
    if not rows:
        raise ValueError("no finite, non-duplicate points available for plotting")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    xs = [row.link_count_per_node for row in rows]
    ys = [row.average_link_bandwidth_gbps for row in rows]
    zs = [row.cpu_gpu_compute_ratio for row in rows]
    updates = [row.update for row in rows]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(xs, ys, zs, c=updates, cmap="viridis", s=48, depthshade=True)
    ax.plot(xs, ys, zs, color="#333333", linewidth=1.3, alpha=0.65)

    x_lower, x_upper, x_shadow = _adaptive_axis_range(xs, projection_side="low")
    y_lower, y_upper, y_shadow = _adaptive_axis_range(ys, projection_side="high")
    z_lower, z_upper, z_shadow = _adaptive_axis_range(zs, projection_side="low")

    ax.plot(
        xs,
        ys,
        [z_shadow] * len(rows),
        color="#6b7280",
        linewidth=1.4,
        linestyle="--",
        alpha=0.42,
        label="XY projection",
    )
    ax.scatter(xs, ys, [z_shadow] * len(rows), color="#6b7280", s=24, alpha=0.35)
    ax.plot(
        xs,
        [y_shadow] * len(rows),
        zs,
        color="#7c3aed",
        linewidth=1.4,
        linestyle="--",
        alpha=0.36,
        label="XZ projection",
    )
    ax.scatter(xs, [y_shadow] * len(rows), zs, color="#7c3aed", s=24, alpha=0.28)
    ax.plot(
        [x_shadow] * len(rows),
        ys,
        zs,
        color="#0891b2",
        linewidth=1.4,
        linestyle="--",
        alpha=0.36,
        label="YZ projection",
    )
    ax.scatter([x_shadow] * len(rows), ys, zs, color="#0891b2", s=24, alpha=0.28)

    for x, y, z in zip(xs, ys, zs):
        ax.plot([x, x], [y, y], [z_shadow, z], color="#9ca3af", linewidth=0.8, alpha=0.28)
        ax.plot([x, x], [y_shadow, y], [z, z], color="#a78bfa", linewidth=0.8, alpha=0.18)
        ax.plot([x_shadow, x], [y, y], [z, z], color="#67e8f9", linewidth=0.8, alpha=0.18)

    ax.scatter([xs[0]], [ys[0]], [zs[0]], marker="o", s=110, color="#2ca02c", label=f"start u{updates[0]}")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], marker="X", s=130, color="#d62728", label=f"end u{updates[-1]}")

    ax.set_xlabel("link_count / node_count")
    ax.set_ylabel("average link bandwidth (Gbps)")
    ax.set_zlabel("CPU peak TFLOPS / GPU peak TFLOPS")
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.set_zlim(z_lower, z_upper)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.15, 1.15, 0.9))
    ax.set_title(title)
    ax.legend(loc="upper left")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    colorbar.set_label("update")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the best topology for each TG-RL update, compute topology metrics, "
            "write a CSV, and render a 3D iteration trajectory."
        )
    )
    parser.add_argument(
        "artifact_dir",
        type=Path,
        help="TG-RL artifact directory, e.g. optimizer/artifacts/tgrl_2rack_topo_new_new.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="CSV output path. Defaults to <artifact_dir>/best_topology_metrics.csv.",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <artifact_dir>/best_topology_trajectory.png.",
    )
    parser.add_argument(
        "--topology-out-dir",
        type=Path,
        default=None,
        help="Directory for copied per-update best topologies. Defaults to <artifact_dir>/best_topologies_by_update.",
    )
    parser.add_argument(
        "--no-copy-topologies",
        action="store_true",
        help="Do not copy per-update best topology JSON files.",
    )
    parser.add_argument(
        "--dedupe-tol",
        type=float,
        default=1e-12,
        help="Tolerance for skipping consecutive duplicate 3D points.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output plot DPI.")
    parser.add_argument("--title", default=None, help="Plot title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    if not artifact_dir.exists():
        raise FileNotFoundError(f"artifact directory not found: {artifact_dir}")

    csv_out = args.csv_out or artifact_dir / "best_topology_metrics.csv"
    plot_out = args.plot_out or artifact_dir / "best_topology_trajectory.png"
    topology_out_dir = (
        None
        if args.no_copy_topologies
        else (args.topology_out_dir or artifact_dir / "best_topologies_by_update")
    )

    rows = collect_metrics(artifact_dir, topology_out_dir=topology_out_dir)
    plotted = mark_plotted_rows(rows, dedupe_tol=args.dedupe_tol)
    write_csv(rows, csv_out)
    plot_trajectory(
        plotted,
        plot_out,
        title=args.title or f"TG-RL Best Topology Trajectory: {artifact_dir.name}",
        dpi=args.dpi,
    )

    duplicate_count = sum(row.skipped_reason == "same_as_previous_point" for row in rows)
    non_finite_count = sum(row.skipped_reason == "non_finite_point" for row in rows)
    print(f"updates: {len(rows)}")
    print(f"plotted points: {len(plotted)}")
    print(f"skipped consecutive duplicates: {duplicate_count}")
    print(f"skipped non-finite points: {non_finite_count}")
    print(f"csv: {csv_out}")
    print(f"plot: {plot_out}")
    if topology_out_dir is not None:
        print(f"per-update best topologies: {topology_out_dir}")


if __name__ == "__main__":
    main()
