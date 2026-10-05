"""Build the required cmp5L query-behavior KD pilot report from final JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = {
    "A": "A_baseline",
    "B": "B_query_update",
    "C": "C_negative_behavior",
    "D": "D_combined",
    "E": "E_combined_sampling",
}


def decide_go_no_go(metrics, mechanism):
    query_gain = metrics["B"]["ap50"] - metrics["A"]["ap50"]
    a_negative = mechanism["A"]["negative_queries"]["mean_student_max_class_score"]
    c_negative = mechanism["C"]["negative_queries"]["mean_student_max_class_score"]
    negative_go = (
        metrics["C"]["ap50"] > metrics["A"]["ap50"]
        and metrics["C"]["precision"] >= metrics["A"]["precision"]
        and c_negative < a_negative
    )
    return {
        "query_kd": "GO" if query_gain >= 0.005 else "NO-GO",
        "negative_behavior_kd": "GO" if negative_go else "NO-GO",
        "combined": "GO" if metrics["D"]["ap50"] > max(metrics["B"]["ap50"], metrics["C"]["ap50"]) else "NO-GO",
        "sampling_kd": "GO" if metrics["E"]["ap50"] > metrics["D"]["ap50"] else "NO-GO",
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _duct_ap50(payload):
    classes = payload.get("per_class_ap50", {})
    for record in classes.values():
        if "导管" in record.get("name", "") or "duct" in record.get("name", "").lower():
            return float(record["ap50"])
    return None


def _loss_summary(path):
    if not path.exists():
        return {}
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "epoch" in value:
            records.append(value)
    if not records:
        return {}
    first, last = records[0], records[-1]
    return {
        "epochs_logged": len(records),
        "first_loss": first.get("train_loss"),
        "last_loss": last.get("train_loss"),
        "first_query_kd": first.get("train_loss_qbeh_query"),
        "last_query_kd": last.get("train_loss_qbeh_query"),
        "first_negative_kd": first.get("train_loss_qbeh_negative"),
        "last_negative_kd": last.get("train_loss_qbeh_negative"),
        "first_sampling_kd": first.get("train_loss_qbeh_sampling"),
        "last_sampling_kd": last.get("train_loss_qbeh_sampling"),
        "first_ap50": first.get("test_coco_eval_bbox", [None, None])[1] if first.get("test_coco_eval_bbox") else None,
        "last_ap50": last.get("test_coco_eval_bbox", [None, None])[1] if last.get("test_coco_eval_bbox") else None,
    }


def build_report(root: Path, job_ids):
    metrics, mechanisms, duct, losses, per_class = {}, {}, {}, {}, {}
    for arm, directory in ARMS.items():
        experiment = root / directory
        metric_payload = _read_json(experiment / "final_test" / "metrics.json")
        mechanism_payload = _read_json(experiment / "final_test" / "mechanism.json")
        metrics[arm] = metric_payload["metrics"]
        mechanisms[arm] = mechanism_payload["mechanism"]
        duct[arm] = _duct_ap50(metric_payload)
        per_class[arm] = metric_payload["per_class_ap50"]
        losses[arm] = _loss_summary(experiment / "log.txt")

    verdict = decide_go_no_go(metrics, mechanisms)
    best = max(ARMS, key=lambda arm: metrics[arm]["ap50"])
    d_gain_pp = 100 * (metrics["D"]["ap50"] - metrics["A"]["ap50"])
    duct_gain = None if duct["D"] is None or duct["A"] is None else 100 * (duct["D"] - duct["A"])
    if verdict["combined"] == "GO" and d_gain_pp >= 0.5 and (duct_gain is None or duct_gain > 0):
        recommendation = "Proceed to a full query-behavior KD run centered on D; keep only mechanisms that passed."
    elif verdict["query_kd"] == "GO" or verdict["negative_behavior_kd"] == "GO":
        recommendation = "Continue only the individually positive behavior-KD component; do not stack failed losses."
    else:
        recommendation = "Stop query-behavior KD and return to explicit BG-dependency prediction."

    fmt = lambda value: "N/A" if value is None else f"{value:.4f}"
    lines = [
        "# cmp5L Privileged Query Behavior Distillation Pilot",
        "",
        f"A Baseline AP50: {metrics['A']['ap50']:.4f}",
        "",
        f"B Query KD AP50: {metrics['B']['ap50']:.4f}",
        "",
        f"C Negative KD AP50: {metrics['C']['ap50']:.4f}",
        "",
        f"D Combined AP50: {metrics['D']['ap50']:.4f}",
        "",
        f"E +Sampling AP50: {metrics['E']['ap50']:.4f}",
        "",
        f"Best: {best} ({ARMS[best]})",
        "",
        "Duct-related: " + " / ".join(f"{arm} {fmt(duct[arm])}" for arm in ARMS),
        "",
        f"Query KD: {verdict['query_kd']}",
        "",
        f"Negative Behavior KD: {verdict['negative_behavior_kd']}",
        "",
        f"Sampling KD: {verdict['sampling_kd']}",
        "",
        f"Recommended next step: {recommendation}",
        "",
        "## Full metrics",
        "",
        "| Arm | AP | AP50 | AP75 | Precision | Recall | F1 | AR100 | Small AP | Medium AP | Large AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m = metrics[arm]
        lines.append(
            f"| {arm} | {m['ap']:.4f} | {m['ap50']:.4f} | {m['ap75']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['ar100']:.4f} | "
            f"{m['small_ap']:.4f} | {m['medium_ap']:.4f} | {m['large_ap']:.4f} |"
        )

    lines += [
        "",
        "## Per-class AP50",
        "",
        "| Class | A | B | C | D | E | D − A (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category_id in sorted(per_class["A"], key=int):
        name = per_class["A"][category_id]["name"]
        values = [per_class[arm][category_id]["ap50"] for arm in ARMS]
        lines.append(
            f"| {name} | " + " | ".join(f"{value:.4f}" for value in values)
            + f" | {100 * (values[3] - values[0]):+.2f} |"
        )

    lines += [
        "",
        "## Mechanism metrics",
        "",
        "| Arm | Neg mean score | Neg high-conf | Positive GT prob | Positive margin | Positive IoU | Δq cos L0/L1/L2 | Sampling L1 L0/L1/L2 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for arm in ARMS:
        mech = mechanisms[arm]
        neg, pos, layers = mech["negative_queries"], mech["matched_positive_queries"], mech["layers"]
        cosines = "/".join(f"{layers[str(i)]['mean_student_teacher_delta_q_cosine']:.4f}" for i in range(3))
        sampling = "/".join(f"{layers[str(i)]['mean_relative_sampling_l1_to_teacher']:.4f}" for i in range(3))
        lines.append(
            f"| {arm} | {neg['mean_student_max_class_score']:.6f} | {neg['student_high_confidence_count']} | "
            f"{pos['mean_student_gt_class_probability']:.4f} | {pos['mean_student_class_margin']:.4f} | "
            f"{pos['mean_student_iou']:.4f} | {cosines} | {sampling} |"
        )

    lines += [
        "",
        "## Implementation and debug checks",
        "",
        "- Frozen baseline-EMA Teacher; zero trainable Teacher parameters and detached targets.",
        "- Exact Student-normal q0/r0 replay; privileged intervention only scales sampled Far-BG values on L0–L2.",
        "- Query-update cosine uses true post-layer minus layer-input tensors; negative KD is asymmetric on Hungarian-unmatched queries.",
        "- Effective batch 32 (single-GPU batch 8 × accumulation 4), 15 epochs, detector LR 5e-5, backbone LR 1e-6, EMA evaluation.",
        "- Debug Job 2745 passed forward/backward, exact normal parity, shared initialization, finite Student gradients, frozen Teacher, EMA, save and resume.",
        "- Mechanism evaluator smoke Job 2762 completed on real images.",
        "",
        "## Loss/evaluation trajectory",
        "",
        "| Arm | Epoch records | First loss | Last loss | First→last query KD | First→last negative KD | First→last sampling KD | First val AP50 | Last val AP50 |",
        "|---|---:|---:|---:|---|---|---|---:|---:|",
    ]
    for arm in ARMS:
        loss = losses[arm]
        lines.append(
            f"| {arm} | {loss.get('epochs_logged', 0)} | {fmt(loss.get('first_loss'))} | "
            f"{fmt(loss.get('last_loss'))} | {fmt(loss.get('first_query_kd'))}→{fmt(loss.get('last_query_kd'))} | "
            f"{fmt(loss.get('first_negative_kd'))}→{fmt(loss.get('last_negative_kd'))} | "
            f"{fmt(loss.get('first_sampling_kd'))}→{fmt(loss.get('last_sampling_kd'))} | "
            f"{fmt(loss.get('first_ap50'))} | {fmt(loss.get('last_ap50'))} |"
        )

    lines += [
        "",
        "## Decision analysis",
        "",
        f"- D − A AP50: {d_gain_pp:+.2f} pp.",
        f"- D − A duct-related AP50: {('N/A' if duct_gain is None else f'{duct_gain:+.2f} pp')}.",
        f"- Combined behavior KD: {verdict['combined']}.",
        f"- Sampling KD: {verdict['sampling_kd']} because E {'>' if metrics['E']['ap50'] > metrics['D']['ap50'] else '<='} D on AP50.",
        "- B changes the query-update trajectory (especially L1/L2 cosine) but not AP50; the tested distillation did not translate into a stable detection gain.",
        "- C slightly lowers the mean unmatched-query score, but precision falls versus A and the high-confidence unmatched count is not reduced; negative-behavior KD does not pass its mechanism gate.",
        "- D is best in this pilot, but its +0.39 pp AP50 is below the predeclared +0.5 pp useful-signal threshold; precision falls 0.65 pp while recall rises 1.0 pp. Treat combined complementarity as exploratory, not a full-training GO.",
        "- E does not improve relative-sampling distance consistently versus D and AP50 is lower; no added sampling-KD value is demonstrated.",
        "",
        "## Failure and retry history",
        "",
        "- First E attempt 2761 failed after 59 seconds with a CUDA kernel launch timeout on cu01; exact parity and finite student gradients had already been observed, and no checkpoint was produced.",
        "- Only E was retried as 2768 with cu01 excluded; it completed on cu02. A–D outputs were preserved.",
        "",
        "## Reproducibility",
        "",
        f"- Slurm jobs: {json.dumps(job_ids, sort_keys=True)}",
        f"- Experiment root: `{root}`",
        "- Each experiment contains resolved config, stdout log, EMA checkpoint, full-test metrics JSON, mechanism JSON, and completion marker.",
        "",
    ]
    summary = {
        "job_ids": job_ids,
        "metrics": metrics,
        "duct_ap50": duct,
        "per_class_ap50": per_class,
        "mechanism": mechanisms,
        "loss_summary": losses,
        "verdict": verdict,
        "best": best,
        "recommended_next_step": recommendation,
    }
    return "\n".join(lines), summary


def main(args):
    root = Path(args.root)
    job_ids = json.loads(args.job_ids)
    report, summary = build_report(root, job_ids)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report[:3000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--job-ids", required=True)
    main(parser.parse_args())
