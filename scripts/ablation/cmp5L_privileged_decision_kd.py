"""Auditable targets and losses for cmp5L privileged decision distillation.

This module intentionally contains only pure target-selection and loss helpers.
The verified A–F/GH decoder replay remains the sole implementation of the
privileged value intervention.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F


DEFAULT_STAGE1_COUNTS = {
    "empty": 128,
    "duct": 128,
    "lymph": 96,
    "cystic": 96,
    "solid": 96,
    "hard": 96,
}
CATEGORY_STRATA = (("duct", 3), ("lymph", 2), ("cystic", 1), ("solid", 0))


def build_counterfactual_gates(full_gate, query_ids_by_image):
    """Return selected-only and selected-excluded gates without mutating full_gate."""
    if full_gate.ndim != 2 or len(query_ids_by_image) != full_gate.shape[0]:
        raise ValueError("counterfactual query ids must align with a [B,Q] full gate")
    chosen = torch.zeros_like(full_gate, dtype=torch.bool)
    for batch_index, query_ids in enumerate(query_ids_by_image):
        for query_id in query_ids:
            query_id = int(query_id)
            if query_id < 0 or query_id >= full_gate.shape[1]:
                raise IndexError(f"query id {query_id} outside [0,{full_gate.shape[1]})")
            chosen[batch_index, query_id] = True
    selected_only = full_gate.to(torch.bool) & chosen
    selected_excluded = full_gate.to(torch.bool) & ~chosen
    return selected_only, selected_excluded


def max_named_buffer_change(before, named_buffers):
    """Return a dtype-safe maximum change over a named-buffer snapshot."""
    maximum = 0.0
    observed = set()
    for name, value in named_buffers:
        if name not in before:
            raise KeyError(f"unexpected buffer after forward: {name}")
        observed.add(name)
        reference = before[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise RuntimeError(f"buffer metadata changed: {name}")
        if not value.numel():
            continue
        if value.dtype == torch.bool:
            difference = float(torch.logical_xor(value, reference).any())
        else:
            difference = float((value - reference).abs().max())
        maximum = max(maximum, difference)
    missing = sorted(set(before) - observed)
    if missing:
        raise KeyError(f"buffers disappeared after forward: {missing[:10]}")
    return maximum


def select_counterfactual_cases(rows, per_category=4):
    """Lock a small deterministic case set before counterfactual inference."""
    if per_category <= 0:
        raise ValueError("per_category must be positive")
    candidates = {"repair": [], "protect": [], "suppress": [], "mixed": []}
    for row in rows:
        image_id = int(row["image_id"])
        edits = []
        for item in row.get("positive", []):
            kind = item["kind"]
            if kind not in ("repair", "protect"):
                continue
            record = {
                "category": kind,
                "image_id": image_id,
                "query_ids": [int(item["teacher_query"])],
                "gt_index": int(item["gt_index"]),
                "class_index": int(item["gt_label"]),
            }
            candidates[kind].append(record)
            edits.append(record)
        for item in row.get("negative", []):
            record = {
                "category": "suppress",
                "image_id": image_id,
                "query_ids": [int(item["teacher_query"])],
                "gt_index": None,
                "class_index": int(item["class_index"]),
            }
            candidates["suppress"].append(record)
            edits.append(record)
        if len(edits) > 1:
            candidates["mixed"].append({
                "category": "mixed",
                "image_id": image_id,
                "query_ids": sorted({query for edit in edits for query in edit["query_ids"]}),
                "components": edits,
                "reason": "multiple preselected edits in one image",
            })

    def key(item):
        return (
            item["image_id"],
            -1 if item.get("gt_index") is None else item["gt_index"],
            item["query_ids"],
            item.get("class_index", -1),
        )

    locked = {}
    for category, values in candidates.items():
        values = sorted(values, key=key)
        chosen, seen_images = [], set()
        for item in values:
            if item["image_id"] in seen_images:
                continue
            chosen.append(item)
            seen_images.add(item["image_id"])
            if len(chosen) == per_category:
                break
        if len(chosen) < per_category:
            chosen_keys = {json.dumps(item, sort_keys=True) for item in chosen}
            for item in values:
                marker = json.dumps(item, sort_keys=True)
                if marker in chosen_keys:
                    continue
                chosen.append(item)
                chosen_keys.add(marker)
                if len(chosen) == per_category:
                    break
        locked[category] = chosen
    return locked


def run_with_all_score_heads(ec_transformer, branch_callable):
    """Capture one call to each score head while preserving eval child state.

    The verified manual decoder only appends the eval layer unless its
    container's ``training`` flag is true.  Toggling that single flag (without
    calling ``train()``) exercises every score head while all child modules
    remain in eval mode.
    """
    decoder = ec_transformer.decoder
    lqe_layers = getattr(decoder, "lqe_layers", None)
    capture_modules = lqe_layers if lqe_layers is not None else ec_transformer.dec_score_head
    if len(capture_modules) != len(ec_transformer.dec_score_head):
        raise RuntimeError("classification capture layer count differs from score heads")
    captured = [[] for _ in capture_modules]
    handles = []
    for index, head in enumerate(capture_modules):
        def hook_factory(layer_index):
            def hook(_module, _inputs, output):
                captured[layer_index].append(output)
            return hook
        handles.append(head.register_forward_hook(hook_factory(index)))
    original_training = decoder.training
    decoder.training = True
    try:
        output = branch_callable()
    finally:
        decoder.training = original_training
        for handle in handles:
            handle.remove()
    calls = [len(values) for values in captured]
    if calls != [1] * len(captured):
        raise RuntimeError(f"expected one score-head call per layer, observed {calls}")
    return output, [values[0] for values in captured]


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def _box_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.numel() == 0 or right.numel() == 0:
        return left.new_zeros((left.shape[0], right.shape[0]))
    left, right = _cxcywh_to_xyxy(left), _cxcywh_to_xyxy(right)
    top_left = torch.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    left_area = (left[:, 2:] - left[:, :2]).clamp(min=0).prod(-1)
    right_area = (right[:, 2:] - right[:, :2]).clamp(min=0).prod(-1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / union.clamp(min=1e-12)


def bernoulli_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Mean KL(Bernoulli teacher || student), with a differentiable empty zero."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student = student_logits.float() / float(temperature)
    teacher = teacher_logits.detach().float() / float(temperature)
    probability = teacher.sigmoid()
    elementwise = (
        probability * (F.logsigmoid(teacher) - F.logsigmoid(student))
        + (1.0 - probability)
        * (F.logsigmoid(-teacher) - F.logsigmoid(-student))
    ) * float(temperature) ** 2
    if valid_mask is None:
        valid_mask = torch.ones_like(elementwise, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(device=elementwise.device, dtype=torch.bool)
        valid_mask = torch.broadcast_to(valid_mask, elementwise.shape)
    count = valid_mask.sum()
    if int(count.detach()) == 0:
        return student_logits.sum() * 0.0
    return elementwise.masked_select(valid_mask).sum() / count.to(elementwise.dtype)


def mean_layer_kd(student_layers, teacher_layers, valid_masks, temperature=1.0):
    """Normalize within each layer and then average layers."""
    if not (len(student_layers) == len(teacher_layers) == len(valid_masks)):
        raise ValueError("student, teacher, and mask layer counts differ")
    if not student_layers:
        raise ValueError("at least one layer is required")
    losses = [
        bernoulli_kl_from_logits(student, teacher, mask, temperature)
        for student, teacher, mask in zip(student_layers, teacher_layers, valid_masks)
    ]
    return torch.stack(losses).mean()


def _fixed_query_matches(logits, boxes, labels, gt_boxes, score_threshold, iou_threshold):
    """Historical score-order, class-constrained, one-to-one matching in query space."""
    scores = logits.sigmoid()
    query_count, class_count = scores.shape
    flat_scores = scores.flatten()
    keep = min(query_count, flat_scores.numel())
    values, indices = torch.topk(flat_scores, keep, sorted=True)
    ious = _box_iou(boxes, gt_boxes)
    unmatched = set(range(int(labels.numel())))
    matches = {}
    false_candidates = []
    for value, flat_index in zip(values.tolist(), indices.tolist()):
        if value < score_threshold:
            break
        query = int(flat_index // class_count)
        category = int(flat_index % class_count)
        candidates = [index for index in unmatched if int(labels[index]) == category]
        if candidates:
            gt_index = max(candidates, key=lambda index: float(ious[query, index]))
            overlap = float(ious[query, gt_index])
        else:
            gt_index, overlap = None, -1.0
        if gt_index is not None and overlap >= iou_threshold:
            matches[int(gt_index)] = query
            unmatched.remove(gt_index)
        else:
            false_candidates.append(
                {"query": query, "class_index": category, "score": float(value)}
            )
    return matches, false_candidates, ious


def select_outcome_targets(
    normal_outputs,
    privileged_outputs,
    targets,
    score_threshold=0.5,
    iou_threshold=0.5,
    negative_iou_threshold=0.3,
):
    """Select protect/repair positives and conservative class-only suppressions.

    N/P share query identity because they replay one frozen teacher
    initialization. Student pairing is intentionally handled elsewhere through
    GT identity (positives) or detached box matching (negatives).
    """
    normal_logits = normal_outputs["pred_logits"].detach()
    normal_boxes = normal_outputs["pred_boxes"].detach()
    privileged_logits = privileged_outputs["pred_logits"].detach()
    privileged_boxes = privileged_outputs["pred_boxes"].detach()
    if normal_logits.shape != privileged_logits.shape:
        raise ValueError("N/P logit shapes differ")
    if normal_boxes.shape != privileged_boxes.shape:
        raise ValueError("N/P box shapes differ")
    if normal_logits.shape[0] != len(targets):
        raise ValueError("batch target count differs from outputs")

    selected = []
    class_count = normal_logits.shape[-1]
    for batch_index, target in enumerate(targets):
        labels = target["labels"].detach()
        gt_boxes = target["boxes"].detach()
        n_match, n_false, n_ious = _fixed_query_matches(
            normal_logits[batch_index], normal_boxes[batch_index], labels, gt_boxes,
            score_threshold, iou_threshold,
        )
        p_match, _, _ = _fixed_query_matches(
            privileged_logits[batch_index], privileged_boxes[batch_index], labels, gt_boxes,
            score_threshold, iou_threshold,
        )
        positive = []
        counts = Counter(
            stable=0, repair=0, protect=0, stable_missed=0,
            suppress=0, uncertain_negative=0,
        )
        for gt_index in range(int(labels.numel())):
            n_query, p_query = n_match.get(gt_index), p_match.get(gt_index)
            if n_query is not None and p_query is not None:
                counts["stable"] += 1
            elif n_query is None and p_query is not None:
                counts["repair"] += 1
                positive.append({
                    "kind": "repair",
                    "gt_index": gt_index,
                    "teacher_query": int(p_query),
                    "branch": "P",
                    "class_indices": list(range(class_count)),
                })
            elif n_query is not None and p_query is None:
                counts["protect"] += 1
                positive.append({
                    "kind": "protect",
                    "gt_index": gt_index,
                    "teacher_query": int(n_query),
                    "branch": "N",
                    "class_indices": list(range(class_count)),
                })
            else:
                counts["stable_missed"] += 1

        negative = []
        seen = set()
        for candidate in n_false:
            query = candidate["query"]
            category = candidate["class_index"]
            key = (query, category)
            if key in seen:
                continue
            seen.add(key)
            max_iou = float(n_ious[query].max()) if gt_boxes.numel() else 0.0
            if max_iou >= negative_iou_threshold:
                counts["uncertain_negative"] += 1
                continue
            normal_score = float(normal_logits[batch_index, query, category].sigmoid())
            privileged_score = float(
                privileged_logits[batch_index, query, category].sigmoid()
            )
            if privileged_score < normal_score:
                negative.append({
                    "kind": "suppress",
                    "teacher_query": query,
                    "class_index": category,
                    "branch": "P",
                    "normal_score": normal_score,
                    "privileged_score": privileged_score,
                    "max_gt_iou": max_iou,
                })
                counts["suppress"] += 1
            else:
                counts["uncertain_negative"] += 1
        selected.append({
            "positive": positive,
            "negative": negative,
            "counts": dict(counts),
            "normal_gt_matches": n_match,
            "privileged_gt_matches": p_match,
        })
    return selected


def build_mixed_oracle_output(normal_outputs, privileged_outputs, selected_targets):
    """Apply only selected classification edits while preserving all N boxes."""
    logits = normal_outputs["pred_logits"].clone()
    boxes = normal_outputs["pred_boxes"].clone()
    if logits.shape[0] != len(selected_targets):
        raise ValueError("selected target batch count differs from outputs")
    privileged_logits = privileged_outputs["pred_logits"]
    for batch_index, selected in enumerate(selected_targets):
        touched = set()
        for item in selected["positive"]:
            if item["kind"] != "repair":
                continue
            query = int(item["teacher_query"])
            logits[batch_index, query] = privileged_logits[batch_index, query]
            touched.update((query, index) for index in range(logits.shape[-1]))
        for item in selected["negative"]:
            query, category = int(item["teacher_query"]), int(item["class_index"])
            if (query, category) in touched:
                raise RuntimeError("positive and negative mixed-oracle edits conflict")
            logits[batch_index, query, category] = privileged_logits[
                batch_index, query, category
            ]
            touched.add((query, category))
    return {"pred_logits": logits, "pred_boxes": boxes}


def _stable_rank(seed, stratum, image_id):
    payload = f"{int(seed)}:{stratum}:{int(image_id)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _xywh_iou(left, right):
    lx, ly, lw, lh = map(float, left)
    rx, ry, rw, rh = map(float, right)
    x0, y0, x1, y1 = max(lx, rx), max(ly, ry), min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def build_stage1_subset_manifest(coco, seed=20260920, requested=None):
    """Build the pre-result, annotation-only 640-image Stage-1 manifest."""
    requested = dict(DEFAULT_STAGE1_COUNTS if requested is None else requested)
    required = {"empty", "duct", "lymph", "cystic", "solid", "hard"}
    if set(requested) != required or any(int(value) < 0 for value in requested.values()):
        raise ValueError(f"requested counts must have exactly {sorted(required)}")
    images = {int(image["id"]): image for image in coco["images"]}
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        if not int(annotation.get("iscrowd", 0)):
            annotations[int(annotation["image_id"])].append(annotation)

    all_area_ratios = []
    per_image = {}
    for image_id, image in images.items():
        current = annotations[image_id]
        denominator = float(image["width"] * image["height"])
        ratios = [float(item["bbox"][2] * item["bbox"][3]) / denominator for item in current]
        all_area_ratios.extend(ratios)
        per_image[image_id] = {
            "category_ids": sorted({int(item["category_id"]) for item in current}),
            "area_ratios": ratios,
        }
    ordered_areas = sorted(all_area_ratios)
    q10 = ordered_areas[int((len(ordered_areas) - 1) * 0.10)] if ordered_areas else 0.0
    for image_id, info in per_image.items():
        current = annotations[image_id]
        reasons = []
        if len(current) > 1:
            reasons.append("multi_gt")
        if info["area_ratios"] and min(info["area_ratios"]) <= q10:
            reasons.append("small_q10")
        if any(
            _xywh_iou(current[left]["bbox"], current[right]["bbox"]) >= 0.3
            for left in range(len(current))
            for right in range(left + 1, len(current))
        ):
            reasons.append("overlap_iou30")
        info["hard_reasons"] = reasons

    selected = set()
    rows = []

    def choose(stratum, eligible, count):
        available = sorted(
            (image_id for image_id in eligible if image_id not in selected),
            key=lambda image_id: (_stable_rank(seed, stratum, image_id), image_id),
        )
        if len(available) < count:
            raise ValueError(
                f"stratum {stratum} has {len(available)} available images, needs {count}"
            )
        for image_id in available[:count]:
            selected.add(image_id)
            image = images[image_id]
            info = per_image[image_id]
            rows.append({
                "image_id": image_id,
                "file_name": image["file_name"],
                "stratum": stratum,
                "gt_count": len(annotations[image_id]),
                "category_ids": info["category_ids"],
                "min_area_ratio": min(info["area_ratios"]) if info["area_ratios"] else None,
                "hard_reasons": info["hard_reasons"],
            })

    choose("empty", [image_id for image_id in images if not annotations[image_id]], requested["empty"])
    for stratum, category_id in CATEGORY_STRATA:
        choose(
            stratum,
            [image_id for image_id, info in per_image.items() if category_id in info["category_ids"]],
            requested[stratum],
        )
    choose(
        "hard",
        [image_id for image_id, info in per_image.items() if annotations[image_id] and info["hard_reasons"]],
        requested["hard"],
    )

    population_classes = Counter(
        int(annotation["category_id"])
        for annotation in coco["annotations"]
        if not int(annotation.get("iscrowd", 0))
    )
    payload = {
        "schema_version": 1,
        "seed": int(seed),
        "selection_is_enriched_not_population_estimate": True,
        "requested": requested,
        "population": {
            "images": len(images),
            "empty_images": sum(not annotations[image_id] for image_id in images),
            "annotations": sum(len(value) for value in annotations.values()),
            "annotations_by_category": {str(key): value for key, value in sorted(population_classes.items())},
            "small_area_q10": q10,
        },
        "images": rows,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def subset_coco_from_manifest(coco, manifest):
    """Return a COCO dataset restricted to the manifest's image IDs."""
    selected = {int(row["image_id"]) for row in manifest["images"]}
    available = {int(image["id"]) for image in coco["images"]}
    missing = sorted(selected - available)
    if missing:
        raise ValueError(f"manifest references unknown image IDs: {missing[:10]}")
    return {
        **{key: value for key, value in coco.items() if key not in ("images", "annotations")},
        "images": [image for image in coco["images"] if int(image["id"]) in selected],
        "annotations": [
            annotation
            for annotation in coco["annotations"]
            if int(annotation["image_id"]) in selected
        ],
    }
