"""Pre-registered EC-V decision losses; selection and teacher values are detached."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.ablation.cmp5L_shared_query_kd import (
    _pairwise_cxcywh_iou, candidate_ranking_kd, final_layer_hungarian_matches,
    normal_query_slice,
)


def layerwise_ranking_kd(outputs, teacher_logits, targets, matcher, *,
                         global_image_count, ddp_world_size, normal_query_count=300,
                         negative_topk=20):
    """V2: each layer selects its own Hungarian lesions and far candidates."""
    layers = list(outputs['aux_outputs']) + [
        {'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}
    ]
    if len(layers) != 4 or len(teacher_logits) != 4:
        raise ValueError('V2 requires four aligned decoder layers')
    results = []
    for layer, teacher in zip(layers, teacher_logits):
        matches = final_layer_hungarian_matches(
            matcher, layer, targets, normal_query_count=normal_query_count)
        results.append(candidate_ranking_kd(
            layer['pred_logits'].float(), teacher.float(), layer['pred_boxes'],
            targets, matches, normal_query_count=normal_query_count,
            negative_topk=negative_topk, negative_max_iou=.3,
            global_image_count=global_image_count, ddp_world_size=ddp_world_size))
    return {
        'loss': torch.stack([item['loss'] for item in results]).mean(),
        'layer_losses': [item['loss'] for item in results],
        'valid_pair_count': sum(item['valid_pair_count'] for item in results),
        'reverse_pair_count': sum(item['reverse_pair_count'] for item in results),
        'matched_class_counts': {
            str(k): sum(item['matched_class_counts'][str(k)] for item in results)
            for k in range(layers[-1]['pred_logits'].shape[-1])},
        'valid_class_counts': {
            str(k): sum(item['valid_class_counts'][str(k)] for item in results)
            for k in range(layers[-1]['pred_logits'].shape[-1])},
        'empty_image_count': results[-1]['empty_image_count'],
    }


def candidate_set_kd(student_logits, teacher_logits, student_boxes, targets, matches, *,
                     global_image_count, ddp_world_size, normal_query_count=300,
                     negative_topk=20):
    """V3: temperature-one KL across lesion + classwise top far candidates."""
    student = normal_query_slice(student_logits, normal_query_count).float()
    teacher = normal_query_slice(teacher_logits, normal_query_count).detach().float()
    if student.shape != teacher.shape or student_boxes.shape != (*student.shape[:2], 4):
        raise ValueError('V3 student/teacher/boxes not aligned')
    if len(targets) != len(matches) or len(targets) != len(student):
        raise ValueError('V3 targets/matches batch mismatch')
    if global_image_count <= 0 or ddp_world_size <= 0:
        raise ValueError('V3 invalid normalization')
    total = student.sum() * 0
    matched_counts = {str(i): 0 for i in range(student.shape[-1])}
    valid_counts = dict(matched_counts)
    valid_pairs = reverse_pairs = empty_images = 0
    for b, (target, (query_ids, target_ids)) in enumerate(zip(targets, matches)):
        with torch.no_grad():
            gt_boxes = target['boxes'].detach().to(student_boxes.device).float()
            labels = target['labels'].detach().to(student.device).long()
            if not len(gt_boxes):
                empty_images += 1
                continue
            eligible = torch.ones(normal_query_count, device=student.device, dtype=torch.bool)
            eligible[query_ids] = False
            eligible &= (_pairwise_cxcywh_iou(student_boxes[b].detach().float(), gt_boxes)
                         .max(1).values < .3)
            pool = eligible.nonzero(as_tuple=False).flatten()
        lesions = []
        for query, target_id in zip(query_ids.tolist(), target_ids.tolist()):
            category = int(labels[target_id])
            matched_counts[str(category)] += 1
            if not len(pool):
                continue
            with torch.no_grad():
                order = torch.argsort(student[b, pool, category].detach(),
                                      descending=True, stable=True)
                negatives = pool[order[:negative_topk]]
                candidates = torch.cat((torch.tensor([query], device=student.device), negatives))
                t = teacher[b, candidates, category]
                if t[0] <= t[1:].max():
                    reverse_pairs += 1
                    continue
            s = student[b, candidates, category]
            lesions.append(F.kl_div(F.log_softmax(s, dim=0), F.softmax(t, dim=0),
                                    reduction='sum'))
            valid_counts[str(category)] += 1
            valid_pairs += len(negatives)
        if lesions:
            total = total + torch.stack(lesions).mean()
    return {'loss': total * (float(ddp_world_size) / global_image_count),
            'valid_pair_count': valid_pairs, 'reverse_pair_count': reverse_pairs,
            'matched_class_counts': matched_counts, 'valid_class_counts': valid_counts,
            'empty_image_count': empty_images}


def np_increment_ranking_kd(student_logits, np_logits, normal_logits, student_boxes,
                            targets, matches, *, global_image_count, ddp_world_size,
                            normal_query_count=300, negative_topk=20):
    """V5: teacher-positive ranking pairs weighted by detached NP-only margin gain."""
    student = normal_query_slice(student_logits, normal_query_count).float()
    np_teacher = normal_query_slice(np_logits, normal_query_count).detach().float()
    normal_teacher = normal_query_slice(normal_logits, normal_query_count).detach().float()
    if (student.shape != np_teacher.shape or student.shape != normal_teacher.shape
            or student_boxes.shape != (*student.shape[:2], 4)):
        raise ValueError('V5 query view alignment failed')
    if len(targets) != len(matches) or len(targets) != len(student):
        raise ValueError('V5 targets/matches batch mismatch')
    if global_image_count <= 0 or ddp_world_size <= 0:
        raise ValueError('V5 invalid normalization')
    total = student.sum() * 0
    matched_counts = {str(i): 0 for i in range(student.shape[-1])}
    valid_counts = dict(matched_counts)
    valid_pairs = reverse_pairs = empty_images = 0
    weight_sum = 0.
    for b, (target, (query_ids, target_ids)) in enumerate(zip(targets, matches)):
        with torch.no_grad():
            gt_boxes = target['boxes'].detach().to(student_boxes.device).float()
            labels = target['labels'].detach().to(student.device).long()
            if not len(gt_boxes):
                empty_images += 1
                continue
            eligible = torch.ones(normal_query_count, device=student.device, dtype=torch.bool)
            eligible[query_ids] = False
            eligible &= (_pairwise_cxcywh_iou(student_boxes[b].detach().float(), gt_boxes)
                         .max(1).values < .3)
            pool = eligible.nonzero(as_tuple=False).flatten()
        lesions = []
        for query, target_id in zip(query_ids.tolist(), target_ids.tolist()):
            category = int(labels[target_id])
            matched_counts[str(category)] += 1
            if not len(pool):
                continue
            with torch.no_grad():
                order = torch.argsort(student[b, pool, category].detach(),
                                      descending=True, stable=True)
                negative = pool[order[:negative_topk]]
                d_np = np_teacher[b, query, category] - np_teacher[b, negative, category]
                d_normal = (normal_teacher[b, query, category]
                            - normal_teacher[b, negative, category])
                valid = d_np > 0
                reverse_pairs += int((~valid).sum())
                valid_pairs += int(valid.sum())
                weights = (d_np - d_normal).clamp(0, 1)[valid]
                weight_sum += float(weights.sum())
            if bool(valid.any()):
                chosen = negative[valid]
                d_student = student[b, query, category] - student[b, chosen, category]
                pair = F.binary_cross_entropy_with_logits(
                    d_student, d_np[valid].sigmoid(), reduction='none')
                lesions.append((pair * weights).sum() / int(valid.sum()))
                valid_counts[str(category)] += 1
        if lesions:
            total = total + torch.stack(lesions).mean()
    return {'loss': total * (float(ddp_world_size) / global_image_count),
            'valid_pair_count': valid_pairs, 'reverse_pair_count': reverse_pairs,
            'matched_class_counts': matched_counts, 'valid_class_counts': valid_counts,
            'empty_image_count': empty_images, 'weight_sum': weight_sum}


def _xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)


def _aligned_giou(first, second):
    left, right = _xyxy(first), _xyxy(second)
    intersection_wh = (torch.minimum(left[..., 2:], right[..., 2:])
                       - torch.maximum(left[..., :2], right[..., :2])).clamp(min=0)
    intersection = intersection_wh.prod(-1)
    area_left = (left[..., 2:] - left[..., :2]).clamp(min=0).prod(-1)
    area_right = (right[..., 2:] - right[..., :2]).clamp(min=0).prod(-1)
    union = (area_left + area_right - intersection).clamp(min=1e-12)
    outer_wh = (torch.maximum(left[..., 2:], right[..., 2:])
                - torch.minimum(left[..., :2], right[..., :2])).clamp(min=0)
    outer = outer_wh.prod(-1).clamp(min=1e-12)
    return intersection / union - (outer - union) / outer


def quality_filtered_box_kd(student_boxes, teacher_boxes, targets, matches, *,
                            global_image_count, ddp_world_size):
    """V4: detach teacher, only on matched GT whose teacher box is reliably better."""
    teacher_boxes = teacher_boxes.detach().float()
    if student_boxes.shape != teacher_boxes.shape or student_boxes.ndim != 3:
        raise ValueError('V4 student/teacher boxes differ')
    if len(targets) != len(matches) or len(targets) != len(student_boxes):
        raise ValueError('V4 target/match batch mismatch')
    if global_image_count <= 0 or ddp_world_size <= 0:
        raise ValueError('V4 invalid normalization')
    zero = student_boxes.sum() * 0
    total = zero
    active = matched_count = 0
    active_classes = {str(i): 0 for i in range(4)}
    for b, (target, (query_ids, target_ids)) in enumerate(zip(targets, matches)):
        with torch.no_grad():
            query_ids = query_ids.to(student_boxes.device)
            target_ids = target_ids.to(student_boxes.device)
            gt = target['boxes'].detach().float().to(student_boxes.device)
            if not len(gt):
                continue
            matched_count += len(query_ids)
            if not len(query_ids):
                continue
            teacher_iou = _pairwise_cxcywh_iou(teacher_boxes[b, query_ids], gt)
            student_iou = _pairwise_cxcywh_iou(student_boxes[b, query_ids].detach(), gt)
            row = torch.arange(len(query_ids), device=student_boxes.device)
            quality = teacher_iou[row, target_ids]
            improvement = quality - student_iou[row, target_ids]
            good = ((quality >= .5) & (improvement >= .05)
                    & (teacher_iou.argmax(-1) == target_ids))
            for target_id in target_ids[good].tolist():
                active_classes[str(int(target['labels'][target_id]))] += 1
        if bool(good.any()):
            s = student_boxes[b, query_ids[good]].float()
            t = teacher_boxes[b, query_ids[good]]
            loss = (s - t).abs().mean(-1) + (1 - _aligned_giou(s, t))
            total = total + loss.mean()
            active += int(good.sum())
    return {'loss': total * (float(ddp_world_size) / global_image_count),
            'active_count': active, 'matched_count': matched_count,
            'active_class_counts': active_classes}
