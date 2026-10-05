"""Opt-in V3_4L: original V3 sets independently built at four decoder layers."""
import torch
from scripts.ablation.cmp5L_ec_v_losses import candidate_set_kd
from scripts.ablation.cmp5L_shared_query_kd import final_layer_hungarian_matches


def four_layer_candidate_set_kd(outputs, teacher_logits, targets, matcher, *,
                                global_image_count, ddp_world_size,
                                normal_query_count=300, negative_topk=20):
    layers = list(outputs['aux_outputs']) + [
        {'pred_logits': outputs['pred_logits'], 'pred_boxes': outputs['pred_boxes']}]
    if len(layers) != 4 or len(teacher_logits) != 4:
        raise ValueError('V3_4L requires four aligned normal decoder layers')
    results = []
    for layer, teacher in zip(layers, teacher_logits):
        matches = final_layer_hungarian_matches(matcher, layer, targets,
                                                normal_query_count=normal_query_count)
        results.append(candidate_set_kd(
            layer['pred_logits'], teacher, layer['pred_boxes'], targets, matches,
            global_image_count=global_image_count, ddp_world_size=ddp_world_size,
            normal_query_count=normal_query_count, negative_topk=negative_topk))
    return {
        'loss': torch.stack([r['loss'] for r in results]).mean(),
        'layer_losses': [r['loss'] for r in results],
        'layer_valid_pair_counts': [r['valid_pair_count'] for r in results],
        'layer_matched_class_counts': [r['matched_class_counts'] for r in results],
        'valid_pair_count': sum(r['valid_pair_count'] for r in results),
        'reverse_pair_count': sum(r['reverse_pair_count'] for r in results),
        'empty_image_count': results[-1]['empty_image_count'],
        'matched_class_counts': {k:sum(r['matched_class_counts'][k] for r in results)
                                 for k in results[-1]['matched_class_counts']},
        'valid_class_counts': {k:sum(r['valid_class_counts'][k] for r in results)
                               for k in results[-1]['valid_class_counts']}}
