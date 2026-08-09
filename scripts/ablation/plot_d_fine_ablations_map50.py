#!/usr/bin/env python3
"""Plot AP50 training curves for the three D-FINE ablations."""

from pathlib import Path

from plot_cdn_map50 import OUTPUT_DIR, ROOT, load_ap50, plot, write_csv


RUNS = {
    "no-FDR decode": ROOT
    / "outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_decode_ignore9/log.txt",
    "no-GO-DDF": ROOT
    / "outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_go_ddf_ignore9/log.txt",
    "no-FDR + no-GO-DDF": ROOT
    / "outputs/ablation/ecdet_l_dinov2s_patch16_dec3_liver_no_fdr_no_go_ddf_ignore9/log.txt",
}


def main():
    series = {name: load_ap50(path) for name, path in RUNS.items()}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "d_fine_ablations_map50_training_curve.png"
    csv_path = OUTPUT_DIR / "d_fine_ablations_map50_training_curve.csv"
    plot(
        series,
        plot_path,
        "D-FINE ablations: AP50 during training",
        "Strict 9-class setting | one evaluation per epoch | raw values (no smoothing)",
        "Source: outputs/ablation/.../log.txt | Metric: test_coco_eval_bbox[1] (COCO AP50)",
    )
    write_csv(series, csv_path)
    for name, values in series.items():
        best_epoch, best_value = max(values, key=lambda item: item[1])
        print(
            f"{name}: {len(values)} epochs (0-{values[-1][0]}), "
            f"best AP50={best_value:.6f} at epoch {best_epoch}"
        )
    print(plot_path)
    print(csv_path)


if __name__ == "__main__":
    main()
