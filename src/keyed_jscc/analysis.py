from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path

import numpy as np


SEEDS = (1001, 1002, 1003)
WEIGHTS = (0.0, 0.2)
CASE_NAMES = tuple(f"seed{s}_lambda{w:g}" for s in SEEDS for w in WEIGHTS)
IMAGE_METRICS = (
    "bob_mse", "bob_ssim_valid", "bob_ssim_legacy", "eve_mse",
    "eve_ssim_valid", "eve_ssim_legacy", "accuracy", "prior_mse",
    "prior_ssim_valid", "prior_ssim_legacy",
)
SSIM_VARIANTS = (
    "legacy_zero_clipped", "zero_unclipped",
    "valid_same_guards_unclipped", "reference_valid_unclipped",
)


def read_rows(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("The output table is empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pooled_psnr(mse):
    values = np.asarray(mse, dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("MSE values must be finite, nonnegative, and nonempty")
    mean = float(values.mean())
    return float("inf") if mean == 0 else float(-10 * np.log10(mean))


def wilson_interval(correct, count):
    if count < 1 or not 0 <= correct <= count:
        raise ValueError("Invalid binomial counts")
    z = 1.959963984540054
    rate = correct / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return center - radius, center + radius


def load_matched_images(data_dir):
    rows = read_rows(Path(data_dir) / "matched_per_image.csv.gz")
    grouped = {}
    for case in CASE_NAMES:
        subset = [r for r in rows if r["case"] == case]
        indices = np.asarray([int(r["test_index"]) for r in subset])
        if not np.array_equal(indices, np.arange(6400, 10000)):
            raise ValueError("The matched experiment requires 3600 ordered holdout images per codec")
        grouped[case] = {m: np.asarray([float(r[m]) for r in subset]) for m in IMAGE_METRICS}
    if len(rows) != 3600 * len(CASE_NAMES):
        raise ValueError("Unexpected matched-experiment row count")
    return grouped


def matched_estimates(images):
    out = {}
    for case, arrays in images.items():
        mean = {metric: float(values.mean()) for metric, values in arrays.items()}
        mean["bob_dataset_psnr"] = pooled_psnr(arrays["bob_mse"])
        mean["eve_dataset_psnr"] = pooled_psnr(arrays["eve_mse"])
        mean["eve_minus_prior_valid_ssim"] = mean["eve_ssim_valid"] - mean["prior_ssim_valid"]
        mean["eve_minus_prior_legacy_ssim"] = mean["eve_ssim_legacy"] - mean["prior_ssim_legacy"]
        mean["eve_minus_prior_mse"] = mean["eve_mse"] - mean["prior_mse"]
        out[case] = mean
    return out


def conditional_matched_intervals(images, draws=2000, seed=2026101509):
    if draws < 1:
        raise ValueError("Bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    out = {}
    for case, arrays in images.items():
        samples = {m: [] for m in ("bob_ssim_valid", "eve_ssim_valid", "accuracy", "bob_dataset_psnr", "eve_minus_prior_valid_ssim", "eve_minus_prior_mse")}
        n = len(arrays["bob_mse"])
        for start in range(0, draws, 100):
            index = rng.integers(0, n, (min(100, draws - start), n))
            for metric in ("bob_ssim_valid", "eve_ssim_valid", "accuracy"):
                samples[metric].extend(arrays[metric][index].mean(1))
            samples["bob_dataset_psnr"].extend(-10 * np.log10(arrays["bob_mse"][index].mean(1)))
            for metric, left, right in (
                ("eve_minus_prior_valid_ssim", "eve_ssim_valid", "prior_ssim_valid"),
                ("eve_minus_prior_mse", "eve_mse", "prior_mse"),
            ):
                samples[metric].extend((arrays[left] - arrays[right])[index].mean(1))
        out[case] = {metric: np.quantile(values, [0.025, 0.975]) for metric, values in samples.items()}
    return out


def paired_seed_contrasts(estimates):
    rows = []
    metrics = ("bob_dataset_psnr", "bob_ssim_valid", "eve_ssim_valid", "eve_minus_prior_valid_ssim", "accuracy")
    for metric in metrics:
        differences = np.asarray([estimates[f"seed{s}_lambda0.2"][metric] - estimates[f"seed{s}_lambda0"][metric] for s in SEEDS])
        mean = float(differences.mean())
        sd = float(differences.std(ddof=1))
        radius = 4.302652729911275 * sd / math.sqrt(len(SEEDS))
        for seed, difference in zip(SEEDS, differences):
            rows.append({"unit": f"seed{seed}", "metric": metric, "difference": float(difference), "ci_low": "", "ci_high": ""})
        rows.append({"unit": "mean_of_3_seed_pairs", "metric": metric, "difference": mean, "ci_low": mean - radius, "ci_high": mean + radius})
    return rows


def verify_reference(actual, expected, key_fields, tolerance=1e-12):
    missing = []
    maximum = 0.0
    for row in expected:
        key = tuple(row[k] for k in key_fields)
        if key not in actual:
            missing.append("/".join(key))
            continue
        error = abs(float(actual[key]) - float(row["value"]))
        maximum = max(maximum, error)
    if missing or not np.isfinite(maximum) or maximum > tolerance:
        raise ValueError(f"Numerical reference mismatch: missing={missing}, maximum_absolute_error={maximum:.17g}")
    return maximum


def reproduce_matched(data_dir, output_dir, draws=2000):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    images = load_matched_images(data_dir)
    estimates = matched_estimates(images)
    intervals = conditional_matched_intervals(images, draws)
    selections = {r["case"]: r for r in read_rows(data_dir / "codec_selections.csv")}
    table = [{**selections[case], **values} for case, values in estimates.items()]
    write_rows(output_dir / "matched_codec_summary.csv", table)
    interval_rows = [{"case": case, "metric": metric, "estimate": estimates[case][metric], "ci_low": float(ci[0]), "ci_high": float(ci[1]), "resamples": draws, "bootstrap_seed": 2026101509, "independent_image_count": 3600, "key_noise_cells_per_image": 9} for case, metrics in intervals.items() for metric, ci in metrics.items()]
    write_rows(output_dir / "matched_conditional_intervals.csv", interval_rows)
    contrasts = paired_seed_contrasts(estimates)
    write_rows(output_dir / "paired_seed_contrasts.csv", contrasts)
    actual = {(case, metric): value for case, metrics in estimates.items() for metric, value in metrics.items()}
    for row in interval_rows:
        for suffix in ("ci_low", "ci_high"):
            actual[(row["case"], row["metric"] + "__" + suffix)] = row[suffix]
    for row in contrasts:
        case = "paired_mean" if row["unit"] == "mean_of_3_seed_pairs" else "paired_" + row["unit"]
        actual[(case, row["metric"])] = row["difference"]
        if row["ci_low"] != "":
            for suffix in ("ci_low", "ci_high"):
                actual[(case, row["metric"] + "__" + suffix)] = row[suffix]
    reference = read_rows(data_dir / "reference_matched_values.csv")
    if draws != 2000:
        reference = [r for r in reference if not r["metric"].endswith(("__ci_low", "__ci_high"))]
    error = verify_reference(actual, reference, ("case", "metric"))
    return estimates, intervals, {"analysis": "matched_codecs", "reference_values_checked": len(reference), "maximum_absolute_error": error}


def summarize_ssim(data_dir, output_dir, draws=10000):
    rows = read_rows(Path(data_dir) / "ssim_per_image.csv.gz")
    out = []
    for dataset in ("cifar", "kodak"):
        subset = [r for r in rows if r["dataset"] == dataset]
        models = list(dict.fromkeys(r["model"] for r in subset))
        arrays = {model: {m: np.asarray([float(r[m]) for r in subset if r["model"] == model]) for m in ("mse",) + SSIM_VARIANTS} for model in models}
        prior = "fitting_mean_image" if dataset == "cifar" else "mean_training_image"
        n = len(arrays[prior]["mse"])
        rng = np.random.default_rng(20261015)
        samples = {model: {m: [] for m in ("mse",) + SSIM_VARIANTS} for model in models}
        for start in range(0, draws, 100):
            indices = rng.integers(0, n, (min(100, draws - start), n))
            for model in models:
                for metric in samples[model]:
                    samples[model][metric].extend(arrays[model][metric][indices].mean(1))
        for model in models:
            mse_boot = np.asarray(samples[model]["mse"])
            psnr_ci = np.quantile(-10 * np.log10(mse_boot), [0.025, 0.975])
            out.append({"dataset": dataset, "model": model, "metric": "dataset_psnr_db", "estimate": pooled_psnr(arrays[model]["mse"]), "ci_low": psnr_ci[0], "ci_high": psnr_ci[1], "minimum": "", "maximum": "", "negative_images": "", "above_one_images": "", "minus_prior": "", "minus_prior_ci_low": "", "minus_prior_ci_high": "", "image_count": n})
            for metric in SSIM_VARIANTS:
                values = arrays[model][metric]
                boot = np.asarray(samples[model][metric])
                ci = np.quantile(boot, [0.025, 0.975])
                difference = values - arrays[prior][metric]
                difference_boot = boot - np.asarray(samples[prior][metric])
                delta_ci = np.quantile(difference_boot, [0.025, 0.975])
                out.append({"dataset": dataset, "model": model, "metric": metric, "estimate": float(values.mean()), "ci_low": ci[0], "ci_high": ci[1], "minimum": float(values.min()), "maximum": float(values.max()), "negative_images": int((values < 0).sum()), "above_one_images": int((values > 1).sum()), "minus_prior": float(difference.mean()), "minus_prior_ci_low": delta_ci[0], "minus_prior_ci_high": delta_ci[1], "image_count": n})
    write_rows(Path(output_dir) / "ssim_definition_sensitivity.csv", out)
    actual = {}
    for row in out:
        base = row["dataset"], row["model"]
        if row["metric"] == "dataset_psnr_db":
            actual[base + (row["metric"],)] = row["estimate"]
        else:
            for field, suffix in (("estimate", "mean"), ("ci_low", "ci_low"), ("ci_high", "ci_high"), ("minus_prior", "minus_prior"), ("minus_prior_ci_low", "minus_prior_ci_low"), ("minus_prior_ci_high", "minus_prior_ci_high")):
                actual[base + (row["metric"] + "__" + suffix,)] = row[field]
    reference = read_rows(Path(data_dir) / "reference_ssim_values.csv")
    error = verify_reference(actual, reference, ("dataset", "model", "metric"))
    return out, {"analysis": "ssim_definitions", "reference_values_checked": len(reference), "maximum_absolute_error": error}


def summarize_kodak(data_dir, output_dir, draws=10000):
    rows = read_rows(Path(data_dir) / "kodak_per_image.csv")
    models = list(dict.fromkeys(r["model"] for r in rows))
    ss = np.column_stack([[float(r["ssim"]) for r in rows if r["model"] == model] for model in models])
    mm = np.column_stack([[float(r["mse"]) for r in rows if r["model"] == model] for model in models])
    prior = models.index("mean_training_image")
    index = np.random.default_rng(20261015).integers(0, len(ss), (draws, len(ss)))
    bs, bm = ss[index].mean(1), mm[index].mean(1)
    out = []
    for j, model in enumerate(models):
        metrics = {
            "mean_ssim": (float(ss[:, j].mean()), bs[:, j]),
            "dataset_psnr_db": (pooled_psnr(mm[:, j]), -10 * np.log10(bm[:, j])),
            "ssim_minus_prior": (float((ss[:, j] - ss[:, prior]).mean()), bs[:, j] - bs[:, prior]),
            "mse_reduction_from_prior": (float((mm[:, prior] - mm[:, j]).mean()), bm[:, prior] - bm[:, j]),
            "dataset_psnr_gain_over_prior_db": (float(-10 * np.log10(mm[:, j].mean() / mm[:, prior].mean())), -10 * np.log10(bm[:, j] / bm[:, prior])),
            "positive_ssim_improvement_images": (int(np.sum(ss[:, j] > ss[:, prior])), None),
        }
        for metric, (mean, bootstrap) in metrics.items():
            low, high = ("", "") if bootstrap is None else np.quantile(bootstrap, [0.025, 0.975])
            out.append({"model": model, "metric": metric, "estimate": mean, "ci_low": low, "ci_high": high, "source_images": len(ss)})
    write_rows(Path(output_dir) / "kodak_source_image_summary.csv", out)
    actual = {(r["model"], r["metric"]): r["estimate"] for r in out}
    reference = read_rows(Path(data_dir) / "reference_kodak_values.csv")
    error = verify_reference(actual, reference, ("model", "metric"))
    return out, {"analysis": "kodak_source_images", "reference_values_checked": len(reference), "maximum_absolute_error": error}


def summarize_resources(data_dir, output_dir, draws=10000):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    rows = read_rows(data_dir / "equal_resource_per_image.csv.gz")
    out, actual = [], {}
    for snr in (6, 12):
        for group in (1, 8, 32, 128):
            subset = [r for r in rows if int(r["source_snr_db"]) == snr and int(r["group_size"]) == group]
            arrays = {scheme: {metric: np.asarray([float(r[metric]) for r in subset if r["scheme"] == scheme]) for metric in ("mse", "ssim_legacy", "ssim_valid")} for scheme in ("keyed", "reference")}
            group_ids = np.asarray([int(r["group_index"]) for r in subset if r["scheme"] == "keyed"])
            sizes = np.bincount(group_ids).astype(np.float64)
            sums = {scheme: {metric: np.bincount(group_ids, weights=values) for metric, values in metrics.items()} for scheme, metrics in arrays.items()}
            rng = np.random.default_rng(20261016)
            samples = {scheme: {metric: [] for metric in metrics} for scheme, metrics in arrays.items()}
            for start in range(0, draws, 128):
                indices = rng.integers(0, len(sizes), (min(128, draws - start), len(sizes)))
                denominator = sizes[indices].sum(1)
                for scheme in arrays:
                    for metric in arrays[scheme]:
                        samples[scheme][metric].extend(sums[scheme][metric][indices].sum(1) / denominator)
            for scheme in arrays:
                metrics = {"mse": (float(arrays[scheme]["mse"].mean()), np.asarray(samples[scheme]["mse"])), "dataset_psnr_db": (pooled_psnr(arrays[scheme]["mse"]), -10 * np.log10(samples[scheme]["mse"]))}
                for metric, name in (("ssim_legacy", "legacy_ssim"), ("ssim_valid", "valid_ssim")):
                    metrics[name] = float(arrays[scheme][metric].mean()), np.asarray(samples[scheme][metric])
                for metric, (estimate, bootstrap) in metrics.items():
                    ci = np.quantile(bootstrap, [0.025, 0.975])
                    out.append({"source_snr_db": snr, "group_size": group, "scheme": scheme, "metric": metric, "estimate": estimate, "ci_low": ci[0], "ci_high": ci[1], "bootstrap_groups": len(sizes), "last_group_images": int(sizes[-1])})
            for metric, name in (("ssim_legacy", "legacy_ssim"), ("ssim_valid", "valid_ssim")):
                estimate = float(arrays["keyed"][metric].mean() - arrays["reference"][metric].mean())
                bootstrap = np.asarray(samples["keyed"][metric]) - np.asarray(samples["reference"][metric])
                ci = np.quantile(bootstrap, [0.025, 0.975])
                out.append({"source_snr_db": snr, "group_size": group, "scheme": "keyed_minus_reference", "metric": name, "estimate": estimate, "ci_low": ci[0], "ci_high": ci[1], "bootstrap_groups": len(sizes), "last_group_images": int(sizes[-1])})
    write_rows(output_dir / "equal_resource_summary.csv", out)
    for row in out:
        base = str(row["source_snr_db"]), str(row["group_size"]), row["scheme"]
        if row["metric"] in ("mse", "dataset_psnr_db"):
            actual[base + (row["metric"],)] = row["estimate"]
        else:
            for field, suffix in (("estimate", "mean"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
                actual[base + (row["metric"] + "__" + suffix,)] = row[field]
    reference = read_rows(data_dir / "reference_resource_values.csv")
    error = verify_reference(actual, reference, ("source_snr_db", "group_size", "scheme", "metric"))
    return out, {"analysis": "equal_transmission_resources", "reference_values_checked": len(reference), "maximum_absolute_error": error}


def summarize_headers(data_dir, output_dir):
    rows = []
    for row in read_rows(Path(data_dir) / "header_monte_carlo.csv"):
        n = int(row["frames"])
        for count_field, metric in (("wrong_headers", "header_error"), ("undetected_wrong_metadata", "undetected_metadata")):
            count = int(row[count_field])
            low, high = wilson_interval(count, n)
            rows.append({"snr_db": int(row["snr_db"]), "metric": metric, "events": count, "frames": n, "rate": count / n, "wilson_low": low, "wilson_high": high, "zero_event_one_sided_95_upper": 1 - 0.05 ** (1 / n) if count == 0 else ""})
    write_rows(Path(output_dir) / "header_reliability.csv", rows)
    return rows


def generate_figures(data_dir, output_dir, estimates, intervals, ssim, kodak, resource, headers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.hashsalt": "keyed-jscc", "savefig.facecolor": "white"})
    palette = {1001: "#0072B2", 1002: "#D55E00", 1003: "#009E73"}

    def save(fig, name):
        fig.savefig(output_dir / f"{name}.png", dpi=200)
        fig.savefig(output_dir / f"{name}.svg", metadata={"Date": None})
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    panels = (("bob_dataset_psnr", "Bob PSNR", "dB", 1), ("bob_ssim_valid", "Bob reconstruction", "Valid-window SSIM", 1), ("accuracy", "Selected class attacker", "Accuracy (%)", 100), ("eve_ssim_valid", "Selected reconstruction attacker", "Valid-window SSIM", 1), ("eve_minus_prior_valid_ssim", "Attacker SSIM relative to image prior", "Paired SSIM difference", 1), ("eve_minus_prior_mse", "Attacker MSE relative to image prior", "Paired MSE difference", 1))
    for ax, (metric, title, ylabel, scale) in zip(axes.flat, panels):
        for j, seed in enumerate(SEEDS):
            x = np.asarray([0, 1], dtype=float) + (j - 1) * 0.05
            y = np.asarray([estimates[f"seed{seed}_lambda{weight:g}"][metric] for weight in WEIGHTS]) * scale
            ci = np.asarray([intervals[f"seed{seed}_lambda{weight:g}"][metric] for weight in WEIGHTS]).T * scale
            ax.errorbar(x, y, yerr=np.maximum(np.stack([y - ci[0], ci[1] - y]), 0), marker="o", capsize=3, color=palette[seed], label=f"Seed {seed}")
        if metric == "accuracy":
            ax.axhline(10, color="0.4", linestyle=":", linewidth=1)
        if metric.startswith("eve_minus_prior"):
            ax.axhline(0, color="0.4", linestyle=":", linewidth=1)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set(title=title, ylabel=ylabel, xlabel="Defense weight", xticks=[0, 1], xticklabels=["0", "0.2"], xlim=(-0.2, 1.2))
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False)
    save(fig, "matched_codec_grid")

    curves = read_rows(data_dir / "codec_training_curves.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True)
    for row, weight in enumerate(WEIGHTS):
        for col, seed in enumerate(SEEDS):
            ax = axes[row, col]
            subset = [r for r in curves if r["case"] == f"seed{seed}_lambda{weight:g}"]
            epochs = [int(r["epoch"]) for r in subset]
            ax.plot(epochs, [float(r["fitting_mse"]) for r in subset], "--", color=palette[seed], label="Fitting")
            ax.plot(epochs, [float(r["validation_mse"]) for r in subset], color=palette[seed], label="Validation")
            selected = int(subset[0]["selected_epoch"])
            ax.plot(selected, float(subset[selected - 1]["validation_mse"]), "k*", markersize=10, label="Selected epoch")
            ax.set(title=f"Seed {seed}, defense weight {weight:g}", yscale="log", xlabel="Codec epoch", ylabel="Reconstruction MSE")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    save(fig, "codec_training_grid")

    candidates = read_rows(data_dir / "attacker_candidates.csv")
    classifiers = list(dict.fromkeys(r["candidate"] for r in candidates if r["kind"] in ("class", "sets")))
    reconstructors = list(dict.fromkeys(r["candidate"] for r in candidates if r["kind"] in ("mlp", "cnn")))
    labels = {"raw_classifier_51": "Raw 51", "raw_classifier_52": "Raw 52", "raw_classifier_53": "Raw 53", "raw_classifier_54": "Raw 54", "raw_classifier_55": "Raw 55", "invariant_cached_61": "Inv cached 61", "invariant_cached_62": "Inv cached 62", "invariant_surrogate_61": "Inv resampled 61", "invariant_surrogate_62": "Inv resampled 62", "deepsets_surrogate": "DeepSets", "mlp_reconstruction_71": "MLP 71", "mlp_reconstruction_72": "MLP 72", "mlp_reconstruction_73": "MLP 73", "cnn_cached_nonce": "CNN + nonce", "cnn_cached_no_nonce": "CNN", "cnn_surrogate_no_nonce": "CNN32 resampled", "cnn_large_surrogate_no_nonce": "CNN256 resampled"}
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
    for ax, names, metric, scale, title in ((axes[0], classifiers, "accuracy", 100, "Class accuracy (%)"), (axes[1], reconstructors, "valid_gain", 1000, "Valid SSIM gain over image prior (×10⁻³)"), (axes[2], reconstructors, "mse_gain", 100000, "MSE reduction from image prior (×10⁻⁵)")):
        matrix = np.asarray([[float(next(r[metric] for r in candidates if r["case"] == case and r["candidate"] == name)) * scale for name in names] for case in CASE_NAMES])
        center = 10 if metric == "accuracy" else 0
        half = max(float(np.max(np.abs(matrix - center))) * 1.05, 1e-8)
        norm = TwoSlopeNorm(vmin=center - half, vcenter=center, vmax=center + half)
        im = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
        for i, case in enumerate(CASE_NAMES):
            for j, name in enumerate(names):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if abs(norm(matrix[i, j]) - 0.5) > 0.3 else "black", fontsize=8)
                if int(next(r["selected"] for r in candidates if r["case"] == case and r["candidate"] == name)):
                    ax.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fill=False, edgecolor="black", linewidth=1.5))
        ax.set(title=title, xticks=np.arange(len(names)), xticklabels=[labels.get(n, n) for n in names], yticks=np.arange(6), yticklabels=[f"{s} / {w:g}" for s in SEEDS for w in WEIGHTS], ylabel="Codec seed / defense weight")
        ax.tick_params(axis="x", labelrotation=20)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    save(fig, "attacker_family_grid")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    hdr = [r for r in headers if r["metric"] == "header_error"]
    axes[0, 0].errorbar([r["snr_db"] for r in hdr], [r["rate"] for r in hdr], yerr=[[max(r["rate"] - r["wilson_low"], 0) for r in hdr], [r["wilson_high"] - r["rate"] for r in hdr]], marker="o", capsize=3, color="#0072B2")
    axes[0, 0].set(title="Header reliability", xlabel="Header signal-to-noise ratio (dB)", ylabel="Decoded header error rate")
    profiles = read_rows(data_dir / "resource_profiles.csv")
    for snr, color in ((6, "#D55E00"), (12, "#0072B2")):
        subset = [r for r in profiles if int(r["source_snr_db"]) == snr]
        x = [int(r["group_size"]) for r in subset]
        axes[0, 1].plot(x, [float(r["complex_uses_per_image"]) for r in subset], "o-", color=color, label=f"{snr} dB")
        rr = [r for r in resource if r["source_snr_db"] == snr and r["scheme"] == "keyed_minus_reference" and r["metric"] == "valid_ssim"]
        axes[1, 0].errorbar([r["group_size"] for r in rr], [r["estimate"] for r in rr], yerr=[[r["estimate"] - r["ci_low"] for r in rr], [r["ci_high"] - r["estimate"] for r in rr]], marker="o", capsize=3, color=color, label=f"{snr} dB")
        axes[1, 1].plot(x, [float(r["payload_snr_max_db"]) for r in subset], "o-", color=color, label=f"{snr} dB source")
    axes[0, 1].set(title="Total transmission resources", ylabel="Complex channel uses / image", xlabel="Images per header", xscale="log")
    axes[1, 0].set(title="Equal-resource reconstruction", ylabel="Keyed minus reference valid SSIM", xlabel="Images per header", xscale="log")
    axes[1, 1].set(title="Payload power allocation", ylabel="Payload signal-to-noise ratio (dB)", xlabel="Images per header", xscale="log")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    save(fig, "header_resource_grid")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, dataset in zip(axes[:2], ("cifar", "kodak")):
        models = list(dict.fromkeys(r["model"] for r in ssim if r["dataset"] == dataset))
        x = np.arange(len(models))
        for metric, offset, label, color in (("legacy_zero_clipped", -0.16, "Zero-padded", "#0072B2"), ("reference_valid_unclipped", 0.16, "Valid-window", "#D55E00")):
            vals = [next(r["estimate"] for r in ssim if r["dataset"] == dataset and r["model"] == model and r["metric"] == metric) for model in models]
            ax.bar(x + offset, vals, width=0.3, label=label, color=color)
        ax.set(title=f"{dataset.upper()} SSIM definitions", ylabel="Mean SSIM", xticks=x, xticklabels=[m.replace("_", " ") for m in models])
        ax.tick_params(axis="x", labelrotation=55)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    models = list(dict.fromkeys(r["model"] for r in kodak))
    rows = [next(r for r in kodak if r["model"] == model and r["metric"] == "ssim_minus_prior") for model in models]
    x = np.arange(len(rows))
    axes[2].errorbar(x, [r["estimate"] for r in rows], yerr=[[r["estimate"] - r["ci_low"] for r in rows], [r["ci_high"] - r["estimate"] for r in rows]], fmt="o", capsize=4, color="#009E73")
    axes[2].axhline(0, color="0.5", linestyle=":")
    axes[2].set(title="Kodak source-image comparison", ylabel="SSIM difference from image prior", xticks=x, xticklabels=[m.replace("_", " ") for m in models])
    axes[2].tick_params(axis="x", labelrotation=55)
    axes[2].grid(axis="y", alpha=0.2)
    save(fig, "metric_and_kodak_grid")


def reproduce_analysis(data_dir, output_dir, plots=True):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimates, intervals, matched_check = reproduce_matched(data_dir, output_dir)
    ssim, ssim_check = summarize_ssim(data_dir, output_dir)
    kodak, kodak_check = summarize_kodak(data_dir, output_dir)
    resource, resource_check = summarize_resources(data_dir, output_dir)
    headers = summarize_headers(data_dir, output_dir)
    checks = [matched_check, ssim_check, kodak_check, resource_check]
    write_rows(output_dir / "numerical_checks.csv", checks)
    if plots:
        generate_figures(data_dir, output_dir, estimates, intervals, ssim, kodak, resource, headers)
    return checks


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated"))
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()
    checks = reproduce_analysis(arguments.data_dir, arguments.output_dir, plots=not arguments.no_plots)
    for row in checks:
        print(f"{row['analysis']}: {row['reference_values_checked']} values, maximum absolute error {row['maximum_absolute_error']:.3g}")
    print(f"Results saved to {arguments.output_dir}")
