"""Regenerate all figures from updated CSV files."""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = '/mnt/shared-storage-user/zhupengyu1/tangling/grpo/CURE/analysis'
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(BASE_DIR, 'train_curve.csv'))
eval_df = pd.read_csv(os.path.join(BASE_DIR, 'eval_summary.csv'))

train_rows = train.to_dict('records')
eval_rows = eval_df.to_dict('records')


def rolling_avg(values, window=10):
    arr = np.array(values, dtype=float)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            start = max(0, i - window + 1)
            seg = arr[start:i+1]
            valid = seg[~np.isnan(seg)]
            if len(valid):
                out[i] = np.mean(valid)
    return out


def setup_plot_style():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "lines.linewidth": 1.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def plot_single_metric(steps, values, title, ylabel, save_path,
                       window=10, eval_steps=None, eval_values=None,
                       extra_series=None):
    setup_plot_style()
    fig, ax = plt.subplots()
    arr = np.array(values, dtype=float)
    ax.plot(steps, arr, alpha=0.3, color="C0", linewidth=0.8)
    smoothed = rolling_avg(values, window)
    ax.plot(steps, smoothed, color="C0", linewidth=2.0,
            label=f"{ylabel} (smoothed, w={window})")
    if extra_series:
        for label, vals, color in extra_series:
            arr2 = np.array(vals, dtype=float)
            ax.plot(steps, arr2, alpha=0.3, color=color, linewidth=0.8)
            sm2 = rolling_avg(vals, window)
            ax.plot(steps, sm2, color=color, linewidth=2.0,
                    label=f"{label} (smoothed)")
    if eval_steps and eval_values:
        ax.scatter(eval_steps, eval_values, color="red", zorder=5, s=60,
                   marker="D", label="Eval on CodeContests")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


steps = [r["step"] for r in train_rows]

def col(key):
    return [r.get(key, float("nan")) for r in train_rows]

eval_valid = [r for r in eval_rows[1:] if pd.notna(r.get("code_acc"))]
eval_steps_all = [r["step"] for r in eval_valid]
eval_code_acc = [r["code_acc"] for r in eval_valid]
eval_case_acc_vals = [r.get("case_acc") for r in eval_valid]
eval_bon16 = [r.get("bon_16_16_acc") for r in eval_valid]

setup_plot_style()
print("Regenerating figures from updated CSV data...\n")

# A: Code Accuracy
plot_single_metric(
    steps, col("code_acc"),
    "Code Accuracy vs Training Steps", "Code Accuracy",
    os.path.join(FIG_DIR, "A_code_acc_vs_steps.png"),
    eval_steps=eval_steps_all, eval_values=eval_code_acc,
)

# B: UT Accuracy
plot_single_metric(
    steps, col("case_acc"),
    "Unit Test Accuracy vs Training Steps", "UT Accuracy",
    os.path.join(FIG_DIR, "B_case_acc_vs_steps.png"),
    eval_steps=[s for s, v in zip(eval_steps_all, eval_case_acc_vals) if v],
    eval_values=[v for v in eval_case_acc_vals if v],
)

# C: BoN Accuracy
fig, ax = plt.subplots(figsize=(10, 6))
bon44 = col("bon_4_4_acc")
bon1616 = col("bon_16_16_acc")
ax.plot(steps, np.array(bon44, dtype=float), alpha=0.25, color="C0", linewidth=0.8)
ax.plot(steps, rolling_avg(bon44), color="C0", linewidth=2.0, label="BoN(4,4) smoothed")
ax.plot(steps, np.array(bon1616, dtype=float), alpha=0.25, color="C1", linewidth=0.8)
ax.plot(steps, rolling_avg(bon1616), color="C1", linewidth=2.0, label="BoN(16,16) smoothed")
valid_eval_bon = [(s, v) for s, v in zip(eval_steps_all, eval_bon16) if v]
if valid_eval_bon:
    ax.scatter([s for s, v in valid_eval_bon], [v for s, v in valid_eval_bon],
               color="red", zorder=5, s=60, marker="D", label="Eval BoN(16,16)")
ax.set_xlabel("Training Step")
ax.set_ylabel("BoN Accuracy")
ax.set_title("Best-of-N Accuracy vs Training Steps")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
path = os.path.join(FIG_DIR, "C_bon_acc_vs_steps.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# D: Response Length
plot_single_metric(
    steps, col("mean_code_len"),
    "Response Length vs Training Steps", "Token Length",
    os.path.join(FIG_DIR, "D_response_length_vs_steps.png"),
    extra_series=[("Case Response Length", col("mean_case_len"), "C1")],
)

# E: p01/p00
fig, ax = plt.subplots(figsize=(10, 6))
p01 = col("p_01")
p00 = col("p_00")
ax.plot(steps, np.array(p01, dtype=float), alpha=0.25, color="C0", linewidth=0.8)
ax.plot(steps, rolling_avg(p01), color="C0", linewidth=2.0, label="p_01 (smoothed)")
ax.plot(steps, np.array(p00, dtype=float), alpha=0.25, color="C3", linewidth=0.8)
ax.plot(steps, rolling_avg(p00), color="C3", linewidth=2.0, label="p_00 (smoothed)")
ax.set_xlabel("Training Step")
ax.set_ylabel("Proportion")
ax.set_title("p_01 / p_00 vs Training Steps")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
path = os.path.join(FIG_DIR, "E_p01_p00_vs_steps.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {path}")

# F: Reward Mean
plot_single_metric(
    steps, col("code_reward_mean"),
    "Reward Mean vs Training Steps", "Reward Mean",
    os.path.join(FIG_DIR, "F_reward_vs_steps.png"),
    extra_series=[("Case Reward Mean", col("case_reward_mean"), "C1")],
)

# G: Baseline vs Trained bar chart
step0 = train_rows[0] if train_rows else {}
best_eval = {}
for r in eval_rows[1:]:
    ca = r.get("code_acc")
    if pd.notna(ca) and (not best_eval or ca > best_eval.get("code_acc", 0)):
        best_eval = r

if step0 and best_eval:
    labels = ["Code Acc", "UT Acc", "BoN(4,4)", "BoN(16,16)"]
    keys = ["code_acc", "case_acc", "bon_4_4_acc", "bon_16_16_acc"]
    baseline_vals = [float(step0.get(k, 0) or 0) for k in keys]
    trained_vals = [float(best_eval.get(k, 0) or 0) for k in keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Baseline (Step 0)", color="C7")
    bars2 = ax.bar(x + width/2, trained_vals, width,
                   label=f"Trained (Step {int(best_eval.get('step', 0))})", color="C0")
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline vs Trained: Key Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(max(baseline_vals), max(trained_vals)) * 1.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "G_baseline_vs_trained.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# Also update comparison_table.txt
sep = "=" * 80
header = f"{sep}\n  CURE 复现实验: Baseline vs Trained 对比表\n{sep}\n\n"

def fmt(v, pct=True):
    if v is None or v == "" or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{float(v):.4f}" if pct else f"{float(v):.1f}"

metrics = [
    ("Code Accuracy",       "code_acc"),
    ("Code Accumulate Acc", "code_acc_acc"),
    ("UT Accuracy",         "case_acc"),
    ("UT Accumulate Acc",   "case_acc_acc"),
    ("p_01",                "p_01"),
    ("p_00",                "p_00"),
    ("BoN(4,4) Acc",        "bon_4_4_acc"),
    ("BoN(16,16) Acc",      "bon_16_16_acc"),
    ("Code Resp Length",    "mean_code_len"),
    ("Case Resp Length",    "mean_case_len"),
]

lines = [header]
lines.append(f"{'Metric':<25s} {'Baseline (Step 0)':>18s} {'Best Eval':>18s} {'Eval Step':>10s}\n")
lines.append("-" * 75 + "\n")
for label, key in metrics:
    is_len = key in ("mean_code_len", "mean_case_len")
    b_val = fmt(step0.get(key), pct=not is_len)
    e_val = fmt(best_eval.get(key), pct=not is_len)
    e_step = str(int(best_eval.get("step", 0)))
    lines.append(f"{label:<25s} {b_val:>18s} {e_val:>18s} {e_step:>10s}\n")
lines.append("\n" + "-" * 75 + "\n")
lines.append("Note: Baseline uses Step 0 training-set metrics (not eval-set).\n")
lines.append("      Best Eval is the checkpoint with highest code_acc on CodeContests eval.\n")
lines.append("      To get a true baseline, run eval.py on the pretrained model.\n")

table_path = os.path.join(BASE_DIR, "comparison_table.txt")
with open(table_path, "w") as f:
    f.writelines(lines)
print(f"\n  Updated: {table_path}")

# Update experiment_config.txt training steps
config_path = os.path.join(BASE_DIR, "experiment_config.txt")
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config_text = f.read()
    config_text = config_text.replace(
        "总步数:           150 (论文 7B 模型: 350 steps)",
        "总步数:           350 (论文 7B 模型: 350 steps)"
    )
    config_text = config_text.replace(
        "1. 训练步数: 150 步 (论文 7B 模型建议 350 步)",
        "1. 训练步数: 350 步 (与论文 7B 模型一致)"
    )
    with open(config_path, 'w') as f:
        f.write(config_text)
    print(f"  Updated: {config_path}")

print("\nAll 7 figures regenerated successfully!")
