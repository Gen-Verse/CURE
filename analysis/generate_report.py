#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from cure_profiles import get_profile_thresholds


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_experiment_dir(project_root: str) -> str:
    exp_base = os.path.join(project_root, "experiments")
    candidates = sorted(p for p in glob.glob(os.path.join(exp_base, "*")) if os.path.isdir(p))
    if not candidates:
        raise SystemExit(f"No experiment directories found under {exp_base}")
    return candidates[-1]


def parse_rl_blocks(filepath: str) -> list[dict]:
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    blocks = []
    i = 0
    while i + 8 < len(lines):
        m0 = re.search(r"code response length:\s*([\d.]+).*case response length:\s*([\d.]+)", lines[i])
        m1 = re.search(r"code acc:\s*([\d.]+).*code accumulate acc:\s*([\d.]+)", lines[i + 1])
        m2 = re.search(r"case acc:\s*([\d.]+).*case accumulate acc:\s*([\d.]+)", lines[i + 2])
        if not (m0 and m1 and m2):
            i += 1
            continue
        block = {
            "mean_code_len": float(m0.group(1)),
            "mean_case_len": float(m0.group(2)),
            "code_acc": float(m1.group(1)),
            "code_acc_acc": float(m1.group(2)),
            "case_acc": float(m2.group(1)),
            "case_acc_acc": float(m2.group(2)),
            "p_01": float(lines[i + 3].split(":")[1].strip()),
            "p_00": float(lines[i + 4].split(":")[1].strip()),
        }
        m6 = re.search(r"acc:\s*([\d.]+).*accumulate acc:\s*([\d.]+)", lines[i + 6])
        m8 = re.search(r"acc:\s*([\d.]+).*accumulate acc:\s*([\d.]+)", lines[i + 8])
        block["bon_4_4_acc"] = float(m6.group(1)) if m6 else 0.0
        block["bon_4_4_acc_acc"] = float(m6.group(2)) if m6 else 0.0
        block["bon_16_16_acc"] = float(m8.group(1)) if m8 else 0.0
        block["bon_16_16_acc_acc"] = float(m8.group(2)) if m8 else 0.0
        blocks.append(block)
        i += 9
    return blocks


def parse_reward_blocks(filepath: str) -> list[dict]:
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    blocks = []
    i = 0
    while i + 1 < len(lines):
        if not lines[i].startswith("estimated_code_reward:"):
            i += 1
            continue
        m_code = re.search(r"mean=([\d.-]+),\s*std=([\d.-]+)", lines[i])
        m_case = re.search(r"mean=([\d.-]+),\s*std=([\d.-]+)", lines[i + 1])
        if m_code and m_case:
            blocks.append(
                {
                    "code_reward_mean": float(m_code.group(1)),
                    "code_reward_std": float(m_code.group(2)),
                    "case_reward_mean": float(m_case.group(1)),
                    "case_reward_std": float(m_case.group(2)),
                }
            )
            i += 2
        else:
            i += 1
    return blocks


def parse_eval_blocks(filepath: str) -> list[dict]:
    if not filepath or not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    blocks = []
    i = 0
    while i + 10 < len(lines):
        if not lines[i].startswith("code acc"):
            i += 1
            continue
        block = {
            "code_acc": float(lines[i].split(":")[-1].strip()),
            "code_acc_acc": float(lines[i + 1].split(":")[-1].strip()),
            "case_acc": float(lines[i + 2].split(":")[-1].strip()),
            "case_acc_acc": float(lines[i + 3].split(":")[-1].strip()),
            "p_01": float(lines[i + 4].split(":")[-1].strip()),
            "p_00": float(lines[i + 5].split(":")[-1].strip()),
        }
        m7 = re.search(r"acc:\s*([\d.]+).*accumulate acc:\s*([\d.]+)", lines[i + 7])
        m9 = re.search(r"acc:\s*([\d.]+).*accumulate acc:\s*([\d.]+)", lines[i + 9])
        m10 = re.search(
            r"code average response length:\s*([\d.]+).*unit test average response length:\s*([\d.]+)",
            lines[i + 10],
        )
        block["bon_4_4_acc"] = float(m7.group(1)) if m7 else 0.0
        block["bon_4_4_acc_acc"] = float(m7.group(2)) if m7 else 0.0
        block["bon_16_16_acc"] = float(m9.group(1)) if m9 else 0.0
        block["bon_16_16_acc_acc"] = float(m9.group(2)) if m9 else 0.0
        block["mean_code_len"] = float(m10.group(1)) if m10 else 0.0
        block["mean_case_len"] = float(m10.group(2)) if m10 else 0.0
        blocks.append(block)
        i += 11
    return blocks


def select_result_file(files: list[str], preferred_tokens: list[str], fallback_token: str = "") -> str:
    for file in files:
        basename = os.path.basename(file)
        if all(token in basename for token in preferred_tokens if token):
            return file
    if fallback_token:
        for file in files:
            if fallback_token in os.path.basename(file):
                return file
    return files[0] if files else ""


def build_train_curve(exp_dir: str, run_metadata: dict) -> list[dict]:
    opt_results = os.path.join(exp_dir, "optimization", "results")
    rl_files = sorted(glob.glob(os.path.join(opt_results, "results-rl-*.txt")))
    reward_files = [
        f
        for f in sorted(glob.glob(os.path.join(opt_results, "results-*.txt")))
        if "results-rl-" not in os.path.basename(f) and "results-eval-" not in os.path.basename(f)
    ]
    pretrain_token = run_metadata.get("pretrain_model", "").replace("/", ".")
    optimized_token = run_metadata.get("optimized_model_path", "").replace("/", ".")
    dataset = run_metadata.get("train_dataset", "")

    pretrain_rl = parse_rl_blocks(select_result_file(rl_files, [pretrain_token, dataset]))
    optimized_rl = parse_rl_blocks(select_result_file(rl_files, [optimized_token, dataset], "optimized"))
    pretrain_rw = parse_reward_blocks(select_result_file(reward_files, [pretrain_token, dataset]))
    optimized_rw = parse_reward_blocks(select_result_file(reward_files, [optimized_token, dataset], "optimized"))

    total_rows = []
    if pretrain_rl:
        row = {"step": 0}
        row.update(pretrain_rl[-1])
        if pretrain_rw:
            row.update(pretrain_rw[-1])
        total_rows.append(row)

    for idx, block in enumerate(optimized_rl, start=1):
        row = {"step": idx}
        row.update(block)
        if idx - 1 < len(optimized_rw):
            row.update(optimized_rw[idx - 1])
        total_rows.append(row)
    return total_rows


def build_eval_summary(exp_dir: str, run_metadata: dict) -> list[dict]:
    eval_results = os.path.join(exp_dir, "evaluation", "results")
    dataset = run_metadata.get("eval_dataset", "")
    eval_files = sorted(glob.glob(os.path.join(eval_results, f"results-*{dataset}*.txt")))
    eval_file = select_result_file(eval_files, [dataset], "results-eval-")
    blocks = parse_eval_blocks(eval_file)
    eval_interval = int(run_metadata.get("eval_interval", 1))
    rows = []
    rows.append(
        {
            "model": f"baseline ({Path(run_metadata.get('pretrain_model', 'model')).name})",
            "checkpoint": "step0-train-metric",
            "dataset": dataset,
            "step": 0,
        }
    )
    for idx, block in enumerate(blocks, start=1):
        row = {
            "model": f"trained_step{idx * eval_interval}",
            "checkpoint": f"iter{idx * eval_interval}",
            "dataset": dataset,
            "step": idx * eval_interval,
        }
        row.update(block)
        rows.append(row)
    return rows


def build_policy_curve(raw_dir: str) -> list[dict]:
    records = read_jsonl(os.path.join(raw_dir, "policy_metrics.jsonl"))
    merged = {}
    for record in records:
        key = (record.get("step", 0), record.get("tag", "all"))
        merged.setdefault(key, {"step": key[0], "tag": key[1]})
        merged[key].update(record)
    return [merged[key] for key in sorted(merged)]


def write_csv(path: str, rows: list[dict], columns: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rolling_avg(values, window=5):
    arr = np.array(values, dtype=float)
    out = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        seg = arr[start : i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid):
            out[i] = np.mean(valid)
    return out


def save_line_plot(path: str, xs, series, title: str, ylabel: str):
    plt.figure(figsize=(10, 6), dpi=150)
    for label, values, color in series:
        arr = np.array(values, dtype=float)
        plt.plot(xs, arr, color=color, alpha=0.25, linewidth=0.9)
        plt.plot(xs, rolling_avg(arr), color=color, linewidth=2.0, label=label)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def build_comparison_table(train_rows: list[dict], eval_rows: list[dict]) -> str:
    step0 = train_rows[0] if train_rows else {}
    best_eval = {}
    for row in eval_rows[1:]:
        if row.get("code_acc") is not None and (not best_eval or row["code_acc"] > best_eval.get("code_acc", -1)):
            best_eval = row

    def fmt(value, is_len=False):
        if value is None or value == "":
            return "N/A"
        return f"{float(value):.1f}" if is_len else f"{float(value):.4f}"

    metrics = [
        ("Code Accuracy", "code_acc", False),
        ("UT Accuracy", "case_acc", False),
        ("BoN(4,4) Acc", "bon_4_4_acc", False),
        ("BoN(16,16) Acc", "bon_16_16_acc", False),
        ("Code Resp Length", "mean_code_len", True),
        ("Case Resp Length", "mean_case_len", True),
    ]
    lines = ["Metric                    Baseline(step0)        Best Eval      Eval Step\n"]
    lines.append("-" * 70 + "\n")
    for label, key, is_len in metrics:
        lines.append(
            f"{label:<24s} {fmt(step0.get(key), is_len):>18s} {fmt(best_eval.get(key), is_len):>15s} {str(best_eval.get('step', 'N/A')):>12s}\n"
        )
    return "".join(lines)


def copy_artifacts(exp_dir: str, output_dir: str):
    artifact_dir = os.path.join(output_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    for name in ["config_snapshot.py", "train.log"]:
        src = os.path.join(exp_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, name))
    for subdir in ["optimization/results", "evaluation/results"]:
        src_dir = os.path.join(exp_dir, subdir)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(artifact_dir, subdir.replace("/", "_"))
        os.makedirs(dst_dir, exist_ok=True)
        for src in glob.glob(os.path.join(src_dir, "*.txt")):
            shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))


def build_summary(
    run_metadata: dict,
    train_rows: list[dict],
    eval_rows: list[dict],
    bootstrap_rows: list[dict],
    validation_rows: list[dict],
    policy_rows: list[dict],
) -> str:
    profile = run_metadata.get("profile", "default")
    thresholds = get_profile_thresholds(profile)
    latest_bootstrap = bootstrap_rows[-1] if bootstrap_rows else {}
    latest_validation = validation_rows[-1] if validation_rows else {}
    step0 = train_rows[0] if train_rows else {}
    best_eval = {}
    for row in eval_rows[1:]:
        if row.get("code_acc") is not None and (not best_eval or row["code_acc"] > best_eval.get("code_acc", -1)):
            best_eval = row

    checks = []
    if profile == "bootstrap_smoke":
        checks.append(("code_spearman_mean >= 0.60", latest_validation.get("code_spearman_mean", 0.0) >= thresholds.get("code_spearman_mean", 0.60)))
        checks.append(("skip_rate <= 0.35", latest_bootstrap.get("skip_rate", 1.0) <= thresholds.get("skip_rate", 0.35)))
        checks.append(("avg_reliable_tests >= 2", latest_bootstrap.get("avg_reliable_tests", 0.0) >= thresholds.get("avg_reliable_tests", 2.0)))
    elif profile == "bootstrap_pilot":
        checks.append(("best eval code_acc > step0", best_eval.get("code_acc", -1) > step0.get("code_acc", -1)))
        checks.append(("best eval BoN(4,4) > step0", best_eval.get("bon_4_4_acc", -1) > step0.get("bon_4_4_acc", -1)))
        checks.append(("code_spearman_mean not below smoke bar", latest_validation.get("code_spearman_mean", 0.0) >= thresholds.get("code_spearman_mean_floor", 0.60)))
    elif profile == "bootstrap_full":
        checks.append(("best eval code_acc > step0", best_eval.get("code_acc", -1) > step0.get("code_acc", -1)))
        checks.append(("best eval case_acc > step0", best_eval.get("case_acc", -1) > step0.get("case_acc", -1)))
        checks.append(("best eval BoN(16,16) > step0", best_eval.get("bon_16_16_acc", -1) > step0.get("bon_16_16_acc", -1)))
        reward_collapse = latest_validation.get("code_spearman_mean", 0.0) < 0.30 or latest_bootstrap.get("avg_reliable_tests", 0.0) < 1.0
        checks.append(("no reward-collapse alert", not reward_collapse))

    passed = all(flag for _, flag in checks) if checks else True
    top_policy = policy_rows[-1] if policy_rows else {}
    lines = [
        f"# Bootstrap Experiment Summary\n\n",
        f"- Profile: `{profile}`\n",
        f"- Status: `{'PASS' if passed else 'REVIEW'}`\n",
        f"- Train dataset: `{run_metadata.get('train_dataset', 'unknown')}`\n",
        f"- Eval dataset: `{run_metadata.get('eval_dataset', 'unknown')}`\n",
        f"- Total steps: `{run_metadata.get('total_steps', 'unknown')}`\n",
        f"- Eval interval: `{run_metadata.get('eval_interval', 'unknown')}`\n\n",
        "## Gate Checks\n",
    ]
    if checks:
        for label, flag in checks:
            lines.append(f"- `{'PASS' if flag else 'FAIL'}` {label}\n")
    else:
        lines.append("- No profile gate configured.\n")
    lines.extend(
        [
            "\n## Best Eval Snapshot\n",
            f"- Best eval step: `{best_eval.get('step', 'N/A')}`\n",
            f"- Code acc: `{best_eval.get('code_acc', 'N/A')}`\n",
            f"- UT acc: `{best_eval.get('case_acc', 'N/A')}`\n",
            f"- BoN(4,4): `{best_eval.get('bon_4_4_acc', 'N/A')}`\n",
            f"- BoN(16,16): `{best_eval.get('bon_16_16_acc', 'N/A')}`\n",
            "\n## Latest Bootstrap Diagnostics\n",
            f"- code_spearman_mean: `{latest_validation.get('code_spearman_mean', 'N/A')}`\n",
            f"- case_spearman_mean: `{latest_validation.get('case_spearman_mean', 'N/A')}`\n",
            f"- skip_rate: `{latest_bootstrap.get('skip_rate', 'N/A')}`\n",
            f"- avg_reliable_tests: `{latest_bootstrap.get('avg_reliable_tests', 'N/A')}`\n",
            f"- duplicate_rate: `{latest_bootstrap.get('duplicate_rate', 'N/A')}`\n",
            f"- timeout_rate: `{latest_bootstrap.get('timeout_rate', 'N/A')}`\n",
            f"- exec_error_rate: `{latest_bootstrap.get('exec_error_rate', 'N/A')}`\n",
            "\n## Latest Policy Snapshot\n",
            f"- tag: `{top_policy.get('tag', 'N/A')}`\n",
            f"- policy_loss: `{top_policy.get('policy_loss', 'N/A')}`\n",
            f"- kl_loss: `{top_policy.get('kl_loss', 'N/A')}`\n",
            f"- entropy: `{top_policy.get('entropy', 'N/A')}`\n",
            f"- group_raw_reward: `{top_policy.get('group_raw_reward', 'N/A')}`\n",
            "\n## Suggested Next Step\n",
        ]
    )
    if passed:
        if profile == "bootstrap_smoke":
            lines.append("- 进入 `pilot` 阶段。\n")
        elif profile == "bootstrap_pilot":
            lines.append("- 进入 `full` 阶段。\n")
        else:
            lines.append("- 可以把 `report/package_for_codex.tar.gz` 回传给 Codex 做结果分析和调优。\n")
    else:
        lines.append("- 先不要进入下一阶段；把 `report/package_for_codex.tar.gz` 或 `summary.md + CSV + train.log` 回传给 Codex。\n")
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate per-experiment bootstrap report")
    parser.add_argument("--exp_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--project_root", type=str, default=ROOT_DIR)
    args = parser.parse_args()

    exp_dir = args.exp_dir or find_experiment_dir(args.project_root)
    output_dir = args.output_dir or os.path.join(exp_dir, "report")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    copy_artifacts(exp_dir, output_dir)

    raw_dir = os.path.join(output_dir, "raw")
    run_metadata = read_json(os.path.join(raw_dir, "run_metadata.json"), {})
    bootstrap_rows = read_jsonl(os.path.join(raw_dir, "bootstrap_step_metrics.jsonl"))
    validation_rows = read_jsonl(os.path.join(raw_dir, "reward_validation.jsonl"))
    policy_rows = build_policy_curve(raw_dir)
    hard_cases = read_jsonl(os.path.join(raw_dir, "hard_cases_raw.jsonl"))

    train_rows = build_train_curve(exp_dir, run_metadata)
    eval_rows = build_eval_summary(exp_dir, run_metadata)

    train_columns = [
        "step",
        "code_acc",
        "code_acc_acc",
        "case_acc",
        "case_acc_acc",
        "p_01",
        "p_00",
        "bon_4_4_acc",
        "bon_4_4_acc_acc",
        "bon_16_16_acc",
        "bon_16_16_acc_acc",
        "mean_code_len",
        "mean_case_len",
        "code_reward_mean",
        "code_reward_std",
        "case_reward_mean",
        "case_reward_std",
    ]
    eval_columns = [
        "model",
        "checkpoint",
        "dataset",
        "step",
        "code_acc",
        "code_acc_acc",
        "case_acc",
        "case_acc_acc",
        "p_01",
        "p_00",
        "bon_4_4_acc",
        "bon_4_4_acc_acc",
        "bon_16_16_acc",
        "bon_16_16_acc_acc",
        "mean_code_len",
        "mean_case_len",
    ]
    bootstrap_columns = [
        "step",
        "profile",
        "problems_total",
        "problems_used_code",
        "problems_used_case",
        "skip_rate",
        "avg_reliable_tests",
        "avg_reliable_ratio",
        "avg_confidence",
        "avg_discriminability",
        "duplicate_rate",
        "empty_test_rate",
        "timeout_rate",
        "exec_error_rate",
        "anchor_coverage",
        "anchor_mean",
        "code_reward_raw_std",
        "case_reward_raw_std",
        "skip_reasons",
    ]
    validation_columns = [
        "step",
        "profile",
        "code_spearman_mean",
        "code_spearman_median",
        "case_spearman_mean",
        "case_spearman_median",
        "valid_problem_count",
        "gt_available_problem_count",
    ]
    policy_columns = [
        "step",
        "profile",
        "tag",
        "policy_loss",
        "kl_loss",
        "total_loss",
        "clip_ratio",
        "entropy",
        "actor_lr",
        "policy_update_steps",
        "group_raw_reward",
        "group_relative_reward",
        "group_reward_std",
        "avg_custom_rewards",
        "avg_response_length",
        "avg_advantages",
        "avg_advantages_abs",
        "avg_group_raw_reward",
        "avg_group_relative_reward",
        "avg_group_reward_std",
    ]
    write_csv(os.path.join(output_dir, "train_curve.csv"), train_rows, train_columns)
    write_csv(os.path.join(output_dir, "eval_summary.csv"), eval_rows, eval_columns)
    write_csv(os.path.join(output_dir, "bootstrap_step_summary.csv"), bootstrap_rows, bootstrap_columns)
    write_csv(os.path.join(output_dir, "reward_validation.csv"), validation_rows, validation_columns)
    write_csv(os.path.join(output_dir, "policy_curve.csv"), policy_rows, policy_columns)

    comparison = build_comparison_table(train_rows, eval_rows)
    with open(os.path.join(output_dir, "comparison_table.txt"), "w", encoding="utf-8") as f:
        f.write(comparison)

    with open(os.path.join(output_dir, "hard_cases.jsonl"), "w", encoding="utf-8") as f:
        for row in hard_cases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(run_metadata, train_rows, eval_rows, bootstrap_rows, validation_rows, policy_rows)
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)

    figures_dir = os.path.join(output_dir, "figures")
    if train_rows:
        steps = [row["step"] for row in train_rows]
        save_line_plot(
            os.path.join(figures_dir, "A_code_acc_vs_steps.png"),
            steps,
            [("Code Acc", [row.get("code_acc", np.nan) for row in train_rows], "C0")],
            "Code Accuracy vs Steps",
            "Code Accuracy",
        )
        save_line_plot(
            os.path.join(figures_dir, "B_case_acc_vs_steps.png"),
            steps,
            [("UT Acc", [row.get("case_acc", np.nan) for row in train_rows], "C1")],
            "UT Accuracy vs Steps",
            "UT Accuracy",
        )
        save_line_plot(
            os.path.join(figures_dir, "C_bon_acc_vs_steps.png"),
            steps,
            [
                ("BoN(4,4)", [row.get("bon_4_4_acc", np.nan) for row in train_rows], "C2"),
                ("BoN(16,16)", [row.get("bon_16_16_acc", np.nan) for row in train_rows], "C3"),
            ],
            "BoN Accuracy vs Steps",
            "Accuracy",
        )
        save_line_plot(
            os.path.join(figures_dir, "D_response_length_vs_steps.png"),
            steps,
            [
                ("Code Resp Len", [row.get("mean_code_len", np.nan) for row in train_rows], "C4"),
                ("Case Resp Len", [row.get("mean_case_len", np.nan) for row in train_rows], "C5"),
            ],
            "Response Length vs Steps",
            "Tokens",
        )
    if validation_rows:
        steps = [row["step"] for row in validation_rows]
        save_line_plot(
            os.path.join(figures_dir, "E_reward_correlation_vs_steps.png"),
            steps,
            [
                ("Code Spearman", [row.get("code_spearman_mean", np.nan) for row in validation_rows], "C0"),
                ("Case Spearman", [row.get("case_spearman_mean", np.nan) for row in validation_rows], "C1"),
            ],
            "Reward Correlation vs Steps",
            "Spearman",
        )
    if bootstrap_rows:
        steps = [row["step"] for row in bootstrap_rows]
        save_line_plot(
            os.path.join(figures_dir, "F_skip_and_reliable_vs_steps.png"),
            steps,
            [
                ("Skip Rate", [row.get("skip_rate", np.nan) for row in bootstrap_rows], "C6"),
                ("Avg Reliable Tests", [row.get("avg_reliable_tests", np.nan) for row in bootstrap_rows], "C7"),
            ],
            "Skip Rate / Reliable Tests vs Steps",
            "Value",
        )
    if policy_rows:
        steps = [row["step"] for row in policy_rows if row.get("policy_loss") is not None or row.get("entropy") is not None]
        if steps:
            save_line_plot(
                os.path.join(figures_dir, "G_policy_dynamics_vs_steps.png"),
                steps,
                [
                    ("Policy Loss", [row.get("policy_loss", np.nan) for row in policy_rows if row.get("policy_loss") is not None or row.get("entropy") is not None], "C8"),
                    ("Entropy", [row.get("entropy", np.nan) for row in policy_rows if row.get("policy_loss") is not None or row.get("entropy") is not None], "C9"),
                ],
                "Policy Dynamics vs Steps",
                "Value",
            )

    print(f"Report generated at: {output_dir}")


if __name__ == "__main__":
    main()
