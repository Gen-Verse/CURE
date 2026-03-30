import argparse
import json
import os
import random
from typing import Any

import numpy as np

import optimization_config
from bootstrap_reward_utils import (
    EMPTY_LABEL,
    EXEC_ERROR_LABEL,
    TIMEOUT_LABEL,
    BootstrapState,
    build_test_keys,
    canonicalize_execution_output,
    canonicalize_expected_output,
    compute_gt_case_reward,
    duplicate_counts,
    majority_vote,
    run_em,
    safe_mean,
    safe_spearman,
    utility_from_pass_rate,
)
from reporting import append_jsonl, get_current_step, get_profile_name


SPECIAL_ERROR_LABELS = {TIMEOUT_LABEL, EXEC_ERROR_LABEL}


max_generation_len = optimization_config.max_generation_token
max_len_threshold = optimization_config.max_len_threshold
min_len_threshold = optimization_config.min_len_threshold
separate_training = optimization_config.separate_training
enable_efficient = optimization_config.enable_efficient
post_stage = optimization_config.post_stage
bootstrap_em_iterations = optimization_config.bootstrap_em_iterations
bootstrap_anchor_strength = optimization_config.bootstrap_anchor_strength
bootstrap_min_confidence = optimization_config.bootstrap_min_confidence
bootstrap_min_reliable_ratio = optimization_config.bootstrap_min_reliable_ratio
bootstrap_min_reliable_tests = optimization_config.bootstrap_min_reliable_tests
bootstrap_duplicate_public_penalty = optimization_config.bootstrap_duplicate_public_penalty
bootstrap_confidence_weight_power = optimization_config.bootstrap_confidence_weight_power
bootstrap_use_anchor = optimization_config.bootstrap_use_anchor
bootstrap_hard_case_limit = optimization_config.bootstrap_hard_case_limit
bootstrap_log_gt_diagnostics = optimization_config.bootstrap_log_gt_diagnostics
bootstrap_min_disc = optimization_config.bootstrap_min_disc


def str2bool(x):
    return x.lower() in ("1", "true", "yes")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--max_generation_len", type=int, default=optimization_config.max_generation_token)
    parser.add_argument("--max_len_threshold", type=int, default=optimization_config.max_len_threshold)
    parser.add_argument("--min_len_threshold", type=int, default=optimization_config.min_len_threshold)
    parser.add_argument("--separate_training", type=str2bool, default=optimization_config.separate_training)
    parser.add_argument("--enable_efficient", type=str2bool, default=optimization_config.enable_efficient)
    parser.add_argument("--post_stage", type=str2bool, default=optimization_config.post_stage)
    parser.add_argument("--use_self_bootstrap", type=str2bool, default=optimization_config.use_self_bootstrap)
    parser.add_argument("--bootstrap_em_iterations", type=int, default=optimization_config.bootstrap_em_iterations)
    parser.add_argument("--bootstrap_anchor_strength", type=float, default=optimization_config.bootstrap_anchor_strength)
    parser.add_argument("--bootstrap_min_confidence", type=float, default=optimization_config.bootstrap_min_confidence)
    parser.add_argument("--bootstrap_min_reliable_ratio", type=float, default=optimization_config.bootstrap_min_reliable_ratio)
    parser.add_argument("--bootstrap_min_reliable_tests", type=int, default=optimization_config.bootstrap_min_reliable_tests)
    parser.add_argument("--bootstrap_duplicate_public_penalty", type=float, default=optimization_config.bootstrap_duplicate_public_penalty)
    parser.add_argument("--bootstrap_confidence_weight_power", type=float, default=optimization_config.bootstrap_confidence_weight_power)
    parser.add_argument("--bootstrap_use_anchor", type=str2bool, default=optimization_config.bootstrap_use_anchor)
    parser.add_argument("--bootstrap_hard_case_limit", type=int, default=optimization_config.bootstrap_hard_case_limit)
    parser.add_argument("--bootstrap_log_gt_diagnostics", type=str2bool, default=optimization_config.bootstrap_log_gt_diagnostics)
    parser.add_argument("--bootstrap_min_disc", type=float, default=optimization_config.bootstrap_min_disc)
    return parser.parse_args()


def normalize_reward(reward_arr):
    reward_arr = np.asarray(reward_arr, dtype=float)
    if np.all(reward_arr == 1) and enable_efficient:
        return reward_arr
    mean = np.mean(reward_arr)
    std = np.std(reward_arr)
    if std.item() == 0:
        return None
    return (reward_arr - mean) / std


def normalize_balance_std(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    pos_mask = x > 0
    neg_mask = x < 0
    sum_pos = x[pos_mask].sum()
    sum_neg_abs = abs(x[neg_mask].sum())
    if sum_pos * sum_neg_abs == 0:
        return None
    scale_factor = sum_neg_abs / sum_pos
    x[pos_mask] *= scale_factor
    return x / x.std()


def length_regularize(reward_arr, response_length_list):
    reward_arr = np.sign(reward_arr)
    pos_list = np.where(reward_arr == 1)[0].tolist()
    neg_list = np.where(reward_arr == -1)[0].tolist()
    if len(pos_list) == 0 or len(neg_list) == 0:
        return None
    pos_response_length = np.array([response_length_list[j] for j in pos_list])
    threshold = np.median(pos_response_length).item()
    if np.sum((pos_response_length - threshold) ** 2) == 0:
        return normalize_balance_std(np.sign(reward_arr))
    threshold = max(min(threshold, max_len_threshold), min_len_threshold)
    length_reg_reward = np.zeros(len(reward_arr), float)
    length_reg_reward[pos_list] = -pos_response_length + threshold
    length_reg_reward[neg_list] = np.min(length_reg_reward).copy()
    return normalize_balance_std(length_reg_reward)


def bool_split_from_scores(scores: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
    n = len(scores)
    if n < 2:
        return None, None
    order = np.argsort(scores)
    split = n // 2
    if split == 0 or split == n:
        return None, None
    bad_idx = order[:split]
    good_idx = order[n - split :]
    good_mask = np.zeros(n, dtype=bool)
    bad_mask = np.zeros(n, dtype=bool)
    good_mask[good_idx] = True
    bad_mask[bad_idx] = True
    if not good_mask.any() or not bad_mask.any():
        return None, None
    return good_mask, bad_mask


def build_novelty_info(
    generated_inputs: list[str],
    generated_outputs: list[str],
    public_inputs: list[str],
    public_outputs: list[str],
    duplicate_public_penalty: float,
) -> dict[str, Any]:
    keys = build_test_keys(generated_inputs, generated_outputs)
    counts = duplicate_counts(keys)
    public_keys = set(build_test_keys(public_inputs, public_outputs))
    novelty = np.ones(len(keys), dtype=float)
    duplicate_mask = np.zeros(len(keys), dtype=bool)
    empty_mask = np.zeros(len(keys), dtype=bool)
    public_duplicate_mask = np.zeros(len(keys), dtype=bool)
    for idx, (norm_input, norm_output) in enumerate(keys):
        is_empty = norm_input == "" or norm_output == EMPTY_LABEL
        is_duplicate = counts[(norm_input, norm_output)] > 1
        is_public_duplicate = (norm_input, norm_output) in public_keys
        empty_mask[idx] = is_empty
        duplicate_mask[idx] = is_duplicate
        public_duplicate_mask[idx] = is_public_duplicate
        if is_empty or is_duplicate:
            novelty[idx] = 0.0
        elif is_public_duplicate:
            novelty[idx] = duplicate_public_penalty
    return {
        "keys": keys,
        "novelty": novelty,
        "duplicate_mask": duplicate_mask,
        "empty_mask": empty_mask,
        "public_duplicate_mask": public_duplicate_mask,
    }


def infer_test_statistics(
    state: BootstrapState,
    label_matrix: np.ndarray,
    declared_bool_table: np.ndarray,
    novelty: np.ndarray,
) -> dict[str, np.ndarray]:
    good_mask, bad_mask = bool_split_from_scores(state.q)
    if good_mask is None or bad_mask is None:
        return {
            "good_mask": None,
            "bad_mask": None,
            "disc": np.zeros(label_matrix.shape[1], dtype=float),
            "util": np.zeros(label_matrix.shape[1], dtype=float),
            "pass_rate": np.zeros(label_matrix.shape[1], dtype=float),
        }
    disc = np.zeros(label_matrix.shape[1], dtype=float)
    util = np.zeros(label_matrix.shape[1], dtype=float)
    pass_rate = np.zeros(label_matrix.shape[1], dtype=float)
    for j in range(label_matrix.shape[1]):
        good_pass = float(np.mean(declared_bool_table[good_mask, j])) if good_mask.any() else 0.0
        bad_pass = float(np.mean(declared_bool_table[bad_mask, j])) if bad_mask.any() else 0.0
        disc[j] = max(good_pass - bad_pass, 0.0)
        pass_rate[j] = float(np.mean(label_matrix[:, j] == state.inferred_labels[j]))
        util[j] = utility_from_pass_rate(pass_rate[j]) if novelty[j] > 0 else 0.0
    return {
        "good_mask": good_mask,
        "bad_mask": bad_mask,
        "disc": disc,
        "util": util,
        "pass_rate": pass_rate,
    }


def compute_code_raw_rewards(
    label_matrix: np.ndarray,
    state: BootstrapState,
    novelty: np.ndarray,
    disc: np.ndarray,
    util: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, list[str]]:
    n_code, n_case = label_matrix.shape
    if n_case == 0:
        return None, np.zeros(n_case, dtype=bool), ["no_generated_tests"]

    raw_rewards = np.zeros(n_code, dtype=float)
    reliable_counts = np.zeros(n_code, dtype=int)
    skip_reasons = []
    reliable_mask_global = np.zeros(n_case, dtype=bool)
    for i in range(n_code):
        other_indices = [idx for idx in range(n_code) if idx != i]
        if len(other_indices) == 0:
            skip_reasons.append("single_code_only")
            return None, reliable_mask_global, skip_reasons
        weights = np.clip(state.q[other_indices], 1e-6, None)
        total_weight = 0.0
        total_score = 0.0
        for j in range(n_case):
            if novelty[j] <= 0:
                continue
            label, conf, _ = majority_vote(label_matrix[other_indices, j].tolist(), weights)
            if conf < bootstrap_min_confidence:
                continue
            if label in SPECIAL_ERROR_LABELS:
                continue
            if disc[j] < bootstrap_min_disc:
                continue
            weight = (conf**bootstrap_confidence_weight_power) * disc[j] * util[j] * novelty[j]
            if weight <= 0:
                continue
            total_weight += weight
            total_score += weight * float(label_matrix[i, j] == label)
            reliable_counts[i] += 1
            reliable_mask_global[j] = True
        if total_weight == 0:
            skip_reasons.append("no_reliable_tests")
            return None, reliable_mask_global, skip_reasons
        raw_rewards[i] = total_score / total_weight
    return raw_rewards, reliable_mask_global, skip_reasons


def compute_case_raw_rewards(
    label_matrix: np.ndarray,
    declared_bool_table: np.ndarray,
    declared_output_labels: list[str],
    anchor_scores: np.ndarray | None,
    novelty: np.ndarray,
) -> np.ndarray | None:
    _, n_case = label_matrix.shape
    raw_rewards = np.zeros(n_case, dtype=float)
    for j in range(n_case):
        if novelty[j] <= 0:
            raw_rewards[j] = -1.0
            continue
        if n_case == 1:
            q_without = anchor_scores if anchor_scores is not None else np.full(label_matrix.shape[0], 0.5, dtype=float)
            state_without = BootstrapState(
                q=q_without,
                inferred_labels=[declared_output_labels[j]],
                confidence=np.array([0.0], dtype=float),
                support=np.array([0.0], dtype=float),
            )
        else:
            reduced_labels = np.delete(label_matrix, j, axis=1)
            state_without = run_em(
                reduced_labels,
                anchor_scores,
                bootstrap_em_iterations,
                bootstrap_anchor_strength,
            )
        good_mask, bad_mask = bool_split_from_scores(state_without.q)
        if good_mask is None or bad_mask is None:
            return None
        vote_label, conf, _ = majority_vote(label_matrix[:, j].tolist(), np.clip(state_without.q, 1e-6, None))
        good_pass = float(np.mean(declared_bool_table[good_mask, j])) if good_mask.any() else 0.0
        bad_pass = float(np.mean(declared_bool_table[bad_mask, j])) if bad_mask.any() else 0.0
        disc = max(good_pass - bad_pass, 0.0)
        util = utility_from_pass_rate(float(np.mean(label_matrix[:, j] == vote_label)))
        magnitude = conf * max(disc, bootstrap_min_disc) * util * novelty[j]
        if vote_label in SPECIAL_ERROR_LABELS:
            raw_rewards[j] = -abs(magnitude)
            continue
        raw_rewards[j] = (1.0 if declared_output_labels[j] == vote_label else -1.0) * magnitude
    return raw_rewards


def truncate_text(value: str, limit: int = 240) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def build_group_entry(prompt: str, responses: list[str], rewards: np.ndarray, response_lengths: list[int]) -> dict[str, Any] | None:
    group_entry = {
        "prompt": prompt,
        "responses": [],
        "rewards": [],
    }
    for idx, reward_j in enumerate(rewards.tolist()):
        response_j = responses[idx] + ("<|im_end|>" if response_lengths[idx] < max_generation_len else "")
        group_entry["responses"].append(response_j)
        group_entry["rewards"].append(float(reward_j))
    return group_entry if group_entry["responses"] else None


def analyze_problem(problem: dict[str, Any]) -> dict[str, Any]:
    result = {
        "code_group": None,
        "case_group": None,
        "raw_code_reward": None,
        "raw_case_reward": None,
        "code_corr": None,
        "case_corr": None,
        "used_code": False,
        "used_case": False,
        "skip_reasons": [],
        "summary": {},
        "hard_case": None,
    }

    if problem.get("all_case_bool_table") is None:
        result["skip_reasons"].append("missing_execution_table")
        return result

    all_bool_table = np.array(problem["all_case_bool_table"], dtype=bool)
    all_exe_table = np.array(problem["all_case_exe_results"], dtype=object)
    t = int(problem.get("num_ground_truth_test", 0))
    gen_bool_table = all_bool_table[:, t:].copy()
    gen_exe_table = all_exe_table[:, t:].copy()
    gen_outputs = problem.get("all_case_output", [])[t:]
    gen_inputs = problem.get("all_case_input", [])[t:]

    if gen_bool_table.size == 0 or gen_bool_table.shape[1] == 0:
        result["skip_reasons"].append("no_generated_tests")
        return result

    label_matrix = np.vectorize(canonicalize_execution_output)(gen_exe_table)
    declared_output_labels = [canonicalize_expected_output(item) for item in gen_outputs]
    novelty_info = build_novelty_info(
        gen_inputs,
        gen_outputs,
        problem.get("example_input", []),
        problem.get("example_output", []),
        bootstrap_duplicate_public_penalty,
    )
    novelty = novelty_info["novelty"]

    anchor_scores = None
    public_table = problem.get("public_example_bool_table")
    if bootstrap_use_anchor and public_table is not None:
        public_arr = np.array(public_table, dtype=bool)
        if public_arr.size > 0:
            anchor_scores = public_arr.mean(axis=1)

    state = run_em(
        label_matrix,
        anchor_scores,
        bootstrap_em_iterations,
        bootstrap_anchor_strength,
    )
    test_stats = infer_test_statistics(state, label_matrix, gen_bool_table, novelty)
    disc = test_stats["disc"]
    util = test_stats["util"]

    raw_code_reward, reliable_mask, code_skip = compute_code_raw_rewards(label_matrix, state, novelty, disc, util)
    result["skip_reasons"].extend(code_skip)
    raw_case_reward = compute_case_raw_rewards(label_matrix, gen_bool_table, declared_output_labels, anchor_scores, novelty)
    if raw_case_reward is None:
        result["skip_reasons"].append("cannot_split_good_bad_for_case")

    reliable_tests = int(np.sum(reliable_mask))
    generated_tests = int(gen_bool_table.shape[1])
    reliable_ratio = reliable_tests / max(generated_tests, 1)
    duplicate_rate = float(np.mean(novelty_info["duplicate_mask"])) if generated_tests else 0.0
    empty_test_rate = float(np.mean(novelty_info["empty_mask"])) if generated_tests else 0.0
    public_duplicate_rate = float(np.mean(novelty_info["public_duplicate_mask"])) if generated_tests else 0.0
    timeout_rate = float(np.mean(label_matrix == TIMEOUT_LABEL)) if label_matrix.size else 0.0
    exec_error_rate = float(np.mean(label_matrix == EXEC_ERROR_LABEL)) if label_matrix.size else 0.0

    summary = {
        "generated_tests": generated_tests,
        "reliable_tests": reliable_tests,
        "reliable_ratio": reliable_ratio,
        "avg_confidence": safe_mean(state.confidence.tolist()),
        "avg_discriminability": safe_mean(disc.tolist()),
        "duplicate_rate": duplicate_rate,
        "empty_test_rate": empty_test_rate,
        "timeout_rate": timeout_rate,
        "exec_error_rate": exec_error_rate,
        "anchor_mean": float(np.mean(anchor_scores)) if anchor_scores is not None else 0.0,
        "anchor_available": anchor_scores is not None,
        "public_duplicate_rate": public_duplicate_rate,
        "code_reward_raw_std": safe_mean([float(np.std(raw_code_reward))]) if raw_code_reward is not None else 0.0,
        "case_reward_raw_std": safe_mean([float(np.std(raw_case_reward))]) if raw_case_reward is not None else 0.0,
    }
    result["summary"] = summary

    min_required_reliable = max(bootstrap_min_reliable_tests, int(np.ceil(bootstrap_min_reliable_ratio * generated_tests)))
    if reliable_tests < min_required_reliable:
        result["skip_reasons"].append("too_few_reliable_tests")

    if raw_code_reward is not None and np.std(raw_code_reward) > 1e-8 and reliable_tests >= min_required_reliable:
        normalized = normalize_reward(raw_code_reward)
        if normalized is not None:
            if enable_efficient:
                normalized = length_regularize(normalized, problem["code_response_length"])
            if normalized is not None:
                group_entry = build_group_entry(
                    problem["code_generation_prompt"],
                    problem["full_code_generation"],
                    normalized,
                    problem["code_response_length"],
                )
                if group_entry is not None:
                    result["code_group"] = group_entry
                    result["raw_code_reward"] = raw_code_reward
                    result["used_code"] = True
    else:
        result["skip_reasons"].append("code_reward_zero_std_or_none")

    if raw_case_reward is not None and np.std(raw_case_reward) > 1e-8 and reliable_tests >= min_required_reliable:
        normalized = normalize_reward(raw_case_reward)
        if normalized is not None:
            if enable_efficient:
                normalized = length_regularize(normalized, problem["case_response_length"])
            if normalized is not None:
                group_entry = build_group_entry(
                    problem["case_generation_prompt"],
                    problem["full_case_generation"],
                    normalized,
                    problem["case_response_length"],
                )
                if group_entry is not None:
                    result["case_group"] = group_entry
                    result["raw_case_reward"] = raw_case_reward
                    result["used_case"] = True
    else:
        result["skip_reasons"].append("case_reward_zero_std_or_none")

    if bootstrap_log_gt_diagnostics and t > 0:
        gt_test_table = all_bool_table[:, :t].copy()
        gt_code_reward = gt_test_table.mean(axis=1)
        if raw_code_reward is not None:
            result["code_corr"] = safe_spearman(raw_code_reward, gt_code_reward)
        if raw_case_reward is not None:
            gt_case_reward = compute_gt_case_reward(gen_bool_table, gt_test_table, post_stage)
            if gt_case_reward is not None and len(gt_case_reward) == len(raw_case_reward):
                result["case_corr"] = safe_spearman(raw_case_reward, gt_case_reward)

    severity = (
        1 if not result["used_code"] or not result["used_case"] else 0,
        -float(result["code_corr"]) if result["code_corr"] is not None else 1.0,
        duplicate_rate,
    )
    result["hard_case"] = {
        "step": get_current_step(),
        "profile": get_profile_name(),
        "severity": severity,
        "skip_reasons": sorted(set(result["skip_reasons"])),
        "code_corr": result["code_corr"],
        "case_corr": result["case_corr"],
        "summary": summary,
        "question": truncate_text(problem.get("question", ""), 500),
        "sample_code": [truncate_text(x, 320) for x in problem.get("generated_code", [])[:2]],
        "sample_test_input": [truncate_text(x, 200) for x in gen_inputs[:2]],
        "sample_test_output": [truncate_text(x, 200) for x in gen_outputs[:2]],
    }
    return result


def main():
    args = parse_args()
    globals().update(vars(args))

    outputs_name = pretrained_model.replace("/", ".") + "-" + dataset
    input_path = "./temp_data/outputs-rl-" + outputs_name + ".json"
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    code_data = []
    case_data = []
    raw_code_rewards_all = []
    raw_case_rewards_all = []
    code_corrs = []
    case_corrs = []
    problems_total = 0
    problems_used_code = 0
    problems_used_case = 0
    problems_used_any = 0
    anchor_available = 0
    reliable_tests_list = []
    reliable_ratio_list = []
    confidence_list = []
    discriminability_list = []
    duplicate_rate_list = []
    empty_rate_list = []
    timeout_rate_list = []
    exec_error_rate_list = []
    anchor_mean_list = []
    code_raw_std_list = []
    case_raw_std_list = []
    hard_cases = []
    skip_reason_counter = {}
    gt_available_problem_count = 0

    for item in data:
        analysis = analyze_problem(item)
        if item.get("all_case_bool_table") is None:
            continue
        problems_total += 1
        if int(item.get("num_ground_truth_test", 0)) > 0:
            gt_available_problem_count += 1
        if analysis["summary"].get("anchor_available"):
            anchor_available += 1
        reliable_tests_list.append(analysis["summary"].get("reliable_tests", 0))
        reliable_ratio_list.append(analysis["summary"].get("reliable_ratio", 0.0))
        confidence_list.append(analysis["summary"].get("avg_confidence", 0.0))
        discriminability_list.append(analysis["summary"].get("avg_discriminability", 0.0))
        duplicate_rate_list.append(analysis["summary"].get("duplicate_rate", 0.0))
        empty_rate_list.append(analysis["summary"].get("empty_test_rate", 0.0))
        timeout_rate_list.append(analysis["summary"].get("timeout_rate", 0.0))
        exec_error_rate_list.append(analysis["summary"].get("exec_error_rate", 0.0))
        anchor_mean_list.append(analysis["summary"].get("anchor_mean", 0.0))
        code_raw_std_list.append(analysis["summary"].get("code_reward_raw_std", 0.0))
        case_raw_std_list.append(analysis["summary"].get("case_reward_raw_std", 0.0))
        hard_cases.append(analysis["hard_case"])
        for reason in analysis["skip_reasons"]:
            skip_reason_counter[reason] = skip_reason_counter.get(reason, 0) + 1
        if analysis["code_group"] is not None:
            code_data.append(analysis["code_group"])
            problems_used_code += 1
        if analysis["case_group"] is not None:
            case_data.append(analysis["case_group"])
            problems_used_case += 1
        if analysis["code_group"] is not None or analysis["case_group"] is not None:
            problems_used_any += 1
        if analysis["raw_code_reward"] is not None:
            raw_code_rewards_all.extend(analysis["raw_code_reward"].tolist())
        if analysis["raw_case_reward"] is not None:
            raw_case_rewards_all.extend(analysis["raw_case_reward"].tolist())
        if analysis["code_corr"] is not None:
            code_corrs.append(analysis["code_corr"])
        if analysis["case_corr"] is not None:
            case_corrs.append(analysis["case_corr"])

    final_data = code_data + case_data
    random.shuffle(final_data)

    if separate_training is False:
        with open("./temp_data/rl_data.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
    else:
        with open("./temp_data/rl_code_data.json", "w", encoding="utf-8") as f:
            json.dump(code_data, f, indent=2, ensure_ascii=False)
        with open("./temp_data/rl_case_data.json", "w", encoding="utf-8") as f:
            json.dump(case_data, f, indent=2, ensure_ascii=False)

    from termcolor import cprint

    os.makedirs(os.path.dirname("./results/results-" + outputs_name + ".txt"), exist_ok=True)
    with open("./results/results-" + outputs_name + ".txt", "a", encoding="utf-8") as f:
        def save_and_print(text):
            cprint(text, color="cyan")
            f.write(text + "\n")

        raw_code_arr = np.array(raw_code_rewards_all) if raw_code_rewards_all else np.array([0.0])
        raw_case_arr = np.array(raw_case_rewards_all) if raw_case_rewards_all else np.array([0.0])

        save_and_print(
            f"estimated_code_reward: mean={raw_code_arr.mean():.4f}, std={raw_code_arr.std():.4f}, "
            f"num_groups={len(code_data)}, num_samples={len(raw_code_rewards_all)}"
        )
        save_and_print(
            f"estimated_case_reward: mean={raw_case_arr.mean():.4f}, std={raw_case_arr.std():.4f}, "
            f"num_groups={len(case_data)}, num_samples={len(raw_case_rewards_all)}"
        )

    hard_cases_sorted = sorted(
        hard_cases,
        key=lambda item: item["severity"],
        reverse=True,
    )[:bootstrap_hard_case_limit]
    for entry in hard_cases_sorted:
        payload = dict(entry)
        payload.pop("severity", None)
        append_jsonl("hard_cases_raw.jsonl", payload)

    step_record = {
        "step": get_current_step(),
        "profile": get_profile_name(),
        "problems_total": problems_total,
        "problems_used_code": problems_used_code,
        "problems_used_case": problems_used_case,
        "skip_rate": 0.0 if problems_total == 0 else 1.0 - (problems_used_any / problems_total),
        "avg_reliable_tests": safe_mean(reliable_tests_list),
        "avg_reliable_ratio": safe_mean(reliable_ratio_list),
        "avg_confidence": safe_mean(confidence_list),
        "avg_discriminability": safe_mean(discriminability_list),
        "duplicate_rate": safe_mean(duplicate_rate_list),
        "empty_test_rate": safe_mean(empty_rate_list),
        "timeout_rate": safe_mean(timeout_rate_list),
        "exec_error_rate": safe_mean(exec_error_rate_list),
        "anchor_coverage": 0.0 if problems_total == 0 else anchor_available / problems_total,
        "anchor_mean": safe_mean(anchor_mean_list),
        "code_reward_raw_std": safe_mean(code_raw_std_list),
        "case_reward_raw_std": safe_mean(case_raw_std_list),
        "skip_reasons": skip_reason_counter,
    }
    validation_record = {
        "step": get_current_step(),
        "profile": get_profile_name(),
        "code_spearman_mean": safe_mean(code_corrs),
        "code_spearman_median": float(np.median(code_corrs)) if code_corrs else 0.0,
        "case_spearman_mean": safe_mean(case_corrs),
        "case_spearman_median": float(np.median(case_corrs)) if case_corrs else 0.0,
        "valid_problem_count": max(problems_used_code, problems_used_case),
        "gt_available_problem_count": gt_available_problem_count,
    }
    append_jsonl("bootstrap_step_metrics.jsonl", step_record)
    append_jsonl("reward_validation.jsonl", validation_record)


if __name__ == "__main__":
    main()
