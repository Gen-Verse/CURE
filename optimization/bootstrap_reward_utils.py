from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr


TIMEOUT_LABEL = "__TIMEOUT__"
EXEC_ERROR_LABEL = "__EXEC_ERROR__"
EMPTY_LABEL = "__EMPTY__"


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def canonicalize_execution_output(output: str | None) -> str:
    text = str(output or "")
    compact = normalize_text(text)
    lowered = compact.lower()
    if compact == "":
        return EMPTY_LABEL
    if compact == "Timeout Error":
        return TIMEOUT_LABEL
    if compact.startswith("Execution Error:") or lowered.startswith("execution error:"):
        return EXEC_ERROR_LABEL
    if compact.startswith("error:") or lowered.startswith("error:"):
        return EXEC_ERROR_LABEL
    return compact


def canonicalize_expected_output(output: str | None) -> str:
    compact = normalize_text(output)
    return compact if compact else EMPTY_LABEL


def utility_from_pass_rate(pass_rate: float) -> float:
    clipped = min(max(pass_rate, 0.0), 1.0)
    return 4.0 * clipped * (1.0 - clipped)


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def safe_std(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float)))


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
        return None
    if np.std(x_arr) < 1e-8 or np.std(y_arr) < 1e-8:
        return None
    corr, _ = spearmanr(x_arr, y_arr)
    if corr is None or np.isnan(corr):
        return None
    return float(corr)


def build_test_keys(test_inputs: list[str], test_outputs: list[str]) -> list[tuple[str, str]]:
    keys = []
    for test_input, test_output in zip(test_inputs, test_outputs):
        keys.append((normalize_text(test_input), canonicalize_expected_output(test_output)))
    return keys


def duplicate_counts(keys: list[tuple[str, str]]) -> Counter:
    return Counter(keys)


def majority_vote(labels: list[str], weights: np.ndarray) -> tuple[str, float, float]:
    vote_totals: dict[str, float] = {}
    for idx, label in enumerate(labels):
        vote_totals[label] = vote_totals.get(label, 0.0) + float(weights[idx])
    if not vote_totals:
        return EMPTY_LABEL, 0.0, 0.0
    ranked = sorted(vote_totals.items(), key=lambda item: (-item[1], item[0]))
    top_label, top_weight = ranked[0]
    second_weight = ranked[1][1] if len(ranked) > 1 else 0.0
    total_weight = sum(vote_totals.values())
    conf = 0.0 if total_weight <= 0 else max(top_weight - second_weight, 0.0) / total_weight
    return top_label, conf, float(top_weight)


def compute_gt_case_reward(all_case_table_i: np.ndarray, all_test_table_i: np.ndarray, post_stage: bool) -> np.ndarray | None:
    correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
    if len(correct_code_list) == 0:
        return None
    correct_code_table = all_case_table_i[correct_code_list, :].copy()
    index_list = np.where(np.all(correct_code_table, axis=0))[0].tolist()
    reward_sign = -np.ones(correct_code_table.shape[1], dtype=float)
    reward_sign[index_list] = 1
    case_reward = reward_sign.copy()
    wrong_code_list = [j for j in range(all_case_table_i.shape[0]) if j not in correct_code_list]
    if len(wrong_code_list) == 0:
        return case_reward
    reward_scale = np.ones(correct_code_table.shape[1], dtype=float)
    correct_case_list = np.where(correct_code_table.all(axis=0))[0].tolist()
    wrong_case_list = [j for j in range(all_case_table_i.shape[1]) if j not in correct_case_list]
    if len(correct_case_list):
        wrong_code_correct_case_table = all_case_table_i[wrong_code_list, :][:, correct_case_list].copy()
        if post_stage is False:
            mean_p01 = np.mean(~wrong_code_correct_case_table, 0)
        else:
            mean_p01 = (~np.any(wrong_code_correct_case_table, axis=0)).astype(float)
        reward_scale[correct_case_list] = reward_scale[correct_case_list] * mean_p01
    if len(wrong_case_list):
        wrong_code_wrong_case_table = all_case_table_i[wrong_code_list, :][:, wrong_case_list].copy()
        if post_stage is False:
            mean_p00 = np.mean(wrong_code_wrong_case_table, 0)
        else:
            mean_p00 = (np.any(wrong_code_wrong_case_table, axis=0)).astype(float)
        reward_scale[wrong_case_list] = reward_scale[wrong_case_list] * mean_p00
    return case_reward * reward_scale


@dataclass
class BootstrapState:
    q: np.ndarray
    inferred_labels: list[str]
    confidence: np.ndarray
    support: np.ndarray


def run_em(
    label_matrix: np.ndarray,
    anchor_scores: np.ndarray | None,
    n_iter: int,
    anchor_strength: float,
) -> BootstrapState:
    k_code, k_case = label_matrix.shape
    q = np.full(k_code, 0.5, dtype=float)
    if anchor_scores is not None and anchor_scores.size == k_code:
        q = anchor_strength * anchor_scores + (1.0 - anchor_strength) * q

    inferred_labels = [EMPTY_LABEL for _ in range(k_case)]
    confidence = np.zeros(k_case, dtype=float)
    support = np.zeros(k_case, dtype=float)
    for _ in range(max(n_iter, 1)):
        weights = np.clip(q, 1e-6, None)
        for j in range(k_case):
            label, conf, top_weight = majority_vote(label_matrix[:, j].tolist(), weights)
            inferred_labels[j] = label
            confidence[j] = conf
            support[j] = top_weight
        weighted_conf = np.clip(confidence, 1e-6, None)
        denom = float(np.sum(weighted_conf))
        if denom <= 0:
            denom = float(k_case) if k_case > 0 else 1.0
            weighted_conf = np.ones(k_case, dtype=float)
        agreement = np.zeros(k_code, dtype=float)
        for i in range(k_code):
            agreement[i] = np.sum(weighted_conf * (label_matrix[i, :] == np.asarray(inferred_labels)))
        q = agreement / denom
        if anchor_scores is not None and anchor_scores.size == k_code:
            q = (1.0 - anchor_strength) * q + anchor_strength * anchor_scores
    return BootstrapState(q=q, inferred_labels=inferred_labels, confidence=confidence, support=support)
