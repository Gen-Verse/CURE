import copy
import os


DEFAULT_PROFILE = "default"


PROFILE_THRESHOLDS = {
    "bootstrap_smoke": {
        "code_spearman_mean": 0.60,
        "skip_rate": 0.35,
        "avg_reliable_tests": 2.0,
    },
    "bootstrap_pilot": {
        "require_eval_code_acc_gain": True,
        "require_eval_bon_4_4_gain": True,
        "code_spearman_mean_floor": 0.60,
    },
    "bootstrap_full": {
        "require_positive_code_acc": True,
        "require_positive_case_acc": True,
        "require_positive_bon_16_16": True,
        "disallow_reward_collapse": True,
    },
}


OPTIMIZATION_PROFILE_OVERRIDES = {
    "bootstrap_smoke": {
        "exp_name": "",
        "train_dataset": "small_train",
        "eval_dataset": "small_eval",
        "total_steps": 10,
        "eval_interval": 2,
        "save_interval": 10,
        "k_code": 4,
        "k_case": 4,
        "n_sample_per_step": 20,
        "num_chunks": 16,
        "scale_tuple_list": [(4, 4)],
        "enable_efficient": False,
        "eval_k_code": 4,
        "eval_k_case": 4,
        "eval_scale_tuple_list": [(4, 4)],
        "eval_num_chunks": 16,
    },
    "bootstrap_pilot": {
        "exp_name": "",
        "train_dataset": "CodeContests_train",
        "eval_dataset": "CodeContests",
        "total_steps": 25,
        "eval_interval": 5,
        "save_interval": 25,
        "k_code": 8,
        "k_case": 8,
        "n_sample_per_step": 20,
        "num_chunks": 32,
        "scale_tuple_list": [(4, 4), (8, 8)],
        "eval_k_code": 8,
        "eval_k_case": 8,
        "eval_scale_tuple_list": [(4, 4), (8, 8)],
        "eval_num_chunks": 32,
    },
    "bootstrap_full": {
        "exp_name": "",
        "train_dataset": "CodeContests_train",
        "eval_dataset": "CodeContests",
        "total_steps": 150,
        "eval_interval": 25,
        "save_interval": 25,
        "k_code": 16,
        "k_case": 16,
        "n_sample_per_step": 20,
        "num_chunks": 32,
        "scale_tuple_list": [(4, 4), (16, 16)],
        "eval_k_code": 16,
        "eval_k_case": 16,
        "eval_scale_tuple_list": [(4, 4), (16, 16)],
        "eval_num_chunks": 32,
    },
}


EVALUATION_PROFILE_OVERRIDES = {
    "bootstrap_smoke": {
        "dataset": "small_eval",
        "k_code": 4,
        "k_case": 4,
        "scale_tuple_list": [(4, 4)],
        "num_chunks": 16,
        "single_eval": False,
        "temp": 1.0,
    },
    "bootstrap_pilot": {
        "dataset": "CodeContests",
        "k_code": 8,
        "k_case": 8,
        "scale_tuple_list": [(4, 4), (8, 8)],
        "num_chunks": 32,
        "single_eval": False,
        "temp": 1.0,
    },
    "bootstrap_full": {
        "dataset": "CodeContests",
        "k_code": 16,
        "k_case": 16,
        "scale_tuple_list": [(4, 4), (16, 16)],
        "num_chunks": 32,
        "single_eval": False,
        "temp": 1.0,
    },
}


def get_active_profile() -> str:
    return os.environ.get("CURE_PROFILE", DEFAULT_PROFILE).strip() or DEFAULT_PROFILE


def get_profile_overrides(kind: str, profile_name: str | None = None) -> dict:
    profile_name = profile_name or get_active_profile()
    if kind == "optimization":
        source = OPTIMIZATION_PROFILE_OVERRIDES
    elif kind == "evaluation":
        source = EVALUATION_PROFILE_OVERRIDES
    else:
        raise ValueError(f"Unsupported profile kind: {kind}")
    return copy.deepcopy(source.get(profile_name, {}))


def apply_profile_overrides(kind: str, config_globals: dict, profile_name: str | None = None) -> dict:
    profile_name = profile_name or get_active_profile()
    overrides = get_profile_overrides(kind, profile_name)
    config_globals.update(overrides)
    config_globals["active_profile"] = profile_name
    return overrides


def get_profile_thresholds(profile_name: str | None = None) -> dict:
    profile_name = profile_name or get_active_profile()
    return copy.deepcopy(PROFILE_THRESHOLDS.get(profile_name, {}))
