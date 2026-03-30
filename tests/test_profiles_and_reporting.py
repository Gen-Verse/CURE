import os

from cure_profiles import apply_profile_overrides
from optimization.reporting import get_current_step, get_profile_name


def test_profile_overrides_apply_smoke_defaults():
    config = {"train_dataset": "CodeContests_train", "total_steps": 150}
    apply_profile_overrides("optimization", config, "bootstrap_smoke")
    assert config["train_dataset"] == "small_train"
    assert config["total_steps"] == 10
    assert config["k_code"] == 4
    assert config["k_case"] == 4


def test_reporting_helpers_read_environment(monkeypatch):
    monkeypatch.setenv("CURE_PROFILE", "bootstrap_pilot")
    monkeypatch.setenv("CURE_STEP", "7")
    assert get_profile_name() == "bootstrap_pilot"
    assert get_current_step() == 7
