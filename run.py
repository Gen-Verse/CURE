import os
import sys
import glob
import json
import shutil
import subprocess
import tarfile
from datetime import datetime
from termcolor import cprint

from optimization import optimization_config


# if you are the first time to train the model, set this to be True.
# if you have stopped the process and want to keep training, simply set this to be False and run this script.
start_from_scratch = True


eval_interval = optimization_config.eval_interval
save_interval = optimization_config.save_interval
total_steps = optimization_config.total_steps
profile_name = os.environ.get("CURE_PROFILE", "").strip()
pretrain_model = optimization_config.pretrained_model
model = os.path.abspath("") + "/optimization/ckpt/" +  optimization_config.optimized_model_name
if start_from_scratch == False:
    pretrain_model = model
eval_dataset = optimization_config.eval_dataset
train_dataset = optimization_config.train_dataset
gpu_groups = optimization_config.gpu_groups
eval_k_code = optimization_config.eval_k_code
eval_k_case = optimization_config.eval_k_case
eval_scale_tuple_list = optimization_config.eval_scale_tuple_list
eval_num_chunks = optimization_config.eval_num_chunks
eval_no_example = optimization_config.eval_no_example
eval_max_test = optimization_config.eval_max_test


# ===================== Experiment Directory Setup =====================

def _build_exp_name():
    raw = optimization_config.exp_name
    if raw:
        return raw
    model_short = os.path.basename(pretrain_model.rstrip("/"))
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    if profile_name and profile_name != "default":
        return f"{ts}_{profile_name}_{model_short}_{train_dataset}"
    return f"{ts}_{model_short}_{train_dataset}"

exp_name = _build_exp_name()
exp_dir = os.path.join("experiments", exp_name)

os.makedirs(os.path.join(exp_dir, "optimization", "results"), exist_ok=True)
os.makedirs(os.path.join(exp_dir, "optimization", "tb_logs"), exist_ok=True)
os.makedirs(os.path.join(exp_dir, "optimization", "ckpt"), exist_ok=True)
os.makedirs(os.path.join(exp_dir, "evaluation", "results"), exist_ok=True)
os.makedirs(os.path.join(exp_dir, "report", "raw"), exist_ok=True)

exp_tb_dir = os.path.abspath(os.path.join(exp_dir, "optimization", "tb_logs"))
report_dir = os.path.abspath(os.path.join(exp_dir, "report"))

shutil.copy2("optimization/optimization_config.py",
             os.path.join(exp_dir, "config_snapshot.py"))
with open(os.path.join(report_dir, "raw", "run_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(
        {
            "profile": profile_name,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "pretrain_model": pretrain_model,
            "optimized_model_path": model,
            "exp_dir": os.path.abspath(exp_dir),
            "total_steps": total_steps,
            "eval_interval": eval_interval,
            "save_interval": save_interval,
            "k_code": optimization_config.k_code,
            "k_case": optimization_config.k_case,
            "eval_k_code": eval_k_code,
            "eval_k_case": eval_k_case,
            "scale_tuple_list": optimization_config.scale_tuple_list,
            "eval_scale_tuple_list": eval_scale_tuple_list,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

cprint(f"{'='*60}", color="yellow")
cprint(f"  Experiment: {exp_name}", color="yellow")
cprint(f"  Directory:  {os.path.abspath(exp_dir)}", color="yellow")
cprint(f"{'='*60}", color="yellow")


def archive_optimization_results():
    keep_tokens = [
        pretrain_model.replace("/", "."),
        model.replace("/", "."),
        train_dataset,
    ]
    for f in glob.glob("optimization/results/*.txt"):
        basename = os.path.basename(f)
        if all(token not in basename for token in keep_tokens):
            continue
        shutil.copy2(f, os.path.join(exp_dir, "optimization", "results", basename))

def archive_eval_results():
    keep_tokens = [
        pretrain_model.replace("/", "."),
        model.replace("/", "."),
        eval_dataset,
    ]
    for f in glob.glob("evaluation/results/*.txt"):
        basename = os.path.basename(f)
        if all(token not in basename for token in keep_tokens):
            continue
        shutil.copy2(f, os.path.join(exp_dir, "evaluation", "results", basename))

def archive_log_symlink():
    log_file = os.environ.get("CURE_LOG_FILE")
    if log_file:
        abs_log = os.path.abspath(log_file)
        link_path = os.path.join(exp_dir, "train.log")
        if os.path.lexists(link_path):
            os.remove(link_path)
        os.symlink(abs_log, link_path)


def build_subprocess_env(step):
    env = os.environ.copy()
    env["CURE_PROFILE"] = profile_name
    env["CURE_EXP_DIR"] = os.path.abspath(exp_dir)
    env["CURE_REPORT_DIR"] = report_dir
    env["CURE_STEP"] = str(step)
    env["CURE_TB_DIR"] = exp_tb_dir
    return env


def generate_report():
    subprocess.run(
        [
            sys.executable,
            "analysis/generate_report.py",
            "--exp_dir",
            os.path.abspath(exp_dir),
            "--output_dir",
            report_dir,
        ],
        check=False,
        cwd=os.path.abspath(""),
        env=build_subprocess_env(i),
    )


def package_report():
    archive_path = os.path.join(report_dir, "package_for_codex.tar.gz")
    temp_archive_path = os.path.join(exp_dir, "package_for_codex.tar.gz")
    for path in (archive_path, temp_archive_path):
        if os.path.exists(path):
            os.remove(path)
    with tarfile.open(temp_archive_path, "w:gz") as tar:
        for name in sorted(os.listdir(report_dir)):
            if name == "package_for_codex.tar.gz":
                continue
            tar.add(os.path.join(report_dir, name), arcname=os.path.join("report", name))
    shutil.move(temp_archive_path, archive_path)
    return archive_path


# ===================== Pipeline Steps =====================

def begin_with(file_name):
    with open(file_name, "w") as f:
        f.write("")

if start_from_scratch:
    os.makedirs("evaluation/results", exist_ok=True)
    os.makedirs("optimization/results", exist_ok=True)
    begin_with("evaluation/results/results-eval-" + model.replace("/", ".") + "-" + eval_dataset + ".txt")
    begin_with("optimization/results/results-rl-" + model.replace("/", ".") + "-" + train_dataset + ".txt")

def evaluation(model, eval_dataset, gpu_groups):
    cprint(f"This is the {i}-th step for evaluation.", color = "green")
    subprocess.run(
        f'python eval.py '
        f'--pretrained_model {model} '
        f'--dataset {eval_dataset} '
        '--use_api False '
        '--exe_verbose False '
        '--is_final_eval False '
        '--single_eval False '
        f'--k_code {eval_k_code} '
        f'--k_case {eval_k_case} '
        f'--scale_tuple_list "{repr(eval_scale_tuple_list)}" '
        f'--num_chunks {eval_num_chunks} '
        f'--no_example {eval_no_example} '
        f'--max_test {eval_max_test} '
        f'--gpu_groups "{repr(gpu_groups)}" ',
        shell=True,
        cwd='evaluation',
        check=True,
        env=build_subprocess_env(i),
    )
    archive_eval_results()

def sample(model):
    cprint(f"This is the {i}-th step for sampling.", color = "green")
    subprocess.run(
        f'python sample.py '
        f'--pretrained_model {model} ',
        shell=True,
        cwd='optimization',
        check=True,
        env=build_subprocess_env(i),
    )

def execute(model):
    cprint(f"This is the {i}-th step for execution.", color = "green")
    subprocess.run(
        f'python execute.py '
        f'--pretrained_model {model} ',
        shell=True,
        cwd='optimization',
        check=True,
        env=build_subprocess_env(i),
    )

def assign_reward(model):
    subprocess.run(
        f'python reward.py '
        f'--pretrained_model {model} ',
        shell=True,
        cwd='optimization',
        check=True,
        env=build_subprocess_env(i),
    )
    archive_optimization_results()

def train(model):
    cprint(f"This is the {i}-th step for training.", color = "green")
    subprocess.run(
        f'python -m train '
        f'--pretrain {model} '
        f'--step {i} '
        f'--tb_dir {exp_tb_dir} ',
        shell=True,
        cwd='optimization',
        check=True,
        env=build_subprocess_env(i),
    )

def save(model_from, model_to):
    os.makedirs(model_to, exist_ok=True)
    subprocess.run(f"rm -rf {model_to}/*", shell=True, check=True)
    subprocess.run(f"cp -r {model_from}/* {model_to}/", shell=True, check=True)

run_failed = False
failure_message = ""

# the first step if train from scratch
i = 0
try:
    sample(pretrain_model)
    execute(pretrain_model)
    assign_reward(pretrain_model)
    train(pretrain_model)
    i += 1

    while i <= total_steps:
        if i % eval_interval == 0:
            evaluation(model, eval_dataset, gpu_groups)
        if i % save_interval == 0:
            exp_ckpt = os.path.join(exp_dir, "optimization", "ckpt", f"iter{i}")
            save(model, exp_ckpt)

        if i == total_steps:
            break

        sample(model)
        execute(model)
        assign_reward(model)
        train(model)

        i += 1

    exp_final_ckpt = os.path.join(exp_dir, "optimization", "ckpt", "final")
    save(model, exp_final_ckpt)
except Exception as exc:
    run_failed = True
    failure_message = str(exc)
    raise
finally:
    archive_optimization_results()
    archive_eval_results()
    archive_log_symlink()
    with open(os.path.join(report_dir, "raw", "run_status.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "profile": profile_name,
                "failed": run_failed,
                "failure_message": failure_message,
                "last_step": i,
                "exp_dir": os.path.abspath(exp_dir),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    generate_report()
    package_report()

cprint(f"{'='*60}", color="yellow")
cprint(f"  Experiment finished: {exp_name}", color="yellow")
cprint(f"  All artifacts saved to: {os.path.abspath(exp_dir)}", color="yellow")
cprint(f"{'='*60}", color="yellow")
