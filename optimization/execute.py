import io
import os
import sys
import ast
import json
import time
import argparse
import numpy as np
import multiprocessing
from termcolor import cprint

import optimization_config
from bootstrap_reward_utils import canonicalize_execution_output
from reporting import append_jsonl, get_current_step, get_profile_name



####### execute the scripts with unit tests #########

def worker(script, input_val, output_queue):
    # Create an iterator over the input lines.
    input_lines = iter(input_val.splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")
    
    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val)  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin


def run_scripts_with_timeout(scripts, inputs, time_limits, worker):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    for i in range(len(scripts)):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=worker, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + time_limits[i])

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results


def test_if_eq(x, y):
    return " ".join(x.split()) == " ".join(y.split())

def get_chunk_indices(n, num_chunks):
    chunk_size = n // num_chunks   
    remainder = n % num_chunks 
    indices = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        indices.append((start, end))
        start = end
    return indices

def run_scripts_with_chunk(code_list, test_input_list, time_limit_list, worker, num_chunks):

    chunks = get_chunk_indices(len(code_list), num_chunks)
    exe_results = []
    i = 0
    for start, end in chunks:
        sub_code_list = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]
        sub_exe_results = run_scripts_with_timeout(sub_code_list, sub_test_input_list, sub_time_limit_list, worker)
        exe_results = exe_results + sub_exe_results
        i += 1
    return exe_results

def execute_scripts(outputs_name, num_chunks):

    os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + '.json'), exist_ok=True)
    with open("./temp_data/outputs-" + outputs_name + '.json', 'r') as f:
        data = json.load(f)

    # get input lists
    index_list = []
    position_list = []
    code_list = []
    case_input_list = []
    case_output_list = []
    time_limit_list = []
    public_index_list = []
    public_position_list = []
    public_code_list = []
    public_input_list = []
    public_output_list = []
    public_time_limit_list = []
    for i in range(len(data)):
        data[i].setdefault("example_input", [])
        data[i].setdefault("example_output", [])
        data[i].setdefault("test_input", [])
        data[i].setdefault("test_output", [])
        if len(data[i]["all_case_input"]) * len(data[i]["generated_code"]) == 0:
            data[i]["all_case_exe_results"] = None
            data[i]["all_case_bool_table"] = None
        else:
            n_row = len(data[i]["generated_code"])
            n_col = len(data[i]["all_case_input"])
            data[i]["all_case_exe_results"] = [["" for _ in range(n_col)] for _ in range(n_row)]
            data[i]["all_case_bool_table"] = np.full((n_row, n_col), False, dtype=bool)
        if len(data[i]["example_input"]) * len(data[i]["generated_code"]) == 0:
            data[i]["public_example_exe_results"] = None
            data[i]["public_example_bool_table"] = None
        else:
            n_row = len(data[i]["generated_code"])
            n_col = len(data[i]["example_input"])
            data[i]["public_example_exe_results"] = [["" for _ in range(n_col)] for _ in range(n_row)]
            data[i]["public_example_bool_table"] = np.full((n_row, n_col), False, dtype=bool)

        data_i = data[i]
        time_limit = data_i["test_time_limit"] if "test_time_limit" in data_i else 1
        for j in range(len(data_i["generated_code"])):
            for k in range(len(data_i["all_case_input"])):
                code = data_i["generated_code"][j]
                case_input = data_i["all_case_input"][k]
                case_output = data_i["all_case_output"][k]
                code_list.append(code)
                case_input_list.append(case_input)
                case_output_list.append(case_output)
                time_limit_list.append(time_limit)
                index_list.append(i)
                position_list.append((j, k))
            for k in range(len(data_i["example_input"])):
                code = data_i["generated_code"][j]
                case_input = data_i["example_input"][k]
                case_output = data_i["example_output"][k]
                public_code_list.append(code)
                public_input_list.append(case_input)
                public_output_list.append(case_output)
                public_time_limit_list.append(time_limit)
                public_index_list.append(i)
                public_position_list.append((j, k))
    
    # execute 
    exe_results = run_scripts_with_chunk(code_list, case_input_list, time_limit_list, worker, num_chunks)
    public_results = run_scripts_with_chunk(public_code_list, public_input_list, public_time_limit_list, worker, num_chunks) if public_code_list else []

    for i in range(len(index_list)):
        index_i = index_list[i]
        j, k = position_list[i]
        data[index_i]["all_case_exe_results"][j][k] = exe_results[i]
        data[index_i]["all_case_bool_table"][j][k] = test_if_eq(exe_results[i], case_output_list[i])
    for i in range(len(public_index_list)):
        index_i = public_index_list[i]
        j, k = public_position_list[i]
        data[index_i]["public_example_exe_results"][j][k] = public_results[i]
        data[index_i]["public_example_bool_table"][j][k] = test_if_eq(public_results[i], public_output_list[i])
    
    # get stats
    stats_single = {
        "BoN_score": 0,
        "BoN_num": 0,
        "BoN_acc_score": 0,
        "BoN_acc_num": 0
    }
    stats = []
    for i in range(len(scale_tuple_list)):
        stats_i = stats_single.copy()
        stats_i["tuple"] = scale_tuple_list[i]
        stats.append(stats_i)
    code_score = 0
    code_num = 0
    code_acc_score = 0
    code_acc_num = 0
    case_score = 0
    case_num = 0
    case_acc_score = 0
    case_acc_num = 0
    p_01_score = 0
    p_01_num = 0
    p_00_score = 0
    p_00_num = 0
    total_generated_exec = 0
    timeout_exec = 0
    exec_error_exec = 0
    empty_exec = 0
    public_example_score = 0
    public_example_num = 0
    for i in range(len(data)):
        if data[i]["all_case_exe_results"] is None:
            continue
        t = data[i]["num_ground_truth_test"]
        all_bool_table_i = np.array(data[i]["all_case_bool_table"])
        all_test_table_i = all_bool_table_i[:, :t].copy()
        all_case_table_i = all_bool_table_i[:, t:].copy()
        gen_results_i = np.array(data[i]["all_case_exe_results"], dtype=object)[:, t:].copy()
        gen_labels_i = np.vectorize(canonicalize_execution_output)(gen_results_i) if gen_results_i.size else np.array([])
        total_generated_exec += gen_labels_i.size
        timeout_exec += int(np.sum(gen_labels_i == "__TIMEOUT__"))
        exec_error_exec += int(np.sum(gen_labels_i == "__EXEC_ERROR__"))
        empty_exec += int(np.sum(gen_labels_i == "__EMPTY__"))
        if data[i]["public_example_bool_table"] is not None:
            public_table_i = np.array(data[i]["public_example_bool_table"])
            public_example_score += int(np.sum(public_table_i))
            public_example_num += int(public_table_i.size)
        if t > 0:
            correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
            code_score += len(correct_code_list)
            code_num += all_test_table_i.shape[0]
            code_acc_score += np.sum(all_test_table_i).item()
            code_acc_num += all_test_table_i.shape[0] * all_test_table_i.shape[1]
            sub_case_table_i = all_case_table_i[correct_code_list, :].copy()
            correct_case_list = np.where(sub_case_table_i.all(axis=0))[0].tolist()
            if len(correct_code_list) > 0:
                case_score += len(correct_case_list)
                case_num += sub_case_table_i.shape[1]
                case_acc_score += np.sum(sub_case_table_i).item()
                case_acc_num += sub_case_table_i.shape[0] * sub_case_table_i.shape[1]
                wrong_code_list = [j for j in range(all_case_table_i.shape[0]) if j not in correct_code_list]
                wrong_case_list = [j for j in range(all_case_table_i.shape[1]) if j not in correct_case_list]
                if len(wrong_code_list) > 0:
                    if len(correct_case_list) > 0:
                        wrong_code_correct_case_table_i = all_case_table_i[wrong_code_list, :][:, correct_case_list].copy()
                        p_01_score += np.sum(~wrong_code_correct_case_table_i).item()
                        p_01_num += wrong_code_correct_case_table_i.shape[0] * wrong_code_correct_case_table_i.shape[1]
                    if len(wrong_case_list) > 0:
                        wrong_code_wrong_case_table_i = all_case_table_i[wrong_code_list, :][:, wrong_case_list].copy()
                        p_00_score += np.sum(wrong_code_wrong_case_table_i).item()
                        p_00_num += wrong_code_wrong_case_table_i.shape[0] * wrong_code_wrong_case_table_i.shape[1]
            index_id = 0
            for scale_num_code, scale_num_case in scale_tuple_list:
                case_table_i = all_case_table_i[:scale_num_code, :scale_num_case].copy()
                test_table_i = all_test_table_i[:scale_num_code, :].copy()
                if case_table_i.size == 0 or test_table_i.size == 0:
                    index_id += 1
                    continue
                best_code_index = np.sum(case_table_i, 1).argmax()
                sub_test_table_i = test_table_i[best_code_index, :].copy()
                stats[index_id]["BoN_score"] = stats[index_id]["BoN_score"] + int(all(sub_test_table_i))
                stats[index_id]["BoN_num"] = stats[index_id]["BoN_num"] + 1
                stats[index_id]["BoN_acc_score"] = stats[index_id]["BoN_acc_score"] + np.sum(sub_test_table_i).item()
                stats[index_id]["BoN_acc_num"] = stats[index_id]["BoN_acc_num"] + len(sub_test_table_i)
                assert int(all(sub_test_table_i)) / 1 <= np.sum(sub_test_table_i).item() / len(sub_test_table_i), "error"
                index_id += 1

    os.makedirs(os.path.dirname("./results/results-" + outputs_name + ".txt"), exist_ok=True)
    with open("./results/results-" + outputs_name + ".txt", "a") as f:
        # Save + print
        def save_and_print(text):
            cprint(text, color="green")
            f.write(text + "\n")

        # Your values
        def safe_divide(d1, d2):
            if d2 == 0:
                return 0
            return d1/d2
        code_acc = safe_divide(code_score, code_num)
        code_acc_acc = safe_divide(code_acc_score, code_acc_num)
        case_acc = safe_divide(case_score, case_num)
        case_acc_acc = safe_divide(case_acc_score, case_acc_num)
        p_01 = safe_divide(p_01_score, p_01_num)
        p_00 = safe_divide(p_00_score, p_00_num)

        save_and_print(f"code acc: {code_acc}, code accumulate acc: {code_acc_acc}")
        save_and_print(f"case acc: {case_acc}, case accumulate acc: {case_acc_acc}")
        save_and_print(f"p_01: {1 - p_01}")
        save_and_print(f"p_00: {p_00}")

        for i in range(len(stats)):
            tuple_name = stats[i]["tuple"]
            save_and_print(f"BoN setting {tuple_name}:")
            acc = safe_divide(stats[i]["BoN_score"], stats[i]["BoN_num"])
            acc_acc = safe_divide(stats[i]["BoN_acc_score"], stats[i]["BoN_acc_num"])
            save_and_print(f"acc: {acc}, accumulate acc: {acc_acc}")

    def safe_divide(d1, d2):
        if d2 == 0:
            return 0
        return d1 / d2

    append_jsonl(
        "execute_step_metrics.jsonl",
        {
            "step": get_current_step(),
            "profile": get_profile_name(),
            "problems": len(data),
            "generated_exec_total": total_generated_exec,
            "timeout_rate": safe_divide(timeout_exec, total_generated_exec),
            "exec_error_rate": safe_divide(exec_error_exec, total_generated_exec),
            "empty_exec_rate": safe_divide(empty_exec, total_generated_exec),
            "public_example_coverage": safe_divide(
                sum(1 for item in data if item.get("public_example_bool_table") is not None),
                len(data),
            ),
            "public_example_acc": safe_divide(public_example_score, public_example_num),
        },
    )


    # convert np to list
    for i in range(len(data)):
        if data[i]["all_case_exe_results"] is None:
            continue
        data[i]["all_case_bool_table"] = data[i]["all_case_bool_table"].tolist()
        if data[i]["public_example_bool_table"] is not None:
            data[i]["public_example_bool_table"] = data[i]["public_example_bool_table"].tolist()

    # output the data
    os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + ".json"), exist_ok=True)
    with open("./temp_data/outputs-" + outputs_name + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)




# read the configurations and convert them to global variables

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--num_chunks", type=int, default=optimization_config.num_chunks)
    parser.add_argument("--scale_tuple_list", type=ast.literal_eval, default=optimization_config.scale_tuple_list)
    return parser.parse_args()

args = parse_args()
globals().update(vars(args))


# read processed data
outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset

# execute
execute_scripts(outputs_name, num_chunks)






