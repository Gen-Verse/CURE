import os
import re
import ast
import json
import random
import argparse
from jinja2 import Template
from termcolor import cprint
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import optimization_config





os.environ["TOKENIZERS_PARALLELISM"] = "false" 





####### vllm inference #######

def worker_fn(pretrained_model, gpu_ids, task_queue, result_queue, max_model_len, max_generation_token):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    print(f"Loading model on GPUs {gpu_ids}...")
    llm = LLM(
        model=pretrained_model,
        dtype="bfloat16",
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len
    )

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=1.0,
        top_k=-1,
        min_p=0.0,
        max_tokens=max_generation_token,
        stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"]
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            print("Stopping worker...")
            break
        task_id, prompts = task
        outputs = llm.generate(prompts, sampling_params)
        result_texts = [out.outputs[0].text for out in outputs]
        result_queue.put((task_id, result_texts))

# To run the worker setup:
def start_workers(pretrained_model, gpu_configs, max_model_len, max_generation_token):
    task_queues = []
    result_queues = []
    processes = []

    for i, gpu_ids in enumerate(gpu_configs):
        task_q = mp.Queue()
        result_q = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(pretrained_model, gpu_ids, task_q, result_q, max_model_len, max_generation_token)
        )
        p.start()
        task_queues.append(task_q)
        result_queues.append(result_q)
        processes.append(p)
    
    return task_queues, result_queues, processes

# Submit tasks
def submit_prompt_set(task_queues, prompt_sets):
    for i, prompts in enumerate(prompt_sets):
        task_queues[i].put((i, prompts))

# Collect results
def collect_results(result_queues, num_sets):
    results = [None] * num_sets
    for q in result_queues:
        task_id, result = q.get()
        results[task_id] = result
    return results

# Stop workers
def stop_workers(task_queues, processes):
    for q in task_queues:
        q.put("STOP")
    for p in processes:
        p.join()

# Split prompts into N chunks
def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

# vllm inference
def generate_results(all_prompts, gpu_groups,task_queues, result_queues):
    prompt_sets = split_prompts(all_prompts, len(gpu_groups))
    submit_prompt_set(task_queues, prompt_sets)
    results = collect_results(result_queues, len(prompt_sets))
    result_list = []
    for result_set in results:
        for r in result_set:
            result_list.append(r)
    return result_list

def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output

import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list




# read the condiguration and convert them into global variables

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default=optimization_config.pretrained_model)
    parser.add_argument("--dataset", type=str, default=optimization_config.train_dataset)
    parser.add_argument("--k_code", type=int, default=optimization_config.k_code)
    parser.add_argument("--k_case", type=int, default=optimization_config.k_case)
    parser.add_argument("--max_model_len", type=int, default=optimization_config.max_model_len)
    parser.add_argument("--max_generation_token", type=int, default=optimization_config.max_generation_token)
    parser.add_argument("--temp", type=float, default=optimization_config.temp)
    parser.add_argument("--p_give_example", type=float, default=optimization_config.p_give_example)
    parser.add_argument("--max_input_examples", type=int, default=optimization_config.max_input_examples)
    parser.add_argument("--max_ground_truth_test", type=int, default=optimization_config.max_ground_truth_test)
    parser.add_argument("--random_select_num", type=int, default=optimization_config.n_sample_per_step)
    parser.add_argument("--gpu_groups", type=ast.literal_eval, default=optimization_config.gpu_groups)
    parser.add_argument("--system_prompts", type=str, default=optimization_config.system_prompts)
    parser.add_argument("--system_case_prompts", type=str, default=optimization_config.system_case_prompts)
    parser.add_argument("--special_requirements", type=str, default=optimization_config.special_requirements)
    parser.add_argument("--post_stage", type=str2bool, default=optimization_config.post_stage)
    return parser.parse_args()

args = parse_args()
globals().update(vars(args))

if post_stage == True:
    p_give_example = 0.0









# read dataset
with open("../data/" + dataset + ".json", 'r') as f:
    data = json.load(f)
#data = [data[i] for i in range(10)]
random_select_num = min(random_select_num, len(data))
data = random_select(data, random_select_num)
num = len(data)


# load model, tokenizer, build vllm engines...
task_queues, result_queues, processes = start_workers(pretrained_model, gpu_groups, max_model_len, max_generation_token)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset









def bernoulli(p):
    return 1 if random.random() < p else 0

# obtain prompt
def get_scaling_prompt(data_i, method):
    problem = data_i["question"]
    if method == "sample":
        return Template(system_prompts).render(language = "python", special_requirements = special_requirements, problem = problem)
    if method == "case":
        n_example = len(data_i["example_input"])
        example_input = ", ".join([repr(item) for item in data_i['example_input']])
        example_output = ", ".join([repr(item) for item in data_i['example_output']])
        if n_example == 0:
            example_intro = """ """
        if n_example == 1:
            example_intro = """We already have one test sample:\n Its input is {{example_input}}. Its output is {{example_output}}.\n"""
            example_intro = Template(example_intro).render(example_input = example_input, example_output = example_output)
        if n_example > 1:
            example_intro = """We already have {{n_sample}} test samples:\n The inputs are, respectively, {{example_input}}. The corresponding outputs are {{example_output}}.\n"""
            example_intro = Template(example_intro).render(n_sample = n_example, example_input = example_input, example_output = example_output)
        return Template(system_case_prompts).render(problem = problem, example_intro = example_intro)

def modify(c):
    # Remove any occurrences of "plaintext\n"
    c = c.replace("plaintext\n", "")
    
    # Convert literal "\n" to actual newlines
    c = c.replace("\\n", "\n")
    
    # Ensure there's a trailing newline
    if not c.endswith("\n"):
        c += "\n"
    
    return c

# extract the unit tests from responses
def extract_test_cases(full_output):
    # First, try extracting with the updated triple-backtick pattern
    pattern_input_backticks = r'\*\*Test Input:\*\*\s*```(.*?)```'
    pattern_output_backticks = r'\*\*Test Output:\*\*\s*```(.*?)```'
    matches_input = re.findall(pattern_input_backticks, full_output, re.DOTALL)
    matches_output = re.findall(pattern_output_backticks, full_output, re.DOTALL)

    fail_case = [""]
    # For Test Input: either use the updated triple-backtick version or fallback to plain text
    if matches_input:
        test_input = [modify(matches_input[-1].lstrip('\n'))]
    else:
        # Fallback pattern without backticks: capture until **Test Output:**
        pattern_input_plain = r'\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)'
        matches_input_plain = re.findall(pattern_input_plain, full_output, re.DOTALL)
        if matches_input_plain:
            test_input = [modify(matches_input_plain[-1].strip())]
        else:
            test_input = fail_case
    
    # For Test Output: either use the updated triple-backtick version or fallback to plain text
    if matches_output:
        test_output = [modify(matches_output[-1].lstrip('\n'))]
    else:
        # Fallback: capture until the **Explanation:** marker or end-of-string
        pattern_output_plain = r'\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:|\*\*Test Input:|$)'
        matches_output_plain = re.findall(pattern_output_plain, full_output, re.DOTALL)
        if matches_output_plain:
            test_output = [modify(matches_output_plain[-1].strip())]
        else:
            test_output = fail_case
    
    # Also extract from the last occurrence of **Test Input:** to the end
    index = full_output.rfind("**Test Input:**")
    if index != -1:
        example_text = [full_output[index:]]
    else:
        example_text = fail_case
    
    # If any essential piece is missing, return empties
    if example_text == fail_case or test_input == fail_case or test_output == fail_case:
        return fail_case, fail_case, fail_case
    
    return test_input, test_output, example_text






# initialization
code_generation_prompts = []
code_index = []
case_generation_prompts = []
case_index = []
for i in range(num):
    # preprocess
    data[i]["full_code_generation"] = []
    data[i]["code_response_length"] = []
    data[i]["full_case_generation"] = []
    data[i]["case_response_length"] = []
    data[i]["generated_code"] = []
    max_k = min(max_ground_truth_test, len(data[i]["test_input"]))
    data[i]["num_ground_truth_test"] = max_k 
    data[i]["all_case_input"] = (data[i]["test_input"][:max_k]).copy()
    data[i]["all_case_output"] = (data[i]["test_output"][:max_k]).copy()
    data[i]["case_input"] = []
    data[i]["case_output"] = []
    data[i]["case_text"] = []

    data_i = data[i].copy()
    # get code generation prompts
    prompt_i = get_scaling_prompt(data_i, "sample")
    data[i]["code_generation_prompt"] = prompt_i
    code_generation_prompts = code_generation_prompts + [prompt_i] * k_code
    code_index = code_index + [i] * k_code
    # get case generation prompts
    #k_case_generate = k_case - min(k_case, len(data_i["example_input"]))
    k_case_generate = k_case
    if_give_example = bernoulli(p_give_example)
    if if_give_example == 0:
        data_i["example_input"] = []
        data_i["example_output"] = []
        data[i]["no_example"] = True
    else:
        max_input_examples_n = min(max_input_examples, len(data_i["example_input"]))
        data_i["example_input"] = data_i["example_input"][:max_input_examples_n]
        data_i["example_output"] = data_i["example_output"][:max_input_examples_n]
        data[i]["no_example"] = False
    prompt_i = get_scaling_prompt(data_i, "case")
    data[i]["case_generation_prompt"] = prompt_i

    if k_case_generate > 0:
        case_generation_prompts = case_generation_prompts + [prompt_i] * k_case_generate
        case_index = case_index + [i] * k_case_generate








# sampling process

cprint("start generation...", "green")

# shuffle first, to achieve efficiency
all_prompts = code_generation_prompts + case_generation_prompts
N = len(all_prompts)
indices = list(range(N))
shuffled_idx = indices[:]      
random.shuffle(shuffled_idx)
shuffled_prompts = [all_prompts[i] for i in shuffled_idx]
# generate
shuffled_outputs = generate_results(shuffled_prompts, gpu_groups, task_queues, result_queues)
restored_outputs = [None] * N
for out, idx in zip(shuffled_outputs, shuffled_idx):
    restored_outputs[idx] = out
code_generation_result = restored_outputs[:len(code_generation_prompts)]
case_generation_result = restored_outputs[len(code_generation_prompts):]

cprint("generation job done!", "green")










# calculate the response length

def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

code_response_length = get_token_lengths(code_generation_result, tokenizer)
case_response_length = get_token_lengths(case_generation_result, tokenizer)
mean_code = sum(code_response_length)/len(code_response_length)
mean_case = sum(case_response_length)/len(case_response_length)

os.makedirs(os.path.dirname("./results/results-" + outputs_name + ".txt"), exist_ok=True)
with open("./results/results-" + outputs_name + ".txt", "a") as f:
    # Save + print
    def save_and_print(text):
        cprint(text, color="green")
        f.write(text + "\n")
    save_and_print(f"code response length: {mean_code}, case response length: {mean_case}")





# process generated codes
i = 0
for full_output in code_generation_result:
    code_output = extract_code(full_output)
    index_i = code_index[i]
    data[index_i]["full_code_generation"] = data[index_i]["full_code_generation"] + [full_output]
    data[index_i]["generated_code"] = data[index_i]["generated_code"] + [code_output]
    data[index_i]["code_response_length"].append(code_response_length[i])
    i += 1

# process generated unit tests
i = 0
for full_output in case_generation_result:
    test_input, test_output, example_text = extract_test_cases(full_output)
    index_i = case_index[i]
    data[index_i]["full_case_generation"] = data[index_i]["full_case_generation"] + [full_output]
    data[index_i]["case_input"] = data[index_i]["case_input"] + test_input
    data[index_i]["case_output"] = data[index_i]["case_output"] + test_output
    data[index_i]["case_text"] = data[index_i]["case_text"] + example_text
    data[index_i]["all_case_input"] = data[index_i]["all_case_input"] + test_input
    data[index_i]["all_case_output"] = data[index_i]["all_case_output"] + test_output
    data[index_i]["case_response_length"].append(case_response_length[i])
    i += 1

# output the data
os.makedirs(os.path.dirname("./temp_data/outputs-" + outputs_name + ".json"), exist_ok=True)
with open("./temp_data/outputs-" + outputs_name + ".json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)



stop_workers(task_queues, processes)










