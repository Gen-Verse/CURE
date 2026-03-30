# ========================  This is the config for evaluation  =========================
# ======================================================================================


# set True if you want to evaluate API model, set False for vllm inference model
use_api = False

# set to be True if you just want to evaluate the model's one-shot coding accuracy, set False if BoN performance is also wanted
single_eval = True

# dataset you can directly use for evaluation: CodeContests_test, LiveBench, LiveCodeBench, Codeforces, MBPP
dataset = "CodeContests"

# number of codes (k_code) and unit tests (k_case) generated for each task; 
k_code = 16
k_case = 16

# the BoN setting (num_code, num_case) you want to test, num_code <= k_code and num_case <= k_case. The running time is totally decided by k_code and k_case, so you can add as many settings as you like
scale_tuple_list = [(4, 4), (16, 16)]

# how many parts the execution tasks are divided into (too small may get stuck), should be proportion to k_code * k_case
num_chunks = 128 * 4

# if provide public test example in unit test generation prompt
no_example = True

# max ground-truth unit test we can use for evaluation here
max_test = 8

# if output process for execution (sometimes execution may take a very long time, such like LiveCodeBench, given its long time limit)
exe_verbose = True

# set True by default here, no need to change
is_final_eval = True

# set False by default, unless for some specific deepseek models (like coder models)
trust_remote_code = False














# ======================== config for vllm inference model (use_api = False) ========================


# vllm model name
pretrained_model = "Qwen/Qwen2.5-7B-Instruct"

# maximum number of tokens the vLLM engine can handle in a single sequence
max_model_len = 20000

# max token model can generate for each quiry
max_generation_token = 10000

# inference temperature
temp = 0.8

# GPU usage for vllm inference, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3]] represents two engines each with 2 GPUs
# each engine loads a model, so for <=7B model, you can only set GPU numbers for each engine <= 2
gpu_groups = [[0,1],[2,3],[4,5],[6,7]]















# ======================== config for API inference model (use_api = True) ========================


# api_key and base_url
api_key = "Your API Key"
base_url = "Base URL, For Example, https://api.openai.com/v1/chat/completions"

# api model name, such like "gpt-4o", "deepseek-chat"
api_model_name = "gpt-4o-mini"

# temperature
api_temperature = 0.8

# max inquiries submitted at one time
max_workers = 20

# if it's OpenAI's model, and your account is available for batch inference, recommend setting this to be True, it's cheaper
use_openai_batch_api = False

# max token can generate for each task
max_tokens = 2500

# the request per minute limit for your API
rpm_limit = 100













# ======================= the prompt for code generation and unit test generation ============================

system_prompts = """<|im_start|>You are a helpful assistant help user solve problems. \
<|im_end|>\n<|im_start|>User: You need to think first then write {{language}} script. {{special_requirements}}
This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: """


system_case_prompts = """<|im_start|>You are a helpful assistant help user generate test examples for coding tasks. \
<|im_end|>\n<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
This is the problem:\n{{problem}}\n
{{example_intro}}
You need to provide a new test example. A good test example should be completely accurate and conform to the problem’s format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.
Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.
Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n
**Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n**Explanation:**\n\nexplanation here.\n <|im_end|>\n<|im_start|>Assistant: """

# some special requirements for code generation
special_requirements = """You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example."""






