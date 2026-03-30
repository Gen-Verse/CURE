# ========================================= config for optimization process ==========================================
# ====================================================================================================================


# the model you want to optimize
pretrained_model = "/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-7B-Instruct"

# the training data and evaluation data
train_dataset = "small_train"
eval_dataset = "small_eval"

# total steps for optimization
total_steps = 10

# evaluate every eval_interval steps
eval_interval = 2

# save optimized model every save_interval steps
save_interval = 10




# ============= config for sampling in each step =================

# number of codes and unit tests sampled in each step
k_code = 4
k_case = 4

# temperature
temp = 1.0

# number of tasks for sampling in each step
n_sample_per_step = 20

# GPU usage for vllm inference, [[0]] represents only one engine with one GPU; [[0, 1], [2, 3]] represents two engines each with 2 GPUs
# each engine loads a model, so for <=7B model, you can only set GPU numbers for each engine <= 2
gpu_groups = [[0,1]]

# max ground-truth unit test we can use here
max_ground_truth_test = 8

# set to 1 by default
max_input_examples = 1

# maximum number of tokens the vLLM engine can handle in a single sequence
max_model_len = 20000

# max token model can generate for each quiry
max_generation_token = 10000

# the probability for providing public unit test example in prompt
p_give_example = 1.0

# the prompt design for code generation and unit test generation
system_prompts = """<|im_start|>You are a helpful assistant help user solve problems. \
<|im_end|>\n<|im_start|>User: You need to think first then write {{language}} script. {{special_requirements}}
This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: """

system_case_prompts = """<|im_start|>You are a helpful assistant help user generate test examples for coding tasks. \
<|im_end|>\n<|im_start|>User: Given a coding task, instead of providing the final script, your task is to generate a new test example (both input, output and explanation).
This is the problem:\n{{problem}}\n
{{example_intro}}
You need to provide a new test example. A good test example should be completely accurate and conform to the problem's format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.
Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.
Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:\n
**Test Input:**\n```input here```\n\n**Test Output:**\n```output here```\n\n**Explanation:**\n\nexplanation here.\n <|im_end|>\n<|im_start|>Assistant: """

# some special requirements for code generation
special_requirements = """You should use input() to input and print() to output in your script. """



# ============= config for execution in each step =================

# how many parts the execution tasks are divided into (too small may get stuck), should be proportion to k_code * k_case * n_sample_per_step
num_chunks = 16

# the BoN setting you want to see in each step's output
scale_tuple_list = [(4, 4)]



# ============= config for reward assignment in each step =================

# set True by default
separate_training = True

# set False for standard base model, True for long-CoT model
enable_efficient = False
# when enable_efficient = True, responses with length >= max_len_threshold enforce negative reward, responses with length <= min_len_threshold no need for length penalty
max_len_threshold = 8000
min_len_threshold = 1000

# False by default, but suggest True after 100+ optimization steps
post_stage = False



# ============= config for training in each step =================

# number of GPUs
total_num_nodes = 2

# learning rate
actor_learning_rate = 1e-6

# 0 by default
num_warmup_steps = 0

# number of updates each step, 1 by default
policy_update_steps = 1

# KL loss setting
use_kl_loss = True
kl_loss_coef = 0.01
use_kl_estimator_k3 = True

# max prompt (inquiry) length in collected data
prompt_max_len = 2000

# generation token limit
generate_max_len = 8000

# we use packing here instead of batching for training, and we need packing_max_len >= generate_max_len + prompt_max_len
packing_max_len = 20000

# number of epoch for this training, 1 by default
max_epochs = 1

# the output model name
optimized_model_name = "optimized-debug01"



# ============= config for evaluation during the optimization =================

eval_k_code = 4
eval_k_case = 4
eval_scale_tuple_list = [(4, 4)]
eval_num_chunks = 16
eval_no_example = True
eval_max_test = 8
