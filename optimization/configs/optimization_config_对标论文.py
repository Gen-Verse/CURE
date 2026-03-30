# ========================================= config for optimization process ==========================================
# Paper reproduction config — aligned with CURE (arXiv:2506.03136v2) Section 4.1.2
# Model: Qwen3-4B (long-CoT), 50 steps, N=M=16, CodeContests train/eval
# Hardware: 2 GPUs (adapted from paper's 8×A100)
# Backup of previous debug config: optimization_config_debug.py
# ====================================================================================================================


# the model you want to optimize
pretrained_model = "/mnt/shared-storage-user/ai4good2-share/models/Qwen/Qwen2.5-7B-Instruct"

# the training data and evaluation data
# Paper: CodeContests difficulty ≤ 2, 4.5k training + 200 eval
train_dataset = "CodeContests_train"
eval_dataset = "CodeContests"

# Paper Section 4.1.2: 4B model trained for 50 steps
total_steps = 150

# evaluate every eval_interval steps — paper Figure 2 curves need per-step data
# eval at step 10,20,...,50 to draw curves while keeping cost manageable
eval_interval = 25

# save optimized model every save_interval steps
save_interval = 25




# ============= config for sampling in each step =================

# Paper: "we generate 16 rollouts for unit tests and 16 for code"
k_code = 16
k_case = 16

# Paper: "temperature of 1.0, top-p of 1.0"
# For long-CoT model (Qwen3-4B), paper mentions lower temp 0.8 — use 1.0 first for standard reproduction
temp = 1.0

# number of tasks sampled per step — paper uses full training set iteratively
# With 2 GPUs this needs to be practical; 20 tasks × 16 codes × 16 tests = 5120 rollouts/step
n_sample_per_step = 20

# GPU usage for vllm inference
# 2 GPUs: single engine with tensor-parallel=2
gpu_groups = [[0,1]]

# max ground-truth unit test we can use here
max_ground_truth_test = 8

# set to 1 by default
max_input_examples = 1

# maximum number of tokens the vLLM engine can handle in a single sequence
max_model_len = 20000

# max token model can generate for each query
max_generation_token = 10000

# the probability for providing public unit test example in prompt
p_give_example = 1.0

# the prompt design for code generation and unit test generation — same as paper Appendix C.1
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

# should be proportional to k_code * k_case * n_sample_per_step
# 16 * 16 * 20 = 5120, use 32 chunks for parallel execution
num_chunks = 32

# Paper Table 1 / Figure 4: BoN uses (N=16, M=16)
# Also report intermediate scales for Figure 4 curves
scale_tuple_list = [(4, 4), (16, 16)]



# ============= config for reward assignment in each step =================

# set True by default
separate_training = True

# Paper Section 3.4: long-CoT model uses response-length-guided transformation
# Qwen3-4B IS a long-CoT model, so enable_efficient = True
enable_efficient = True
# Paper: "truncate responses longer than 8K tokens"
max_len_threshold = 8000
min_len_threshold = 1000

# False by default, but suggest True after 100+ optimization steps
post_stage = False



# ============= config for training in each step =================

# number of GPUs for training
total_num_nodes = 2

# Paper: "learning rate to 1 × 10^-6"
actor_learning_rate = 1e-6

# 0 by default
num_warmup_steps = 0

# number of updates each step, 1 by default
policy_update_steps = 1

# Paper: "KL coefficient β to 0.01"
use_kl_loss = True
kl_loss_coef = 0.01
use_kl_estimator_k3 = True

# max prompt (inquiry) length in collected data
prompt_max_len = 2000

# generation token limit
generate_max_len = 8000

# packing_max_len >= generate_max_len + prompt_max_len
packing_max_len = 20000

# number of epoch for this training, 1 by default
max_epochs = 1

# the output model name
optimized_model_name = "optimized"



# ============= config for evaluation during the optimization =================

# Eval uses N=M=16 to match paper Table 1
eval_k_code = 16
eval_k_case = 16
eval_scale_tuple_list = [(4, 4), (16, 16)]
eval_num_chunks = 32
eval_no_example = True
eval_max_test = 8
