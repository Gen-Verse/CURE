# Dataset

## Download

The training data is `CodeContests_train`, the evaluation datasets are `LiveBench`, `CodeContests`, `LiveCodeBench`, `CodeForces` and `MBPP`. You can simply download them by
```bash
# download the evaluation data
python download_data.py --dataset LiveBench-ReasonFlux
python download_data.py --dataset CodeContests
python download_data.py --dataset LiveCodeBench-ReasonFlux
python download_data.py --dataset CodeForces
python download_data.py --dataset MBPP-ReasonFlux
# download the training data
python download_data.py --dataset CodeContests_train
```

We use Stdio input/output format here. For example, for the task to calculate the sum of a list, the input and output are in the following format:
```python
input = "5\n1 2 3 4 5\n"
output = "15"
```
CodeContests and CodeForces are using this format, however, MBPP and part of LiveCodeBench are using functional input/output format, such like
```python
assert sum_function([1, 2, 3, 4, 5]) == 15
```
In this project, we have converted the the functional format to the Stdio format to achieve consistency (rules are witten in Appendix of our paper). We also provide the script `transformation.ipynb` to help convert the tasks which is in functional format in original LiveCodeBench/LiveBench to Stdio format. 


## Customize Your Own Dataset

Your JSON dataset must contain the following necessary fields to perform both optimization and evaluation.

1. `question`: This is the coding task description.
2. `test_time_limit`: The time limit for each task's execution, usually set as 1.
3. `example_input` and `example_output` are two lists of the same length, where each corresponding element represents the Stdio-format input and output, respectively. These unit tests are public (can be provided during inference time).
4. `test_input` and `test_output` are two lists of the same length, where each corresponding element represents the Stdio-format input and output, respectively. These private unit tests are used to evaluation.



The original sources for the datasets used in this paper are [LiveBench](https://huggingface.co/datasets/livebench/coding), [LiveCodeBench](https://huggingface.co/datasets/livecodebench/code_generation_lite), [CodeContests](https://huggingface.co/datasets/deepmind/code_contests), [CodeForces](https://huggingface.co/datasets/open-r1/codeforces), and [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp).
