# 自举奖励 GRPO 设计文档

## Self-Bootstrapped Execution Consistency Reward for GRPO

---

## 一、动机

CURE 原方案的训练依赖 ground-truth (GT) 测试用例：

1. **代码 reward**：`code_reward[j] = mean(通过 GT 测试的比例)`
2. **测试 reward**：先用 GT 识别"好代码"，再评价测试的区分力

这意味着训练数据必须带有高质量的 GT 测试用例，限制了可用数据规模。

**我们的目标**：设计一种不依赖 GT 的自举奖励机制，让代码生成器和测试生成器通过交叉执行的一致性信号完成完全自监督的协同进化，并结合 GRPO（Group Relative Policy Optimization）进行训练。

---

## 二、核心思想

### 2.1 执行一致性（Execution Consistency）

对每道题生成 k 份代码和 k 份测试，交叉执行得到布尔矩阵 M：

```
M[i,j] = 1 if code_i passes test_j, else 0

shape: (k_code, k_case)
```

**关键假设**：在足够多样的采样下，**多数代码对某个测试的行为一致** → 该测试的期望结果可以通过多数投票确定，无需 GT。

### 2.2 自举循环

```
代码质量 ← 在"可靠测试"上的表现
测试质量 ← 对"不同质量代码"的区分能力
         ↑_________________________________↓
```

这是一个"鸡生蛋"的关系，通过迭代 EM（Expectation-Maximization）求解。

---

## 三、算法详细设计

### 3.1 输入

对每道编程题 p，`sample.py` 生成：
- k_code 份代码 `{c_1, ..., c_k}`
- k_case 份测试 `{t_1, ..., t_k}`（模型生成的，无 GT）

`execute.py` 交叉执行得到：
- 布尔矩阵 `M[i,j]`，shape = (k_code, k_case)
- **不再拼接 GT 列**（与 CURE 原版的区别）

### 3.2 可选锚定：Public Example

每道题通常有 1-2 个公开的 example_input/example_output。如果存在，可以用它们作为弱锚定：

```
anchor_score[i] = code_i 通过所有 public example 的比例
```

这比 GT 测试的依赖弱得多（public example 是题目本身的一部分，不是额外标注）。

### 3.3 EM 迭代求解代码质量与测试质量

**符号定义**：
- `M`: (k_code, k_case) 布尔矩阵
- `q[i]`: 代码 i 的质量分数，∈ [0, 1]
- `w[j]`: 测试 j 的可靠性权重，∈ [0, 1]
- `e[j]`: 测试 j 的期望行为（多数投票），∈ {0, 1}

**初始化**：
```python
# 测试的期望行为：多数投票
pass_rate[j] = mean(M[:, j])      # 第 j 列的通过率
e[j] = round(pass_rate[j])        # >0.5 则期望通过，否则期望不通过

# 测试的初始权重：共识程度
w[j] = max(pass_rate[j], 1 - pass_rate[j])

# 如果有 public example，初始化代码质量锚定
if has_public_examples:
    q[i] = α * anchor_score[i] + (1 - α) * 0.5    # α 为锚定强度，建议 0.3
else:
    q[i] = 0.5    # 无锚定时均匀初始化
```

**E 步 — 更新代码质量**：
```python
for i in range(k_code):
    # 代码 i 与期望行为的加权一致性
    agreement[i] = sum(w[j] * (M[i,j] == e[j]) for j in range(k_case))
    q[i] = agreement[i] / sum(w[j] for j in range(k_case))
```

**M 步 — 更新测试质量**：
```python
# 将代码按质量分为"好代码"和"坏代码"
threshold = median(q)
good_codes = {i : q[i] > threshold}
bad_codes  = {i : q[i] <= threshold}

for j in range(k_case):
    # 区分力：好代码和坏代码在该测试上的行为差异
    good_pass_rate = mean(M[good_codes, j])
    bad_pass_rate  = mean(M[bad_codes, j])
    discriminability[j] = abs(good_pass_rate - bad_pass_rate)

    # 共识：多大比例的代码行为一致
    consensus[j] = max(pass_rate[j], 1 - pass_rate[j])

    # 更新权重
    w[j] = discriminability[j] * consensus[j]

    # 更新期望行为（好代码的多数行为）
    e[j] = round(good_pass_rate)
```

**迭代**：重复 E 步和 M 步 2-3 轮即可收敛（矩阵很小，计算开销可忽略）。

### 3.4 计算最终 Reward

**代码 reward（per-problem group）**：

```python
# 方案 A：基于加权一致性的连续 reward
code_reward[i] = sum(w[j] * (M[i,j] == e[j]) for j in range(k_case)) / sum(w)

# GRPO 归一化
code_reward = (code_reward - mean(code_reward)) / std(code_reward)
```

**测试 reward（per-problem group）**：

```python
# 方向分：好测试 = 被好代码通过、被坏代码拒绝
if e[j] == 1:  # 期望通过的测试
    direction[j] = mean(M[good, j]) - mean(M[bad, j])   # 好代码通过多、坏代码通过少 → 正向
else:           # 期望不通过的测试
    direction[j] = mean(1 - M[good, j]) - mean(1 - M[bad, j])  # 反转

# 最终 reward = 区分力 × 方向
case_reward[j] = discriminability[j] * sign(direction[j])

# GRPO 归一化
case_reward = (case_reward - mean(case_reward)) / std(case_reward)
```

### 3.5 跳过低信心样本

当信号太弱时，该道题不参与训练：

```python
# 条件 1：所有代码行为完全相同（全通过或全不通过每个测试）→ 无区分信号
if all(pass_rate[j] in {0.0, 1.0} for j in range(k_case)):
    skip this problem

# 条件 2：EM 后所有测试权重都很低
if max(w) < min_weight_threshold:   # 建议 0.1
    skip this problem

# 条件 3：代码质量方差为零（所有代码表现一样）
if std(q) < 1e-6:
    skip this problem (for code reward)
```

---

## 四、与 CURE 原版的对比

| 维度 | CURE（原版） | 自举 GRPO（本方案） |
|------|-------------|-------------------|
| 代码 reward 来源 | GT 测试通过率 | EM 迭代估计的加权一致性 |
| 测试 reward 来源 | 用 GT 识别好代码 → 评价测试区分力 | 用共识识别好代码 → 评价测试区分力 |
| GT 依赖 | 强依赖（必须有 test_input/test_output） | 无依赖（可选 public example 弱锚定） |
| 训练算法 | PPO-like（无 critic，reward 当 advantage） | GRPO（显式 group normalization） |
| reward 数据格式 | `{prompt, response, reward}` | `{prompt, responses[], rewards[]}` |
| 可用数据 | 仅有 GT 标注的编程题 | 任何编程题（甚至无标注的） |

---

## 五、需要修改的文件

### 5.1 `execute.py` — 小改

**改动点**：执行矩阵不再拼接 GT 列，仅使用模型生成的测试。

具体：
- 当前 `sample.py` 将 GT 测试拼在 `all_case_input/all_case_output` 的前面，生成测试在后面
- 当前 `execute.py` 构建完整矩阵后，`reward.py` 用 `[:, :t]` 和 `[:, t:]` 拆分
- **新方案**：`sample.py` 中不再拼接 GT；或者保留拼接但在 `reward.py` 中忽略 GT 列

**推荐**：保留 `sample.py` 和 `execute.py` 不动（仍然拼接 GT、仍然执行完整矩阵），这样：
1. GT 列仍可用于**监控指标**（code acc 等），只是不参与 reward 计算
2. 切换回 CURE 原版只需换 `reward.py`
3. 对比实验更方便

唯一改动：`execute.py` 的统计输出部分增加一行自举指标的打印。

### 5.2 `reward.py` — 重写

这是核心改动。当前文件 196 行，新版本预计 250 行左右。

**输入**：和现在一样，读取 `temp_data/outputs-rl-*.json`

**输出**：和现在一样，写入 `temp_data/rl_code_data.json` + `temp_data/rl_case_data.json`，但 reward 值的计算逻辑完全不同。

输出格式沿用 grpo 版本的分组格式：
```json
{
  "prompt": "...",
  "responses": ["resp_1", "resp_2", ...],
  "rewards": [0.8, -0.3, ...]
}
```

### 5.3 `optimization_config.py` — 增加参数

新增的配置项：

```python
# ============= config for self-bootstrapped reward =================

# 是否使用自举 reward（False 则退回 CURE 原版 GT-based reward）
use_self_bootstrap = True

# EM 迭代次数（2-3 次足够）
em_iterations = 3

# 是否使用 public example 锚定
use_anchor = True

# 锚定强度 α，仅在 use_anchor=True 时生效
anchor_strength = 0.3

# 低信心样本的过滤阈值
min_weight_threshold = 0.1

# consensus 低于此值的测试不参与 reward 计算
min_consensus_threshold = 0.55
```

### 5.4 `train.py` / `trainer.py` — 不改

当前 grpo 版本的 `trainer.py` 已经实现了 GRPO 的 group normalization：

```python
# trainer.py line 108-118
reward_tensor = torch.tensor(rewards, dtype=torch.float32)
group_mean = reward_tensor.mean()
group_std = reward_tensor.std(unbiased=False)
centered = reward_tensor - group_mean
if group_std > 1e-6:
    centered = centered / (group_std + 1e-8)
```

这已经完全满足需求，不需要改。

### 5.5 `sample.py` — 不改

保持现有逻辑（仍然拼接 GT 到 `all_case_input`）。

### 5.6 `run.py` — 不改

主循环逻辑不变。

---

## 六、新 `reward.py` 的伪代码

```python
import numpy as np

def em_estimate(M, anchor_scores=None, n_iter=3, anchor_strength=0.3):
    """
    EM 迭代估计代码质量 q 和测试权重 w。

    Args:
        M: (k_code, k_case) 布尔矩阵
        anchor_scores: (k_code,) 可选，public example 通过率
        n_iter: 迭代次数
        anchor_strength: 锚定权重 α

    Returns:
        q: (k_code,) 代码质量
        w: (k_case,) 测试权重
        e: (k_case,) 测试期望行为 (0 or 1)
    """
    k_code, k_case = M.shape
    pass_rate = M.mean(axis=0)             # (k_case,)
    e = (pass_rate > 0.5).astype(float)    # (k_case,) 初始期望
    w = np.maximum(pass_rate, 1 - pass_rate)  # (k_case,) 初始权重

    # 初始化代码质量
    if anchor_scores is not None:
        q = anchor_strength * anchor_scores + (1 - anchor_strength) * 0.5
    else:
        q = np.full(k_code, 0.5)

    for iteration in range(n_iter):
        # === E 步：更新代码质量 ===
        agreement = np.zeros(k_code)
        for i in range(k_code):
            agreement[i] = np.sum(w * (M[i, :] == e))
        w_sum = np.sum(w)
        if w_sum > 0:
            q = agreement / w_sum
        # 混入锚定
        if anchor_scores is not None:
            q = anchor_strength * anchor_scores + (1 - anchor_strength) * q

        # === M 步：更新测试质量 ===
        threshold = np.median(q)
        good_mask = q > threshold
        bad_mask = q <= threshold

        if good_mask.sum() == 0 or bad_mask.sum() == 0:
            break

        for j in range(k_case):
            good_pass = M[good_mask, j].mean()
            bad_pass = M[bad_mask, j].mean()
            discriminability = abs(good_pass - bad_pass)
            consensus = max(pass_rate[j], 1 - pass_rate[j])
            w[j] = discriminability * consensus
            e[j] = round(good_pass)  # 好代码多数行为决定期望

    return q, w, e


def compute_code_reward(M, q, w, e):
    """计算代码 reward"""
    k_code, k_case = M.shape
    code_reward = np.zeros(k_code)
    w_sum = np.sum(w)
    if w_sum == 0:
        return None
    for i in range(k_code):
        code_reward[i] = np.sum(w * (M[i, :] == e)) / w_sum
    # GRPO normalization
    mean_r = code_reward.mean()
    std_r = code_reward.std()
    if std_r < 1e-6:
        return None
    return (code_reward - mean_r) / std_r


def compute_case_reward(M, q, w, e):
    """计算测试 reward"""
    k_code, k_case = M.shape
    threshold = np.median(q)
    good_mask = q > threshold
    bad_mask = q <= threshold
    if good_mask.sum() == 0 or bad_mask.sum() == 0:
        return None

    case_reward = np.zeros(k_case)
    for j in range(k_case):
        good_pass = M[good_mask, j].mean()
        bad_pass = M[bad_mask, j].mean()
        discriminability = abs(good_pass - bad_pass)
        if e[j] == 1:
            direction = good_pass - bad_pass
        else:
            direction = (1 - good_pass) - (1 - bad_pass)
        case_reward[j] = discriminability * np.sign(direction)

    # GRPO normalization
    mean_r = case_reward.mean()
    std_r = case_reward.std()
    if std_r < 1e-6:
        return None
    return (case_reward - mean_r) / std_r


# === 主流程 ===
for i in range(len(data)):
    M_full = np.array(data[i]["all_case_bool_table"])  # (k_code, GT+Gen)
    t = data[i]["num_ground_truth_test"]

    # 拆出只有生成测试的子矩阵（GT 列仅用于监控，不参与 reward）
    M = M_full[:, t:]  # (k_code, k_case)

    # 可选锚定：用 public example 计算锚定分
    if use_anchor and len(data[i].get("example_input", [])) > 0:
        # anchor 用 GT 前几列（public example 是题目的一部分，不算额外标注）
        n_example = min(len(data[i]["example_input"]), t)
        if n_example > 0:
            anchor_scores = M_full[:, :n_example].mean(axis=1)
        else:
            anchor_scores = None
    else:
        anchor_scores = None

    # 跳过空矩阵
    if M.shape[0] == 0 or M.shape[1] == 0:
        continue

    # EM 求解
    q, w, e = em_estimate(M, anchor_scores, n_iter=em_iterations, anchor_strength=anchor_strength)

    # 跳过低信心样本
    if np.max(w) < min_weight_threshold:
        continue

    # 代码 reward
    code_reward = compute_code_reward(M, q, w, e)
    if code_reward is not None:
        # ... 构建 group_entry，同现有格式 ...

    # 测试 reward
    case_reward = compute_case_reward(M, q, w, e)
    if case_reward is not None:
        # ... 构建 group_entry，同现有格式 ...
```

---

## 七、监控与诊断指标

在 `execute.py` 或 `reward.py` 中输出以下指标，用于监控自举 reward 的质量：

### 7.1 与 GT 的相关性（仅在有 GT 时可用，用于验证）

```python
# GT-based code reward（CURE 原版计算方式）
gt_code_reward = M_full[:, :t].mean(axis=1)
# Self-bootstrapped code reward
sb_code_reward = code_reward  # EM 输出

# Spearman 相关系数
from scipy.stats import spearmanr
corr, pval = spearmanr(gt_code_reward, sb_code_reward)
print(f"Code reward correlation with GT: {corr:.3f} (p={pval:.3f})")
```

**预期**：相关系数 > 0.7 说明自举 reward 有效捕捉了代码质量。

### 7.2 EM 收敛监控

```python
print(f"Problem {i}: EM converged in {n_iter} iterations")
print(f"  Code quality std: {q.std():.3f}")        # 应 > 0，说明有区分
print(f"  Test weight max: {w.max():.3f}")          # 应 > 0.1
print(f"  Test weight mean: {w.mean():.3f}")
print(f"  Skipped: {skipped_count}/{total_count}")  # 跳过比例不应过高
```

### 7.3 每步汇总

```python
print(f"Step {step}: {used_problems}/{total_problems} problems used for training")
print(f"  Code groups: {len(code_data)}, Case groups: {len(case_data)}")
print(f"  Avg code reward std (within group): {avg_code_std:.3f}")
print(f"  Avg case reward std (within group): {avg_case_std:.3f}")
print(f"  GT correlation (code): {mean_corr:.3f}")  # 仅在有 GT 时
```

---

## 八、实验计划

### Phase 1：验证自举 reward 的信号质量（不训练）

1. 用现有模型（如 Qwen3-4B）跑一次 `sample.py` + `execute.py`
2. 同时计算 GT-based reward（CURE 原版）和 self-bootstrapped reward（新方案）
3. 比较两者的 Spearman 相关系数

**通过标准**：代码 reward 相关系数 > 0.6，测试 reward 相关系数 > 0.5

### Phase 2：小规模训练对比

在 `small_train` 数据集上，用相同的模型和步数，分别跑：
- CURE 原版（GT-based reward）
- 自举 GRPO（self-bootstrapped reward）
- 自举 GRPO + anchor（加 public example 锚定）

比较训练后模型在 `small_eval` 上的 code acc 和 BoN acc。

### Phase 3：扩大到无 GT 数据

收集一批无 GT 的编程题（如从 LeetCode、Codeforces 抓题但不要测试用例），用自举 GRPO 训练，验证是否能提升模型能力。

---

## 九、风险与缓解策略

| 风险 | 触发条件 | 缓解策略 |
|------|---------|---------|
| **多数皆错** | 难题上大部分代码错误，共识指向错误答案 | 用 public example 锚定；consensus < 阈值时跳过该题 |
| **退化解** | 模型学会生成 trivial 测试（如空输入） | 过滤空测试 / 过短测试；监控测试多样性 |
| **共谋退化** | code 和 test 共同退化到互相"放水" | separate_training 保持分开训练；KL 约束防止偏离原始模型 |
| **信号过弱** | k 太小，矩阵中几乎没有区分信号 | 增大 k_code/k_case（建议 ≥ 8）；增大 n_sample_per_step |
| **奖励震荡** | EM 不稳定导致 reward 在步间震荡 | 固定 EM 迭代次数；对 reward 做 EMA 平滑 |

---

## 十、配置推荐

### 初始调试配置

```python
# reward 相关
use_self_bootstrap = True
em_iterations = 3
use_anchor = True
anchor_strength = 0.3
min_weight_threshold = 0.1
min_consensus_threshold = 0.55

# 采样（建议比 CURE 默认大，以获取更多信号）
k_code = 8
k_case = 8
n_sample_per_step = 5

# 训练
separate_training = True
use_kl_loss = True
kl_loss_coef = 0.01       # 防止偏离原始模型
```

### 正式训练配置

```python
k_code = 16
k_case = 16
n_sample_per_step = 50+
em_iterations = 3
use_anchor = True
anchor_strength = 0.2     # 训练后期可降低锚定
```

---

## 十一、总结

| 项目 | 内容 |
|------|------|
| 核心改动文件 | `reward.py`（重写）、`optimization_config.py`（增加参数） |
| 不改的文件 | `sample.py`、`execute.py`（保持 GT 拼接用于监控）、`train.py`/`trainer.py`、`run.py` |
| 算法 | EM 迭代估计代码质量 + 测试权重 → GRPO group normalization |
| 关键假设 | 足够多样的采样下，多数代码行为反映正确答案 |
| 最大风险 | 难题上多数代码都错 → 用 anchor + 低信心过滤缓解 |
| 验证路径 | 先验证 reward 与 GT 的相关性 → 再跑训练对比 |
