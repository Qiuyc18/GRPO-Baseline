# Replay Buffer / Spec-Verify 实验对比记录

日期：2026-05-26  
项目：GRPO-Baseline / DeepScaleR / Qwen3-4B-Base / 4 GPU

## 结论摘要

这几组实验里，最稳的折中仍然是 **loose spec-verify + p_fresh=0.5**：它在扣除 test 开销后比 no-replay 快约 13%，同时 final validation score 仍接近 no-replay。

**strict spec-verify + p_fresh=0.5** 更保守，确实降低了 cache 接受率，但没有带来更好的 final score，反而牺牲了大部分加速。

**strict spec-verify + p_fresh=0.0** 能恢复到接近 loose spec-verify 的速度，但 score 明显下降。这说明即使有 verification，完全取消 fresh mixing 也会削弱训练稳定性或探索。

**naive replay** 虽然有一定加速，但 final score 崩到 0.1125，基本可以作为反例：不能无条件复用旧 trajectory。

正在跑的 **NAT RPC** 目前只有早期结果，actor update 明显更快，但还不能判断最终质量。

## 对比口径

为了避免 `test_freq` 不同带来的 wall-time 偏差，下面主要使用：

```text
adjusted_step_time = timing_s/step - timing_s/testing
```

其中 strict 系列使用 `test_freq=5`，测试更频繁；no-replay、naive replay、loose spec-verify 主要是 `test_freq=20`。因此 strict 的 raw wall time 会被 test 放大，必须看扣除 test 后的时间。

## 完整实验主表

| 实验 | p_fresh | verify | 阈值 mean / min_seq | cache rate | verify accept | 扣 test 后 mean step | 扣 test 后总时长 | best val | final val |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| no-replay | - | - | - | 0.0% | - | 201.6s | 6.61h | 0.4750 | 0.4750 |
| naive replay | 0.5 | no | - | 48.3% | - | 187.8s | 6.16h | 0.4450 | 0.1125 |
| spec-verify loose | 0.5 | yes | -1.0 / -5.0 | 46.6% | 93.2% | 174.9s | 5.73h | 0.4775 | 0.4550 |
| spec-verify strict | 0.5 | yes | -0.2 / -2.0 | 39.8% | 71.2% | 192.2s | 6.30h | 0.4675 | 0.4300 |
| spec-verify strict p_fresh=0 | 0.0 | yes | -0.2 / -2.0 | 53.4% | 53.8% | 174.6s | 5.72h | 0.4400 | 0.3775 |

## 时间分析

相对 no-replay，扣除 test 后：

| 实验 | adjusted mean step | 相对 no-replay |
|---|---:|---:|
| no-replay | 201.6s | baseline |
| naive replay | 187.8s | 快 6.8% |
| spec-verify loose | 174.9s | 快 13.3% |
| spec-verify strict | 192.2s | 快 4.7% |
| spec-verify strict p_fresh=0 | 174.6s | 快 13.4% |

strict p_fresh=0 的速度接近 loose spec-verify，原因是它每一步都尝试 cache，因此 cache rate 从 strict p_fresh=0.5 的 39.8% 提升到 53.4%。不过这不是无条件复用：117 次 verify attempt 中只有 63 次通过，accept rate 为 53.8%，其余会 fallback 到 fresh generation。

strict p_fresh=0.5 raw wall time 看起来更慢，其中 test 总开销约 0.36h；扣除 test 后仍比 loose 慢，主要原因是 strict threshold 降低了 cache 接受率。

## 质量分析

no-replay final score 是 0.475，是当前质量基线。

loose spec-verify final score 0.455，best 0.4775，说明在较宽松的 logprob verification 下可以保留大部分质量，同时获得明显加速。

strict spec-verify final score 0.430，best 0.4675。它更保守，但没有换来更高质量；可能原因是 reject 增多后 fresh step 增多，速度优势下降，但 verification 本身并不直接提升训练信号。

strict p_fresh=0 final score 0.3775，best 0.44。这个结果说明：即使 cache trajectory 经过 strict verification，完全取消 fresh mixing 仍然会损害训练稳定性。`p_fresh` 不只是性能参数，也是保持探索和 on-policy 更新比例的稳定性参数。

naive replay final score 0.1125，并且 final response length clip ratio 接近 0.97，说明直接复用旧 trajectory 会导致严重退化。

## Cache / Verification 行为

| 实验 | fresh steps | cache steps | verify attempts | accepts | rejects |
|---|---:|---:|---:|---:|---:|
| naive replay | 61 | 57 | 0 | - | - |
| spec-verify loose | 63 | 55 | 59 | 55 | 4 |
| spec-verify strict | 71 | 47 | 66 | 47 | 19 |
| spec-verify strict p_fresh=0 | 55 | 63 | 117 | 63 | 54 |

可以看到：

- loose threshold 几乎只过滤掉极少数 stale trajectory。
- strict threshold 明显降低接受率，但质量没有更好。
- p_fresh=0 会让 verify 几乎每步都参与；不过 reject 后仍会 fresh，因此不是完全 cache。
- p_fresh=0 的 final score 下降说明“verify 通过”不等价于“训练分布足够健康”。

## NAT RPC 早期观察

当前 `train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_nat_rpc_20260526_083556.log` 仍在运行，解析时只有 15 个 step，不能作为最终结果。

早期指标：

| 指标 | nat_rpc partial |
|---|---:|
| steps | 15 / 118 |
| adjusted mean step | 167.3s |
| cache rate | 33.3% |
| verify accept rate | 83.3% |
| val best / final | 0.3900 / 0.3825 |
| actor update mean | 78.1s |
| nat_keep_ratio mean | 0.356 |
| nat_response_len_ratio mean | 0.221 |

NAT RPC 的主要信号是 actor update 时间明显降低：相对 strict p_fresh=0.5 的 104.8s，早期 actor update mean 约 78.1s。但 early validation 还低，且 step 数太少，需要等完整 run 结束后再判断最终质量。

## 建议

1. 当前最适合用于汇报的主结果是 **loose spec-verify p_fresh=0.5**：加速明显，质量接近 no-replay。
2. strict threshold 可以作为 ablation：它证明 verification 更严格会降低 cache 接受率，但不一定提高质量。
3. p_fresh=0 可以作为关键负例：说明完全依赖 cache-first 即使有 strict verification，也会损害 final score。
4. 后续如果继续探索，优先尝试中间值，例如 `p_fresh=0.25`，而不是直接 0。
5. NAT RPC 值得继续观察完整结果；如果 final score 能保持，它可能和 spec-verify 是互补方向：一个减少 generation，一个减少 actor update token cost。

## 日志来源

- `logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_no_replay_20260524_074312.log`
- `logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_20260523_051105.log`
- `logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_spec_verify_20260525_034821.log`
- `logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_spec_verify_strict_20260525_105548.log`
- `logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_spec_verify_strict_pfresh0_20260526_015944.log`
- `logs/train_qwen3_4B-base_deepscaler_grpo_4gpu_replay_buffer_nat_rpc_20260526_083556.log`
