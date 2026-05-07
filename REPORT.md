# Lab 21 Report

## 1. Setup

- **Base model**: `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`
- **Fine-tuning method**: QLoRA 4-bit with Unsloth + TRL `SFTTrainer`
- **Dataset**: `tatsu-lab/alpaca`
- **Dataset size used for this lab**: 500 cleaned samples, split into 450 train / 50 eval
- **GPU**: Google Colab T4 (16 GB)
- **LoRA baseline config**: `r=16`, `lora_alpha=32`, `target_modules=["q_proj","v_proj"]`, `lora_dropout=0`
- **Training hyperparameters**: 3 epochs, learning rate `2e-4`, cosine schedule, warmup ratio `0.10`, effective batch size `8` (`per_device_train_batch_size=1`, `gradient_accumulation_steps=8`), `adamw_8bit`
- **Memory-saving choices**: gradient checkpointing enabled, `packing=False`, evaluation during training disabled on T4 to avoid OOM
- **Max sequence length**: `256`, chosen from the dataset token-length analysis with `p95 = 179` and a T4 cap of `1024`

**Option B links**

- GitHub: <https://github.com/Wan1302/2A202600081_HoTrongDuyQuang_Lab21.git>
- Hugging Face adapter: <https://huggingface.co/Wan1302/lab21-llama-3.2-3b-r16>
- W&B run: <https://wandb.ai/duyquangho1302/lab21-lora-qlora/runs/21t70npg?nw=nwuserduyquangho1302>

**Training cost estimate**

The three required LoRA runs (`r=8`, `r=16`, `r=64`) took about `20.49` minutes in total. Using the notebook's T4 estimate of `$0.35/hour`, the equivalent cloud cost is about **$0.12**. Including the stretch run (`r=16` target-all-layers), total adapter training time becomes about `27.59` minutes, or about **$0.16** equivalent T4 cost. On Colab Free, the direct out-of-pocket cost was effectively `$0`.

## 2. Rank Experiment Results

Perplexity was computed as `exp(eval_loss)`.

| Run | Rank | Target modules | Trainable params | Train time (min) | Peak VRAM (GB) | Eval loss | Eval perplexity |
|---|---:|---|---:|---:|---:|---:|---:|
| Base | 0 | none | 0 | 0.00 | 7.00 | 2.2002 | 9.0272 |
| LoRA r=8 | 8 | `q_proj + v_proj` | 2,293,760 | 6.62 | 6.44 | 1.5210 | 4.5767 |
| LoRA r=16 | 16 | `q_proj + v_proj` | 4,587,520 | 7.22 | 5.68 | 1.5109 | 4.5310 |
| LoRA r=64 | 64 | `q_proj + v_proj` | 18,350,080 | 6.64 | 7.43 | 1.4872 | 4.4246 |

**Stretch goal result**

| Run | Rank | Target modules | Trainable params | Train time (min) | Peak VRAM (GB) | Eval loss | Eval perplexity |
|---|---:|---|---:|---:|---:|---:|---:|
| LoRA r=16 all layers | 16 | `q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj` | 24,313,856 | 7.10 | 8.28 | 1.5181 | 4.5635 |

**Key observations**

- Fine-tuning helped a lot compared with the frozen base model: perplexity dropped from `9.03` to the `4.42-4.58` range for all LoRA runs.
- `r=64` achieved the best perplexity (`4.4246`), but the gain over `r=16` (`4.5310`) was small relative to the 4x increase in trainable parameters.
- `r=8` was the lightest useful adapter and already delivered most of the improvement.
- The stretch run with target-all-layers used the most memory and parameters, but its perplexity (`4.5635`) was actually slightly worse than the baseline `r=16 q/v` run. On this dataset, more target modules did not automatically improve quality.

## 3. Loss Curve Analysis

The W&B loss curve for the baseline `r=16 q/v` run shows a steady downward training trend from a little above `2.1` at the beginning to around `1.5` near the end of epoch 3. The learning-rate curve follows the intended cosine schedule, warming up early and then decaying smoothly toward zero. Gradient norm stayed in a reasonable range and did not show instability or explosion.

Because the notebook was tuned for a T4 GPU, evaluation during training was intentionally disabled (`eval_strategy="no"`) to reduce OOM risk. As a result, W&B contains a dense `train/loss` curve but only a final `eval/loss` point after training. This means I cannot diagnose overfitting from an eval-loss trajectory across epochs. Still, the final evaluation numbers are consistent: every LoRA run improved substantially over the frozen base model, which suggests the adapters converged normally without obvious training instability.

The saved loss curve image is `loss_curve.png`, and the corresponding W&B run is linked above.

## 4. Qualitative Comparison

I used the five prompts saved in `qualitative_comparison.csv` and kept both favorable and unfavorable cases instead of cherry-picking only wins.

| Prompt | Base model summary | Fine-tuned `r=16` summary | Comment |
|---|---|---|---|
| Explain the difference between LoRA and QLoRA | Discusses them as dimensionality-reduction methods; still wrong, but at least in the right ML area | Hallucinates "LoRA = Long Short-Term Memory" and "QLoRA = Quantum Long Short-Term Memory" | Fine-tuned output is clearly worse here |
| Write a Python function that returns the nth Fibonacci number | Gives a complete recursive solution and then suggests an improved version | Gives a valid recursive function, but the response becomes noisy with many `print(...)` lines and is cut off | Base model is better structured |
| List five practical tips for preparing for a job interview | Detailed five-point answer with explanation for each tip | Shorter five-point list with direct, practical advice | Fine-tuned output is acceptable and more concise, but not clearly better overall |
| Summarize why gradient checkpointing helps train large models on small GPUs | Mentions lower memory use but explains the mechanism inaccurately | Also mentions lower memory use and efficiency, but still gives an inaccurate mechanism | Both are partially wrong; fine-tuning did not fix factual precision |
| Create a concise study plan for learning machine learning in one month | Gives a structured week-by-week plan | Starts with short bullet points but stops early and is incomplete | Base model is clearly stronger |

**Qualitative takeaway**

The perplexity improvements did **not** translate into consistently better free-form answers on these prompts. The adapter improved the model statistically on the eval split, but the qualitative results remain mixed and sometimes worse than the base model. This is a useful reminder that lower perplexity does not guarantee better factuality, completeness, or instruction-following on every downstream prompt.

## 5. Conclusion về Rank Trade-off

For this lab, the best overall ROI was the baseline **`r=16` with `q_proj` and `v_proj` only**. It reached a perplexity of `4.5310`, very close to the best run `r=64` at `4.4246`, while using only `4,587,520` trainable parameters instead of `18,350,080`. That trade-off matters on a T4 because memory headroom is limited, and the goal is not just to win a single metric, but to get a reliable, repeatable training setup. The `r=8` adapter was even cheaper and still strong, so it is a valid budget option. However, `r=16` is a better middle point because it preserves most of the efficiency of `r=8` while slightly improving perplexity.

The `r=64` result shows classic diminishing returns. It delivered the best quantitative score, but the improvement over `r=16` was modest relative to the 4x parameter increase and higher VRAM usage. The stretch run is even more instructive: targeting all linear layers increased trainable parameters to `24,313,856` and peak VRAM to `8.28 GB`, but perplexity (`4.5635`) was slightly worse than the simpler `r=16 q/v` baseline. That indicates that for a small generic Alpaca subset, simply adding more LoRA capacity or more target modules does not guarantee better adaptation. My recommendation for this dataset and hardware is therefore: use `r=16 q/v` as the default, `r=8` if budget is extremely tight, and `r=64` only when a small extra perplexity gain is worth the added complexity.

## 6. What I Learned

- Gradient checkpointing, 4-bit loading, and disabling eval during training are not cosmetic tweaks on T4; they are the difference between a stable run and frequent OOM failures.
- Lower eval perplexity is useful, but it is not enough by itself. My qualitative comparison showed several prompts where the fine-tuned adapter was shorter, less precise, or more hallucinated than the base model.
- More LoRA capacity is not automatically better. In this experiment, `r=64` only slightly improved perplexity over `r=16`, and the target-all-layers stretch run actually underperformed the simpler `q_proj + v_proj` baseline despite using much more memory and many more trainable parameters.
