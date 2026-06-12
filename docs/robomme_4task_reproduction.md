# RoboMME Four-Task Evaluation Reproduction

This note records a local evaluation-only reproduction on a selected four-task
RoboMME subset. It is meant to make the exact run auditable without committing
datasets, checkpoints, videos, or logs.

## Scope

- Date: 2026-06-12.
- Host: `L20_5`.
- GPU: physical GPU 2.
- Policy repo base: `RoboMME/robomme_policy_learning` at `ecf086c`.
- Benchmark submodule: `856bc3`.
- ManiSkill fork: `07be6f`.
- Seed: `7`.
- Checkpoint id: `79999`.
- Episodes: 50 per task.
- Tasks: `PickXtimes`, `VideoUnmask`, `VideoRepick`, `MoveCube`.
- Models:
  - `pi05_baseline`
  - `perceptual-framesamp-modul`

The official leaderboard evaluates 16 tasks and 800 episodes, with 50 episodes
per task and multiple random seeds recommended:
https://robomme.github.io/leaderboard.html

This run is therefore a focused four-task reproduction, not a replacement for
the full leaderboard protocol.

## Assets

The run used local copies of the released checkpoints:

- `runs/ckpts/pi05_baseline/pi05_baseline/79999`
- `runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999`

The `pi05_base` initializer was linked from an existing local OpenPI cache. The
evaluation run also used the local RoboMME simulator environment, cached
tokenizers, norm stats under `runs/assets`, and a Vulkan setup with local NVIDIA
GL libraries.

These assets are intentionally not committed.

## Launcher

The evaluation was run with `scripts/run_chunked_robomme_eval.py`.

The launcher starts the normal policy server and repeatedly invokes the official
`examples/robomme/eval.py` in short episode chunks. This keeps the rollout logic
inside the official evaluator while restarting the simulator process often
enough to avoid long-lived Vulkan renderer failures observed on this machine.

Temporary progress markers used to isolate a chunk are cleaned after each chunk.
The final `progress.json`, `log.json`, and `chunked_summary.json` contain only
real evaluated episodes.

Command used:

```bash
source /home/zhangyu/datasets/robomme_reproduction/env.sh
cd "$ROBOMME_POLICY_REPO"

RUN_ID=20260611_193611
LOGROOT=runs/evaluation_4task_full/logs/chunked_resume_v3_20260612_103700

"$ROBOMME_POLICY_VENV/bin/python" scripts/run_chunked_robomme_eval.py \
  --run-id "$RUN_ID" \
  --log-root "$LOGROOT" \
  --models pi05_baseline,perceptual-framesamp-modul \
  --tasks PickXtimes,VideoUnmask,VideoRepick,MoveCube \
  --episodes 50 \
  --chunk-size 5 \
  --retries 4 \
  --gpu 2 \
  --save-dir runs/evaluation_4task_full
```

## Result Paths

Final JSON outputs:

- `runs/evaluation_4task_full/pi05_baseline_4tasks50ep_gpu2_20260611_193611/ckpt79999/seed7/chunked_summary.json`
- `runs/evaluation_4task_full/perceptual-framesamp-modul_4tasks50ep_gpu2_20260611_193611/ckpt79999/seed7/chunked_summary.json`

Each result directory also contains `progress.json`, `log.json`, and 200 rollout
videos. These outputs are intentionally not committed.

## Results

| Model | PickXtimes | VideoUnmask | VideoRepick | MoveCube | Four-task total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pi05_baseline` | 20/50 = 40.0% | 12/50 = 24.0% | 0/50 = 0.0% | 20/50 = 40.0% | 52/200 = 26.0% |
| `perceptual-framesamp-modul` | 47/50 = 94.0% | 19/50 = 38.0% | 16/50 = 32.0% | 41/50 = 82.0% | 123/200 = 61.5% |

The `pi05_baseline` numbers are broadly consistent with the manually checked
official leaderboard reference on these tasks:

| Task | Official `pi0.5` reference | Local `pi05_baseline` |
| --- | ---: | ---: |
| `PickXtimes` | 42.89% | 40.0% |
| `VideoUnmask` | 20.44% | 24.0% |
| `VideoRepick` | 0.44% | 0.0% |
| `MoveCube` | 26.00% | 40.0% |

The selected four-task average is not directly comparable to a full 16-task
average. The released `perceptual-framesamp-modul` model card reports a 44.51%
full-suite average:
https://huggingface.co/Yinpei/mme_vla_suite

## Interpretation

The run confirms the expected qualitative story for these tasks:

- The no-history `pi05_baseline` is weak on memory-heavy tasks, especially
  `VideoRepick`.
- `perceptual-framesamp-modul` is a strong released baseline on the selected
  subset, improving from 26.0% to 61.5% overall.
- Future method work should treat FrameSamp+Modul as the primary strong
  baseline for this subset. Improvements against very weak recurrent baselines
  alone would be less convincing.

## Training Status

No training reproduction has been completed in this run.

The official training configuration uses 80k steps, four-way FSDP, and global
batch size 64 for `mme_vla_suite`. It also requires the full preprocessed
RoboMME dataset, norm stats, and the `pi05_base` checkpoint. The current machine
only has the sample preprocessed dataset staged, so a faithful training
reproduction needs a separate data-preparation step before launching a full
fine-tune.

Recommended next step before full training is a short training smoke test on the
sample dataset to validate dataloader, checkpoint writing, and memory usage.
After that, run a full FrameSamp+Modul training reproduction only if the method
change requires training from scratch or fine-tuning a modified memory module.
