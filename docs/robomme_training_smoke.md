# RoboMME Training Smoke Test

This note records a short FrameSamp+Modul training smoke test. The goal was to
validate the training chain, not to reproduce a full training run.

## Scope

- Date: 2026-06-12.
- Host: `L20_5`.
- Policy repo base: `RoboMME/robomme_policy_learning` at `ecf086c`, plus local
  reproduction utility commits.
- Model config: `mme_vla_suite`.
- History config: `perceptual-framesamp-modul.yaml`.
- Dataset: `data/robomme_preprocessed_data_sample`.
- Initial weights: local `pi05_base` OpenPI checkpoint.
- W&B: disabled.

## Command

The successful checkpoint-writing run used idle GPUs 0 and 1:

```bash
source /home/zhangyu/datasets/robomme_reproduction/env.sh
cd "$ROBOMME_POLICY_REPO"

EXP=perceptual_framesamp_modul_train_smoke_homeckpt_20260612_125535

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
TRANSFORMERS_OFFLINE=1 \
HF_HUB_OFFLINE=1 \
"$ROBOMME_POLICY_VENV/bin/python" scripts/train.py mme_vla_suite \
  --exp-name "$EXP" \
  --overwrite \
  --seed 7 \
  --batch-size 2 \
  --num-workers 1 \
  --fsdp-devices 2 \
  --num-train-steps 1 \
  --log-interval 1 \
  --save-interval 1 \
  --keep-period 1 \
  --dataset-path data/robomme_preprocessed_data_sample \
  --checkpoint-base-dir /home/zhangyu/robomme_train_smoke_ckpts \
  --model.use-history \
  --model.history-config perceptual-framesamp-modul.yaml \
  --no-wandb-enabled
```

The upstream `scripts/train.py` runs a tentative pass and then a normal pass in
its `__main__`, so this command performs the same one-step smoke twice.

## Results

The training path is valid:

- Data loader initialized correctly.
- FrameSamp perceptual memory with modulation was selected.
- `pi05_base` loaded from the local cache.
- Missing memory-specific weights were initialized and merged.
- FSDP sharding worked on two L20 GPUs.
- Step loss was finite: `loss=0.1176`.
- Peak observed memory was about 38.5 GB per GPU for `batch_size=2`,
  `fsdp_devices=2`.

The successful checkpoint was finalized under:

```text
/home/zhangyu/robomme_train_smoke_ckpts/mme_vla_suite/perceptual_framesamp_modul_train_smoke_homeckpt_20260612_125535/0
```

It was copied to the public project disk at:

```text
runs/train_smoke/ckpts_from_home/mme_vla_suite/perceptual_framesamp_modul_train_smoke_homeckpt_20260612_125535/0
```

The checkpoint directory is about 12 GB. Source and copied directory both had
25 files at the time of verification.

## Public Disk Checkpoint Issue

Writing Orbax/TensorStore checkpoints directly under the public project disk
failed with:

```text
ENOLCK No locks available
```

This happened after the training steps completed, during async checkpoint
serialization. The public disk appears usable for finalized checkpoint storage
but not as the active Orbax checkpoint-writing target.

Recommended workflow for future training:

1. Write active checkpoints to a local path such as
   `/home/zhangyu/robomme_train_ckpts`.
2. After each checkpoint finalizes, copy it to public storage with `rsync`.
3. Keep only a small rolling window on the home disk.
4. Use the public copy for longer-term storage and evaluation.

## Full Training Implications

The official `mme_vla_suite` training config uses 80k steps, `batch_size=64`,
and `fsdp_devices=4`. This smoke test validates the code path, but it does not
prove that the official batch size fits alongside other jobs.

Before a full FrameSamp+Modul training reproduction:

- Reserve four idle L20 GPUs.
- Use a home/local checkpoint staging directory.
- Download and stage the full preprocessed RoboMME dataset.
- Run a short batch-size ramp before committing to 80k steps.
