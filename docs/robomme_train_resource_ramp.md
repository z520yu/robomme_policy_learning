# RoboMME FrameSamp+Modul Training Resource Ramp

This note records a short resource ramp for reproducing the official
FrameSamp+Modul training configuration on L20 GPUs.

## Scope

- Date: 2026-06-13.
- Host: `L20_5`.
- GPUs: physical GPUs 0, 1, 2, 3.
- Model config: `mme_vla_suite`.
- History config: `perceptual-framesamp-modul.yaml`.
- Dataset: `data/robomme_preprocessed_data_sample`.
- Checkpoint writing: disabled for timing via `scripts/run_train_no_ckpt.py`.
- Ramp logs: `runs/train_ramp/logs/20260613_111633`.

The resource ramp validates train-step memory and timing. It does not replace a
full 80k-step training reproduction on the complete dataset.

## Scripts

Two helper scripts were used:

- `scripts/run_train_no_ckpt.py`: calls the normal training stack but monkey
  patches checkpoint saving. This avoids measuring Orbax write time in every
  short ramp run.
- `scripts/run_train_batch_ramp.sh`: runs a sequence of batch sizes and samples
  GPU memory/utilization with `nvidia-smi`.

## Ramp Results

All tested batch sizes completed 5 train steps with `fsdp_devices=4`.

| Batch size | Status | Peak GPU memory |
| ---: | ---: | ---: |
| 8 | pass | 38,491 MiB |
| 16 | pass | 38,537 MiB |
| 32 | pass | 38,601 MiB |
| 64 | pass | 38,475 MiB |

The peak memory is dominated by the sharded model, optimizer, and EMA state.
The incremental activation memory from batch size 8 to 64 was small in this
short test. L20 cards have enough headroom for the official batch size, assuming
the four GPUs are otherwise idle.

## Steady-State Batch-64 Timing

After the 5-step ramp, a 30-step batch-64 run reused the same experiment name to
benefit from the JAX compilation cache:

```text
runs/train_ramp/logs/20260613_111633/batch_64_steady30.log
```

It completed 30 steps in 2:29 according to the training progress log:

```text
30.0it/30.0it rate:4.2s/it elapsed:02:29
```

Observed GPU memory during this run:

| GPU | Peak memory | Peak utilization |
| ---: | ---: | ---: |
| 0 | 38,463 MiB | 100% |
| 1 | 38,479 MiB | 100% |
| 2 | 38,463 MiB | 100% |
| 3 | 38,479 MiB | 100% |

## Full-Train Estimate

Official FrameSamp+Modul training uses 80,000 steps. Based on the 30-step
batch-64 run:

- 4.2 s/step gives about 93.3 hours, or 3.9 days.
- 5.0 s/step gives about 111.1 hours, or 4.6 days.
- 7.0 s/step gives about 155.6 hours, or 6.5 days.

A realistic planning estimate is 5-7 days on four idle L20 GPUs, including
checkpoint overhead, occasional stalls, and dataloader variance.

## Storage Notes

The prior training smoke test showed that active Orbax checkpoint writing fails
on the public NFS-backed project disk with `ENOLCK No locks available`.

For a full training run:

1. Write active checkpoints to a local/home path, for example
   `/home/zhangyu/robomme_train_ckpts`.
2. Copy each finalized checkpoint to public storage with `rsync`.
3. Keep only a rolling window on home to avoid filling the local disk.
4. Keep public storage as the long-term copy and evaluation source.

Each checkpoint is about 12 GB. Eight official 10k-step checkpoints would need
about 96 GB if all are retained.

## Recommendation

The machine is suitable for a full FrameSamp+Modul train reproduction when four
L20 GPUs are reserved. Before starting the full run, stage the complete
preprocessed dataset and launch with checkpoint staging to home plus periodic
public sync.
