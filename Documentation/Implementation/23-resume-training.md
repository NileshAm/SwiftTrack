# Continue training from the last completed epoch

The current config is ready to continue `mot17_appearance_residual_v1` from its
saved epoch 20 through epoch 40:

```python
RESUME_TRAINING = True
EPOCHS = 40
LEARNING_RATE = 1e-4
```

Run from the project folder in the existing environment:

```bash
conda activate dragon-track
python train.py
```

`EPOCHS` is the **total target**, not the number to add. Epoch 20 plus a target of
40 runs epochs 21–40. To add ten epochs instead, use 30. When the saved epoch has
already reached the target, the script exits without modifying run files.
With no checkpoint/history files in the selected run, training starts at epoch 1.
To force a fresh experiment, choose a new `RUN_NAME` and optionally set
`RESUME_TRAINING=False`; false refuses existing run files rather than overwriting.

## What is restored

- Association weights, both residual projections and the learned matching parameter.
- Adam optimizer moments and step counters; trainable encoder weights when applicable.
- Completed epoch number, prior history and lowest validation loss/best epoch.
- Python, NumPy and PyTorch CPU/CUDA random states in newly written checkpoints.

The existing epoch-20 checkpoint predates random-state saving. It can resume its
weights and optimizer, but the script prints that exact random-stream continuation
is unavailable for that first legacy resume. Device changes and nondeterministic
GPU operations can also prevent bit-for-bit reproduction.

The current `LEARNING_RATE` is applied after restoring Adam. This deliberately
allows a lower rate for continuation without discarding its accumulated moments.
Other training-recipe changes (architecture, dataset split/detection source,
encoder preprocessing, loss, batches, supervision, frame limit, motion/history
and spatial gate) are rejected. Use a fresh experiment for those changes.
Tracking-only settings such as `MATCH_THRESHOLD` can differ.

## Saving and interruption recovery

Before continuing, the script copies the existing best checkpoint, last checkpoint
and history into `outputs/<run>/resume_backups/<timestamp>/`. These preserve the
earlier epoch-20 weights used for your completed evaluation. They consume extra
disk space, particularly if the encoder is trainable.

`last.pt` now saves **every completed training-plus-validation epoch**, even when
the legacy `SAVE_EVERY_EPOCH` flag is false. `model.pt` only updates when validation
loss improves across the whole run, including epochs before resume. No numbered
checkpoint per epoch is added. This does not implement HOTA-based checkpoint selection.

Each checkpoint and CSV uses a temporary file followed by an atomic rename. New
checkpoints contain the full loss history. If interruption occurs during CSV
writing, the resumed run restores that history without duplicate epochs, backing
up the previous CSV first. If a newly improved best checkpoint was saved before
an interruption prevented `last.pt` from updating, the newer complete best
checkpoint supplies that completed epoch instead. Corrupt checkpoints are reported
rather than silently skipped.

An interruption halfway through an epoch repeats that unfinished epoch; there is
no mid-frame/batch recovery. The ETA counts only remaining epochs. Legacy checkpoints
require their original CSV; history after the chosen checkpoint is backed up and
recomputed if it was never checkpointed.

Existing tracking/evaluation outputs are retained and still describe the earlier
evaluated weights. When evaluating the newly continued model, choose a fresh
`TRACKING_VARIANT` (for example `residual_epoch40`) so the existing output protections
do not block tracking. Changing it in config also updates the derived output paths
when the next Python process starts. Run `test.py` again for a new timestamped
feature report; use the backed-up checkpoint if reproducing the earlier model.

## Parameter search

`search.py` forces `RESUME_TRAINING=False` inside each fresh training trial. Search
resume still skips completed trials, archives unfinished trial outputs and restarts
that trial. It does not accidentally continue your main production run. Tracking
mode continues to reuse its chosen checkpoint without training.

## Implementation

| File/function | Responsibility |
| --- | --- |
| `Utils/training_resume.py`: `load_resume` | Check compatible best/latest weights and reconstruct history from checkpoint or legacy CSV. |
| `random_state`, `restore_random_state` | Save/restore supported random-generator state without unsafe pickle loading. |
| `save_checkpoint`, `save_history` | Atomic publication of each completed checkpoint/history file. |
| `backup_run` | Preserve pre-continuation checkpoint/history files. |
| `train.py`: `main` | Restore model/encoder/optimizer, retain best loss, loop from the next epoch and show remaining ETA. |
| `Utils/config.py` | Enable continuation and set the total target epoch. |
| `search.py`: `trial_settings` | Keep training trials independent of main-run resume. |

## Verification

The existing full epoch-20 checkpoint and its history passed compatibility checks
without modification. Temporary controlled runs verified that uninterrupted and
resumed model/Adam states agree, including the trainable-encoder path. Checks also
covered retaining the earlier best, history recovery, backups, completed-target
no-op behavior, learning-rate changes, incompatible recipes and legacy checkpoints.
A separate full-resolution CPU DETR check trained on two real MOT17-04 graphs,
validated on two MOT17-02 graphs, saved, reloaded and completed the next epoch.
Production checkpoints/history remained unchanged. Full continuation has not been
launched by this implementation task; these short checks do not measure accuracy.
