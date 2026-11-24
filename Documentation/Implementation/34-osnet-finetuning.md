# OSNet identity fine-tuning experiment

This trains only the existing MSMT17-pretrained OSNet appearance encoder. The
YOLOX detector stays on its published pretrained weights, and the direct-cosine
tracker still has no GCN or learned matcher. The original frozen OSNet baseline
remains the default when no `--osnet-checkpoint` is supplied.

## Train and compare learning rates

From the project directory:

```bash
conda activate dragon-track
python osnet-train.py --lr 1e-5 --epochs 5
python osnet-train.py --lr 3e-6 --epochs 5
```

Each learning rate gets a separate folder under `outputs/osnet_finetune/`. The
number given to `--epochs` is the total target; rerunning the same command with
a larger target resumes from `last.pt`. `best.pt` may be **epoch 0**, meaning
fine-tuning failed to improve the frozen pretrained features on the sampled
validation triplets. Use `--output-dir` for other parameter variations.

The trainer reads marked person identities only from `TRAIN_SEQUENCES`. Identity
keys include the video name because MOT identity numbers repeat across videos.
It samples same-person crops from distinct frames and prefers a different
person from the same frame as the negative. Validation pairs use only
`VAL_SEQUENCES` and a fixed seed. It trains with normalized-feature triplet
loss; no detector or graph weights are updated. Default learning rate is 1e-5,
batch size 8 triplets, 2,000 training pairs per epoch, 500 validation pairs,
and every fifth frame. `--lr`, `--batch-size`, `--pairs-per-epoch`,
`--val-pairs`, and `--frame-stride` are exposed. Checkpoints and `history.csv`
include the split and ground-truth hashes.

## Measure tracking, not just triplet accuracy

After training, run a separate full-video YOLOX + OSNet tracking evaluation for
each checkpoint. Example for the first learning rate:

```bash
python track-yolox-osnet.py --evaluate --detection-min-score 0.5 --new-track-min-score 0.7 --osnet-checkpoint outputs/osnet_finetune/lr_1e-05/best.pt
```

Use the same detection cache, similarity, motion/history and thresholds as the
frozen baseline. Each tracking invocation creates a fresh timestamped result
folder. Compare HOTA, IDF1, MOTA and identity switches with the frozen OSNet
result. Validation triplet accuracy is only a proxy; it does not prove better
tracking. The trainer crops **GT boxes**, whereas tracking crops YOLOX detection
boxes, so a train/inference crop mismatch remains. All validation videos have
been reused for tuning; they are not an untouched final test.

For a quick pipeline check only:

```bash
python osnet-train.py --lr 1e-5 --epochs 1 --pairs-per-epoch 8 --val-pairs 8 --batch-size 2 --frame-stride 30 --output-dir /tmp/osnet-check
```

The tiny check cannot establish accuracy. The trained checkpoint loader has
been checked on two real frames; a full fine-tuned tracking result is still to
be measured.

Files: `osnet-train.py` trains OSNet; `track-yolox-osnet.py` accepts the optional
fine-tuned checkpoint; `Utils/reid.py` supplies the original frozen encoder.
