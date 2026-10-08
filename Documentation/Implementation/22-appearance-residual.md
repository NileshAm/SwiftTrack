# Preserve each node's own appearance during graph attention

The full validation diagnostic for `mot17_tuned_full_v1` found strict correct-match
ranking fell from 84.66% before the graph to 27.87% after it; 50.35% of competing
queries became ties. This is a conditional feature diagnostic, not tracking accuracy.
An isolated one-track/two-detection graph reproduced a structural cause: each
detection received the same track's message and lost its own distinguishing input.

## Implemented change

`Utils/attention.py`, `GraphAttention.forward`, now computes:

```text
neighbour_message = ReLU(attention_weighted_sum / incoming_degree)
output = residual_projection(own_features) + neighbour_message
```

Each layer has a separate, trainable, bias-free `residual` linear projection.
It maps input to output width (260→512 then 512→128 with DETR). The first layer
preserves a projection of the node's own appearance and geometry; the second
preserves a projection of its own first-layer representation. The self term is
added after message ReLU so it can retain signed information. Neighbour attention,
edge relationships and degree division otherwise keep their previous behavior.

There are no new candidate edges or self-loops. An isolated node retains its own
projected features, but forbidden matches remain zero and it remains unmatched.
Final embeddings are still normalized once in `AssociationModel.forward`, and
matching and contrastive loss use these same embeddings. The frozen DETR encoder,
motion and appearance-history helpers, training video split, tuned thresholds,
loss settings and epoch budget remain configured as before.

This is a learned self-feature path, not a lossless copy of all 256 raw appearance
coordinates. Training can still learn poor representations; the residual removes
the forced shared-neighbour symmetry but does not guarantee correct rankings or
better HOTA/IDF1. It is an experimental extension, not a claimed exact paper reproduction.

## Training and checkpoint compatibility

`MODEL_VERSION` is now `dragontrack_projected_self_residual_v5`.
The new residual parameters are automatically included in the existing optimizer
and checkpoint state dictionaries. Train, validation, tracking and the feature
diagnostic all construct the same `AssociationModel`; no separate enable switch
is needed. Old checkpoints are rejected rather than combined with random residual
weights. Do not bypass this using `strict=False`.

Config selects the new run `mot17_appearance_residual_v1`. The previous tuned run,
checkpoint, evaluation and diagnostic reports remain intact. From the project root:

```bash
conda activate dragon-track
python train.py
python test.py
python tracking.py
python evaluate.py
```

Training saves the best validation-loss checkpoint as `model.pt` and now always
saves `last.pt` each epoch. Tracking and the diagnostic use the configured best
checkpoint. Compatible runs can now continue with `RESUME_TRAINING=True`; see the
[resume guide](23-resume-training.md). Periodic HOTA-based checkpoint selection is
not part of this change.

Search manifests now include the model version. Automatic resume starts a separate
search for the residual model; an explicitly selected incompatible `RESUME_DIR`
fails clearly. Existing older manifests without a version are incompatible with
the new model. `MODE="tracking"` requires a newly trained compatible checkpoint.

## Compare after training

Keep the previous full diagnostic and tracking scores as a baseline. Check the new
full-video report for fewer ties and improved strict ranking, particularly on
MOT17-02 where most competing queries occurred. Read same/different-person means
alongside ranking; an improved global mean gap alone was misleading in the old run.
Then compare tracking HOTA, IDF1 and identity switches with the same video split,
detections and tracker settings. Residual attention cannot recover missing detections.

## Files and responsibilities

| File/function | Responsibility |
| --- | --- |
| `Utils/attention.py`: `GraphAttention.__init__`, `forward` | Own-feature projection plus neighbour messages in both layers. |
| `model.py`: `MODEL_VERSION`, `AssociationModel.forward` | New architecture identity; common normalized output for matching and loss. |
| `Utils/config.py`: `RUN_NAME` | Separate checkpoint and output folder for fresh training. |
| `search.py`: `search_signature`, `main` | Include architecture version in saved manifests and resume comparison. |
| `test.py`: `load_models` | Explain that old checkpoints require fresh residual training. |

## Verification completed

- A controlled layer check retained signed self features with no neighbours.
- The one-track/two-detection regression produced distinct embeddings and finite
  gradients through both residual projections. This is a structural check, not an
  accuracy measurement of trained residual weights.
- Empty sides and fully masked graphs kept forbidden matching scores at zero;
  checkpoint save/reload reproduced scores and old checkpoint loading was rejected.
- Search signatures distinguished the new version from older unversioned searches.
- A CPU check used full-resolution pretrained DETR on frames 1–3 of MOT17-04 for
  training and MOT17-02 for validation. Both residual projections updated, the
  encoder stayed frozen, losses were finite and validation did not update weights.
- The previous tuned checkpoint's SHA256 remained unchanged. No full residual
  training, tracking or accuracy evaluation has been run yet.
