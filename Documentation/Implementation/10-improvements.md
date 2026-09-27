# Optional work after the four-day essentials

[Back to TODO](README.md)

The working tracker, saved model, measured evaluation, comparisons, and demo come first. No extra accuracy algorithm is required for the minimal delivery.

If time remains, choose one small change based on an observed validation failure: for example, confidence-weighted appearance history or a low-confidence second matching pass. Add it inside the existing `track.py`, controlled by one setting in `config.py`. Do not create a new improvements package or ablation runner.

Run the same checkpoint, detections, and held-out frames with the setting off and on. Record accuracy/speed and whether retraining was needed in `results.md`. A worse or inconclusive result is still worth reporting. Restoring real scores, repairing geometry, and reconnecting missing losses are fixes, not new algorithms.

Leave these for later unless the main delivery is already complete: transformer-to-box feature integration, unverified paper components, motion/camera compensation, learned long-term memory, graph pruning, extra comparison trackers, and full benchmark/multiple-seed training. Name missing paper components plainly when describing the four-day result.
