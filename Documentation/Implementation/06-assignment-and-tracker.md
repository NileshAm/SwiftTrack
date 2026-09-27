# track.py — IDs, assignment, output, and demo

[Back to TODO](README.md)

| Function | What to do |
| --- | --- |
| `match_tracks(scores, allowed_mask, match_threshold)` | Use the CPU Hungarian solver with the selected score/cost convention and an explicit unmatched option. Return matched row pairs plus unmatched track/detection rows. Respect invalid candidates; handle empty matrices before calling the solver. |
| `update_tracks(tracks, matches, unmatched_tracks, unmatched_detections, frame)` | Update matched boxes/features/scores, age unmatched tracks, remove expired tracks, and create new IDs for eligible detections. Return current visible observations. |
| `run_sequence(config, model, encoder)` | Read frames in order, extract features, build/score the graph, normalize if required, assign/update tracks, and write MOT rows. Use evaluation mode with no gradients. Clear state at each new sequence. |
| `draw_tracks(image, observations)` | Draw boxes and IDs for the optional demo. Use OpenCV video writing inside the sequence loop when enabled; no separate video script. |
| `main()` | Load paths/settings and the saved model, run selected sequences, and print output locations and timing. |

Use a list or dictionary of tracks and an increasing integer ID counter. Store each track's last box, feature, confidence, last-seen frame, and missed count. Keep lost tracks eligible until a configured short expiry. Never reuse a removed ID within a sequence. Output observed matches/new tracks under a stated birth rule; do not silently emit predictions for invisible people.

Every detection/track can match at most once. Handle no tracks, no detections, no allowed pairs, and all-unmatched frames. Allow unmatched choices in assignment rather than forcing bad matches and assuming thresholding afterward is equivalent. Use original frame gaps when aging tracks.

Write MOT rows with original frame ID, positive track ID, xywh box, output score, and the expected placeholder fields. Reverse the internal coordinate-origin conversion exactly once. Avoid duplicate `(frame,id)` rows. Keep timing and optional rendering in this file.

**Quick check:** run a short clip and inspect IDs, boxes, a person disappearing/reappearing, and a frame with no detections. Read a few exported rows. Restart Python and reload the same checkpoint to repeat the clip. Meaningful accuracy requires a trained checkpoint; random-model output is only a plumbing check.
