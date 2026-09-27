# features.py — frozen appearance features

[Back to TODO](README.md)

| Function | What to do |
| --- | --- |
| `load_encoder(device)` | Load the selected pretrained DenseNet121 feature encoder, select the documented feature layer/pooling, freeze its parameters, and put it in evaluation mode. Return the encoder. |
| `extract_features(image, boxes, encoder, batch_size)` | Clip boxes safely, crop people, apply the encoder's expected resize/color/normalization, process small crop batches, and return `[D,F]` features in detection order without gradients. |

Reuse the upstream crop encoder where suitable. Read each frame once. Start with a small crop batch such as 16 and reduce it if dense frames exceed memory. Keep the encoder in evaluation mode even while the graph model trains. Handle empty boxes with an empty feature tensor. Invalid crops need an explicit mask/filter applied to all aligned rows; never silently shift feature rows.

Keep only needed features on the GPU. Reuse a track's stored historical feature instead of re-encoding its old crop each frame. This is enough for the first implementation; a separate cache module is unnecessary.

If repeated training extraction is too slow, optionally save per-sequence CPU features in a `.pt` file with frame/box order, weight identity, and preprocessing settings. Load it only when those inputs match; otherwise regenerate. A simple save/load is sufficient.

**Quick check:** extract features for a few boxes, print shape and finite values, and check that box order is preserved. Two runs in evaluation mode should agree. Try a crowded frame before increasing batch size.

Transformer features are deferred unless the working pipeline and evaluation are complete. DenseNet is a repaired-code starting point, not the paper's DETR path.
