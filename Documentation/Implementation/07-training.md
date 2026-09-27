# train.py — a small training loop

[Back to TODO](README.md)

| Function | What to do |
| --- | --- |
| `train_one_epoch(model, frame_pairs, encoder, optimizer)` | Load pairs from training sequences, obtain frozen features, build graphs/identity targets, call the model and `compute_loss`, then zero gradients, backward, and update. Print total/BCE/contrastive loss and elapsed time. |
| `validate(model, frame_pairs, encoder)` | Switch the association model to evaluation mode and disable gradients; compute loss on validation pairs only. Restore training mode afterward. Keep the frozen extractor in evaluation mode throughout. |
| `main()` | Read `config.py`, set a seed, create model/Adam, first run a tiny sample, then loop for the selected epoch budget and save checkpoints. |

Start with one graph at a time, FP32, Adam at the chosen fixed learning rate, and no scheduler, gradient accumulation, or worker pipeline. Skip/report batches without valid labels. Match labels only for supervision; the network should receive the same kind of graph inputs it will see when tracking.

Save directly with `torch.save`: model state, optimizer state if resuming, epoch, settings, seed, feature weights/preprocessing, and the selected sequences. Load with `torch.load` and `load_state_dict` in training/tracking. No checkpoint utility module is needed. Store settings as simple values/dictionaries, not the imported Python module. Keep weights in the output folder.

Train a tiny labelled sample first. Both loss terms must be finite, trainable parameters must receive gradients, and the sample should become easier to associate. Then measure epoch duration and start a run that fits the remaining time. Save periodically and use validation, not final held-out evaluation, to choose a checkpoint.

**Quick check:** save, restart Python, load, and compare the same evaluation-mode output. Record actual epochs/time and training curves or printed losses in the run folder. A shortened run is acceptable when labelled; do not claim the paper's full training schedule.
