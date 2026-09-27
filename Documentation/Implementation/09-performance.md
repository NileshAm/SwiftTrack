# Simple timing and speed improvements

[Back to TODO](README.md)

Add timing inside **`track.py::run_sequence`**. No profiler file or profiling framework is needed.

1. Warm up the model on a few frames, then reset track state before measuring the actual sequence. Synchronize CUDA before starting and after finishing the full timed run. Use elapsed wall-clock time and the number of processed frames to calculate FPS.
2. Keep video rendering off. Record whether features are live or cached. The full supplied-box measurement includes image loading, features, graph/model, assignment, and track updates; external detection is excluded.
3. Record PyTorch peak GPU memory after resetting the peak counter. Repeat a short run and note variation. Use the same plugged-in/power conditions and no competing GPU job.
4. If needed, temporarily time a few coarse stages to identify the slowest one. GPU stage timing needs CUDA events or synchronization; label diagnostic timings separately because synchronization can change throughput.

Change only the slow part: batch crops in `features.py`, reuse unchanged historical features in `track.py`, vectorize pair calculations in `graph.py`/`model.py`, or remove repeated image loading and device transfers. Keep the same checkpoint, candidate set, thresholds, detections, and frames for before/after runs.

**Quick check:** compare scores/output on a saved small sample and rerun held-out metrics after an optimization. Investigate changed matches even if the score difference is small. Report the timing scope and peak memory in `results.md`; no automated performance report is needed.

Defer background queues, multiprocessing, mixed precision, compilation, and candidate pruning. If dense graphs run out of memory, reduce crop/batch sizes and remove duplicate tensors first. Chunking must preserve attention normalization; dropping candidate edges changes the algorithm.
