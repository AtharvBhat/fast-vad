# Benchmark Notes

These plots compare `fast_vad` against `silero_vad` (ONNX) and `webrtcvad` on **AVA-Speech** and **LibriVAD**.

The goal here isn't to claim `fast_vad` is the best VAD. It isn't, at least not on every metric. The point is that it's competitive enough to be a real option, and it's much faster than both baselines.

## Processing Speed

Benchmarks were run on a Ryzen 7700X (8 cores, 16 threads). Your results will vary. Since `fast_vad` uses `rayon` for parallelism, the offline path scales with core count. On higher core machines you should see even larger gaps over the baselines.

![Processing speed](assets/processing-speed.png)

- `fast_vad` offline: **721x** faster than Silero ONNX
- `fast_vad` streaming: **129x** faster than Silero ONNX
- `webrtcvad`: **63x** faster than Silero ONNX
- `silero_vad` (ONNX): **1x** baseline

Even the streaming path is about 2x faster than WebRTC VAD. The offline path is in a different class entirely.

## Real-world Audio: AVA-Speech

![AVA-Speech comparison](assets/ava-speech.png)

AVA-Speech is where `fast_vad` doesn't win on raw F1:

- `silero_vad`: F1 = 0.738, precision = 0.803, recall = 0.712
- `fast_vad`: F1 = 0.680, precision = 0.659, recall = 0.785
- `webrtcvad`: F1 = 0.669, precision = 0.619, recall = 0.825

`fast_vad` sits between the two baselines on the precision/recall tradeoff — less trigger-happy than WebRTC, less conservative than Silero. Not a clear win, but not an embarrassment either.

## Cleaner Speech: LibriVAD

![LibriVAD comparison](assets/librivad.png)

On cleaner speech the picture is better. At **-5 dB SNR**, `fast_vad` and `webrtcvad` are nearly identical (~0.928 F1) while `silero_vad` drops to ~0.878. From 0 to 20 dB, `silero_vad` pulls ahead on F1, but `fast_vad` stays close (0.93-0.94) with very stable precision (~0.908 across the whole range). `webrtcvad` matches on F1 but with much weaker precision.

## fast_vad Mode Tradeoffs

![fast_vad mode comparison](assets/fast-vad-modes.png)

- `permissive`: highest recall, strongest F1 across SNR values (~0.93-0.94)
- `normal`: best default balance, ~0.847 F1 at -5 dB
- `aggressive`: highest precision but poor recall in noise (~0.684 F1 at -5 dB due to recall of only 0.574)

The presets were tuned against LibriVAD-style evaluation, so they're biased toward read speech. For anything that differs from that, it's worth tuning with `with_config()` directly rather than assuming the defaults are optimal for your data.

## Takeaways

Pick `fast_vad` when throughput matters or you want more control over the precision/recall tradeoff than `webrtcvad` offers. Pick `silero_vad` if you need the best possible F1 on harder real-world audio and the runtime cost is acceptable. Either way, benchmark on your own data before deciding.
