# fast-vad

Extremely fast voice activity detection in Rust with Python bindings and streaming mode support. Significantly faster than WebRTC VAD and orders of magnitude faster than Silero ONNX. See [benchmark comparisons](docs/README.md).

Supports 16 kHz and 8 kHz sample rates.

## Architecture

Audio is split into non-overlapping 32 ms frames (512 samples at 16 kHz, 256 at 8 kHz), Hann-windowed, FFT'd, and collapsed into 8 log-energy bands covering roughly 94-4000 Hz.

Per frame, the detector builds 32 features: 8 raw log-energies, 8 noise-normalised values (raw minus a running noise floor), and their first and second order deltas. A logistic regression model with weights compiled into the crate scores these features and compares the result to a mode-specific threshold. The noise floor is a per-band exponential moving average that only updates on silence frames, so it adapts to background noise without being contaminated by speech.

Raw frame labels are then post-processed: short speech bursts below `min_speech_ms` are dropped, short silence gaps below `min_silence_ms` are filled, and voiced regions are extended by `hangover_ms` to avoid clipping word endings.

`VAD` processes all frames in parallel with `rayon`. `VadStateful` processes one frame at a time with reused FFT scratch buffers for low-latency streaming. Hot loops are SIMD-accelerated via the `wide` crate.

## Install

### Python

```bash
pip install fast-vad
```

Or with `uv`:

```bash
uv add fast-vad
```

### Rust

```bash
cargo add fast-vad
```

## Build from source

### Python

Requires a Rust toolchain and [maturin](https://github.com/PyO3/maturin).

```bash
git clone https://github.com/AtharvBhat/fast-vad
cd fast-vad
maturin develop --release
```

### Rust

```bash
cargo build --release
```

## Python usage

Fast vad comes with a few modes.`VAD()` and `VadStateful()` default to `fast_vad.mode.normal` for offline and streaming mode respectively. To customize parameters use `with_mode` or `with_config` for even finer control.

```python
import numpy as np
import soundfile as sf
import fast_vad

audio, sr = sf.read("audio.wav", dtype="float32")
assert sr in (8000, 16000)

# Default (Normal mode)
vad = fast_vad.VAD(sr)

# Explicit mode
vad = fast_vad.VAD.with_mode(sr, fast_vad.mode.aggressive) # choose permissive, normal or aggressive 

# Custom parameters
vad = fast_vad.VAD.with_config(
    sr,
    threshold_probability=0.7,
    min_speech_ms=100,
    min_silence_ms=300,
    hangover_ms=100,
)

# Per-sample labels
labels = vad.detect(audio)

# Per-frame labels
frame_labels = vad.detect_frames(audio)

# Speech segments as a (N, 2) uint64 numpy array of [start, end] sample indices
segments = vad.detect_segments(audio)
for start, end in segments:
    print(f"speech: {start/sr:.2f}s – {end/sr:.2f}s")
```

### Streaming

```python
# Default (Normal mode)
vad = fast_vad.VadStateful(sr)

# Explicit mode
vad = fast_vad.VadStateful.with_mode(sr, fast_vad.mode.normal)

# Custom parameters
vad = fast_vad.VadStateful.with_config(sr, 0.7, 100, 300, 100)

frame_size = vad.frame_size  # 512 at 16 kHz, 256 at 8 kHz

for i in range(0, len(audio) - frame_size + 1, frame_size):
    is_speech = vad.detect_frame(audio[i : i + frame_size])
    print(f"frame {i // frame_size}: {'speech' if is_speech else 'silence'}")

vad.reset_state()  # reuse for another stream
```

### Feature extraction

You can also use fast vad as a feature extractor.

```python
fe = fast_vad.FeatureExtractor(sr)

# 8 log-energy band features per frame
features = fe.extract_features(audio)  # shape: (num_frames, 8)

# 24-dimensional features per frame: raw bands + first- and second-order deltas
features = fe.feature_engineer(audio)  # shape: (num_frames, 24)
```

### Modes

| Constant                   | Description                                   |
|----------------------------|-----------------------------------------------|
| `fast_vad.mode.permissive` | Low false-negative rate; more speech accepted |
| `fast_vad.mode.normal`     | Balanced, general-purpose                     |
| `fast_vad.mode.aggressive` | Low false-positive rate; stricter             |

The built-in modes were tuned against LibriVAD, so they work best on read speech. For other domains (phone calls, meetings, noisy environments, etc.) you'll likely get better results tuning `with_config()` against your own data.

## Rust usage

Config is set at construction. `VAD::new` and `VadStateful::new` default to Normal
mode; use `with_mode` or `with_config` to customise.

```rust
use fast_vad::vad::detector::{VAD, VADModes, VadConfig};

fn main() -> Result<(), fast_vad::VadError> {
    let audio = vec![0.0f32; 16000]; // 1 second of silence

    // Default (Normal mode)
    let vad = VAD::new(16000)?;

    // Explicit mode
    let vad = VAD::with_mode(16000, VADModes::Aggressive)?;

    // Custom parameters
    let vad = VAD::with_config(16000, VadConfig {
        threshold_probability: 0.7,
        min_speech_ms: 100,
        min_silence_ms: 300,
        hangover_ms: 100,
    })?;

    let labels = vad.detect(&audio);           // one bool per sample
    let frame_labels = vad.detect_frames(&audio); // one bool per frame
    let segments = vad.detect_segments(&audio);   // Vec<[start, end]>

    Ok(())
}
```

### Streaming

```rust
use fast_vad::vad::detector::{VadStateful, VADModes, VadConfig};

fn main() -> Result<(), fast_vad::VadError> {
    let audio = vec![0.0f32; 16000];

    // Default (Normal mode)
    let mut vad = VadStateful::new(16000)?;

    // Explicit mode
    let mut vad = VadStateful::with_mode(16000, VADModes::Normal)?;

    // Custom parameters
    let mut vad = VadStateful::with_config(16000, VadConfig {
        threshold_probability: 0.7,
        min_speech_ms: 100,
        min_silence_ms: 300,
        hangover_ms: 100,
    })?;

    let frame_size = vad.frame_size();
    for frame in audio.chunks_exact(frame_size) {
        let is_speech = vad.detect_frame(frame)?;
        println!("{is_speech}");
    }

    vad.reset_state(); // reuse for another stream
    Ok(())
}
```

## Benchmarking

```bash
cargo bench --manifest-path bench_rs/Cargo.toml
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
