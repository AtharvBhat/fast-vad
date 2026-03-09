# fast-vad

Extremely fast voice activity detection in Rust with Python bindings and streaming mode support.

Supports 16 kHz and 8 kHz audio. Fixed frame width of 32 ms (512 samples at 16 kHz and 256 samples at 8 kHz).

If you are interested in benchmark comparisons, see [docs/README.md](docs/README.md).

## Architecture

`fast_vad` is a small fixed-frame DSP pipeline with a hardcoded lightweight classifier.

```text
audio
  -> 32 ms frames
     - 16 kHz: 512 samples
     - 8 kHz: 256 samples
  -> Hann window
  -> real FFT
  -> 8 log-energy bands
  -> feature engineering
     - raw bands          (8)
     - noise-normalized   (8)
     - first deltas       (8)
     - second deltas      (8)
     = 32 total features
  -> hardcoded logistic regression
  -> threshold + smoothing
  -> speech / silence labels
```

At a glance:

- `VAD` (offline / batch) splits audio into 32 ms frames and uses `rayon` to process complete frames in parallel while extracting the 8-band features.
- `VadStateful` (streaming) runs the same per-frame pipeline one frame at a time and reuses scratch buffers instead of paying thread-pool overhead.
- The detector keeps a running 8-band noise floor, then derives 32 total features from each frame: raw band energies, noise-normalized energies, first-order deltas, and second-order deltas.
- Classification is a tiny hardcoded logistic-regression-style model with fixed weights and bias compiled into the crate.
- The final decision is shaped by simple temporal rules: thresholding, minimum speech length, minimum silence length, and hangover.
- Hot loops are SIMD-accelerated with the `wide` crate for windowing, spectral power computation, band-energy math, and detector feature math.

```text
frame features (8 bands)
    | raw
    | raw - noise_floor
    | delta
    | delta2
    v
32 engineered features
    v
linear score + bias
    v
speech / silence
```

## Build from source

### Python (with uv)

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain.

```bash
git clone https://github.com/AtharvBhat/fast-vad
cd fast-vad
uv venv
uv pip install maturin
uv run maturin develop --release
```

The package is then importable inside the virtual environment.

### Rust

```bash
cargo build --release
```

Add as a dependency in another crate:

```toml
[dependencies]
fast-vad = { path = "/path/to/fast-vad" }
```

## Python usage

Config is set at construction time. `VAD()` and `VadStateful()` default to Normal
mode; use `with_mode` or `with_config` to customise.

```python
import numpy as np
import soundfile as sf
import fast_vad

audio, sr = sf.read("audio.wav", dtype="float32")
assert sr in (8000, 16000)

# Default (Normal mode)
vad = fast_vad.VAD(sr)

# Explicit mode
vad = fast_vad.VAD.with_mode(sr, fast_vad.mode.aggressive)

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

```python
fe = fast_vad.FeatureExtractor(sr)
features = fe.extract_features(audio)  # shape: (num_frames, 8)
```

### Modes

| Constant                   | Description                                   |
|----------------------------|-----------------------------------------------|
| `fast_vad.mode.permissive` | Low false-negative rate; more speech accepted |
| `fast_vad.mode.normal`     | Balanced, general-purpose                     |
| `fast_vad.mode.aggressive` | Low false-positive rate; stricter             |

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

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
