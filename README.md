# fast-vad

Fast voice activity detection (VAD) in Rust with Python bindings.

- 8-band log-filterbank features with SIMD-accelerated FFT
- Logistic regression classifier with noise floor tracking
- Batch and streaming APIs
- Supports 8000 and 16000 Hz audio

## Build from source

### Python (with uv)

Requires [uv](https://docs.astral.sh/uv/) and a Rust toolchain.

```bash
git clone https://github.com/youruser/fast-vad
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
