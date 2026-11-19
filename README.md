<div align="center">

## 🎙️ VibeVoice-ASR (trimmed fork)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-VibeVoice--ASR-orange?logo=huggingface)](https://huggingface.co/microsoft/VibeVoice-ASR)
[![ASR Report](https://img.shields.io/badge/ASR-Report-yellow?logo=arxiv)](https://arxiv.org/pdf/2601.18184)

</div>


<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="Figures/VibeVoice_logo_white.png">
  <img src="Figures/VibeVoice_logo.png" alt="VibeVoice Logo" width="300">
</picture>
</div>

## Overview

This is a trimmed fork of [microsoft/VibeVoice](https://github.com/microsoft/VibeVoice) that keeps only the
speech-recognition (ASR) side, set up for an AMD Strix Halo (gfx1151 / Radeon 8060S) GPU with ROCm.
The TTS, realtime TTS, vLLM, Gradio/FastAPI demos and finetuning code from upstream have been removed.

**VibeVoice-ASR** is a unified speech-to-text model that handles up to **60 minutes** of audio in a single pass,
producing structured transcriptions with **Who (Speaker), When (Timestamps), and What (Content)**, with support for
**customized hotwords** and over 50 languages. See [docs/vibevoice-asr.md](docs/vibevoice-asr.md) for details,
and [docs/vibevoice-asr-streaming.md](docs/vibevoice-asr-streaming.md) for the streaming variant.

| Model | Weights |
|-------|---------|
| VibeVoice-ASR-7B | [HF Link](https://huggingface.co/microsoft/VibeVoice-ASR) |
| VibeVoice-ASR-Streaming | [HF Link](https://huggingface.co/microsoft/VibeVoice-ASR-Streaming-7B) |

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml` pins the ROCm 7.13 build of
PyTorch for gfx1151 from AMD's wheel index (`https://repo.amd.com/rocm/whl/gfx1151/`), and only resolves for
Linux x86_64. Python 3.10–3.14 all work.

```bash
uv sync            # or: uv sync --python 3.14
```

System dependencies:
- a C compiler (`cc`/`gcc`): Triton compiles a small HIP helper at runtime
- `ffmpeg`: optional, used to decode audio formats other than WAV/FLAC (falls back to `soundfile`)

## Usage

### Fast inference: `fast_asr.py`

Uses the stock VibeVoice speech encoders, but replaces HF `generate()` with a static-shape, graph-captured decoder
(`fast_decode.py` + Triton kernels in `fast_kernels.py`), with int8 weight-only quantization by default.

```bash
uv run python fast_asr.py meeting.wav \
  --context "Meeting between Alice and Bob, discussing Kubernetes and Terraform" \
  --json out.json
```

See `fast_asr.py --help` for all options (quantization, batching, max tokens, ...).
The first run compiles the decoder for each KV-cache size bucket (cached afterwards).

The model handles at most ~60 minutes per pass. For longer recordings, split into overlapping chunks
(e.g. 45 min with 5 min overlap), offset each chunk's timestamps, and cut at the overlap midpoints.
Speaker IDs are assigned per chunk.

### Reference inference (HF `generate()`)

```bash
uv run python demo/vibevoice_asr_inference_from_file.py --model_path microsoft/VibeVoice-ASR --audio_files [audio path]
```

### Streaming ASR

```bash
uv run python demo/vibevoice_asr_streaming_inference_from_file.py --model_path [checkpoint] --audio_files [audio path]
```

## Evaluation

<p align="center">
  <img src="Figures/DER.jpg" alt="DER" width="50%"><br>
  <img src="Figures/cpWER.jpg" alt="cpWER" width="50%"><br>
  <img src="Figures/tcpWER.jpg" alt="tcpWER" width="50%">
</p>

## ⚠️ Risks and Limitations

VibeVoice inherits any biases, errors, or omissions produced by its base model (Qwen2.5). Transcripts may be
unexpected, biased, or inaccurate; check content accuracy before relying on it.
Upstream does not recommend using VibeVoice in commercial or real-world applications without further testing and
development; it is intended for research and development purposes.

## License

MIT, see [LICENSE](LICENSE). Contribution guidelines: [CONTRIBUTING.md](CONTRIBUTING.md).
