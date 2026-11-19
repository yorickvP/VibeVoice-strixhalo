# VibeVoice-ASR-Streaming

**VibeVoice-ASR-Streaming** transcribes while the audio is still arriving, instead of waiting for it to end. It emits text once per audio chunk, so a transcript appears as the speaker talks.

**Non-streaming:** [VibeVoice-ASR](./vibevoice-asr.md)<br>

## Installation
Dependencies are managed with uv; `pyproject.toml` pins the ROCm 7.13 PyTorch build for gfx1151 (Strix Halo).
```bash
uv sync
```
Triton needs a C compiler (`cc`/`gcc`) at runtime.
Install ffmpeg as well, which the demo uses to decode audio.

## Usage

### Inference from files
```bash
python demo/vibevoice_asr_streaming_inference_from_file.py --model_path [add the checkpoint path here] --audio_files [add an audio path here]
```

Each chunk is printed as soon as the model emits it. Add `--context_info "Microsoft,VibeVoice"` to bias recognition toward specific terms, the same way hotwords work on the non-streaming model.

The chunk and lookahead are read from the checkpoint's `preprocessor_config.json`, so a checkpoint always runs at the chunk it was trained on.

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
