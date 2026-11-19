#!/usr/bin/env python
"""Fast VibeVoice-ASR inference for ROCm (tested on Strix Halo / gfx1151).

Uses the stock VibeVoice speech encoders, but replaces HF generate() with the
static-shape graphed decoder in fast_decode.py (optionally int8 weight-only).
Multiple files are decoded together (--batch) so each weight read is shared.
"""
import os

# AOTriton flash attention is gated behind this on gfx11xx; without it SDPA falls back to math
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import argparse
import json
import re
import time

import torch
from tqdm import tqdm

from fast_decode import FastDecoder
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor

BUCKET = 4096  # KV cache length is rounded up to this; each distinct size compiles once
END_RE = re.compile(r'"End"\s*:\s*([0-9.]+)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio_files", nargs="+")
    ap.add_argument("--model_path", default="microsoft/VibeVoice-ASR")
    ap.add_argument("--quant", choices=["int8", "bf16"], default="int8")
    ap.add_argument("--batch", type=int, default=4,
                    help="max files decoded concurrently (capped at the number of files)")
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="default: ~15 tokens per second of audio + 512")
    ap.add_argument("--context", default=None, help="hotwords / extra info passed to the prompt")
    ap.add_argument("--json", help="write results to this JSON file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_loop_recovery", action="store_true",
                    help="don't roll back and resample when greedy decoding gets stuck repeating")
    args = ap.parse_args()

    t = time.time()
    proc = VibeVoiceASRProcessor.from_pretrained(args.model_path, language_model_pretrained_name="Qwen/Qwen2.5-7B")
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    lm = model.model.language_model
    eos = proc.tokenizer.eos_token_id
    print(f"model loaded in {time.time() - t:.1f}s")

    # 1. encode all audio up front
    jobs, durations = [], []
    for path in args.audio_files:
        t0 = time.time()
        inputs = proc(audio=[path], sampling_rate=None, return_tensors="pt", padding=True,
                      add_generation_prompt=True, context_info=args.context)
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        duration = inputs["speech_tensors"].shape[1] / 24000
        max_new = args.max_new_tokens or int(duration * 15) + 512
        with torch.no_grad():
            torch.manual_seed(args.seed)
            feats = model.encode_speech(inputs["speech_tensors"], inputs.get("speech_masks"))
            emb = model.get_input_embeddings()(inputs["input_ids"])[0].clone()
            emb[inputs["acoustic_input_mask"][0]] = feats.to(emb.dtype)
        del inputs, feats
        torch.cuda.synchronize()
        print(f"encoded {path} ({duration:.1f}s audio, {emb.shape[0]} prompt tokens) in {time.time() - t0:.1f}s")
        jobs.append((emb, max_new))
        durations.append(duration)

    # 2. one decoder sized for the longest job
    B = max(1, min(args.batch, len(jobs)))
    max_len = -(-max(e.shape[0] + m for e, m in jobs) // BUCKET) * BUCKET
    print(f"building {args.quant} decoder: batch {B}, KV length {max_len} (compiles on first use)...")
    dec = FastDecoder(lm, model.lm_head, max_len, quant=None if args.quant == "bf16" else args.quant, batch=B)
    tc = time.time()
    dec.capture()
    print(f"compiled in {time.time() - tc:.1f}s")

    # 3. decode
    bars, starts = {}, {}

    def on_start(j, slot):
        starts[j] = time.time()
        bars[j] = tqdm(total=round(durations[j], 1), unit="s", position=slot, leave=False,
                       desc=os.path.basename(args.audio_files[j]), dynamic_ncols=True,
                       bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.1f}/{total:.1f}s audio "
                                  "[{elapsed}<{remaining}{postfix}]")

    def progress(j, toks):
        # the model emits {"Start":..,"End":..} segments in order; track the latest End
        bar = bars[j]
        ends = END_RE.findall(proc.decode(toks[-64:], skip_special_tokens=True))
        if ends:
            bar.n = min(float(ends[-1]), bar.total)
        bar.set_postfix_str(f"{len(toks)} tok, {len(toks) / max(time.time() - starts[j], 1e-3):.1f} tok/s",
                            refresh=False)
        bar.refresh()

    t1 = time.time()
    # tokens that close a segment and open the next one ('},{"', '"},{"'): safe rollback points
    tok = proc.tokenizer
    boundary_ids = [i for i in range(len(tok)) if "},{" in tok.decode([i])]
    outs = dec.generate_many(jobs, eos, callback=progress, on_start=on_start, boundary_ids=boundary_ids,
                             loop_window=0 if args.no_loop_recovery else 384)
    t2 = time.time()
    for bar in bars.values():
        bar.close()
    total = sum(len(o) for o in outs)
    print(f"\ndecoded {total} tokens for {len(jobs)} file(s) in {t2 - t1:.1f}s "
          f"({total / (t2 - t1):.1f} tok/s aggregate, batch {B})")

    results = []
    for j, (path, duration, ids) in enumerate(zip(args.audio_files, durations, outs)):
        text = proc.decode(ids, skip_special_tokens=True)
        looped = j in dec.looped
        if dec.recoveries[j]:
            print(f"\nnote: {path}: recovered from {dec.recoveries[j]} repetition loop(s) by resampling")
        if looped:
            ends = END_RE.findall(text)
            print(f"\nwarning: {path}: decoding stayed stuck in a repetition loop "
                  f"(after ~{ends[-1] if ends else '?'}s of audio) and was stopped; transcript is truncated")
        try:
            segments = proc.post_process_transcription(text)
        except Exception as e:
            print(f"warning: could not parse structured output: {e}")
            segments = []
        print(f"\n=== {path} ({duration:.1f}s audio) — {len(ids)} tok")
        if segments:
            for s in segments:
                print(f"[{s.get('start_time', '?')} - {s.get('end_time', '?')}] "
                      f"Speaker {s.get('speaker_id', '?')}: {s.get('text', '')}")
        else:
            print(text)
        results.append({"file": path, "raw_text": text, "segments": segments, "looped": looped,
                        "loop_recoveries": dec.recoveries[j]})

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
