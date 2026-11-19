"""Minimal static-shape Qwen2 decoder for VibeVoice-ASR, gpt-fast style.

Takes the weights out of the HF Qwen2Model, optionally quantizes linears to
int8 weight-only (per output channel), and runs greedy decoding with a
preallocated KV cache and a single captured HIP/CUDA graph per decode step.

Supports batched decoding: `batch` slots share each weight read. Slots are
refilled from a queue as sequences finish (simple continuous batching), and the
attention kernel only reads each row's KV up to its own position.
"""
import torch
import torch.nn.functional as F
import torch._inductor.config

from fast_kernels import decode_attention, w8_linear

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.fx_graph_cache = True


class Linear:
    def __init__(self, ws, bs=None, quant=None):
        w = torch.cat([x.detach() for x in ws], 0)
        self.b = torch.cat([x.detach() for x in bs], 0) if bs else None
        self.quant = quant
        if quant == "int8":
            wf = w.float()
            scale = wf.abs().amax(1).clamp(min=1e-8) / 127.0
            self.w = torch.round(wf / scale[:, None]).clamp(-127, 127).to(torch.int8)
            self.s = scale.to(w.dtype)
        else:
            self.w = w.contiguous()
            self.s = None

    def __call__(self, x):
        if self.quant == "int8":
            M = x.shape[-2]
            if M == 1:
                # M=1: write it as a reduction so inductor fuses the int8 load (fastest for a GEMV)
                y = (x.unsqueeze(-2) * self.w.to(x.dtype)).sum(-1) * self.s
            elif M <= 64:
                # small batch: Triton kernel streams the int8 weights once for all rows
                return w8_linear(x.contiguous(), self.w, self.s, self.b)
            else:
                y = F.linear(x, self.w.to(x.dtype)) * self.s
        else:
            y = F.linear(x, self.w)
        return y if self.b is None else y + self.b


def rmsnorm(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat((-x2, x1), -1)


def is_looping(toks, window, max_period=64):
    """If the last `window` tokens are periodic with period <= max_period, return that
    period, else 0."""
    if len(toks) < window:
        return 0
    tail = toks[-window:]
    return next((p for p in range(1, max_period + 1) if tail[p:] == tail[:-p]), 0)


def loop_start(toks, period):
    """Index where the periodic run at the end of toks begins."""
    i = len(toks) - period
    while i > 0 and toks[i - 1] == toks[i - 1 + period]:
        i -= 1
    return i


class FastDecoder:
    def __init__(self, hf_lm, lm_head, max_len, quant="int8", batch=1):
        cfg = hf_lm.config
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = cfg.hidden_size // self.nh
        self.eps = cfg.rms_norm_eps
        self.max_len = max_len
        self.B = batch
        dev, dt = lm_head.weight.device, lm_head.weight.dtype
        self.embed = hf_lm.embed_tokens.weight.detach()
        self.layers = []
        for l in hf_lm.layers:
            a, m = l.self_attn, l.mlp
            self.layers.append(dict(
                ln1=l.input_layernorm.weight.detach(),
                ln2=l.post_attention_layernorm.weight.detach(),
                qkv=Linear([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight],
                           [a.q_proj.bias, a.k_proj.bias, a.v_proj.bias], quant),
                o=Linear([a.o_proj.weight], quant=quant),
                gu=Linear([m.gate_proj.weight, m.up_proj.weight], quant=quant),
                down=Linear([m.down_proj.weight], quant=quant),
            ))
        self.norm = hf_lm.norm.weight.detach()
        self.head = Linear([lm_head.weight], quant=quant)
        self.inter = hf_lm.layers[0].mlp.gate_proj.out_features

        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.hd, 2, device=dev).float() / self.hd))
        freqs = torch.outer(torch.arange(max_len, device=dev).float(), inv_freq)
        emb = torch.cat((freqs, freqs), -1)
        self.cos, self.sin = emb.cos().to(dt), emb.sin().to(dt)

        self.kc = [torch.zeros(batch, self.nkv, max_len, self.hd, device=dev, dtype=dt) for _ in self.layers]
        self.vc = [torch.zeros_like(k) for k in self.kc]
        self.bidx = torch.arange(batch, device=dev)[:, None]
        self.hidx = torch.arange(self.nkv, device=dev)[None, :]
        self.graph = None

    def _mlp(self, L, h):
        gu = L["gu"](rmsnorm(h, L["ln2"], self.eps))
        g, u = gu.split(self.inter, -1)
        return h + L["down"](F.silu(g) * u)

    @torch.no_grad()
    def prefill(self, embeds, slot=0):
        """embeds: [T, D] -> fills KV slot `slot`, returns argmax token of the last position ([1])."""
        T = embeds.shape[0]
        cos, sin = self.cos[:T], self.sin[:T]
        h = embeds
        for i, L in enumerate(self.layers):
            qkv = L["qkv"](rmsnorm(h, L["ln1"], self.eps))
            q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], -1)
            q = q.view(T, self.nh, self.hd).transpose(0, 1)
            k = k.view(T, self.nkv, self.hd).transpose(0, 1)
            v = v.view(T, self.nkv, self.hd).transpose(0, 1)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            self.kc[i][slot, :, :T] = k
            self.vc[i][slot, :, :T] = v
            o = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True, enable_gqa=True)[0]
            h = h + L["o"](o.transpose(0, 1).reshape(T, -1))
            h = self._mlp(L, h)
        h = rmsnorm(h[-1:], self.norm, self.eps)
        return self.head(h).argmax(-1)

    def _step(self, tok, pos, temp, noise):
        """tok, pos: [B] int64; temp: [B] f32 (0 = greedy); noise: [B, V] uniform(0,1)
        -> next tok [B]. Static shapes only."""
        B = self.B
        h = self.embed[tok]                                    # [B, D]
        cos, sin = self.cos[pos][:, None], self.sin[pos][:, None]  # [B, 1, hd]
        for i, L in enumerate(self.layers):
            qkv = L["qkv"](rmsnorm(h, L["ln1"], self.eps))
            q, k, v = qkv.split([self.nh * self.hd, self.nkv * self.hd, self.nkv * self.hd], -1)
            q = q.view(B, self.nh, self.hd)
            k = k.view(B, self.nkv, self.hd)
            v = v.view(B, self.nkv, self.hd)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
            idx = (self.bidx, self.hidx, pos[:, None])
            self.kc[i].index_put_(idx, k)
            self.vc[i].index_put_(idx, v)
            o = decode_attention(q.contiguous(), self.kc[i], self.vc[i], pos)  # [B, nh, hd]
            h = h + L["o"](o.reshape(B, -1))
            h = self._mlp(L, h)
        h = rmsnorm(h, self.norm, self.eps)
        logits = self.head(h).float()
        # Gumbel-max: argmax(logits/T + g) == argmax(logits + T*g); T=0 reduces to greedy
        gumbel = -torch.log(-torch.log(noise.clamp(1e-10, 1 - 1e-7)))
        return (logits + temp[:, None] * gumbel).argmax(-1)

    @torch.no_grad()
    def capture(self, compile=True):
        dev = self.embed.device
        self.s_tok = torch.zeros(self.B, dtype=torch.long, device=dev)
        self.s_pos = torch.zeros(self.B, dtype=torch.long, device=dev)
        self.s_inc = torch.zeros(self.B, dtype=torch.long, device=dev)  # 1 for active rows, 0 for idle
        self.s_temp = torch.zeros(self.B, dtype=torch.float32, device=dev)
        self.s_noise = torch.empty(self.B, self.head.w.shape[0], dtype=torch.float32, device=dev)
        fn = torch.compile(self._step, fullgraph=True, dynamic=False) if compile else self._step

        def body():
            self.s_noise.uniform_()
            nxt = fn(self.s_tok, self.s_pos, self.s_temp, self.s_noise)
            self.s_tok.copy_(nxt)
            # rows may overshoot their budget by < check_every steps; clamp keeps writes in bounds
            self.s_pos.add_(self.s_inc).clamp_(max=self.max_len - 1)

        # warmup at pos 0 with inc 0: writes slot 0 of each row, which prefill overwrites
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                body()
        torch.cuda.current_stream().wait_stream(st)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            body()

    @torch.no_grad()
    def generate_many(self, jobs, eos_id, check_every=32, callback=None, on_start=None,
                      loop_window=384, boundary_ids=(), retry_temps=(0.2, 0.3, 0.4),
                      sample_segments=2, sample_max=512):
        """jobs: list of (embeds [T, D], max_new). Runs up to `batch` at a time, refilling
        slots as rows finish. callback(job_idx, tokens_so_far: list[int]) is called after
        prefill and every check_every steps; on_start(job_idx, slot) just before a job's prefill.

        Loop recovery: when a row's last `loop_window` tokens repeat with a short period
        (greedy decoding stuck), it is rolled back to the last token in `boundary_ids`
        (end of a complete segment) before the loop and resumes sampling at
        retry_temps[k] until `sample_segments` more segments are done (or `sample_max`
        tokens), then goes back to greedy. Rollback only resets the row's position, the
        KV past it is simply overwritten. After len(retry_temps) failed retries at the
        same spot the row is truncated there and its index added to self.looped.
        self.recoveries[j] counts rollbacks per job. loop_window=0 disables all of this.

        Returns a list of token lists, in job order."""
        if self.graph is None:
            self.capture()
        boundary_ids = set(boundary_ids)
        self.looped = []
        self.recoveries = [0] * len(jobs)
        results = [None] * len(jobs)
        queue = list(range(len(jobs)))
        slots = [None] * self.B   # per slot: dict(j, toks, max_new, T, retry, last_rb, sample_from)

        def finish(b, toks):
            j = slots[b]["j"]
            results[j] = toks
            if callback:
                callback(j, toks)
            slots[b] = None
            self.s_inc[b] = 0
            self.s_pos[b] = 0
            self.s_temp[b] = 0

        def fill(b):
            while queue:
                j = queue.pop(0)
                emb, max_new = jobs[j]
                T = emb.shape[0]
                max_new = min(max_new, self.max_len - T)
                assert max_new > 0, f"job {j}: prompt of {T} tokens doesn't fit max_len {self.max_len}"
                if on_start:
                    on_start(j, b)
                first = self.prefill(emb, b)
                slots[b] = dict(j=j, toks=[int(first[0])], max_new=max_new, T=T,
                                retry=0, last_rb=-1, sample_from=None)
                if callback:
                    callback(j, slots[b]["toks"])
                if slots[b]["toks"][-1] == eos_id or max_new <= 1:
                    finish(b, slots[b]["toks"])
                    continue
                self.s_tok[b] = first[0]
                self.s_pos[b] = T
                self.s_inc[b] = 1
                self.s_temp[b] = 0
                return

        def recover(b, period):
            """Roll row b back to before its loop. Returns False if out of retries."""
            s, toks = slots[b], slots[b]["toks"]
            start = loop_start(toks, period)
            r = next((k + 1 for k in range(start - 1, -1, -1) if toks[k] in boundary_ids), 1)
            s["retry"] = s["retry"] + 1 if r <= s["last_rb"] else 1
            s["last_rb"] = r
            del toks[r:]
            if s["retry"] > len(retry_temps):
                return False
            self.recoveries[s["j"]] += 1
            s["sample_from"] = r
            # toks[k] sits at position T + k; re-feed the last kept token at its own position
            self.s_tok[b] = toks[-1]
            self.s_pos[b] = s["T"] + r - 1
            self.s_temp[b] = retry_temps[s["retry"] - 1]
            return True

        for b in range(self.B):
            fill(b)
        buf = torch.empty(self.B, check_every, dtype=torch.long, device=self.embed.device)
        while any(s is not None for s in slots):
            for j in range(check_every):
                self.graph.replay()
                buf[:, j].copy_(self.s_tok)
            chunk = buf.tolist()
            for b, s in enumerate(slots):
                if s is None:
                    continue
                toks = s["toks"]
                done = False
                for t in chunk[b]:
                    toks.append(t)
                    if t == eos_id or len(toks) >= s["max_new"]:
                        done = True
                        break
                period = 0 if done or not loop_window else is_looping(toks, loop_window)
                if period and not recover(b, period):
                    self.looped.append(s["j"])
                    done = True
                if not done and s["sample_from"] is not None:
                    new = toks[s["sample_from"]:]
                    if len(new) >= sample_max or sum(t in boundary_ids for t in new) >= sample_segments:
                        s["sample_from"] = None
                        self.s_temp[b] = 0
                if done:
                    finish(b, toks)
                    fill(b)
                elif callback:
                    callback(s["j"], toks)
        return results

    @torch.no_grad()
    def generate(self, embeds, eos_id, max_new, check_every=32, callback=None):
        """Single-sequence convenience wrapper; returns a 1-D token tensor."""
        cb = (lambda j, toks: callback(torch.tensor(toks))) if callback else None
        toks = self.generate_many([(embeds, max_new)], eos_id, check_every, cb)[0]
        return torch.tensor(toks, dtype=torch.long)
