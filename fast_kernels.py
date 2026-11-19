"""Triton kernels for the static-shape decoder in fast_decode.py.

- decode_attention: flash-decoding over a preallocated KV cache. Reads each row's
  position from device memory and skips KV blocks past it, so a captured graph
  costs O(pos) instead of O(max_len), and rows in a batch can be at different
  positions.
- w8_linear: int8 weight-only matmul for small M (batched decode). Dequantizes in
  registers so the weights are streamed once regardless of M.
"""
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# decode attention
# ---------------------------------------------------------------------------

ATTN_SPLIT = 512    # KV positions per program
ATTN_BLOCK = 64     # KV positions per inner iteration


@triton.jit
def _attn_split_kernel(q_ptr, k_ptr, v_ptr, pos_ptr, o_ptr, m_ptr, l_ptr,
                       L,
                       NKV: tl.constexpr, G: tl.constexpr, G_PAD: tl.constexpr, HD: tl.constexpr,
                       SPLIT: tl.constexpr, BLOCK: tl.constexpr, NSPLIT: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    pos = tl.load(pos_ptr + b)
    start = sp * SPLIT
    rg = tl.arange(0, G_PAD)
    rd = tl.arange(0, HD)
    part = ((b * NKV + h) * NSPLIT + sp)
    if start > pos:
        # nothing to attend in this split
        tl.store(m_ptr + part * G_PAD + rg, tl.full([G_PAD], float("-inf"), tl.float32))
        tl.store(l_ptr + part * G_PAD + rg, tl.zeros([G_PAD], tl.float32))
        return
    # q: [B, NKV*G, HD] -> rows h*G .. h*G+G-1
    q = tl.load(q_ptr + (b * NKV * G + h * G + rg[:, None]) * HD + rd[None, :],
                mask=rg[:, None] < G, other=0.0)
    kv_base = (b * NKV + h) * L * HD
    m_i = tl.full([G_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([G_PAD], tl.float32)
    acc = tl.zeros([G_PAD, HD], tl.float32)
    end = tl.minimum(start + SPLIT, pos + 1)
    for n in range(start, end, BLOCK):
        rn = n + tl.arange(0, BLOCK)
        valid = rn < end
        k = tl.load(k_ptr + kv_base + rn[:, None] * HD + rd[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k))  # q is pre-scaled by 1/sqrt(hd)
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + kv_base + rn[:, None] * HD + rd[None, :], mask=valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(o_ptr + (part * G_PAD + rg[:, None]) * HD + rd[None, :], acc)
    tl.store(m_ptr + part * G_PAD + rg, m_i)
    tl.store(l_ptr + part * G_PAD + rg, l_i)


def decode_attention(q, kc, vc, pos):
    """q: [B, nh, hd]; kc/vc: [B, nkv, L, hd]; pos: [B] int64 (last valid index, inclusive).
    Returns [B, nh, hd] in q.dtype."""
    B, nh, hd = q.shape
    _, nkv, L, _ = kc.shape
    G = nh // nkv
    G_PAD = max(16, triton.next_power_of_2(G))
    nsplit = triton.cdiv(L, ATTN_SPLIT)
    o = torch.empty(B, nkv, nsplit, G_PAD, hd, device=q.device, dtype=torch.float32)
    m = torch.empty(B, nkv, nsplit, G_PAD, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    _attn_split_kernel[(B, nkv, nsplit)](
        (q * hd ** -0.5).contiguous(), kc, vc, pos, o, m, l, L,
        NKV=nkv, G=G, G_PAD=G_PAD, HD=hd, SPLIT=ATTN_SPLIT, BLOCK=ATTN_BLOCK, NSPLIT=nsplit,
        num_warps=4)
    # combine splits (tiny tensors; inductor fuses this)
    o, m, l = o[:, :, :, :G], m[:, :, :, :G], l[:, :, :, :G]
    mx = m.amax(2, keepdim=True)
    w = torch.exp(m - mx)                                   # empty splits -> 0
    out = (o * w[..., None]).sum(2) / (l * w).sum(2)[..., None]
    return out.reshape(B, nh, hd).to(q.dtype)


# ---------------------------------------------------------------------------
# int8 weight-only linear, small M
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=nw, num_stages=ns)
             for bn in (32, 64, 128) for bk in (128, 256) for nw in (4, 8) for ns in (1, 2)],
    key=["N", "K", "BLOCK_M"],
)
@triton.jit
def _w8_kernel(x_ptr, w_ptr, s_ptr, b_ptr, y_ptr, M, N, K,
               HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rm = tl.arange(0, BLOCK_M)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    x_ptrs = x_ptr + rm[:, None] * K + rk[None, :]
    w_ptrs = w_ptr + rn[None, :] * K + rk[:, None]
    for _ in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=rm[:, None] < M, other=0.0)
        w = tl.load(w_ptrs)
        acc += tl.dot(x, w.to(x.dtype))
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    acc = acc * tl.load(s_ptr + rn).to(tl.float32)[None, :]
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn).to(tl.float32)[None, :]
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(y_ptr.dtype.element_ty), mask=rm[:, None] < M)


def w8_linear(x, w, s, b=None):
    """x: [M, K] bf16 (M <= 64), w: [N, K] int8, s: [N], b: [N] or None -> [M, N]."""
    M, K = x.shape
    N = w.shape[0]
    assert K % 256 == 0 and N % 128 == 0, (N, K)
    BLOCK_M = max(16, triton.next_power_of_2(M))
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (N // meta["BLOCK_N"],)
    _w8_kernel[grid](x, w, s, b if b is not None else s, y, M, N, K,
                     HAS_BIAS=b is not None, BLOCK_M=BLOCK_M)
    return y
