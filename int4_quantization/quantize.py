from dataclasses import dataclass

import torch
import triton

from .kernels import compute_scale_kernel, quantize_pack_int4_kernel


def quantize_pack(x: torch.Tensor, group_size=128) -> (torch.Tensor, torch.Tensor):
    assert x.dtype == torch.float16 and x.dim() == 2
    M, N = x.shape
    assert N % 2 == 0 and N % group_size == 0

    G = N // group_size
    scales = torch.empty((M, G), dtype=torch.float16, device=x.device)
    packed = torch.empty((M, N // 2), dtype=torch.int8, device=x.device)

    BLOCK_M, BLOCK_G = 16, 4
    grid_s = (triton.cdiv(M, BLOCK_M), triton.cdiv(G, BLOCK_G))
    compute_scale_kernel[grid_s](
        x, scales, M, N, x.stride(0), x.stride(1),
        GROUP_SIZE=group_size, BLOCK_M=BLOCK_M, BLOCK_G=BLOCK_G,
    )

    BLOCK_M, BLOCK_N = 32, 64
    grid_q = (triton.cdiv(M, BLOCK_M), triton.cdiv(N // 2, BLOCK_N))
    quantize_pack_int4_kernel[grid_q](
        x, packed, scales, M, N,
        x.stride(0), x.stride(1),
        packed.stride(0), packed.stride(1),
        scales.stride(0), scales.stride(1),
        GROUP_SIZE=group_size, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return packed, scales


@dataclass
class QuantizedWeight:
    # int8  [K//2, N]
    q: torch.Tensor
    # bf16  [K//GROUP_SIZE, N]      
    scales: torch.Tensor
    # bf16  [K//GROUP_SIZE, N]
    zeros: torch.Tensor
    group_size: int = 128
    orig_dtype: torch.dtype = torch.bfloat16

    @property
    def shape(self):  # (K, N)
        return (self.q.shape[0] * 2, self.q.shape[1])

    @property
    def dtype(self):
        return self.orig_dtype


def quantize_weight(W: torch.Tensor, group_size: int = 128) -> QuantizedWeight:
    assert W.dim() == 2
    K, N = W.shape
    assert K % group_size == 0 and K % 2 == 0

    Wt_fp16 = W.detach().T.contiguous().to(torch.float16)   # [N, K]
    packed_t, scales_t = quantize_pack(Wt_fp16, group_size=group_size)

    q = packed_t.T.contiguous()
    scales = scales_t.T.contiguous().to(torch.bfloat16)
    zeros = torch.zeros_like(scales, dtype=torch.bfloat16)

    return QuantizedWeight(
        q=q, scales=scales, zeros=zeros,
        group_size=group_size, orig_dtype=torch.bfloat16,
    )