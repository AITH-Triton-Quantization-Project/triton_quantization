import torch
import triton
import triton.language as tl

from int4_quantization.kernels import QuantizedWeight

# общие константы тайлинга
_BLOCK_M, _BLOCK_N, _BLOCK_K, _GROUP_M = 64, 64, 64, 8


@triton.jit
def _w4_gemm_kernel(
    x_ptr, wq_ptr, s_ptr, z_ptr, out_ptr,
    M, N, K,
    sxm, sxk, swqk, swqn, ssg, ssn, szg, szn, som, son,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    BLOCK_KH: tl.constexpr = BLOCK_K // 2
    offs_kh = tl.arange(0, BLOCK_KH)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):

        a = tl.load(
            x_ptr + offs_m[:, None] * sxm + (k0 + offs_k)[None, :] * sxk,
            mask=(offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K),
            other=0.0,
        )
        a = a.to(tl.bfloat16)   # страховка, если X пришёл fp16

        wq = tl.load(
            wq_ptr + (k0 // 2 + offs_kh)[:, None] * swqk + offs_n[None, :] * swqn,
            mask=((k0 // 2 + offs_kh)[:, None] < (K // 2)) & (offs_n[None, :] < N),
            other=0,
        )
        w_lo = (wq & 0x0F).to(tl.int32)
        w_hi = ((wq >> 4) & 0x0F).to(tl.int32)

        g = k0 // GROUP_SIZE
        s = tl.load(s_ptr + g * ssg + offs_n * ssn,
                    mask=offs_n < N, other=1.0).to(tl.bfloat16)
        z = tl.load(z_ptr + g * szg + offs_n * szn,
                    mask=offs_n < N, other=0.0).to(tl.bfloat16)

        w_lo_bf = (w_lo.to(tl.bfloat16) - z[None, :]) * s[None, :]
        w_hi_bf = (w_hi.to(tl.bfloat16) - z[None, :]) * s[None, :]

        w_bf = tl.reshape(tl.join(w_lo_bf, w_hi_bf), (BLOCK_K, BLOCK_N))
        acc += tl.dot(a, w_bf, out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _w16_gemm_kernel():
    pass


def _grid(M, N):
    return (triton.cdiv(M, _BLOCK_M) * triton.cdiv(N, _BLOCK_N),)

def _launch_w16_gemm(X, W, out):
    M, K = X.shape; _, N = W.shape
    _w16_gemm_kernel[_grid(M, N)](
        X, W, out, M, N, K,
        X.stride(0), X.stride(1), W.stride(0), W.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K, GROUP_M=_GROUP_M,
    )

def _launch_w4_gemm(X, W: QuantizedWeight, out):
    M, K = X.shape
    Kh, N = W.q.shape
    assert Kh * 2 == K

    _w4_gemm_kernel[_grid(M, N)](
        X, W.q, W.scales, W.zeros, out, M, N, K,
        X.stride(0), X.stride(1),
        W.q.stride(0),      W.q.stride(1),
        W.scales.stride(0), W.scales.stride(1),
        W.zeros.stride(0),  W.zeros.stride(1),
        out.stride(0), out.stride(1),
        GROUP_SIZE=W.group_size,
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K, GROUP_M=_GROUP_M,
    )


def matmul(X: torch.Tensor, W, kind: str, out_dtype=torch.bfloat16):
    assert X.dim() == 2 and kind in ("w16", "w4")
    M, K = X.shape
    K_w, N = W.shape
    assert K == K_w, f"K mismatch: X={K}, W={K_w}"

    out = torch.empty((M, N), dtype=out_dtype, device=X.device)

    if kind == "w16":
        W_bf = W if W.dtype == torch.bfloat16 else W.to(torch.bfloat16)
        _launch_w16_gemm(X, W_bf, out)
    else:
        assert isinstance(W, QuantizedWeight)
        _launch_w4_gemm(X, W, out)
    return out
