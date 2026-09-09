# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import math

import numpy as np
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)

# hipblaslt-bench-style benchmark methodology. Set from argparse in __main__.
#   INIT         -> hipblaslt-bench `initialization` (trig_float)
#   ROTATING_MIB -> hipblaslt-bench `rotating` (MiB of rotating operand buffers)
#   FLUSH        -> hipblaslt-bench `flush` (GPU cache flush before each dispatch)
#   COLD/HOT_ITERS -> hipblaslt-bench `cold_iters` / `iters`
INIT = "trig_float"
ROTATING_MIB = 512
FLUSH = True
COLD_ITERS = 20
HOT_ITERS = 100


def _trig_float(shape, dtype, kind):
    """trig_float initialization, matching hipblaslt-bench: A = sin(idx),
    B = cos(idx) over the flattened linear index (deterministic, bounded)."""
    n = 1
    for s in shape:
        n *= s
    idx = torch.arange(n, dtype=torch.float32, device="cuda")
    v = torch.sin(idx) if kind == "sin" else torch.cos(idx)
    return v.reshape(shape).to(dtype)


def _make_flush_buffer():
    """Cold buffer sized above the L2 so writing it evicts the GEMM operands
    from cache (mirrors hipblaslt-bench `flush: true`)."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    l2 = int(getattr(props, "L2_cache_size", 0) or 0)
    nbytes = max(l2 * 2, 256 * 1024 * 1024)
    return torch.zeros(nbytes // 4, dtype=torch.int32, device="cuda")


def _rotating_sets(base_args, rotating_mib):
    """Build a rotating pool of operand sets. Total extra allocation is capped
    at ~rotating_mib (hipblaslt-bench `rotating`); each timed dispatch consumes a
    different set so its inputs are cache-cold."""
    tensors = [t for t in base_args if isinstance(t, torch.Tensor)]
    per_iter = sum(t.nbytes for t in tensors) or 1
    rot_bytes = int(rotating_mib) * 1024 * 1024
    n_sets = max(1, math.ceil(rot_bytes / per_iter)) if rot_bytes > 0 else 1
    sets = [tuple(base_args)]
    for _ in range(n_sets - 1):
        sets.append(
            tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in base_args)
        )
    return sets


def _bench_cold(call, base_args, *, rotating_mib, flush, cold_iters, hot_iters):
    """Time `call(args)` the hipblaslt-bench way: rotating operand buffers, an
    optional GPU cache flush before every timed dispatch, and per-dispatch CUDA
    event timing (use_gpu_timer). Returns (last_output, avg_us)."""
    sets = _rotating_sets(base_args, rotating_mib)
    n = len(sets)
    flush_buf = _make_flush_buffer() if flush else None

    for i in range(cold_iters):
        call(sets[i % n])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    out = None
    for i in range(hot_iters):
        s = sets[i % n]
        if flush_buf is not None:
            flush_buf.zero_()
        start.record()
        out = call(s)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return out, float(np.mean(times)) * 1000.0  # ms -> us


@perftest(num_iters=5)
def run_torch(x, w, x_scales, w_scales, dtype):
    m, _k = x.shape
    n, _k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, out):
    return aiter.gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out)


@perftest()
def run_triton(x, w, x_scales, w_scales, out, dtype=dtypes.bf16):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, out)
    return out


@perftest()
def run_gemm_asm(
    x,
    weightshuffle,
    x_scale,
    w_scale,
    out,
    kernelName="",
    bias=None,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    log2_k_split=None,
):
    # if log2_k_split is not None and log2_k_split > 0:
    #     out_reset = torch.zeros(
    #         (out.shape[0] + 31) // 32 * 32, out.shape[1], dtype=dtype
    #     )
    #     out = out_reset

    aiter.gemm_a4w4_asm(
        x,
        weightshuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=log2_k_split,
    )
    return out


@benchmark()
def test_gemm(dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx

    if get_gfx() not in ["gfx950"]:
        return
    ret = {}
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    if INIT == "trig_float":
        x = _trig_float((M, K), dtype, "sin")
        w = _trig_float((N, K), dtype, "cos")
    else:
        x = torch.randn((M, K), dtype=dtype)
        w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)
    a, _avg_a = run_torch(x, w, x_scales, w_scales, dtype)
    # out1 = torch.empty(M, N, dtype=dtype)
    # b, avg_b = run_triton(x, w.T, x_scales, w_scales, out1, dtype)
    # b, avg_b = a, 0
    # err_b = checkAllclose(a, b, msg="triton        ")

    # hipblaslt-bench-style timing: rotating operand buffers + GPU cache flush +
    # CUDA-event timer. Inputs are passed as a tuple that _bench_cold rotates.
    base_args = (x, wshuffle, x_scales_shuffle, w_scales_shuffle)

    def call(args):
        return aiter.gemm_a4w4(*args, bpreshuffle=True)

    c, us = _bench_cold(
        call,
        base_args,
        rotating_mib=ROTATING_MIB,
        flush=FLUSH,
        cold_iters=COLD_ITERS,
        hot_iters=HOT_ITERS,
    )
    err = checkAllclose(a, c, msg="unified api", catastrophic_check=True)
    ret["init"] = INIT
    ret["rotating_MiB"] = ROTATING_MIB
    ret["flush"] = FLUSH
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = err

    # kernelName = "" # "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"
    # log2_k_split = 1
    # out2 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    # d, us = run_gemm_asm(
    #     x,
    #     wshuffle,
    #     x_scales_shuffle,
    #     w_scales_shuffle,
    #     out2,
    #     kernelName,
    #     bias_f32,
    #     bpreshuffle=True,
    #     log2_k_split=log2_k_split,
    # )
    # err = checkAllclose(a, d[:M], msg=f"asm {kernelName} log2_k_split_{log2_k_split}")
    # tag = "asm_dbg"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    # out3 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    # e, us = run_gemm_ck(x, wshuffle, x_scales_shuffle, w_scales_shuffle, out3)
    # err = checkAllclose(a, e[:M], msg="ck            ")
    # tag = "ck"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    choices=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-mnk",
    "--shape",
    type=dtypes.str2tuple,
    nargs="*",
    default=[


# aiter::f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256	32768	6144	4096	552.4	2986	QKV fwd	TN (X·Wᵀ)
# aiter::f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256	32768	4096	4096	289.0	3805	out_proj fwd	TN (X·Wᵀ)
# aiter::f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256	32768	28672	4096	1767.5	4355	FC1 fwd	TN (X·Wᵀ)
# aiter::f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256	32768	4096	14336	712.6	5400	FC2 fwd	TN (X·Wᵀ)
        (32768, 6144, 4096),
        (32768, 4096, 4096),
        (32768, 28672, 4096),
        (32768, 4096, 14336),

        # pure_compute
        # (256, 2048, 8192),
        # (2048, 8192, 8192),
        # (16384, 16384, 16384),
        # (32768, 106496, 16384),
        # (32768, 16384, 53248),
        # (32768, 18432, 16384),
        # (32768, 16384, 16384),
        # (128, 106496, 16384),
        # (128, 16384, 53248),
        # (128, 18432, 16384),
        # (128, 16384, 16384),
        # (64, 106496, 16384),
        # (64, 16384, 53248),
        # (64, 18432, 16384),
        # (64, 16384, 16384),
        # (64, 106496, 16384),
        # (32, 106496, 16384),
        # (32, 16384, 53248),
        # (32, 18432, 16384),
        # (32, 16384, 16384),
        # # qkv_proj
        # (1, 1280, 8192),
        # (64, 1280, 8192),
        # (127, 1280, 8192),
        # (129, 1280, 8192),
        # (65, 1280, 8192),
        # (32, 1280, 8192),
        # (64, 1280, 8192),
        # (128, 1280, 8192),
        # (192, 1280, 8192),
        # (256, 1280, 8192),
        # (320, 1280, 8192),
        # (512, 1280, 8192),
        # (1024, 1280, 8192),
        # (2048, 1280, 8192),
        # (4096, 1280, 8192),
        # (8192, 1280, 8192),
        # # attn_out
        # (1, 8192, 1024),
        # (32, 8192, 1024),
        # (64, 8192, 1024),
        # (128, 8192, 1024),
        # (192, 8192, 1024),
        # (256, 8192, 1024),
        # (320, 8192, 1024),
        # (512, 8192, 1024),
        # (1024, 8192, 1024),
        # (2048, 8192, 1024),
        # (4096, 8192, 1024),
        # (8192, 8192, 1024),
        # (16384, 8192, 1024),
        # # tune
        # (1552, 8192, 8192),
        # (1664, 8192, 8192),
        # (1792, 8192, 8192),
        # (1920, 8192, 8192),
        # (3072, 8192, 8192),
        # (1552, 10240, 8192),
        # (1664, 10240, 8192),
        # (1792, 10240, 8192),
        # (1920, 10240, 8192),
        # (3072, 10240, 8192),
        # (1552, 57344, 8192),
        # (1664, 57344, 8192),
        # (1792, 57344, 8192),
        # (1920, 57344, 8192),
        # (3072, 57344, 8192),
        # (1552, 8192, 28672),
        # (1664, 8192, 28672),
        # (1792, 8192, 28672),
        # (1920, 8192, 28672),
        # (3072, 8192, 28672),
    ],
    help="""Shape of mnk.
    e.g. -mnk 1280,8192,1024""",
)
parser.add_argument(
    "--init",
    choices=["trig_float", "randn"],
    default="trig_float",
    help="operand initialization (default: trig_float, like hipblaslt-bench).\n"
    "  trig_float = A=sin(idx), B=cos(idx)\n"
    "  randn      = N(0,1) gaussian",
)
parser.add_argument(
    "--rotating",
    type=int,
    default=512,
    help="rotating operand buffer size in MiB (hipblaslt-bench `rotating`); "
    "0 disables rotation (default: 512)",
)
parser.add_argument(
    "--flush",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="flush GPU caches before each timed dispatch (hipblaslt-bench "
    "`flush`); use --no-flush to disable (default: on)",
)
parser.add_argument(
    "--cold-iters",
    dest="cold_iters",
    type=int,
    default=20,
    help="untimed warmup dispatches (hipblaslt-bench `cold_iters`, default: 20)",
)
parser.add_argument(
    "--iters",
    dest="iters",
    type=int,
    default=100,
    help="timed dispatches (hipblaslt-bench `iters`, default: 100)",
)

args = parser.parse_args()

INIT = args.init
ROTATING_MIB = args.rotating
FLUSH = args.flush
COLD_ITERS = args.cold_iters
HOT_ITERS = args.iters

df = []
for dtype in args.dtype:
    for m, n, k in args.shape:
        ret = test_gemm(dtype, m, n, k)
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("gemm_a4w4 summary (markdown):\n%s", df_md)
