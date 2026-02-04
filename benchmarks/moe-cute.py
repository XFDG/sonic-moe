import argparse
import random
import time
from typing import Tuple, Type

import cutlass
import torch
import torch.nn.functional as F
from rich import print as print0
from triton.testing import do_bench # 用于精确测量 GPU 时间的工具

from sonicmoe import MoE
from sonicmoe.enums import ActivationType, is_glu
from sonicmoe.functional import moe_TC_softmax_topk_layer

# --- 【可学习点】这些函数定义了 MoE 专家内部使用的各种激活函数 ---
def swiglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return u * F.silu(g)

# ... (其他激活函数省略，逻辑一致)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SonicMoE 性能研究脚本")

    # --- 【可修改项 1】核心维度参数 ---
    # 默认值: T=32768 (Tokens), H=4096 (Hidden), I=1024 (Inter), E=128 (Experts), K=8 (Top-k)
    parser.add_argument(
        "--thiek",
        type=lambda s: tuple([int(x.strip()) for x in s.split(",")]),
        #default=(32768, 4096, 1024, 128, 8),
        default=(32768*2, 4096, 1024, 128, 8), #A
        #default=(32768, 4096, 1024, 128, 8), #B
        help="修改这 5 个值可以测试不同负载下的 H100 表现",
    )

    # --- 【可修改项 2】激活函数选择 ---
    parser.add_argument(
        "--activation", choices=["swiglu", "geglu", "relu", "silu"], default="swiglu"
    )

    # --- 【可修改项 3】是否跳过精度校验 ---
    # 如果你只想看速度，开启此项可以节省大约 1 分钟的 CPU 比对时间
    parser.add_argument("--skip_test", action="store_true", default=False)

    return parser.parse_args()

def run(thiek, activation, skip_test):
    T, H, I, E, K = thiek
    print(f"🚀 开始测试: Tokens={T}, 专家总数={E}, 每次激活专家={K}, 激活函数={activation}")

    # --- 初始化 SonicMoE 层 ---
    # 这里会申请显存并初始化权重
    moe = MoE(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        intermediate_size=I,
        activation_function=ActivationType(activation),
    ).to(dtype=torch.bfloat16).cuda() # H100 推荐使用 bfloat16

    x = 0.2 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    dout = 0.2 * torch.randn_like(x)

    # --- 【核心逻辑】精度校验 (Reference Check) ---
    if not skip_test:
        print("正在进行数学精度校验 (与标准 PyTorch 结果对比)...")
        # 调用 SonicMoE 的加速内核
        o, _, _ = moe_TC_softmax_topk_layer(x, router_w, w1.permute(1, 2, 0), None, w2.permute(1, 2, 0), None, K, 0, ActivationType(activation))
        # ... (此处省略比对逻辑，成功则打印 PASS)

    # --- 【性能测量】计算量 (FLOPs) 统计 ---
    # 对于 SwiGLU，计算量公式为 6 * T * I * H * K
    flops_fwd = 6 * T * I * H * K 
    
    # --- 【可修改项 4】测试循环次数 ---
    repeats = 500 # 增加次数可以获得更稳健的平均值
    warmup = 10   # 预热次数，确保 GPU 频率稳定

    # 测量推理性能 (Inference)
    fwd_timing = do_bench(lambda: moe(x)[0], warmup=warmup, rep=repeats)
    tflops = flops_fwd / (fwd_timing * 1e9)
    print0(f"[bold green]✅ 推理性能: {fwd_timing:.3f} ms, TFLOPS: {tflops:.1f}[/bold green]")

if __name__ == "__main__":
    args = parse_arguments()
    run(args.thiek, args.activation, args.skip_test)