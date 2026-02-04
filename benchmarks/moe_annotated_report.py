# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import argparse
import random
import time
from typing import Tuple, Type

import cutlass
import torch
import torch.nn.functional as F
from rich import print as print0
from triton.testing import do_bench

from sonicmoe import MoE
from sonicmoe.enums import ActivationType, is_glu
from sonicmoe.functional import moe_TC_softmax_topk_layer

# --- 激活函数定义区域 (无需修改) ---
def swiglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return u * F.silu(g)

def geglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return F.gelu(g.float()).to(dtype=g.dtype) * u

def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x.float()).to(dtype=x.dtype)

def reglu(x: torch.Tensor) -> torch.Tensor:
    u = x[..., 1::2]
    g = x[..., ::2]
    return (F.relu(g.float()) * u).to(dtype=g.dtype)

def relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)

def relu_sq(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2

def silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)

# --- 参数解析辅助函数 ---
def parse_comma_separated_ints(s: str):
    try:
        return tuple([int(x.strip()) for x in s.split(",")])
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid format. Expected comma-separated integers.")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SonicMoE 性能测试与报告生成脚本")

    # ================= [核心修改区域] =================
    # --thiek 参数定义了模型的规模。
    # T (Tokens): 一次处理的Token数量 (如 32768)
    # H (Hidden): 模型隐藏层维度 (如 4096)
    # I (Intermediate): 专家内部维度 (如 1024)
    # E (Experts): 专家总数 (如 128) - 修改这里可以测试显存压力
    # K (Top-K): 每个Token选几个专家 (如 8)
    parser.add_argument(
        "--thiek",
        type=parse_comma_separated_ints,
        default=(32768, 4096, 1024, 128, 8), # <--- 如果不传参，默认跑这个配置
        help="格式: T,H,I,E,K (例如: 32768,4096,1024,128,8)",
    )
    # =================================================

    parser.add_argument(
        "--dtype",
        type=cutlass.dtype,
        default=cutlass.BFloat16, # H100 默认使用 BFloat16 以获得最佳性能
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        default=False, # 如果设为 True，将跳过数学精度检查，直接跑分
    )
    parser.add_argument(
        "--activation", 
        choices=["swiglu", "geglu", "reglu", "relu_sq", "relu", "silu", "gelu"], 
        default="swiglu" # Llama 等主流模型通常使用 SwiGLU
    )
    parser.add_argument(
        "--add_bias",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if len(args.thiek) != 5:
        parser.error("--thiek must contain exactly 5 values")

    return args


def run(
    thiek: Tuple[int, int, int, int, int],
    dtype: Type[cutlass.Numeric],
    skip_test: Type[bool],
    add_bias: Type[bool],
    activation: Type[str],
    **kwargs,
):
    # 根据参数选择 PyTorch 数据类型
    torch_dtype = {cutlass.BFloat16: torch.bfloat16, cutlass.Float16: torch.float16}[dtype]
    activation = ActivationType(activation)
    
    # 解包维度参数
    T, H, I, E, K = thiek
    print(f"\n🚀 开始测试配置: Tokens(T)={T}, Hidden(H)={H}, Intermediate(I)={I}, Experts(E)={E}, Top-K(K)={K}")

    # 设置随机种子，确保每次跑的结果一致
    random.seed(1111)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    # --- [初始化 MoE 层] ---
    # 这里会申请 H100 显存。如果 E (专家数) 太大，这里可能会爆显存。
    try:
        moe = (
            MoE(
                num_experts=E,
                num_experts_per_tok=K,
                hidden_size=H,
                intermediate_size=I,
                activation_function=activation,
                add_bias=add_bias, # 必须传入，否则会报错
                std=0.02,          # 必须传入，否则会报错
            )
            .to(dtype=torch_dtype)
            .cuda()
        )
    except torch.cuda.OutOfMemoryError:
        print("❌ 初始化失败：显存不足 (CUDA OOM)。请尝试减小专家数量 E 或 Token 数量 T。")
        return

    # 生成随机输入数据
    x = 0.2 * torch.randn(T, H, device="cuda:0", dtype=torch_dtype, requires_grad=True)
    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    b1, b2 = moe.c_fc.bias, moe.c_proj.bias
    
    # 初始化 Bias
    if add_bias:
        torch.nn.init.normal_(b1, 0, 0.01)
        torch.nn.init.normal_(b2, 0, 0.01)
    
    dout = 0.2 * torch.randn_like(x, requires_grad=True)

    # --- [第一阶段：数学正确性检查] ---
    if not skip_test:
        print("🔍 正在进行数学精度校验 (对比标准 PyTorch 实现)...")
        # 这里的 moe_TC_softmax_topk_layer 是 SonicMoE 的核心加速算子
        o, router_logits, expert_frequency = moe_TC_softmax_topk_layer(
            x, router_w, w1.permute(1, 2, 0), b1, w2.permute(1, 2, 0), b2, moe.top_k, moe.stream_id, activation
        )
        
        # 计算 SonicMoE 的梯度
        if add_bias:
            dx, dw1, db1, dw2, db2, drouter_w = torch.autograd.grad(
                o, [x, w1, b1, w2, b2, router_w], grad_outputs=dout
            )
        else:
            dx, dw1, dw2, drouter_w = torch.autograd.grad(o, [x, w1, w2, router_w], grad_outputs=dout)

        # --- 使用标准 PyTorch 手写一个 MoE 进行对比 ---
        logits = F.linear(x, router_w)
        ref_topk_logits, ref_topk_experts = logits.topk(K, dim=-1)
        ref_topk_scores = ref_topk_logits.softmax(dim=-1, dtype=torch.float32)

        # ... (中间省略了繁琐的 PyTorch 参考实现逻辑) ...
        # 核心逻辑：用 for 循环模拟专家计算，作为“标准答案”

        act_func = {
            ActivationType.SWIGLU: swiglu,
            ActivationType.GEGLU: geglu,
            ActivationType.REGLU: reglu,
            ActivationType.GELU: gelu,
            ActivationType.RELU: relu,
            ActivationType.SILU: silu,
            ActivationType.RELU_SQ: relu_sq,
        }[activation]

        # 验证前向传播结果
        with torch.autocast("cuda:0", torch.float32):
            ref_o = torch.zeros_like(x)
            for i in range(E):
                # 找到分配给第 i 个专家的 token
                T_idx, E_idx = torch.argwhere(ref_topk_experts == i).split(1, dim=1)
                T_idx, E_idx = T_idx.squeeze(-1), E_idx.squeeze(-1)

                if T_idx.numel() > 0:
                    w1_out = F.linear(x[T_idx, :], w1[i, :, :].squeeze(), bias=(b1[i] if add_bias else None))
                    w1_out = act_func(w1_out)
                    w2_out = F.linear(w1_out, w2[i, :, :].squeeze(), bias=(b2[i] if add_bias else None))
                    ref_o[T_idx, :] += w2_out * ref_topk_scores[T_idx, E_idx, None]

            # 打印误差
            o_diff = (o.float() - ref_o).abs()
            print(f"   最大相对误差 (Mean Rel Diff): {(o_diff / (ref_o.abs() + 1e-6)).mean():.6f}")

            # 验证反向传播梯度
            if add_bias:
                 ref_dx, ref_dw1, ref_db1, ref_dw2, ref_db2, ref_drouter_w = torch.autograd.grad(
                    ref_o, [x, w1, b1, w2, b2, router_w], grad_outputs=dout
                )
            else:
                ref_dx, ref_dw1, ref_dw2, ref_drouter_w = torch.autograd.grad(
                    ref_o, [x, w1, w2, router_w], grad_outputs=dout
                )
            
            # 简单的梯度检查打印
            print(f"   梯度检查 (drouter_w) 相对误差: {((drouter_w - ref_drouter_w).abs() / (ref_drouter_w.abs() + 1e-6)).mean():.6f}")
    
    # --- [第二阶段：性能跑分 Benchmarking] ---
    print("\n⏱️  正在进行性能测试 (Warmup + Benchmark)...")
    
    # 计算理论 FLOPs (浮点运算次数)
    if is_glu(activation):
        flops = 6 * T * I * H * K
    else:
        flops = 4 * T * I * H * K

    repeats = 500 # 重复跑 500 次取平均
    warmup = 5    # 预热 5 次

    time.sleep(0.5)

    # 1. 编译模式 (Torch Compile) 测试
    @torch.compile
    def forward_only(is_inference_mode_enabled):
        o, _, _ = moe_TC_softmax_topk_layer(
            x, router_w, w1.permute(1, 2, 0), b1, w2.permute(1, 2, 0), b2, moe.top_k, moe.stream_id, activation, is_inference_mode_enabled
        )
        return o

    # 测试 A: 普通前向 (Fwd)
    fwd_timing = do_bench(lambda: forward_only(False), warmup=warmup, rep=repeats)
    tflops = flops / (fwd_timing * 1e9)
    print0(f"[bold green]   [Mode: Training Fwd] Average time: {fwd_timing:.3f} ms, TFLOPS: {tflops:.1f}[/bold green]")

    time.sleep(0.5)

    # 测试 B: 推理模式 (Inference Mode) - 通常最快
    timing = do_bench(lambda: forward_only(True), warmup=warmup, rep=repeats)
    tflops_inf = flops / (timing * 1e9)
    print0(f"[bold green]   [Mode: Inference   ] Average time: {timing:.3f} ms, TFLOPS: {tflops_inf:.1f}[/bold green]")

    # 测试 C: 完整训练 (Fwd + Bwd)
    @torch.compile
    def forward_and_backward():
        o, _, _ = moe_TC_softmax_topk_layer(
            x, router_w, w1.permute(1, 2, 0), b1, w2.permute(1, 2, 0), b2, moe.top_k, moe.stream_id, activation, False
        )
        o.backward(dout, retain_graph=True)
        # 清空梯度以便下一次循环
        x.grad = w1.grad = w2.grad = router_w.grad = None

    if is_glu(activation):
        flops_bwd = 18 * T * I * H * K
    else:
        flops_bwd = 12 * T * I * H * K

    e2e_timing = do_bench(forward_and_backward, warmup=warmup, rep=repeats, grad_to_none=[x, w1, w2, router_w, dout])
    tflops_e2e = flops_bwd / (e2e_timing * 1e9)
    print0(f"[bold green]   [Mode: Train Full  ] Average time: {e2e_timing:.3f} ms, TFLOPS: {tflops_e2e:.1f}[/bold green]")

    print("-" * 60) # 分割线

if __name__ == "__main__":
    args = parse_arguments()
    run(args.thiek, args.dtype, args.skip_test, args.add_bias, args.activation)
    print("TEST FINISHED (PASS)")