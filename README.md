# AI-MAX-395-Qwen3.6

在 AMD Ryzen AI Max+ 395（Strix Halo）上使用 llama.cpp + Vulkan 部署 Qwen3.6 大语言模型，通过 SSH 反向隧道 + Nginx HTTPS 将推理 API 暴露到公网。

> 语音助手与监控播报子系统已停用，不在本仓库范围内。

---

## 性能基准

所有数据在推理机（AMD Ryzen AI Max+ 395，128 GB LPDDR5X，Radeon 8060S，llama.cpp b10068 Vulkan）上测量。基准测试使用统一脚本串行执行，每轮冷启动、不并发，F16 KV cache，UB=256。

### 35B-A3B MoE（别名 `358`）

MoE 模型，每个 token 仅激活 3B/35B 参数。配置：F16 KV + UB=256 + cache-ram=49152 + parallel=1 + MTP n=2 + reasoning-preserve=1。

| 测试点 | Prompt tokens | Prefill (t/s) | Gen (t/s) | MTP 接受率 | TTFT (s) | 总耗时 (s) |
|--------|-------------|--------------|-----------|-----------|---------|-----------|
| p128   | 159         | 251.3        | 64.6      | 76.3%     | 0.63    | 17.75     |
| p4K    | 7,683       | 666.3        | 63.0      | 77.5%     | 11.53   | 28.80     |
| p32K   | 36,517      | 523.4        | 58.5      | 81.0%     | 69.77   | 86.43     |
| p64K   | 69,310      | 452.1        | 50.4      | 76.5%     | 153.30  | 174.13    |
| p128K  | 132,432     | 354.0        | 43.7      | 81.5%     | 374.10  | 396.05    |
| p256K  | 250,190     | 240.2        | 32.7      | 79.8%     | 1041.37 | 1060.06   |

> 满载温度：GPU 峰值 70°C / CPU 峰值 70°C，远低于节流阈值 90°C。

### 27B Dense Q8（别名 `278`）

Dense 模型，每个 token 激活全部 27B 参数。配置：F16 KV + UB=256 + cache-ram=49152 + parallel=1 + MTP n=3 + reasoning-preserve=1。

| 测试点 | Prompt tokens | Prefill (t/s) | Gen (t/s) | MTP 接受率 | TTFT (s) | 总耗时 (s) |
|--------|-------------|--------------|-----------|-----------|---------|-----------|
| p128   | 159         | 142.0        | 14.7      | 72.7%     | 1.12    | 42.95     |
| p4K    | 7,683       | 245.7        | 14.5      | 73.4%     | 31.26   | 81.43     |
| p32K   | 36,517      | 192.3        | 13.0      | 69.6%     | 189.87  | 235.57    |
| p64K   | 69,310      | 108.5        | 12.4      | 72.3%     | 639.05  | 721.12    |
| p128K  | 132,432     | 47.1         | 10.8      | 72.0%     | 2808.91 | 2864.03   |
| p256K  | 241,822     | ~26.0¹       | —²       | —³        | —       | >8249（GPU 崩溃）|

> ¹ **p256K prefill 部分数据**（来自 llama-server `print_timing` 日志）：prefill 到 212,992/241,822 token（85%）时 Vulkan GPU 崩溃（`ErrorDeviceLost`），未进入生成阶段。瞬时 prefill 速度从 253 t/s（4K）衰减到 25.8 t/s（213K），累计平均 26.0 t/s。这是 Dense 27B attention O(n²) 在超长上下文的物理极限，叠加 Vulkan 驱动在超长 compute shader 链下的稳定性问题。
>
> ² 无生成数据（prefill 未完成）。
>
> ³ 无 MTP 数据（未进入生成阶段）。
>
> 满载温度：GPU 峰值 72°C / CPU 峰值 71°C，远低于节流阈值 90°C。

### 跨模型对比

| 测试点 | 358 Prefill (t/s) | 358 Gen (t/s) | 278 Prefill (t/s) | 278 Gen (t/s) | 358/278 Gen 倍率 |
|--------|-------------------|---------------|-------------------|---------------|------------------|
| p128   | 251.3             | 64.6          | 142.0             | 14.7          | 4.4×             |
| p4K    | 666.3             | 63.0          | 245.7             | 14.5          | 4.3×             |
| p32K   | 523.4             | 58.5          | 192.3             | 13.0          | 4.5×             |
| p64K   | 452.1             | 50.4          | 108.5             | 12.4          | 4.1×             |
| p128K  | 354.0             | 43.7          | 47.1              | 10.8          | 4.0×             |
| p256K  | 240.2             | 32.7          | —                 | —             | —                |

> 358（MoE 3B 激活）gen 速度约为 278（Dense 27B）的 **4-4.5 倍**，prefill 速度约为 **1.8-5.2 倍**。随上下文增长，两者 prefill 差距拉大（MoE 稀疏激活优势在长序列更明显）。278 在 p256K 因 Dense 架构 O(n²) attention 计算量过大无法在超时内完成。
>
> MTP 接受率：278（n=3）平均 72.0%，358（n=2）平均 78.8%。

---

## 部署架构

```
┌──────────────┐  HTTPS (443)  ┌────────────────┐
│   Client     │ ────────────▶ │  Cloud Nginx    │
│  (任意设备)   │               │  {your_server_ip} │
└──────────────┘               └────────┬────────┘
                                        │ proxy_pass 127.0.0.1:8080
┌──────────────┐  SSH 反向隧道           │
│ Inference Box│ ◀──────────────────────┘
│ Ubuntu 26.04 │  127.0.0.1:8080 ←→ :12345
│ AMD AI Max+395│
│ 128 GB RAM   │
└──────────────┘
     └─ llama-router :12345  (278/358, Vulkan GPU)
```

llama.cpp 仅绑定 `127.0.0.1:12345`，云服务器运行 Nginx（端口 443），SSH 反向隧道提供 NAT 穿透。参考 `docs/` 中 Nginx 配置模板，SSE 需关闭 gzip 和 buffering。llama-tunnel.service 通过 SSH 反向隧道暴露推理 API。hw-temp.service 每 60 秒记录硬件温度。

> ASR/TTS/AI Station 语音助手子系统已停用，不在本仓库范围内。

---

## 硬件与系统配置

### 硬件规格

| 组件 | 规格 |
|------|------|
| APU | AMD Ryzen AI Max+ 395（16C，SMT 已禁用） |
| 内存 | 128 GB LPDDR5X（统一内存） |
| iGPU | Radeon 8060S（RDNA 3.5，40 CU） |
| GTT | 120 GB（`amdgpu.gttsize=122880`） |
| 存储 | NVMe SSD |

### BIOS 配置

| 设置 | 值 |
|------|-----|
| SMT | **禁用** |
| iGPU Mem Bar | **ResizableBAR** |
| UMA Version | **Non-Legacy** |
| Dedicated Graphics Memory | **0.5G**（512 MB VRAM） |

### GRUB 内核参数

```bash
GRUB_CMDLINE_LINUX_DEFAULT="amd_iommu=off amdgpu.gttsize=122880 processor.max_cstate=1"
```

应用：`sudo update-grub && sudo reboot`

- `amd_iommu=off` — 禁用 IOMMU，减少内存映射开销
- `amdgpu.gttsize=122880` — GTT 穿透内存设为 120 GB，为 Vulkan 推理提供足够的 GPU 可见内存
- `processor.max_cstate=1` — 限制 CPU 最低 C-State 为 C1，避免深度节能导致推理延迟抖动

### 编译 llama.cpp

```bash
cd ~/llama.cpp && git pull origin master
cmake -B build-vulkan --fresh -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan -j$(nproc)
```

> 编译后二进制位于 `~/llama.cpp/build-vulkan/bin/`。环境变量 `GGML_VK_MAX_NODES_PER_SUBMIT=1`（防 APU GPU job timeout）。

---

## 模型清单

| 别名 | 文件 | 架构 | 大小 | 激活参数 | 角色 |
|------|------|------|------|----------|------|
| **278** | `Qwen3.6-27B-UD-Q8_K_XL.gguf` | Dense | ~33 GB | 27B | 主模型（OpenCode 默认，视觉，语音助手常驻） |
| **358** | `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` | MoE | ~37 GB | 3B | 备用模型（Hermes 默认，快速响应，按需加载） |

**mmproj：**

| 文件 | 用途 |
|------|------|
| `mmproj-Qwen3.6-27B-F16.gguf` | 278 视觉投影 |
| `mmproj-Qwen3.6-35B-A3B-F16.gguf` | 358 视觉投影 |

单模型模式（`--models-max 1`），278 和 358 均设 sleep-idle=1800s。Router 通过 LRU 自动切换。

### 客户端配置

| 客户端 | 默认模型 | max_output_tokens | preserve_thinking | 说明 |
|--------|----------|-------------------|------------------|------|
| Hermes | 358 | 81920 | true | 主对话 + auxiliary 全部用 358，enable_thinking=false |
| OpenCode | 278 | 81920 | true | 飞书编程助手，走 opencode-bridge@3.1.5 |

---

## router-preset.ini

**配置文件：** `~/model/router-preset.ini`

```ini
; Router preset - 所有模型由 Router 统一管理
; models-max=1: 同一时刻只加载一个模型，避免内存竞争
;
; cache-idle-slots: 全部开启（默认），空闲KV cache存入cache-ram释放显存
; timeout: HTTP 并发等待窗口

[Qwen3.6-27B-UD-Q8_K_XL]
n-gpu-layers = 99
flash-attn = auto
kv-unified = 1
parallel = 1
ctx-size = 262144
batch-size = 4096
ubatch-size = 256
spec-type = draft-mtp
spec-draft-n-max = 3
cache-ram = 49152
mmproj = /home/$USER/mmproj/mmproj-Qwen3.6-27B-F16.gguf
image-min-tokens = 2048
mlock = 1
numa = distribute
reasoning-budget = 32768
reasoning-preserve = 1
threads = 8
temp = 0.6
top-p = 0.95
top-k = 20
presence-penalty = 0.0
min-p = 0.0
slot-prompt-similarity = 0.8
sleep-idle-seconds = 1800
alias = 278
timeout = 3600

[Qwen3.6-35B-A3B-UD-Q8_K_XL]
n-gpu-layers = 99
flash-attn = auto
kv-unified = 1
parallel = 1
ctx-size = 262144
batch-size = 4096
ubatch-size = 256
spec-type = draft-mtp
spec-draft-n-max = 2
cache-ram = 49152
mmproj = /home/$USER/mmproj/mmproj-Qwen3.6-35B-A3B-F16.gguf
image-min-tokens = 2048
mlock = 1
numa = distribute
reasoning-budget = 32768
reasoning-preserve = 1
threads = 8
temp = 0.6
top-p = 0.95
top-k = 20
presence-penalty = 0.0
min-p = 0.0
slot-prompt-similarity = 0.8
sleep-idle-seconds = 1800
alias = 358
timeout = 3600
```

**关键参数：**

| 参数 | 278 | 358 | 说明 |
|------|-----|-----|------|
| parallel | 1 | 1 | 单并发，避免显存竞争 |
| cache-ram | 49152 | 49152 | MB 级 KV cache 穿透内存 |
| cache-type | F16 | F16 | KV cache 默认精度 |
| reasoning-preserve | 1 | 1 | 保留思考链（b10068 支持） |
| spec-draft-n-max | 3 | 2 | MTP draft token 数（27B 用 3，MoE 用 2） |
| sleep-idle-seconds | 1800 | 1800 | 空闲卸载时间（均为 30min） |
| reasoning-budget | 32768 | 32768 | 思考模式最大 token 数 |
| temp | 0.6 | 0.6 | 采样温度 |
| top-p | 0.95 | 0.95 | 核采样概率阈值 |
| top-k | 20 | 20 | 核采样候选数 |
| min-p | 0.0 | 0.0 | 最小概率阈值 |
| presence-penalty | 0.0 | 0.0 | 存在惩罚 |
| repetition-penalty | 1.0 | 1.0 | 重复惩罚（默认值，未显式设置） |

---

## llama-router.service

**服务文件：** `~/.config/systemd/user/llama-router.service`

```ini
[Unit]
Description=llama.cpp Router Server
After=network.target

[Service]
Type=simple
ExecStart=/home/$USER/scripts/llama-router.sh
LimitMEMLOCK=infinity
Restart=on-failure
RestartSec=10
TimeoutStartSec=300
TimeoutStopSec=35
KillMode=process

[Install]
WantedBy=default.target
```

> TimeoutStopSec=35 + KillMode=process：Router 需先卸载模型再退出，短超时 + process 级 kill 防止 systemd 误杀子进程导致显存泄漏。

## llama-router.sh

**启动脚本：** `~/scripts/llama-router.sh`

```bash
#!/bin/bash
# Qwen3.6 LLM Router Service (systemd compatible - foreground)
# Vulkan backend

LOGDIR="/home/$USER/logs/llama"
BINDIR="/home/$USER/llama.cpp/build-vulkan/bin"
ROUTER="$BINDIR/llama-server"

export LD_LIBRARY_PATH="$BINDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Vulkan: 限制每批提交节点数，避免 APU GPU job timeout (ErrorDeviceLost)
export GGML_VK_MAX_NODES_PER_SUBMIT=1

mkdir -p "$LOGDIR"

exec "$ROUTER" \
  --host 127.0.0.1 \
  --port 12345 \
  --api-key YOUR_API_KEY \
  -a Qwen3.6 \
  --models-dir /home/$USER/model \
  --models-max 1 \
  --models-preset /home/$USER/model/router-preset.ini \
  --metrics \
  >> "$LOGDIR/router.log" 2>&1
```

---

## 模型切换

客户端在 API 请求中指定 `model` 字段，别名和完整文件名均可：

```bash
curl https://your_domain/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model": "278", ...}'
```

> Router 收到 `model=358` 时，如果当前加载的是 278，会先卸载 278 再加载 358，切换耗时约 30–60 秒。

---

## 验证清单

- [ ] `llama-router.service` 活跃（`--models-max 1`）
- [ ] 外部 `curl https://your_domain/health` 返回 `OK`
- [ ] 别名路由 `curl -d '{"model":"278",...}'` 和 `{"model":"358",...}` 均可工作

**快速测试：**

```bash
curl http://127.0.0.1:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model":"278","messages":[{"role":"user","content":"Hello"}],"max_tokens":50,"stream":true}'
```

---

*推理机 · AMD Ryzen AI Max+ 395 · 128 GB LPDDR5X · Radeon 8060S · llama.cpp b10068 Vulkan*
*基准测试日期：2026-07-26 · F16 KV · UB=256 · Vulkan 后端*
