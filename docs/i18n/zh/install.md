---
title: "fak — 安装与上手（简体中文入口 / Simplified Chinese install guide)"
description: "fak 的简体中文安装入口页：从干净的检出到跑起内核，含分层（Tier）地图、go build / go install 命令，以及中国大陆的 GOPROXY 与 ModelScope 镜像提示；自托管、PIPL 友好、Apache-2.0。"
---

# fak — 安装与上手（简体中文入口）

> 这是一个**本地化入口页**，而非完整文档翻译。完整文档为英文——本页只带你从安装走到跑起内核，
> 更完整的产品介绍见 [README](./README.md)（中文入口）与[英文 README](https://github.com/anthony-chaudhary/fak/blob/main/README.md)。
> **说明：** 本页由机器生成、尚待母语者校对——如发现错误，欢迎提交 issue/PR。
> **其他语言入口：**[i18n hub](../README.md)。

本页是**安装并运行**的入口；一句话理解 fak、60 秒验证与完整定位，请看[中文 README 入口](./README.md)。

fak 是**一个静态 Go 二进制程序**，零外部依赖：它位于 AI 智能体与其所调用的工具之间，在每次
tool call *执行之前*进行审查，并在长会话中复用重复的共享工作。同一个智能体循环因此变得更安全、
更省、更快，且无需重写——你只需把一个 base URL 重新指向 `fak serve`，或用一条命令封装现有智能体：
`fak manage -- claude`。

> **成本与毛利（人民币视角）。** token 账单按上游计价侵蚀你的人民币毛利；在 50 轮 × 5 智能体的
> 会话中，fak 相比一套**已调优的 warm-cache 栈**约**少做 4.1×** 的工作。约 60×（约 19 小时→约
> 19 分钟）**仅在对比 naive 的“每轮重发全部”循环时成立**，绝非头条数字。复用增益为**自托管专属**，
> 适用于读密集型集群。

---

## 0. 先决条件

- **Go 1.26+。** `go.mod` 声明 `go 1.26`；在默认的 `GOTOOLCHAIN=auto` 下，较旧的 `go` 会在首次
  构建时自动下载正确的工具链（需联网一次）。用 `go version` 检查。
- **Tier 0 到这里就够了**：无需 GPU、无需 key、无需网络。
- **Tier 1** 另需任意 OpenAI 兼容的模型服务（如 Ollama）。

---

## 1. 分层地图（Tier）

四个层级，按准备成本从低到高排列，**层与层之间不会再安装任何新组件**：

| 层级 | 你得到什么 | 准备工作 |
|---|---|---|
| **Tier 0 — try the kernel** | 离线运行/度量裁决边界 | `go build`，零下载 |
| **Tier 1 — front a real model** | 把内核挡在你自己部署的模型前（Ollama / vLLM / llama.cpp / 云厂商） | + 一个运行中的 OpenAI 兼容服务 |
| **Tier 1b — local model in one command** | 用现有智能体在内核内跑本地 GGUF 模型——无需 key、无需网络、无需第二个终端 | `fak manage --gguf qwen2.5:7b -- claude` |
| **Tier 2 — the fused in-kernel model** | 内核自有的纯 Go 前向推理（reference forward pass） | + 权重导出 |

若你只想**让 fak 挡在一个真实模型前面**，就选 **Tier 1**。Tier 2 的 in-kernel 模型是与 Hugging Face 逐位一致的*参考前向推理*，
而非面向对话质量的服务引擎（详见英文 GETTING-STARTED 的
honest caveat）。

**关于模型：** fak 只管控并缓存你的模型，而不替换它。**Qwen（通义千问）** 与 **GLM（智谱）**
已在 in-kernel 参考引擎中验证为逐位一致（bit-exact）；**DeepSeek、Yi、百川（Baichuan）、Kimi**
以及任意开源权重模型通过 OpenAI 兼容协议接入（Ollama / vLLM / SGLang / llama.cpp / LM Studio
或任意 OpenAI 兼容 API）。

---

## 2. 安装（拿到二进制）

从克隆构建（贡献者路径）：

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # Windows: go build -o fak.exe ./cmd/fak
```

用 Go 直接安装（模块路径即仓库根，可直接解析）：

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **中国大陆安装提示。** Go 默认的模块代理在境内常不可达——先设置国内代理：
> `GOPROXY=https://goproxy.cn,direct`（用于模块下载）。GGUF 权重在 Hugging Face 不可达时，
> 可在 **ModelScope（魔搭）** 上寻找同名镜像。
>
> ```bash
> export GOPROXY=https://goproxy.cn,direct
> go install github.com/anthony-chaudhary/fak/cmd/fak@latest
> ```

---

## 3. 60 秒验证（无需 key、无需模型、无需 GPU）

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

`preflight` 展示的是**结构性、默认拒绝**的能力闸门（capability floor）：不在允许清单上的动作
无法运行，与模型被如何说服无关——这是 fail-closed。`fak agent --offline` 是注入 A/B：在实测中，
prompt 注入 5/5 攻破了无防护基线，fak 则 5/5 将其隔离在模型上下文之外。

---

## 4. 下一步

- **英文引导教程（friendliest on-ramp）：** [`docs/fak/tutorial.md`](https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/tutorial.md) —— 逐步走过 Tier 0–2，每条命令都附真实捕获的输出。
- [README（完整概览）](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 分钟跑起本地模型](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — 安装与分层参考（英文源页）](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — 接入你的智能体](../../integrations/README.md)
- [数据驻留与合规 — 对应 PIPL / 数据安全法 / 网络安全法](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — 每个数字的出处](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — 哪些是 shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)——免费、自托管，无需信用卡、无需跨境发票、无需主体。
