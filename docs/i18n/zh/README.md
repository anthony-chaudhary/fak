---
title: "fak — 融合式智能体内核（简体中文入门 / Simplified Chinese introduction)"
description: "fak 的简体中文入口页：一个位于 AI 智能体与工具调用之间的 Go 二进制程序——每次 tool call 执行前先审查，长会话复用重复工作；自托管、PIPL 友好、Apache-2.0。"
---

# fak — 融合式智能体内核（简体中文入门）

> 这是一个**本地化入口页**，而非完整文档翻译。完整文档为英文——本页给出 fak 的核心定位、
> 60 秒验证、安装路径，然后把你引向[英文文档](../../../README.md)。
> **说明：** 本翻译由机器生成、尚待母语者校对——如发现错误，欢迎提交 issue/PR。
> **其他语言入口：**[i18n hub](../README.md)（हिन्दी · தமிழ் · తెలుగు · বাংলা · मराठी）。

## 一句话理解 fak

**fak 是一个 Go 二进制程序**，它位于你的 AI 智能体与它调用的工具之间——在每次工具调用
*执行之前*先审查它，并在长会话中复用重复的工作。结果：同一个智能体循环变得**更安全、更省、
更快**，且无需改动其他任何东西。

你不用重写智能体——只需把一个 base URL 指向 `fak serve`，每次工具调用都会先经过能力闸门
（capability floor）。

```bash
fak guard -- claude    # 用一条命令包住你现有的智能体
```

## 为什么中国的创业团队值得一看

- **模型自主：govern 并 cache 你已经在用的国产模型。** fak 不要求你换模型——它把你的模型
  包起来。**Qwen（通义千问）、GLM（智谱）** 已在 in-kernel 参考引擎中验证为逐位一致
  （bit-exact）；**DeepSeek、Yi、百川、Kimi 以及任意开源权重模型**通过 OpenAI 兼容协议
  接入（Ollama / vLLM / SGLang / llama.cpp / LM Studio 或任意 OpenAI 兼容 API）。
- **数据留在境内（PIPL / 数据安全法 / 网络安全法）。** fak 以自托管为先：一个静态二进制
  程序挡在**本地模型**或国产 provider 前面，跨后端 fail-closed 数据驻留、默认拒绝的能力闸门，
  以及每次工具调用的防篡改审计日志。数据不出你的机器。默认拒绝是**结构性**的——模型无法
  请求的动作，也就无法泄露——这比分类器更强。
- **成本以美元计价，收入不是。** fak 在长会话中复用共享工作（system prompt、工具列表的
  KV cache）：在 50×5 的测试中比经过调优的 warm-cache 栈**少做约 4.1× 的工作**（相对
  naive 重发循环约 60×，但诚实的数字是 4.1×）。再加上 per-aspect 路由把便宜的部分发给便宜
  的模型——这是直接作用于毛利的杠杆。
- **零支付摩擦。** fak 采用 **Apache-2.0**、免费、自托管——无需信用卡、无需跨境发票、无需
  主体。`git clone` 与 `go install` 就是全部流程。
- **一个静态二进制，零外部依赖。** 小团队运维简单——没有 sidecar，没有独立授权组件；从笔记本
  到集群是同一个产物，你只加 flag，不加组件。

## 在中国大陆安装（重要）

Go 默认的 `proxy.golang.org` 在境内常不可达。请先设置国内模块代理：

```bash
export GOPROXY=https://goproxy.cn,direct
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

模型权重：Hugging Face 常无法访问时，可在 **ModelScope（魔搭）** 上寻找同名 GGUF/权重镜像。

## 60 秒验证（无需 key、无需模型、无需 GPU）

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY（POLICY_BLOCK）
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # 注入被拦截，任务仍完成
```

## fak 解决什么问题

- **长会话不再越来越贵。** provider 的 prompt-cache 折扣只在 cached prefix 逐字节不变时才
  保留；fak 在剥离中间旧轮次的同时保持 prefix 逐位一致，折扣因而不会失效。
- **prompt 注入 / 结果投毒的隔离。** 可疑的工具*返回结果*被放入独立的 quarantine，完全不进入
  模型上下文——靠结构，而非模型可以绕过的分类器。

## 下一步

- [README（完整概览）](../../../README.md)
- [START-HERE — 10 分钟跑起本地模型](../../../START-HERE.md)
- [Getting Started — 安装二进制](../../../GETTING-STARTED.md)
- [Integrations — 接入你的智能体](../../integrations/README.md)
- [数据驻留与合规 — 对应 PIPL / 数据安全法](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — 每个数字的出处](../../../BENCHMARK-AUTHORITY.md)
- [CLAIMS — 哪些是 shipped/simulated/stub](../../../CLAIMS.md)

License: [Apache-2.0](../../../LICENSE)。
