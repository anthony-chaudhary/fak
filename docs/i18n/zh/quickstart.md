---
title: "fak 快速开始（10 分钟跑起一个受管治的本地模型）"
description: "fak 的简体中文快速开始入口页：约 10 分钟从零到一个受管治的本地 AI——离线、无需 key、无云账单、数据留在本机；含在中国大陆的安装要点（GOPROXY / ModelScope）。"
---

# fak 快速开始（10 分钟跑起一个受管治的本地模型）

> 这是一个**本地化入口页**，而非完整文档翻译。完整文档为英文——本页把你从零带到一个受管治的
> 本地 AI，然后把你引向[英文文档](https://github.com/anthony-chaudhary/fak/blob/main/README.md)。
> **说明：** 本页由机器生成、尚待母语者校对——如发现错误，欢迎提交 issue/PR。
> **相关中文页：**[中文入口页 ./README.md](./README.md) · 更多语言见 [i18n hub](../README.md)。
> 权威英文来源：<https://github.com/anthony-chaudhary/fak/blob/main/README.md>。

## 承诺：约 10 分钟，从零到一个受管治的本地 AI

跟着本页走，你会得到一个跑在**自己机器上**的 AI：离线可用、无需 API key、没有云账单，数据不出
本机——小模型用 CPU 就够，不需要 GPU。fak 不替换你的模型，它在每次工具调用执行前先审查、并在
长会话中复用重复工作。

## 最快路径：一条命令，把本地模型包进你现有的智能体

```bash
fak manage claude
```

这会用 fak 包住你现有的智能体（这里是 `claude`），每次工具调用都先经过内核里的默认拒绝
能力闸门（capability floor）。你**不用重写智能体**；如果走网关方式，只需把一个 base URL 指向
`fak serve`。

### 在中国大陆安装（重要）

Go 默认的 `proxy.golang.org` 在境内常不可达，请先设置国内模块代理再安装：

```bash
export GOPROXY=https://goproxy.cn,direct
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

模型权重：当 Hugging Face 无法访问时，可在 **ModelScope（魔搭）** 上寻找同名 GGUF/权重镜像。

## 60 秒验证（无需 key、无需模型、无需 GPU）

在克隆下来的仓库根目录运行：

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

第一行拒绝、第二行放行，靠的是内核**结构**而非分类器：不在允许清单上的动作根本无法执行——无论
模型被怎样诱导。最后一行在离线状态下拦下注入，任务照样完成。

## 什么是 fak

**fak 是一个静态 Go 二进制程序**，位于你的 AI 智能体与它调用的工具之间——在每次工具调用
*执行之前*先审查它，并在长会话中复用重复的共享工作。结果：同一个智能体循环变得**更安全、更省、
更快**，且无需重写。

- **默认拒绝的能力闸门**在内核内、同一条调用路径上检查，fail-closed。
- **结果隔离（quarantine）**把可疑的工具*返回结果*完全挡在模型上下文之外——靠结构，不靠检测器
  （其内置检测器按设计视为可被绕过，只是加分项，绝不作为底线）。实测：prompt 注入对未保护基线
  命中 5/5，fak 全部拦下 5/5。

模型方面，fak 管治并缓存你的模型，而不替换它：**Qwen2/Qwen3（通义千问）、GLM-MoE（智谱）** 已在内核内置的
参考引擎中验证为逐位一致（bit-exact）；**DeepSeek、Yi、百川、Kimi 以及任意开源权重模型**通过
OpenAI 兼容协议接入（Ollama / vLLM / SGLang / llama.cpp / LM Studio 或任意 OpenAI 兼容 API）。

## 到底有多快（诚实口径）

在一次实测的 50 轮 × 5 智能体会话中，相对一个**经过调优的 warm-cache 栈**，fak 的诚实收益约为
**4.1× 更少的工作量**——这是本页的头条数字。

那个抢眼的 **约 60×**（约 19 小时降到约 19 分钟）**仅**相对**朴素的“每轮重发全部内容”循环**成立；
不要把它当头条，若提及必须明确写成“仅相对 naive 模式”。这项复用收益**仅限自托管**，且适用于
读密集的智能体集群。

关于成本：这会直接体现在以人民币（RMB）计价的毛利上——长会话不再越来越贵。服务商的 prompt-cache
折扣只在 cached prefix 逐字节不变时才保留；fak 在剥离中间旧轮次的同时**保证 prefix 逐位一致**，
折扣因而不会失效。是否真正命中缓存则由服务商决定，fak 只如实转发、不代为断言。

每个数字都可追溯到其 commit 与产物，见
[BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)。

## 许可与成本

fak 采用 **Apache-2.0**、免费、自托管——无需信用卡、无需跨境发票、无需主体。`git clone` 加
`go install github.com/anthony-chaudhary/fak/cmd/fak@latest` 就是全部路径。

## 下一步

- [README（完整概览）](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 分钟跑起本地模型](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — 安装并把控制面放到 AI 前面](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — 每个数字的出处](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — 哪些是 shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [数据驻留与合规 — 对应 PIPL / 数据安全法 / 网络安全法](../../explainers/data-residency-and-compliance.md)
- [Integrations — 接入你的智能体](../../integrations/README.md)
- 返回[中文入口页 ./README.md](./README.md) · [i18n hub](../README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)。
