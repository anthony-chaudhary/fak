---
title: "fak — 首次接触常见问题（简体中文入门 / Simplified Chinese FAQ)"
description: "fak 最常见的入门问题（简体中文）：它是什么、要不要换模型、数据去哪、要不要花钱、快多少、支持哪些模型、如何拦截 prompt 注入、如何安装。自托管、PIPL 友好、Apache-2.0。"
---

# fak — 首次接触常见问题（简体中文入门）

> 这是一个**本地化入口页**，而非完整文档翻译。完整文档为英文——本页只回答最常见的入门问题，
> 然后把你引向[英文文档](https://github.com/anthony-chaudhary/fak/blob/main/README.md)。
> **说明：** 本页由机器生成、尚待母语者校对——如发现错误，欢迎提交 issue/PR。
> **其他语言与入口：**[i18n hub](../README.md) · 本语言入口页 [./README.md](./README.md)。

## Q1. fak 是什么？

一个**静态 Go 二进制程序**，放在你**已经在运行**的 AI 智能体（Claude Code、Codex、
Cursor，或任意 OpenAI / Anthropic / MCP 客户端）前面——只需把一个 base URL 重新指向它即可，
**无需重写**。它能降低长会话的开销（剥离旧的中间轮次，同时保持 provider 的 prompt-cache prefix
逐字节一致），为每一次工具调用做路由，可在进程内运行本地 GGUF 模型，并为每次调用留下一条可审计的裁决记录。

## Q2. 我必须换模型或重写智能体吗？

不用。fak 只是对你**已经在用**的模型做管控与缓存，并不替换它。用一条命令把你现有的智能体包起来即可：

```bash
fak manage claude
```

或者把一个 base URL 指向 `fak serve` 即可。

## Q3. 我的数据去哪里——这合规吗？

**自托管为先**：一个静态二进制程序挡在**本地模型**或境内 provider 前面，具备 fail-closed
的数据驻留、默认拒绝（default-deny）的能力闸门，以及每次工具调用的防篡改审计日志。
**数据不出你的机器。** 这直接对应境内数据保护法规——**《个人信息保护法》（PIPL）、
《数据安全法》与《网络安全法》**：数据留在境内、可审计、动作可控。

## Q4. 要多少钱？真的免费吗？

**Apache-2.0、免费、自托管。** 无需信用卡、无需跨境发票、无需注册主体。
`git clone` 加 `go install` 就是全部流程——没有按人民币计的订阅账单，也没有跨境结算。

## Q5. 到底能省多少、快多少？

在一次**实测的 50 轮 × 5 智能体会话**中，相比一个**经过调优的 warm-cache 技术栈**，
fak **少做约 4.1× 的工作**——这是诚实的头条数字，直接影响你的毛利率（成本以人民币计价时同样成立）。
那个 ~60×（约 19 小时缩短到约 19 分钟）的数字**仅**相对**朴素的『全量重发』循环**才成立，
**绝不**能当作头条。复用带来的收益仅限**自托管**场景，且只对**读密集型集群**适用。

## Q6. 支持哪些模型？

**Qwen2 / Qwen3 与 GLM-MoE** 已在内核内的参考引擎中验证为逐位一致（bit-exact）。
其余一切（DeepSeek、Mistral，以及任意开源权重模型）通过 OpenAI 兼容协议接入：
Ollama / vLLM / SGLang / llama.cpp / LM Studio，或任意 OpenAI 兼容 API。

## Q7. 它如何拦截 prompt 注入？

靠两道**结构性**闸门，而非分类器：

1. **默认拒绝的能力闸门**——危险工具从不在 allow-list 上，因此无论模型被如何诱导都无法运行。
2. **结果隔离（result quarantine）**——可疑的工具返回结果被完全挡在模型上下文之外。

在实测中，注入攻击对**无防护基线**得手 5/5，而 fak 将其挡下 5/5。

## 60 秒自证（自己跑一遍）

不必轻信上面的说法——克隆后自己跑一遍即可自证（能力闸门放行/拦截，注入被挡下且任务照常完成）：

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Q8. 如何安装？

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

或从克隆目录构建：

```bash
go build -o fak ./cmd/fak
```

**中国大陆提示：** Go 默认的 `proxy.golang.org` 常不可达，请先设置国内模块代理
`GOPROXY=https://goproxy.cn,direct` 再执行 `go install`；模型权重（GGUF）可在
**ModelScope（魔搭）** 上寻找镜像。

## Q9. 接下来去哪？

- [README（完整概览）](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 分钟跑起本地模型](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — 安装二进制](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — 每个数字的出处](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — 哪些是 shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [数据驻留与合规 — 对应 PIPL / 数据安全法 / 网络安全法](../../explainers/data-residency-and-compliance.md)
- [Integrations — 接入你的智能体](../../integrations/README.md)
- 本语言 [快速上手 ./quickstart.md](./quickstart.md) · [安装 ./install.md](./install.md)
- 本语言入口页：[./README.md](./README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)。
