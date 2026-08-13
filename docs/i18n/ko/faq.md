---
title: "fak — 첫 접촉 FAQ (한국어 입문 / Korean introduction)"
description: "fak의 한국어 입구 페이지: AI 에이전트와 그 tool call 사이에 앉아 모든 호출을 실행 전에 심사하는 Go 바이너리 — 같은 에이전트 loop를 더 안전하고 저렴하고 빠르게. self-host, PIPA(개인정보보호법) 친화, Apache-2.0."
---

# fak — 첫 접촉 FAQ (한국어 입문)

> 이 페이지는 **현지화된 입구 페이지(entry point)** 이며, 전체 문서의 번역이 아닙니다.
> 전체 문서는 영어로 되어 있습니다 — 이 페이지는 fak의 핵심, 60초 검증, 설치 경로를
> 짧게 짚어 [영문 문서](https://github.com/anthony-chaudhary/fak/blob/main/README.md)로
> 안내합니다.
> **안내:** 이 번역은 기계가 작성했으며 아직 원어민 검토를 거치지 않았습니다 — 오류를
> 발견하면 issue/PR을 열어 주세요.
> **다른 언어 입구:** [i18n hub](../README.md).

## Q1. fak가 무엇인가요?

이미 운영 중인 AI 에이전트(Claude Code, Codex, Cursor, 그 밖의 어떤 OpenAI/Anthropic/MCP
클라이언트든) 앞에 놓는 **하나의 정적 Go 바이너리**입니다. base URL 하나만 다시 가리키면
되고, 에이전트를 다시 쓸 필요가 없습니다. 긴 session을 더 저렴하게 만들고(provider의
prompt-cache prefix를 byte 단위로 동일하게 유지한 채 오래된 turn을 덜어냅니다), 각 tool call을
라우팅하며, 로컬 GGUF 모델을 in-process로 실행할 수 있고, 모든 호출에 대해 감사 가능한
판정(verdict)을 기록합니다.

## Q2. 모델을 바꾸거나 에이전트를 다시 써야 하나요?

아니요. fak는 이미 쓰고 있는 모델을 **govern(통제)하고 cache**할 뿐, 대체하지 않습니다.
한 줄로 감싸거나:

```bash
fak manage claude
```

base URL 하나를 `fak serve`로 다시 가리키면 됩니다.

## Q3. 제 데이터는 어디로 가나요 — 규정을 준수하나요?

**self-host 우선**입니다: 로컬 모델이나 국내 provider 앞에 정적 바이너리 하나가 서서,
fail-closed 데이터 잔류(residency), default-deny capability floor, 그리고 모든 tool call에 대한
변조 감지(tamper-evident) 감사 로그를 제공합니다. 데이터는 당신의 머신을 떠나지 않습니다.
이는 국내 **개인정보보호법(PIPA)** 의 데이터 처리·역외 이전 요건에 맞춰 데이터를 국내에
두려는 팀에게 그대로 들어맞습니다.

## Q4. 비용은 얼마인가요? 정말 무료인가요?

**Apache-2.0**, 무료, self-host입니다. 신용카드도, 국경을 넘는(cross-border) 청구서도, 법인도
필요 없습니다. fak 자체에는 원화(KRW)든 달러든 어떤 청구도 없습니다 — 토큰 비용은 당신이
쓰는 provider가 청구하며, fak는 그 비용을 낮출 뿐입니다. `git clone`과 `go install`이 전체
경로입니다.

## Q5. 얼마나 더 저렴하거나 빠른가요?

측정된 **50-turn × 5-agent** session 기준으로, 잘 **튜닝된 warm-cache 스택 대비 약 4.1배 적은
작업량**입니다. 이는 곧 provider가 원화(KRW)로 청구하는 토큰 비용을 직접 낮추는 지렛대입니다.
~60배 수치(약 19시간 → 약 19분)는 **오직 naive re-send(전부 다시 보내기) loop와 비교했을
때만** 성립하며, 결코 대표 수치로 내세우지 않습니다. 재사용(reuse) 이득은 **self-host
전용**이고, 읽기 중심(read-heavy) 플릿에 적용됩니다.

## Q6. 어떤 모델이 동작하나요?

**Qwen2/Qwen3와 GLM-MoE**는 in-kernel 참조 엔진에서 bit-exact(비트 단위 일치)로 검증되었습니다.
그 밖의 모든 것(DeepSeek, Mistral, 임의의 open-weights 모델)은 OpenAI 호환 wire를 통해
앞단에 붙습니다 — Ollama / vLLM / SGLang / llama.cpp / LM Studio.

## Q7. prompt injection은 어떻게 막나요?

분류기(classifier)가 아니라 **두 개의 구조적 게이트**입니다: default-deny capability floor
(위험한 tool은 애초에 allow-list에 없습니다)와 result quarantine(오염된 tool 결과는 모델
context에 아예 도달하지 않습니다). 실제 테스트에서 injection은 보호받지 않은 baseline에
5/5로 도달했고, fak는 이를 5/5로 차단했습니다.

## Q8. 어떻게 설치하나요?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

또는 clone에서:

```bash
go build -o fak ./cmd/fak
```

모듈 다운로드가 막히는 환경이라면 `GOPROXY`를 설정한 뒤 다시 시도하세요.

### 60초 검증 (key 없이, 모델 없이, GPU 없이)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Q9. 다음은 어디로 가나요?

- [README (전체 개요)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10분 안에 로컬 모델](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — 바이너리 설치](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — 모든 수치의 출처](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — 무엇이 shipped/simulated/stub인지](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [데이터 residency와 규정 준수 — PIPA(개인정보보호법)에 대응](../../explainers/data-residency-and-compliance.md)
- [Integrations — 당신의 에이전트를 연결](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
