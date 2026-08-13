---
title: "fak — 융합형 에이전트 커널 (한국어 입문 / Korean introduction)"
description: "fak 한국어 진입 페이지: AI 에이전트와 tool call 사이에 자리 잡는 하나의 Go 바이너리 — 모든 tool call을 실행 전에 검토하고 긴 세션의 반복 작업을 재사용해, 같은 에이전트 loop를 더 안전하고 저렴하고 빠르게. 자체 호스팅, PIPA 친화, Apache-2.0."
---

# fak — 융합형 에이전트 커널 (한국어 입문)

> 이 문서는 **현지화된 진입 페이지(entry point)**이며, 전체 문서의 번역이 아닙니다.
> 완전한 문서는 영어로 제공됩니다 — 이 페이지는 fak의 핵심 정의, 60초 검증, 설치 경로를
> 요약해 [영문 문서](https://github.com/anthony-chaudhary/fak/blob/main/README.md)로
> 안내합니다.
> **안내:** 이 번역은 기계가 작성했으며 아직 원어민 검수를 거치지 않았습니다 — 오류를
> 발견하시면 issue/PR로 알려 주세요.
> **다른 언어 진입점:** [i18n hub](../README.md) (हिन्दी · 简体中文 · தமிழ் · తెలుగు · বাংলা · मराठी).

## 한 줄로 이해하는 fak

**fak은 하나의 Go 바이너리**로, 여러분의 AI 에이전트와 그것이 호출하는 도구 사이에
자리 잡습니다 — 모든 tool call을 *실행되기 전에* 검토하고, 긴 세션에서 반복되는 공유
작업을 재사용합니다. 결과적으로 같은 에이전트 loop가 **더 안전하고, 더 저렴하고, 더
빨라지며**, 그 밖의 어떤 것도 다시 작성할 필요가 없습니다.

에이전트를 다시 짜지 않습니다 — base URL 하나를 `fak serve`로 돌리기만 하면, 모든 tool
call이 먼저 capability floor를 통과합니다.

```bash
fak manage claude      # 기존 에이전트를 명령어 하나로 감쌉니다
```

## 한국 팀이 주목해야 하는 이유

- **비용은 원(KRW)으로 아프지만, 토큰 청구서는 달러로 옵니다.** fak은 긴 세션에서 공유
  작업(system prompt, tool list의 KV cache)을 재사용합니다 — 튜닝된 warm-cache 스택 대비
  50턴 × 5에이전트 실행에서 **약 4.1배 적은 작업**을 합니다. (naive 재전송 loop 대비로는
  약 60배지만, 이는 **오직 naive 패턴과 비교했을 때만** 성립하며 정직한 대표 수치는 4.1배
  쪽입니다.) 이 재사용 이득은 **자체 호스팅에서만** 유효하며 read-heavy fleet에 적용됩니다 —
  원 단위 마진에 직접 작용하는 지렛대입니다.
- **데이터를 국내에 두기 (개인정보보호법 / PIPA).** fak은 자체 호스팅 우선입니다: **로컬
  모델**이나 국내 provider 앞에 서는 하나의 static 바이너리로, 백엔드 전반에 fail-closed
  residency, default-deny capability floor, 그리고 모든 tool call에 대한 tamper-evident
  감사 로그를 제공합니다. 데이터는 여러분의 머신을 벗어나지 않습니다.
- **국경을 넘는 결제가 필요 없습니다.** fak은 **Apache-2.0**, 무료, 자체 호스팅입니다 —
  카드도, cross-border invoice도, 법인도 필요 없습니다. `git clone`과 `go install`이 전체
  경로입니다.
- **하나의 static 바이너리, 외부 의존성 제로.** 작은 팀에게 운영이 단순합니다 — sidecar도,
  별도 authorizer도 없습니다. 노트북에서 fleet까지 동일한 산출물이며, 여러분은 컴포넌트가
  아니라 flag만 추가합니다.

## fak이 해결하는 문제

- **긴 세션이 계속 비싸지지 않도록.** provider의 prompt-cache 할인은 cached prefix가
  byte-for-byte 동일할 때만 유지됩니다. fak은 중간의 오래된 turn을 걷어내면서도 prefix를
  byte-identical로 유지합니다 — fak은 prefix의 byte-identity를 **보장**하며, provider가
  실제로 캐시를 재사용할지는 provider의 몫이므로 fak은 이를 주장하지 않고 그대로 전달만
  합니다.
- **default-deny 보안.** 권한 정책은 커널 *내부*, 같은 call path에서 검사됩니다. allow-list에
  한 번도 없던 동작은 모델이 무엇에 설득당했든 실행될 수 없습니다 — 이것은 공격을 "탐지"하는
  분류기가 아니라 **fail-closed** 구조입니다.
- **prompt injection / 오염된 결과의 격리.** 의심스러운 tool *결과*는 별도의 quarantine에
  보관되어 모델 context에 아예 들어가지 않습니다 — 분류기가 아니라 구조로 막습니다. 그것을
  표시하는 탐지기는 설계상 ~100% 우회 가능하다고 취급됩니다 — 보너스일 뿐 결코 바닥이
  아닙니다. 실측 테스트에서 prompt injection은 무방비 baseline을 5/5로 뚫었고, fak은 이를
  5/5로 막아냈습니다.

## 60초 검증 (key 없이, 모델 없이, GPU 없이)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## 여러분의 모델과 함께

fak은 여러분의 모델을 대체하지 않습니다 — 모델을 govern하고 cache합니다. **Qwen2/Qwen3와
GLM-MoE**는 in-kernel reference engine에서 bit-exact로 검증되었습니다. 그 외 모든 것
(DeepSeek, Mistral, 임의의 open-weights 모델)은 OpenAI-compatible wire 위에서 연결됩니다 —
Ollama / vLLM / SGLang / llama.cpp / LM Studio 또는 임의의 OpenAI-compatible API를 통해서요.

## 다음으로 갈 곳

- [빠른 시작 — 10분 만에 로컬 모델](./quickstart.md)
- [설치 — 바이너리와 tier 맵](./install.md)
- [FAQ — 첫 접촉 질문](./faq.md)
- [README (전체 개요)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10분 만에 로컬 모델](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — 바이너리 설치](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — 여러분의 에이전트 연결](../../integrations/README.md)
- [데이터 residency와 컴플라이언스 — 개인정보보호법(PIPA) 대응](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — 모든 수치의 출처](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — 무엇이 shipped/simulated/stub인지](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
