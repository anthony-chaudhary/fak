---
title: "fak 빠른 시작 — 10분 만에 로컬 모델 실행 (한국어 입문 / Korean quickstart)"
description: "fak 한국어 빠른 시작: 모든 tool call을 실행 전에 검토하고 긴 session의 공유 작업을 재사용하는 Go 바이너리 — 키 없이, 클라우드 비용 없이 약 10분 만에 관리형 로컬 AI. self-host, PIPA(개인정보보호법) 친화, Apache-2.0."
---

# fak 빠른 시작 — 10분 만에 로컬 모델 (한국어 입문)

> 이 문서는 **현지화된 진입 페이지(entry point)** 이며 전체 문서의 번역이 아닙니다.
> 전체 문서는 영어로 제공됩니다 — 이 페이지는 fak의 핵심, 60초 증명, 설치 경로를 짧게
> 요약해 [영문 문서](https://github.com/anthony-chaudhary/fak/blob/main/README.md)로
> 안내합니다.
> **안내:** 이 번역은 기계가 생성했으며 원어민 검수 대기 중입니다 — 오류를 발견하면
> issue/PR을 열어 주세요.
> **다른 언어 입문:** [i18n hub](../README.md).

## 약속: 약 10분 만에 관리형 로컬 AI

아래 단계를 마치면 여러분의 컴퓨터에서 **오프라인으로 동작하는 관리형(governed) AI**를 갖게
됩니다 — API 키 없이, 클라우드 청구서 없이, 데이터는 여러분의 기기에 머무릅니다. 작은
모델은 CPU만으로 충분합니다.

## 가장 빠른 경로 — 한 줄

기존 에이전트를 로컬 모델로 그대로 감쌉니다. 에이전트를 다시 쓸 필요 없이, 한 명령으로
tool call마다 capability floor를 먼저 통과시킵니다.

```bash
fak manage --gguf qwen2.5:7b -- claude
```

## 60초 증명 (키 없음, 모델 없음, GPU 없음)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

`refund_payment`은 allow-list에 없어 **구조적으로** 거부됩니다(POLICY_BLOCK). 이 판정은
kernel *안에서*, 같은 call path 위에서 이루어지므로 **fail-closed**입니다 — 모델이 어떤
말에 설득되든, allow-list에 없는 동작은 실행될 수 없습니다.

## fak란 무엇인가

**fak은 하나의 정적 Go 바이너리**로, 여러분의 AI 에이전트와 그것이 호출하는 도구 사이에
앉습니다. 모든 tool call을 *실행되기 전에* 검토하고, 긴 session에서 반복되는 공유 작업을
재사용합니다. 결과: **같은 에이전트 loop가 재작성 없이 더 안전하고, 더 저렴하고, 더
빨라집니다.** 에이전트를 다시 쓰지 않습니다 — base URL 하나를 `fak serve`로 돌리거나,
`fak manage claude` 한 명령으로 기존 에이전트를 감싸면 됩니다.

- **default-deny capability floor** — allow-list에 없는 동작은 kernel 안에서 fail-closed로
  차단됩니다. 공격을 "탐지"하는 데 의존하지 않습니다.
- **result quarantine** — 의심스러운 tool *결과*는 모델 context에 아예 들어오지 않도록
  격리됩니다. 이는 우회 가능한 분류기가 아니라 **구조**입니다. 결과를 표시하는 탐지기는
  설계상 ~100% 우회 가능한 것으로 취급하며, 어디까지나 보너스일 뿐 바닥선이 아닙니다.
- **라이브 테스트** — prompt injection이 보호되지 않은 baseline에는 5/5로 도달했고, fak은
  이를 5/5로 차단했습니다.

모델은 fak이 govern·cache할 뿐 대체하지 않습니다. **Qwen2/Qwen3와 GLM-MoE**는 in-kernel
reference engine에서 bit-exact로 검증되었고, 그 밖의 모든 모델(DeepSeek, Mistral, 임의의
open-weights 모델)은 OpenAI-compatible wire로 접속합니다 — Ollama / vLLM / SGLang /
llama.cpp / LM Studio 또는 임의의 OpenAI-compatible API.

## 한국 팀에게 중요한 이유

- **토큰 비용은 원화(KRW) 마진에 직접 반영됩니다.** fak은 긴 session에서 공유 작업
  (system prompt, tool list의 KV cache)을 재사용하므로, 반복 처리에 드는 KRW 비용이
  그만큼 줄어듭니다. 이 재사용 이득은 **self-host 전용**이며 read-heavy fleet에 적용됩니다.
  provider의 prompt-cache 할인은 cached prefix가 바이트 단위로 동일할 때만 유지되는데,
  fak은 중간의 오래된 turn을 덜어내면서도 prefix를 byte-identical하게 유지합니다. fak은
  prefix의 byte-identity를 **보장**합니다. provider가 실제로 cache를 재사용하는지는
  provider의 몫이며, fak은 이를 주장하지 않고 그대로 중계합니다.
- **데이터는 기기를 벗어나지 않습니다 (PIPA / 개인정보보호법).** fak은 self-host 우선의
  static 바이너리로 **로컬 모델** 또는 국내 provider 앞에 앉아, 백엔드 전반에 걸쳐
  fail-closed residency와 tool call마다의 감사 로그를 제공합니다. 데이터가 국외로 이전되지
  않으므로 PIPA의 국외 이전 요건을 근본적으로 단순화합니다.

## 얼마나 빠른가 (정직한 수치)

측정된 **50-turn × 5-agent session**에서, *튜닝된* warm-cache 스택 대비 정직한 이득은
약 **4.1× 적은 작업**입니다 — 이것이 헤드라인 수치입니다.

**모든 것을 매번 다시 보내는 naive loop** 대비로만 같은 session이 약 19시간에서 약 19분으로
줄어듭니다(약 60×). 다만 이 ~60× 수치는 **오직 naive 패턴에 대해서만** 성립하므로 헤드라인이
아닙니다. 모든 수치의 출처는
[BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)에서
commit과 artifact까지 추적할 수 있습니다.

## 라이선스와 비용

fak은 **Apache-2.0**, 무료, self-host입니다 — 신용카드도, 국경 간 인보이스도, 법인 설립도
필요 없습니다. `git clone`과
`go install github.com/anthony-chaudhary/fak/cmd/fak@latest`가 전체 경로입니다.

## 다음으로 갈 곳

- [README (전체 개요)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10분 만에 로컬 모델](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — 도구 호출 제어 평면](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — 모든 수치의 출처](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — shipped/simulated/stub 구분](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [데이터 residency와 규정 준수 — PIPA(개인정보보호법)](../../explainers/data-residency-and-compliance.md)
- [Integrations — 여러분의 에이전트 연결](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
