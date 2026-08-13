---
title: "fak 설치 및 시작하기 (한국어 입문 / Korean getting-started)"
description: "fak 설치·실행 진입점: 모든 tool call을 실행 전에 심사하는 하나의 정적 Go 바이너리 — 같은 에이전트 loop가 더 안전·저렴·빠르게. PIPA 친화 self-host, Apache-2.0."
---

# fak 설치 및 시작하기 (한국어 입문)

> 이 문서는 **로컬라이즈된 진입점(entry point)** 이며, 전체 문서의 번역이 아닙니다.
> 완전한 문서는 영어로 제공됩니다 — 이 페이지는 설치와 실행의 첫 관문만 담고 곧바로
> [영어 원문 문서](https://github.com/anthony-chaudhary/fak/blob/main/README.md)로 안내합니다.
> **참고:** 이 번역은 기계가 작성했으며 아직 원어민 검수를 받지 않았습니다 — 오류를 발견하면
> issue/PR을 열어 주세요.
> **다른 언어 입구:** [i18n hub](../README.md).

## 한눈에

이 페이지는 fak을 **설치하고 실행하는 첫 관문**입니다. 밀도 높은 소개는
[README](https://github.com/anthony-chaudhary/fak/blob/main/README.md)에 있습니다.

fak은 AI 에이전트와 그 에이전트가 호출하는 도구 사이에 놓이는 **하나의 정적 Go 바이너리**
입니다. 모든 tool call을 *실행되기 전에* 심사하고, 긴 세션에서 반복되는 공유 작업을
재사용합니다. 결과: 같은 에이전트 loop가 재작성 없이 **더 안전하고, 더 저렴하고, 더 빠르게**
바뀝니다. 에이전트를 다시 쓰지 않고, base URL 하나를 `fak serve`로 돌리거나 기존 에이전트를
한 명령으로 감싸면 됩니다.

```bash
fak manage claude      # 기존 에이전트를 한 명령으로 감쌉니다
```

## 한국 팀에게 왜 중요한가

- **데이터는 국내에, PIPA(개인정보보호법)에 맞춰.** fak은 self-host 우선입니다 — 로컬 모델
  이나 국내 provider 앞에 서는 정적 바이너리 하나가 백엔드마다 fail-closed 데이터 residency,
  default-deny capability floor, 그리고 모든 tool call에 대한 변조 감지(tamper-evident)
  감사 로그를 제공합니다. 데이터가 기기를 떠나지 않습니다.
- **토큰 비용은 원(KRW)으로 청구되어 마진을 갉아먹습니다.** fak은 긴 세션의 공유 작업
  (system prompt, tool list의 KV cache)을 재사용해, 튜닝된 warm-cache 스택 대비 50턴×5에이전트
  세션에서 **약 4.1× 적은 작업**을 합니다. 이 재사용 이득은 **self-host 전용**이며 read-heavy
  fleet에 적용됩니다. 흔히 인용되는 ~60×(약 19시간 → 약 19분)는 **오직 naive한 전부-재전송
  loop 대비**일 때만 참이며, 헤드라인 수치가 아닙니다.

## 0. 사전 준비 (Prerequisites)

- **Go 1.26+.** `go.mod`가 `go 1.26`을 선언합니다. Go 기본값 `GOTOOLCHAIN=auto`이면 더 오래된
  `go`도 첫 빌드 시 알맞은 toolchain을 자동으로 내려받습니다(최초 1회 네트워크 필요). 그렇지
  않으면 <https://go.dev/dl/>에서 Go 1.26을 설치하세요. `go version`으로 확인합니다.
- **Tier 0에는 그 외 아무것도 필요 없습니다:** GPU 없음, API key 없음, 네트워크 없음.

## 티어 한눈에

설정 비용이 낮은 순서이며, 티어 사이에서 **새로 설치되는 것은 없습니다.**

| 티어 | 무엇을 얻는가 | 설정 | 다운로드 |
|---|---|---|---|
| **0 — 커널 체험** | 심사 경계를 오프라인으로 실행·측정 | `go build` | 없음 |
| **1 — 실제 모델 앞에 세우기** | 다른 곳에서 서빙하는 모델(Ollama / vLLM / llama.cpp / 클라우드) 앞에 커널을 둠 | + OpenAI 호환 서버 실행 | 채팅 모델 하나 |
| **1b — 한 명령으로 로컬 모델** | 기존 에이전트로 로컬 GGUF 모델을 커널 내부에서 실행 — key·네트워크·두 번째 터미널 불필요 | `fak manage --gguf qwen2.5:7b -- claude` | ~5 GB GGUF (캐시됨) |
| **2 — 융합된 커널 내부 모델** | 커널이 소유하는 순수 Go SmolLM2 forward pass | + (실제 가중치) Python export | ~135M 파라미터 |

실용적으로 **쓸모 있는 모델을 fak 뒤에 세우고 싶다면 Tier 1** 입니다. Tier 2의 커널 내부
모델은 HuggingFace와 비트 단위로 검증된 *레퍼런스 forward pass*이지 채팅 품질의 서빙 엔진이
아닙니다.

## 바이너리 받기 (설치)

`git clone` 후 빌드하거나, Go로 바로 설치하면 됩니다 — 그게 전부입니다.

```bash
# 클론에서 빌드 (contributor)
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # -> ./fak   (Windows: go build -o fak.exe ./cmd/fak)
./fak help
```

```bash
# 또는 Go로 직접 설치 — 모듈 경로가 곧 저장소 루트이므로 그대로 설치됩니다
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

카드도, 국경 간 청구서도, 법인도 필요 없습니다: fak은 **Apache-2.0**, 무료, self-host입니다.

## 60초 증명 (key 없음, model 없음, GPU 없음)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

allow-list에 없는 동작은 kernel 안, 같은 call path에서 fail-closed로 막힙니다(분류기가 아니라
구조). 라이브 테스트에서 prompt injection은 보호되지 않은 baseline을 5/5로 뚫었고, fak은 이를
5/5로 차단했습니다.

## 여러분의 모델과 함께

fak은 여러분의 모델을 대체하지 않고, govern하고 cache합니다. **Qwen2/Qwen3와 GLM-MoE**는
in-kernel 레퍼런스 엔진에서 bit-exact로 검증되었고, 그 밖의 모든 것(DeepSeek, Mistral, 임의의
open-weights 모델)은 OpenAI 호환 wire로 접속합니다 — Ollama / vLLM / SGLang / llama.cpp /
LM Studio 또는 임의의 OpenAI 호환 API를 통해.

## 다음 단계로

- [`docs/fak/tutorial.md`](../../fak/tutorial.md) — **안내형 첫 세션**(약 15분): 모든 명령을 실제
  캡처된 출력과 함께, 앞부분은 오프라인·무 key·무 GPU로.
- [README (전체 개요)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10분 만에 로컬 모델](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — 설치·티어 레퍼런스](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [데이터 residency와 규정 준수 — PIPA(개인정보보호법)용](../../explainers/data-residency-and-compliance.md)
- [Integrations — 여러분의 에이전트를 연결](../../integrations/README.md)
- [BENCHMARK-AUTHORITY — 모든 수치의 출처](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — shipped/simulated/stub 구분](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
