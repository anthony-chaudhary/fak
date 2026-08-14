---
title: "fak — நிறுவல் மற்றும் தொடக்கம் (தமிழ் / Tamil install & getting started)"
description: "fak-ஐ ஒரு சுத்தமான checkout-இலிருந்து இயங்கும் kernel வரை கொண்டுசெல்லும் தமிழ் நிறுவல் நுழைவுப் பக்கம்: முன்நிபந்தனைகள், Tier வரைபடம், go build / go install கட்டளைகள்; self-host, DPDP-அனுகூலம்."
---

# fak — நிறுவல் மற்றும் தொடக்கம் (தமிழ்)

> இது ஒரு **உள்ளூர்மயமாக்கப்பட்ட நுழைவுப் பக்கம் (entry point)** — முழு ஆவணத்தொகுப்பின்
> மொழிபெயர்ப்பு அல்ல. முழு ஆவணங்கள் ஆங்கிலத்தில் உள்ளன — இந்தப் பக்கம் உங்களை நிறுவல்
> வழியில் ஏற்றி, பிறகு
> [ஆங்கில உண்மை-மூலம் (README)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
> நோக்கி அழைத்துச் செல்கிறது.
> **குறிப்பு:** இந்த மொழிபெயர்ப்பு இயந்திரத்தால் உருவாக்கப்பட்டது; native சரிபார்ப்பு
> நிலுவையில் — திருத்தங்களுக்கு issue/PR திறக்கவும்.
>
> மற்ற மொழிகள் மற்றும் முழுப் பட்டியல்: [i18n hub](../README.md). தமிழ் அறிமுகச் சாரம்:
> [./README.md](./README.md).

இது **நிறுவி-இயக்கும் முகப்பு (install-and-run front door)** — அடர்த்தியான விளக்கம்
[./README.md](./README.md)-இல் உள்ளது. இந்தப் பக்கம் ஒரு சுத்தமான checkout-இலிருந்து
இயங்கும் kernel வரை, copy-paste செய்யக்கூடிய கட்டளைகளுடன் உங்களைக் கொண்டுசெல்கிறது.

> **இந்திய சந்தைக்கு ஏன் இது பொருந்தும்:** fak self-host முதன்மை — Tier 0-க்கு key,
> GPU, அல்லது network தேவையில்லை; தரவு உங்கள் இயந்திரத்தை விட்டு வெளியேறாது
> (**DPDP Act, 2023**-க்கு ஏற்ப). நீண்ட session-களில் பகிரப்பட்ட வேலையை மறுபயன்பாடு
> செய்வதால் வரும் மிச்சம் — ரூபாய் margin-க்கு நேரடி நெம்புகோல் — self-host, read-heavy
> fleet-களுக்கு மட்டும்; நேர்மையான எண் ஒரு tuned warm-cache stack-ஐ விட 50-turn × 5-agent
> session-இல் **~4.1× குறைவான வேலை** (~60× என்பது naive re-send loop-ஐ விட மட்டுமே). முழு
> விளக்கம் [./README.md](./README.md)-இல்.

---

## 0. முன்நிபந்தனைகள்

- **Go 1.26+.** `fak/go.mod` `go 1.26`-ஐ அறிவிக்கிறது. Go-இன் default
  `GOTOOLCHAIN=auto`-உடன், பழைய `go` முதல் build-இல் சரியான toolchain-ஐத் தானாகவே
  download செய்யும் (ஒருமுறை network தேவை); இல்லையேல் <https://go.dev/dl/>-இலிருந்து
  Go 1.26-ஐ நிறுவவும். `go version`-ஆல் சரிபார்க்கவும்.
- **Tier 0-க்கு இதுவே போதும்:** GPU இல்லை, API key இல்லை, network இல்லை.
- **Tier 1**-க்கு கூடுதலாக எந்த OpenAI-compatible model server (எ.கா. Ollama) தேவை.

---

## 1. Tier வரைபடம்

setup செலவின் ஏறுவரிசையில் நான்கு படிகள் — **அவற்றுக்கு இடையில் புதிதாக எதுவும்
நிறுவப்படுவதில்லை:**

| Tier | என்ன கிடைக்கிறது | எப்படித் தொடங்குவது |
|---|---|---|
| **0 — kernel-ஐ முயற்சிக்க** | adjudication எல்லையை offline-இல் இயக்கி/அளக்க | `go build` — key/GPU/network தேவையில்லை |
| **1 — உண்மையான model-க்கு முன்** | வேறெங்கோ serve செய்யும் model-க்கு (Ollama / vLLM / llama.cpp / cloud) முன் kernel-ஐ வைக்க | + இயங்கும் OpenAI-compatible server |
| **1b — ஒரே கட்டளையில் local model** | இருக்கும் agent-உடன் local GGUF model-ஐ in-kernel இயக்க — key இல்லை, network இல்லை, இரண்டாவது terminal இல்லை | `fak manage --gguf qwen2.5:7b -- claude` |
| **2 — இணைந்த in-kernel model** | kernel சொந்தமாகக் கொண்டிருக்கும் pure-Go SmolLM2 forward pass | + (real weights) Python export |

நீங்கள் வெறுமனே **fak-ஐ முன் வைத்து ஒரு பயனுள்ள model-ஐ serve** செய்ய விரும்பினால்,
உங்களுக்குத் தேவை **Tier 1**. Tier 2-இன் in-kernel model என்பது HuggingFace-க்கு எதிராக
bit-for-bit நிரூபிக்கப்பட்ட ஒரு *reference forward pass* — chat-தர serving engine அல்ல.

---

## 2. binary-ஐப் பெறுங்கள்

fak **ஒரே Go binary** — வெளிச் சார்புகள் பூஜ்ஜியம். clone-இலிருந்து build செய்ய:

```bash
go build -o fak ./cmd/fak
```

> **Windows குறிப்பு:** Windows-இல் `go build -o fak.exe ./cmd/fak` எனப் build செய்யவும்
> (`-o fak` எனில் cmd.exe / PowerShell பெயரால் இயக்க முடியாத extension-இல்லா கோப்பு
> உருவாகும்). இந்த வழிகாட்டி `./fak` என எழுதும் இடங்களில் `.\fak.exe` எனத் தட்டச்சு செய்யவும்.

module path (`github.com/anthony-chaudhary/fak`) களஞ்சிய மூலமாக இருப்பதால், நேரடியாக
நிறுவுகிறது:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

**உரிமம்/செலவு:** fak **Apache-2.0**, இலவசம், self-host — கார்டு இல்லை, cross-border
invoice இல்லை, entity இல்லை. `git clone` மற்றும் `go install` — இதுவே முழு வழி.

---

## 3. 60-வினாடி சான்று (key இல்லை, model இல்லை, GPU இல்லை)

Tier 0 முழுவதும் offline மற்றும் deterministic. `fak/` உள்ளிருந்து இயக்கவும்:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

இது **capability floor**-ஐ செயலில் காட்டுகிறது: allow-list-இல் இல்லாத ஒரு action
model-ஐ எப்படி வற்புறுத்தினாலும் இயங்காது — kernel-இன் உள்ளே, அதே call path-இல்,
**fail-closed**. `./fak agent --offline`-இல் prompt injection தடுக்கப்பட்டாலும் task
நிறைவடைகிறது.

---

## 4. அடுத்து எங்கே

- **படிப்படியான முதல் session (~15 நிமிடம்):**
  [docs/fak/tutorial.md](https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/tutorial.md)
  — ஒவ்வொரு கட்டளையும் அதன் உண்மையான output-உடன், முதல் பகுதிகள் offline, key/GPU இல்லை.
- [README (முழு கண்ணோட்டம்)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 நிமிடத்தில் local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — ஆங்கில நிறுவல் மூலம்](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — உங்கள் ஏஜென்டை இணைக்கவும்](../../integrations/README.md)
- [தரவு residency மற்றும் compliance — DPDP Act-க்கு](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — ஒவ்வொரு எண்ணின் மூலம்](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — எது shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
