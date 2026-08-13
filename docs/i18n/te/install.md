---
title: "fak — install & getting started (తెలుగు / install & getting started)"
description: "fak తెలుగు install-and-run ప్రవేశ పేజీ: ఒకే static Go binary — go build తో Tier 0, local model కోసం fak guard; Go 1.26+, Tier 0-కి GPU/key/network అవసరం లేదు; DPDP-అనుకూల self-host."
---

# fak — install & getting started (తెలుగు)

> ఇది ఒక **స్థానికీకరించిన ప్రవేశ పేజీ (entry point)** — పూర్తి డాక్యుమెంటేషన్ అనువాదం
> కాదు. పూర్తి డాక్యుమెంటేషన్ ఇంగ్లీషులో ఉంది. ఈ పేజీ install మార్గం, tier map ఇచ్చి,
> మిమ్మల్ని [ఇంగ్లీష్ డాక్స్](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
> వైపు తీసుకెళ్తుంది.
> **గమనిక:** ఈ అనువాదం యంత్రంతో తయారైనది; native సమీక్ష పెండింగ్‌లో ఉంది — సవరణల
> కోసం issue/PR తెరవండి.
>
> **ఇతర భాషలు / సాంద్రమైన pitch:** [i18n hub](../README.md) ·
> [తెలుగు README](./README.md).

ఇది **install-and-run ముఖద్వారం** — శుభ్రమైన checkout నుండి నడుస్తున్న kernel వరకూ,
ఆపై దాని వెనుక ఒక model serve చేసే వరకూ తీసుకెళ్తుంది. సాంద్రమైన pitch
[README](./README.md)-లో ఉంది. `fak` **ఒకే Go binary** — సున్నా బాహ్య dependencies
(Python లేదు, CUDA toolkit లేదు) ఉన్న ఒకే static artifact; అదే binaryయే gateway,
KV-cache, policy gate, result quarantine, audit అన్నీ ఒకే process-లో.

మీరు మీ ఏజెంట్‌ను తిరిగి రాయరు — ఒకే base URL-ను `fak serve` వైపు మళ్లిస్తారు, లేదా
ఇప్పటి ఏజెంట్‌ను ఒకే కమాండ్‌లో wrap చేస్తారు:

```bash
fak manage claude
```

---

## 0. ముందస్తు అవసరాలు (Prerequisites)

- **Go 1.26+.** `fak/go.mod` లో `go 1.26` ప్రకటించబడింది. Go default
  `GOTOOLCHAIN=auto` తో పాత `go` కూడా మొదటి build-లో సరైన toolchain-ను స్వయంగా download
  చేసుకుంటుంది (ఒకసారి network అవసరం); లేదంటే <https://go.dev/dl/> నుండి Go 1.26 install
  చేయండి. `go version` తో తనిఖీ చేయండి.
- **Tier 0-కి ఇంతే** — GPU లేదు, API key లేదు, network లేదు.
- **Tier 1**-కి అదనంగా ఏదైనా OpenAI-compatible model server (ఉదా. Ollama) కావాలి.

---

## Tier map (పెరుగుతున్న setup ఖర్చు క్రమంలో — మధ్యలో కొత్తగా ఏదీ install కాదు)

| Tier | ఏమి వస్తుంది | Setup |
|---|---|---|
| **0 — kernel-ను ప్రయత్నించండి** | adjudication boundary-ను offline నడపండి/కొలవండి | `go build` |
| **1 — నిజమైన model ముందు నిలబెట్టండి** | మీరు వేరే చోట serve చేసే model ముందు kernel (Ollama / vLLM / llama.cpp / cloud provider) | + నడుస్తున్న OpenAI-compatible server |
| **1b — ఒకే కమాండ్‌లో local model** | మీ ఇప్పటి ఏజెంట్‌తో local GGUF model in-kernel — key లేదు, network లేదు, రెండో terminal లేదు | `fak manage --gguf qwen2.5:7b -- claude` |
| **2 — fused in-kernel model** | kernel స్వంతం చేసుకునే pure-Go forward pass | + (real weights కోసం) Python export |

మీకు కేవలం **fak వెనుక ఒక ఉపయోగకరమైన model serve** చేయాలంటే **Tier 1** కావాలి.
Tier 2-లోని in-kernel model అనేది HuggingFace-తో bit-for-bit సరిపోల్చిన *reference
forward pass* — chat-quality serving engine కాదు.

---

## Install — binary పొందండి

Contributor (clone నుండి build):

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # -> ./fak   (Windows: go build -o fak.exe ./cmd/fak)
./fak help
```

లేదా Go తో install (module path = repository root, కాబట్టి నేరుగా install అవుతుంది):

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

---

## 60-సెకన్ల రుజువు (Tier 0 — key లేదు, model లేదు, GPU లేదు)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection ఆగింది, task అయినా పూర్తి
```

allow-list మీద లేని action **నడవదు** — model-ను ఎంతగా ఒప్పించినా. ఇది classifier
కాదు; kernel లోపల అదే call path-లో నడిచే **fail-closed** capability floor.

---

## భారత్‌కు ఎందుకు ముఖ్యం

- **డేటా దేశంలోనే ఉంటుంది (DPDP Act, 2023).** fak self-host మొదట — ఒక static binary
  మీ local model లేదా దేశీయ provider ముందు కూర్చుని, ప్రతి tool call-ను నడవడానికి
  ముందు తనిఖీ చేస్తుంది (fail-closed default-deny floor). డేటా మీ మెషిన్‌ను వదిలి
  బయటకు వెళ్లదు.
- **ఖర్చు రూపాయల్లో (INR) కొడుతుంది, token bill డాలర్లలో వస్తుంది.** పొడవైన session-లో
  fak భాగస్వామ్య పనిని మళ్లీ వాడుకుంటుంది — ఒక tuned warm-cache stack-తో పోలిస్తే
  50-turn × 5-agent run మీద **~4.1× తక్కువ పని** (ఇదే నిజాయితీ అంకె; naive re-send
  loop-తో పోలిస్తే ~60×, కానీ అది *కేవలం* ఆ naive నమూనాతో పోలిక — headline కాదు). ఈ
  reuse-గెలుపు self-host-only, read-heavy fleet-లకే వర్తిస్తుంది.
- **ఏ payment rail దాటదు.** fak **Apache-2.0**, ఉచితం, self-host — card లేదు,
  cross-border invoice లేదు, legal entity లేదు. `git clone` మరియు `go install`
  మాత్రమే పూర్తి మార్గం.

అధిక వివరాలు [data residency & compliance](../../explainers/data-residency-and-compliance.md)-లో.

---

## తరువాత ఎక్కడికి

- [tutorial — దశలవారీ మొదటి session](https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/tutorial.md)
  (~15 నిమి; ప్రతి కమాండ్‌కు నిజమైన output, మొదట offline — key/GPU అవసరం లేదు)
- [README (పూర్తి అవలోకనం)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 నిమిషాల్లో local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — పూర్తి install & tier reference](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — మీ ఏజెంట్‌ను అనుసంధానించండి](../../integrations/README.md)
- [డేటా residency & compliance — DPDP Act, 2023 కోసం](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — ప్రతి సంఖ్యకు మూలం](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — ఏది shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE) — ఉచితం, self-host.
