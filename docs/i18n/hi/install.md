---
title: "fak — इंस्टॉल और शुरुआत (हिन्दी / Hindi install guide)"
description: "fak का हिन्दी इंस्टॉल-पृष्ठ: एक static Go बाइनरी को साफ़ checkout से चलते कर्नेल तक ले जाने वाले copy-paste कमांड — Tier 0 के लिए न GPU, न key, न network; DPDP-अनुकूल self-host।"
---

# fak — इंस्टॉल और शुरुआत

> यह एक **स्थानीयकृत प्रवेश-पृष्ठ (entry point)** है, पूरी दस्तावेज़ का अनुवाद नहीं।
> पूरी दस्तावेज़ अंग्रेज़ी में है — यह पृष्ठ आपको इंस्टॉल का रास्ता और tier का नक़्शा देकर
> आगे [अंग्रेज़ी डॉक्स](https://github.com/anthony-chaudhary/fak/blob/main/README.md) तक
> पहुँचाता है।
> **सूचना:** यह अनुवाद मशीन द्वारा तैयार है और native समीक्षा बाकी है — सुधार के लिए
> [issue/PR](https://github.com/anthony-chaudhary/fak) खोलें।
>
> हिन्दी में और: [परिचय / README](./README.md) · पूरी सूची [i18n hub](../README.md) पर।

यह **इंस्टॉल और चलाने का मुख्य द्वार** है। fak का सघन पिच [README](./README.md) में है —
यह पृष्ठ आपको साफ़ checkout से एक चलते कर्नेल तक, और उसके पीछे एक model serve करने तक ले
जाता है।

**fak एक ही Go बाइनरी है:** शून्य बाहरी dependency वाला एक static artifact (न Python, न CUDA)।
आप अपना एजेंट दोबारा नहीं लिखते — बस एक base URL को `fak serve` की ओर मोड़ते हैं, या मौजूदा
एजेंट को एक ही कमांड में wrap कर देते हैं:

```bash
fak manage claude
```

## 0. पूर्वापेक्षाएँ (Prerequisites)

- **Go 1.26+.** `fak/go.mod` में `go 1.26` घोषित है। Go के default `GOTOOLCHAIN=auto` के
  साथ, पुराना `go` पहली बार build पर सही toolchain अपने-आप डाउनलोड कर लेता है (एक बार
  network चाहिए); वरना Go 1.26 <https://go.dev/dl/> से इंस्टॉल करें। जाँच: `go version`।
- **Tier 0 के लिए बस इतना ही** — न GPU, न API key, न network।
- **Tier 1** के लिए इसके अलावा कोई भी OpenAI-compatible model server (जैसे Ollama) भी चाहिए।

## Tier का नक़्शा (setup की बढ़ती लागत के क्रम में)

एक ही बाइनरी से चार काम हो सकते हैं, और **इनके बीच कुछ नया इंस्टॉल नहीं होता**:

| Tier | आपको क्या मिलता है | Setup |
|---|---|---|
| **0 — कर्नेल आज़माएँ** | adjudication boundary को offline चलाएँ/मापें | `go build` |
| **1 — असली model के आगे रखें** | कर्नेल को कहीं और serve किए model (Ollama / vLLM / llama.cpp / cloud) के आगे रखें | + एक चलता OpenAI-compatible server |
| **1b — एक कमांड में local model** | मौजूदा एजेंट के साथ एक local GGUF model in-kernel चलाएँ — न key, न network, न दूसरा terminal | `fak manage --gguf qwen2.5:7b -- claude` |
| **2 — fused in-kernel model** | कर्नेल के अपने address space में चलने वाला pure-Go forward pass | + (असली weights) Python export |

अगर आप बस **fak को सामने रखकर एक उपयोगी model serve** करना चाहते हैं, तो आपको **Tier 1**
चाहिए। Tier 2 का in-kernel model एक *reference forward pass* है (HuggingFace के मुक़ाबले
bit-for-bit सिद्ध), production chat-quality engine नहीं।

## 1. बाइनरी पाएँ (Install)

**Clone से build करें (contributor):**

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak
./fak help
```

**Go से सीधे install करें** — module path `github.com/anthony-chaudhary/fak` ही repository
की जड़ है, तो यह सीधे resolve होता है:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

> **Windows सूचना.** `go build`/`go vet`/`go run` native चलते हैं। Windows पर
> `go build -o fak.exe ./cmd/fak` से build करें, और इस गाइड में जहाँ `./fak` लिखा है वहाँ
> `.\fak.exe` टाइप करें।

## 60-सेकंड का प्रमाण (कोई key नहीं, कोई model नहीं, कोई GPU नहीं)

`fak/` के अंदर से चलाएँ:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

पहली दो पंक्तियाँ **default-deny capability floor** को कर्नेल के अंदर, उसी call path पर,
fail-closed चलते हुए दिखाती हैं — allow-list पर न होने वाली action चल ही नहीं सकती। तीसरी पंक्ति
prompt injection को structure से दीवार के पीछे रखते हुए task पूरा करती है।

## भारत के लिए क्यों मायने रखता है

- **डेटा देश में ही रहे (DPDP Act, 2023)।** fak सबसे पहले self-host है: एक static बाइनरी जो
  किसी local model या घरेलू provider के आगे बैठती है और हर tool call को चलने से पहले
  जाँचती है (fail-closed default-deny floor)। डेटा आपकी मशीन से बाहर नहीं जाता।
- **लागत रुपये में चुभती है।** fak लंबी session में साझा काम दोबारा इस्तेमाल करता है —
  एक tuned warm-cache stack के मुक़ाबले 50×5 run पर **~4.1× कम काम** (यही ईमानदार आँकड़ा
  है; ~60× वाला आँकड़ा *सिर्फ़ naive re-send loop के मुक़ाबले* सच है, headline नहीं)। यह
  reuse-लाभ सिर्फ़ self-host पर मिलता है और read-heavy fleet पर लागू होता है — सीधे margin का लीवर।
- **कोई payment rail पार नहीं।** fak **Apache-2.0**, मुफ़्त, self-host है — न कार्ड,
  न cross-border invoice, न entity। `git clone` और `go install` ही पूरा रास्ता है।

## आगे कहाँ जाएँ

- [परिचय / README (हिन्दी)](./README.md) — सघन पिच और 60-सेकंड प्रमाण।
- [README (पूरा अवलोकन)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 मिनट में local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — विस्तृत इंस्टॉल reference + tutorial](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [डेटा residency और अनुपालन — DPDP Act के लिए](../../explainers/data-residency-and-compliance.md)
- [Integrations — अपने एजेंट को जोड़ें](../../integrations/README.md)
- [BENCHMARK-AUTHORITY — हर आँकड़े का स्रोत](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — क्या shipped/simulated/stub है](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)।
