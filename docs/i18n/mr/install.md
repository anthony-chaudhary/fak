---
title: "fak — install आणि सुरुवात (मराठी / Marathi getting started)"
description: "fak चे मराठी install-पान: स्वच्छ checkout पासून चालणाऱ्या kernel पर्यंत, आणि त्याच्यामागे एक model सर्व्ह करण्यापर्यंत — copy-paste करता येणाऱ्या commands, Go 1.26+, Tier 0 साठी GPU/key/network नाही."
---

# fak — install आणि सुरुवात

> हे एक **स्थानिकीकृत प्रवेश-पान (entry point)** आहे — संपूर्ण दस्तऐवजांचे भाषांतर नाही.
> संपूर्ण दस्तऐवज इंग्रजीत आहेत. हे पान fak install करून चालवण्याचा मार्ग देते; दाट
> ओळख (pitch) [README](https://github.com/anthony-chaudhary/fak/blob/main/README.md) मध्ये आहे.
> **सूचना:** हे भाषांतर यंत्राने तयार केले असून native तपासणी बाकी आहे — दुरुस्तीसाठी
> [issue/PR](https://github.com/anthony-chaudhary/fak) उघडा. इतर भाषांसाठी
> [i18n hub](../README.md), आणि सत्याचा मूळ स्रोत
> [इंग्रजी README](https://github.com/anthony-chaudhary/fak/blob/main/README.md).

हे install-आणि-चालवण्याचे दार आहे. दाट pitch [`README`](./README.md) मध्ये; झटपट सुरुवात
हवी असल्यास [quickstart](./quickstart.md) पाहा. हे पान तुम्हाला स्वच्छ checkout पासून
चालणाऱ्या kernel पर्यंत, आणि त्याच्यामागे एक model सर्व्ह करण्यापर्यंत नेते.

`fak` ही **एकच Go binary** आहे — शून्य बाह्य dependency असलेला एक static artifact (Python
नाही, CUDA toolchain नाही). तीच binary म्हणजे संपूर्ण serving पृष्ठभाग: gateway,
KV-cache आणि routing engine, token-savers — आणि त्याच seam वर policy gate, result
quarantine आणि audit/metrics — एकाच process मध्ये.

तुम्ही तुमचा एजंट पुन्हा लिहीत नाही — फक्त एक base URL `fak serve` कडे वळवता, किंवा सध्याचा
एजंट एका कमांडमध्ये wrap करता:

```bash
fak manage claude
```

## 0. पूर्वअटी (Prerequisites)

- **Go 1.26+.** `fak/go.mod` मध्ये `go 1.26` घोषित आहे. Go च्या default
  `GOTOOLCHAIN=auto` सह, जुनी `go` पहिल्या build वेळी योग्य toolchain आपोआप download
  करते (एकदा network लागते); अन्यथा <https://go.dev/dl/> वरून Go 1.26 install करा.
  `go version` ने तपासा.
- **Tier 0 आणि Tier 2-synthetic साठी एवढेच** — GPU नाही, API key नाही, network नाही.
- **Tier 1** ला याशिवाय कोणताही OpenAI-compatible model server लागतो (उदा. Ollama).
- **Tier 2 खऱ्या weights सह** ला याशिवाय **Python 3.10+** लागते (fetch script venv तयार
  करून `torch`/`transformers` install करते).

## Tier नकाशा

चढत्या setup-खर्चाच्या क्रमाने चार गोष्टी करता येतात, आणि **त्यांच्यामध्ये नवीन काहीही
install होत नाही**:

| Tier | काय मिळते | Setup |
|---|---|---|
| **0 — kernel वापरून पाहा** | adjudication boundary offline चालवा/मोजा | `go build` |
| **1 — खरा model समोर ठेवा** | दुसरीकडे सर्व्ह केलेल्या model समोर kernel ठेवा (Ollama / vLLM / llama.cpp / cloud) | + एक चालू OpenAI-compatible server |
| **1b — एका कमांडमध्ये local model** | तुमच्या सध्याच्या एजंटसह local GGUF model in-kernel चालवा — key नाही, network नाही | `fak guard --gguf qwen2.5:7b -- claude` |
| **2 — fused in-kernel model** | kernel च्या मालकीचा pure-Go forward pass | + (खऱ्या weights साठी) Python export |

फक्त **fak समोर ठेवून एक उपयुक्त model सर्व्ह** करायचे असेल, तर तुम्हाला **Tier 1** हवे.
Tier 2 चा in-kernel model हा HuggingFace विरुद्ध bit-for-bit सिद्ध *reference forward
pass* आहे — production chat-quality serving engine नव्हे.

## Install commands

**Clone मधून build करा (contributor):**

```bash
go build -o fak ./cmd/fak
```

> **Windows टीप:** `go build -o fak.exe ./cmd/fak` ने build करा — `-o fak` (extension
> नसलेले) असे नाव ठेवते जे cmd.exe / PowerShell थेट चालवू शकत नाही. मग या मार्गदर्शकात
> जिथे `./fak` आहे तिथे `.\fak.exe` वापरा.

**Go ने install करा (adopter).** module path `github.com/anthony-chaudhary/fak` हाच
repository root असल्याने तो थेट install होतो:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

कार्ड नाही, cross-border invoice नाही, entity नाही — fak **Apache-2.0**, मोफत, self-host
आहे. `git clone` आणि `go install` हाच पूर्ण मार्ग; खर्च रुपयांतच राहतो — self-host असल्याने
कोणतेही परकीय-चलन (forex) बिल नाही. डेटा तुमच्या मशीनवरच राहतो हे
भारतातील **DPDP Act, 2023** अनुपालनासाठी महत्त्वाचे; तपशील खालील data-residency दुव्यात.

## 60-सेकंदांचा पुरावा (key नाही, model नाही, GPU नाही)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## पुढे कुठे

- [README (मराठी परिचय)](./README.md) · [quickstart (मराठी)](./quickstart.md)
- [README (पूर्ण आढावा, इंग्रजी)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 मिनिटांत local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — मार्गदर्शित पहिली session](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — तुमचा एजंट जोडा](../../integrations/README.md)
- [डेटा residency आणि compliance — DPDP Act, 2023 साठी](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — प्रत्येक आकड्याचा स्रोत](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — काय shipped/simulated/stub आहे](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
