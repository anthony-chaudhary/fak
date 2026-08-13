---
title: "fak — మొదటి పరిచయపు తరచు ప్రశ్నలు (తెలుగు FAQ / Telugu FAQ)"
description: "fak గురించి మొదటిసారి అడిగే ప్రశ్నలకు సంక్షిప్త సమాధానాలు: ప్రతి tool call-ను నడిచే ముందే తనిఖీ చేసే ఒకే Go binary — అదే agent loop మరింత సురక్షితం, చౌక, వేగం; DPDP-అనుకూల self-host."
---

# fak — మొదటి పరిచయపు తరచు ప్రశ్నలు (FAQ)

> ఇది ఒక **స్థానికీకరించిన ప్రవేశ పేజీ (entry point)** — పూర్తి డాక్యుమెంటేషన్ అనువాదం
> కాదు. పూర్తి డాక్యుమెంటేషన్ ఇంగ్లీషులో ఉంది — ఈ పేజీ మొదటిసారి అడిగే ప్రశ్నలకు
> సంక్షిప్త సమాధానాలు ఇచ్చి, మిమ్మల్ని
> [ఇంగ్లీష్ డాక్స్](https://github.com/anthony-chaudhary/fak/blob/main/README.md) వైపు
> తీసుకెళ్తుంది.
> **గమనిక:** ఈ అనువాదం యంత్రంతో తయారైనది; native సమీక్ష పెండింగ్‌లో ఉంది — సవరణల
> కోసం issue/PR తెరవండి.
>
> **ఇతర భాషలు & పేజీలు:** [i18n hub](../README.md) · [ఈ భాషలో పరిచయం (README)](./README.md).

## Q1. fak అంటే ఏమిటి?

మీరు ఇప్పటికే నడుపుతున్న AI ఏజెంట్ (Claude Code, Codex, Cursor, ఏ
OpenAI/Anthropic/MCP client అయినా) ముందు కూర్చునే **ఒకే static Go binary** —
ఒక్క base URL-ను తిప్పితే చాలు, ఏజెంట్‌ను తిరిగి రాయనక్కర్లేదు. ఇది పొడవైన
session-లను చౌక చేస్తుంది (పాత turns-ను తీసేస్తూ provider prompt-cache prefix-ను
byte-identical-గా ఉంచుతుంది), ప్రతి tool call-ను route చేస్తుంది, local GGUF
models-ను in-process నడపగలదు, మరియు ప్రతి call-కు audit చేయదగ్గ verdict-ను
నమోదు చేస్తుంది.

## Q2. నేను models మార్చాలా, లేదా నా ఏజెంట్‌ను తిరిగి రాయాలా?

అవసరం లేదు. మీరు ఇప్పటికే వాడుతున్న model-నే fak govern చేసి cache చేస్తుంది.
ఒక్క కమాండ్‌తో wrap చేయండి:

```bash
fak manage claude
```

లేదా ఒక base URL-ను `fak serve` వైపు తిప్పండి.

## Q3. నా డేటా ఎక్కడికి వెళ్తుంది — ఇది compliant అవుతుందా?

**Self-host ముందు:** ఒక static binary ఒక local model లేదా దేశీయ provider ముందు
కూర్చుంటుంది — fail-closed residency, default-deny capability floor, ప్రతి tool
call-కు tamper-evident audit log. మీ డేటా మీ మెషీన్ నుంచి బయటకు వెళ్ళదు. ఇది
**DPDP Act, 2023** (భారత డేటా-నిల్వ చట్టం) ప్రకారం డేటాను దేశంలోనే ఉంచాలనే
అవసరానికి నేరుగా సరిపోతుంది.

## Q4. దీని ఖర్చు ఎంత? నిజంగా ఉచితమేనా?

**Apache-2.0**, ఉచితం, self-host. కార్డ్ లేదు, cross-border invoice లేదు, entity
లేదు — అంటే రూపాయల్లో ఎలాంటి subscription outflow లేదు. `git clone` మరియు
`go install` — ఇదే మొత్తం దారి.

## Q5. ఇది ఎంత చౌక లేదా వేగం?

token bill డాలర్లలో వస్తుంది కానీ margin రూపాయల్లో బాధిస్తుంది — అక్కడే fak
మీట. కొలిచిన 50-turn × 5-agent session-లో, ఒక **tuned warm-cache stack-తో
పోలిస్తే ~4.1× తక్కువ పని**. ~60× అనే సంఖ్య (సుమారు 19 గంటల నుంచి సుమారు 19
నిమిషాలకు) **naive re-send-everything loop-తో పోలిస్తే మాత్రమే** నిజం — దాన్ని
ఎప్పుడూ headline-గా చూపించకూడదు. ఈ reuse లాభం self-host మాత్రమే, read-heavy
fleet-ల కోసం.

## Q6. ఏ models పనిచేస్తాయి?

**Qwen2/Qwen3, GLM-MoE** in-kernel reference engine-లో bit-exact అని
నిరూపితమయ్యాయి. మిగతావన్నీ (DeepSeek, Mistral, ఏ open-weights model అయినా)
OpenAI-compatible wire మీద front అవుతాయి — Ollama / vLLM / SGLang / llama.cpp /
LM Studio.

## Q7. ఇది prompt injection-ను ఎలా ఆపుతుంది?

classifier కాదు, రెండు structural gates: **default-deny capability floor**
(ప్రమాదకర tool ఎప్పుడూ allow-list-లో ఉండదు) మరియు **result quarantine**
(విషపూరిత tool results model context-లోకి అసలు చేరవు). Live tests-లో injection
రక్షణ లేని baseline-ను 5/5 చేరింది; fak దాన్ని 5/5 అడ్డుకుంది.

## Q8. దీన్ని ఎలా install చేయాలి?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

లేదా clone నుంచి `go build -o fak ./cmd/fak`. (module fetch నెమ్మదిగా ఉంటే
`GOPROXY`-ను set చేసుకోవచ్చు.)

### 60-సెకన్ల రుజువు (key లేదు, model లేదు, GPU లేదు)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Q9. తరువాత ఎక్కడికి వెళ్ళాలి?

- [README (పూర్తి అవలోకనం)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 నిమిషాల్లో local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — binary install చేయండి](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — ప్రతి సంఖ్యకు మూలం](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — ఏది shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [డేటా residency & compliance — DPDP Act, 2023 కోసం](../../explainers/data-residency-and-compliance.md)
- [Integrations — మీ ఏజెంట్‌ను అనుసంధానించండి](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
