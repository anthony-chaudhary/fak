---
title: "fak — త్వరిత ప్రారంభం (10 నిమిషాల్లో ఒక local model)"
description: "fak తెలుగు త్వరిత-ప్రారంభ పేజీ: సున్నా నుంచి governed local AI వరకు ~10 నిమిషాల్లో — offline, key లేదు, cloud bill లేదు, డేటా మీ మెషీన్‌లోనే. DPDP-అనుకూల self-host."
---

# fak — త్వరిత ప్రారంభం (10 నిమిషాల్లో ఒక local model)

> ఇది ఒక **స్థానికీకరించిన ప్రవేశ పేజీ (entry point)** — పూర్తి డాక్యుమెంటేషన్ అనువాదం
> కాదు. పూర్తి డాక్యుమెంటేషన్ ఇంగ్లీషులో ఉంది. ఈ పేజీ యంత్రంతో తయారైనది; native
> సమీక్ష పెండింగ్‌లో ఉంది — సవరణల కోసం issue/PR తెరవండి.
>
> **మరిన్ని:** [i18n hub](../README.md) · [fak తెలుగు పరిచయం](./README.md) ·
> [ఇంగ్లీష్ మూలాధారం](https://github.com/anthony-chaudhary/fak/blob/main/README.md)

## వాగ్దానం

ఈ దశలు అనుసరించాక, మీ సొంత మెషీన్‌లో నడిచే **governed local AI** మీ చేతిలో ఉంటుంది —
ఇది **offline** పని చేస్తుంది, ఏ **key అవసరం లేదు**, ఏ **cloud bill రాదు**, **డేటా మీ
మెషీన్‌లోనే** ఉంటుంది, చిన్న models కోసం **CPU సరిపోతుంది** — GPU అవసరం లేదు. మొత్తం
దారి సుమారు **10 నిమిషాలు**.

భారత్‌లో ఇది సూటిగా margin మీట: token bill డాలర్లలో వస్తుంది కానీ ఖర్చు రూపాయల్లో
బాధిస్తుంది; self-host అయిన fak వద్ద cross-border invoice లేదు, entity లేదు —
[DPDP Act, 2023](../../explainers/data-residency-and-compliance.md) ప్రకారం డేటా
దేశంలోనే ఉంటుంది.

## వేగవంతమైన మార్గం

మీ ఇప్పటి coding agent వెనుక ఒక local model-ను ఒకే కమాండ్‌లో అమర్చండి — ఏజెంట్‌ను
తిరిగి రాయనక్కర్లేదు:

```bash
fak manage claude
```

## 60-సెకన్ల రుజువు (key లేదు, model లేదు, GPU లేదు)

ఏ download-ఉ లేకుండా tool-call సరిహద్దును ఇప్పుడే చూడండి — ఒక structural DENY, ఒక
ALLOW, మరియు offline-లో ఆగిపోయిన injection:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## fak అంటే ఏమిటి

**fak ఒక static Go binary** — ఇది మీ AI ఏజెంట్‌కు, అది పిలిచే tools-కు మధ్య
కూర్చుంటుంది. ప్రతి tool call *నడిచే ముందే* దాన్ని తనిఖీ చేస్తుంది, పొడవైన session-లలో
పునరావృతమయ్యే పంచుకున్న పనిని తిరిగి వాడుకుంటుంది. ఫలితం: అదే agent loop **మరింత
సురక్షితం, చౌక, వేగం** — ఇంకేమీ మార్చకుండా. మీరు ఏజెంట్‌ను తిరిగి రాయరు; ఒక base
URL-ను `fak serve` వైపు తిప్పుతారు, లేదా ఒకే కమాండ్‌లో `fak manage claude`.

సురక్షత structure ద్వారానే వస్తుంది, classifier ద్వారా కాదు: kernel *లోపల*, అదే call
path-లో నడిచే **default-deny capability floor** (fail-closed) — allow-list-లో ఎప్పుడూ
లేని action, model-ను ఎంత మభ్యపెట్టినా, నడవలేదు. అనుమానాస్పద tool *results* విడి
quarantine-లో ఉంచబడతాయి, model context-లోకి అసలు రావు.

## ఎంత వేగం (నిజాయితీగా)

కొలిచిన **50-turn × 5-agent session** మీద, ఒక *tuned* warm-cache stack-తో పోలిస్తే
నిజాయితీగల headline **~4.1× తక్కువ పని**. **naive re-send-everything loop**-తో
**మాత్రమే** పోలిస్తే అదే session ~19 గంటల నుంచి ~19 నిమిషాలకు (~60×) పడిపోతుంది — కానీ
ఆ ~60× సంఖ్య naive pattern-తో పోలిస్తేనే నిజం; అది headline కాదు. ఈ reuse లాభం
**self-host మాత్రమే**, read-heavy fleets-కు వర్తిస్తుంది.

provider-ల prompt-cache డిస్కౌంట్, cached prefix byte-for-byte అలాగే ఉంటేనే
నిలుస్తుంది; fak మధ్యలోని పాత turns-ను తీసేసినా prefix-ను **byte-identical**-గా
ఉంచుతుంది. prefix byte-identity-ను fak హామీ ఇస్తుంది; provider ఆ cache-ను నిజంగా
తిరిగి వాడతాడా అనేది provider నిర్ణయం — fak దాన్ని relay చేస్తుంది, claim చేయదు.

ప్రతి సంఖ్య దాని commit, artifact వరకూ traced —
[BENCHMARK-AUTHORITY](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
చూడండి.

## మీ model-తో

fak మీ model-ను మార్చదు — దాన్ని govern చేసి cache చేస్తుంది. **Qwen2/Qwen3, GLM-MoE**
in-kernel reference engine-లో bit-exact అని నిరూపితమయ్యాయి; మిగతావన్నీ (DeepSeek,
Mistral, ఏ open-weights model అయినా) OpenAI-compatible wire మీద front అవుతాయి —
Ollama / vLLM / SGLang / llama.cpp / LM Studio లేదా ఏ OpenAI-compatible API ద్వారా
అయినా.

## తరువాత ఎక్కడికి

- [README (పూర్తి అవలోకనం)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 నిమిషాల్లో local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — binary install చేయండి](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — ప్రతి సంఖ్యకు మూలం](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — ఏది shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [డేటా residency & compliance — DPDP Act కోసం](../../explainers/data-residency-and-compliance.md)
- [Integrations — మీ ఏజెంట్‌ను అనుసంధానించండి](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE) — ఉచితం, self-host.
