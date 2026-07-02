---
title: "fak — ఫ్యూజ్డ్ ఏజెంట్ కెర్నల్ (తెలుగు పరిచయం / Telugu introduction)"
description: "fak తెలుగు ప్రవేశ పేజీ: ప్రతి tool call-ను నడిచే ముందే తనిఖీ చేసే Go binary — అదే agent loop మరింత సురక్షితం, చౌక, వేగం; DPDP-అనుకూల self-host."
---

# fak — ఫ్యూజ్డ్ ఏజెంట్ కెర్నల్ (తెలుగు పరిచయం)

> ఇది ఒక **స్థానికీకరించిన ప్రవేశ పేజీ (entry point)** — పూర్తి డాక్యుమెంటేషన్ అనువాదం
> కాదు. పూర్తి డాక్యుమెంటేషన్ ఇంగ్లీషులో ఉంది — ఈ పేజీ fak సారాంశం, 60-సెకన్ల రుజువు,
> install మార్గం ఇచ్చి, మిమ్మల్ని [ఇంగ్లీష్ డాక్స్](https://github.com/anthony-chaudhary/fak/blob/main/README.md) వైపు
> తీసుకెళ్తుంది.
> **గమనిక:** ఈ అనువాదం యంత్రంతో తయారైనది; native సమీక్ష పెండింగ్‌లో ఉంది — సవరణల
> కోసం issue/PR తెరవండి.
>
> **ఇతర భారతీయ భాషల్లో:** [हिन्दी (Hindi)](../hi/README.md) ·
> [தமிழ் (Tamil)](../ta/README.md) · [বাংলা (Bengali)](../bn/README.md) ·
> [मराठी (Marathi)](../mr/README.md) — పూర్తి జాబితా [i18n hub](../README.md)-లో.

## fak ఒక్క వాక్యంలో

**fak ఒక Go binary** — ఇది మీ AI ఏజెంట్‌కు, దాని tool calls-కు మధ్య కూర్చుంటుంది —
ప్రతి tool call-ను *నడవక ముందే* తనిఖీ చేస్తుంది, పొడవైన session-లలో పునరావృతమయ్యే
పనిని తిరిగి వాడుకుంటుంది. ఫలితం: అదే agent loop **మరింత సురక్షితం, చౌక, వేగం** —
ఇంకేమీ మార్చకుండా.

మీ ఏజెంట్‌ను తిరిగి రాయనక్కర్లేదు — ఒక base URL-ను `fak serve` వైపు తిప్పితే చాలు;
ప్రతి tool call ముందుగా capability floor గుండా వెళ్తుంది.

```bash
fak guard -- claude    # మీ ఇప్పటి ఏజెంట్‌ను ఒకే కమాండ్‌లో wrap చేస్తుంది
```

## భారతీయ స్టార్టప్‌లకు ఇది ఎందుకు ముఖ్యం

- **ఖర్చు రూపాయల్లో బాధిస్తుంది, token bill డాలర్లలో వస్తుంది.** పొడవైన session-లలో
  పంచుకున్న పనిని (system prompt, tool list-ల KV cache) fak తిరిగి వాడుకుంటుంది —
  ఒక tuned warm-cache stack-తో పోలిస్తే 50×5 run-లో **~4.1× తక్కువ పని** (naive
  re-send loop-తో పోలిస్తే ~60×; నిజాయితీగల సంఖ్య 4.1×). దానితో పాటు per-aspect
  routing చౌక భాగాలను చౌక model-కు పంపుతుంది. ఇది నేరుగా margin-ను పెంచే మీట.
- **డేటా దేశంలోనే ఉండాలి (DPDP Act, 2023).** fak self-host ముందు: ఒక static binary
  ఏ **local model** లేదా దేశీయ provider ముందైనా కూర్చుని, ప్రతి backend-లో
  fail-closed residency, default-deny capability floor, ప్రతి tool call-కు
  tamper-evident audit log ఇస్తుంది. డేటా మీ మెషీన్ నుంచి బయటకు వెళ్ళదు.
- **ఏ payment rail-నూ దాటనక్కర్లేదు.** fak **Apache-2.0**, ఉచితం, self-host — కార్డ్
  లేదు, cross-border invoice లేదు, entity లేదు. `git clone`, `go install` — ఇదే
  మొత్తం దారి.
- **ఒకే static binary, సున్నా బాహ్య dependencies.** చిన్న టీమ్‌కు సులభమైన ops —
  sidecar లేదు, విడి authorizer లేదు. laptop నుంచి fleet వరకూ అదే artifact; మీరు
  components కాదు, flags మాత్రమే జోడిస్తారు.

## fak ఏ సమస్యలను పరిష్కరిస్తుంది

- **పొడవైన session-లు ఖరీదవడం ఆగిపోతుంది.** provider-ల prompt-cache డిస్కౌంట్,
  cached prefix byte-for-byte అలాగే ఉంటేనే నిలుస్తుంది; fak మధ్యలోని పాత turns-ను
  తీసేసినా prefix-ను byte-identical-గా ఉంచుతుంది — కాబట్టి డిస్కౌంట్ విరగదు.
- **default-deny భద్రత.** permission విధానం kernel *లోపల*, అదే call path-లో
  నడుస్తుంది. ఒక irreversible action-ను ఆపడం attack-ను "పట్టుకోవడం"పై ఆధారపడదు —
  ఆ మీట అసలు కలపబడనే లేదు. ఇది **fail-closed**, open కాదు.
- **prompt injection / poisoned results నివారణ.** అనుమానాస్పద tool *results*-ను
  విడి quarantine-లో ఉంచుతారు — అవి model context-లోకి అసలు ప్రవేశించలేవు; ఇది
  structure ద్వారా, ఏ classifier ద్వారానో కాదు.

## 60-సెకన్ల రుజువు (key లేదు, model లేదు, GPU లేదు)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection ఆగింది, task అయినా పూర్తి
```

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
- [Integrations — మీ ఏజెంట్‌ను అనుసంధానించండి](../../integrations/README.md)
- [డేటా residency & compliance — DPDP Act కోసం](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — ప్రతి సంఖ్యకు మూలం](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — ఏది shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
