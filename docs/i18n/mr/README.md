---
title: "fak — फ्यूज्ड एजंट कर्नल (मराठी परिचय / Marathi introduction)"
description: "fak चे मराठी प्रवेश-पान: प्रत्येक tool call चालण्याआधी तपासणारी Go binary — तोच agent loop अधिक सुरक्षित, स्वस्त, वेगवान; DPDP-अनुकूल self-host, Apache-2.0."
---

# fak — फ्यूज्ड एजंट कर्नल (मराठी परिचय)

> हे एक **स्थानिकीकृत प्रवेश-पान (entry point)** आहे — संपूर्ण दस्तऐवजांचे भाषांतर नाही.
> संपूर्ण दस्तऐवज इंग्रजीत आहेत — हे पान fak चा गाभा, 60-सेकंदांचा पुरावा आणि install
> चा मार्ग देऊन तुम्हाला [इंग्रजी डॉक्स](https://github.com/anthony-chaudhary/fak/blob/main/README.md)कडे पोहोचवते.
> **सूचना:** हे भाषांतर यंत्राने तयार केले असून native तपासणी बाकी आहे — दुरुस्तीसाठी
> issue/PR उघडा.
>
> **इतर भारतीय भाषांमध्ये:** [हिन्दी (Hindi)](../hi/README.md) ·
> [தமிழ் (Tamil)](../ta/README.md) · [తెలుగు (Telugu)](../te/README.md) ·
> [বাংলা (Bengali)](../bn/README.md) — पूर्ण यादी [i18n hub](../README.md) वर.

## एका ओळीत fak

**fak ही एक Go binary आहे** जी तुमच्या AI एजंट आणि त्याच्या tool calls च्या मध्ये बसते —
प्रत्येक tool call *चालण्याआधीच* तपासते, आणि लांब session मध्ये पुन्हा-पुन्हा येणारे काम
पुनर्वापरते. परिणाम: तोच agent loop **अधिक सुरक्षित, स्वस्त आणि वेगवान** — बाकी काहीही
न बदलता.

तुम्ही तुमचा एजंट पुन्हा लिहीत नाही — फक्त एक base URL `fak serve` कडे वळवता, आणि
प्रत्येक tool call आधी capability floor मधून जाते.

```bash
fak guard -- claude    # तुमच्या सध्याच्या एजंटला एका कमांडमध्ये wrap करते
```

## भारतीय स्टार्टअप्ससाठी हे का महत्त्वाचे

- **खर्च रुपयांत बोचतो, token bill डॉलरमध्ये येते.** लांब session मध्ये सामायिक काम
  (system prompt, tool list चा KV cache) fak पुनर्वापरते — एका tuned warm-cache
  stack च्या तुलनेत 50×5 run वर **~4.1× कमी काम** (naive re-send loop च्या तुलनेत
  ~60×; प्रामाणिक आकडा 4.1× आहे). शिवाय per-aspect routing स्वस्त भाग स्वस्त model
  कडे पाठवते. हा थेट margin चा लिव्हर आहे.
- **डेटा देशातच राहावा (DPDP Act, 2023).** fak self-host-first आहे: एक static binary
  कोणत्याही **local model** किंवा देशी provider समोर बसते — प्रत्येक backend वर
  fail-closed residency, default-deny capability floor, आणि प्रत्येक tool call चा
  tamper-evident audit log. डेटा तुमच्या मशीनबाहेर जात नाही.
- **कोणतीही payment rail ओलांडावी लागत नाही.** fak **Apache-2.0**, मोफत, self-host —
  कार्ड नाही, cross-border invoice नाही, entity नाही. `git clone` आणि `go install`
  हाच पूर्ण मार्ग.
- **एकच static binary, शून्य बाह्य dependency.** लहान टीमसाठी सोपे ops — sidecar
  नाही, वेगळा authorizer नाही. laptop पासून fleet पर्यंत तोच artifact; तुम्ही
  components नव्हे, फक्त flags जोडता.

## fak कोणत्या समस्या सोडवते

- **लांब session महाग होणे थांबते.** provider ची prompt-cache सूट तेव्हाच टिकते जेव्हा
  cached prefix byte-for-byte तोच राहतो; fak मधले जुने turns काढूनही prefix
  byte-identical ठेवते — त्यामुळे सूट तुटत नाही.
- **default-deny सुरक्षा.** permission धोरण kernel च्या *आत*, त्याच call path वर चालते.
  एखादी irreversible action थांबवणे attack "पकडण्यावर" अवलंबून नाही — तो लिव्हर कधी
  जोडलाच गेला नव्हता. हे **fail-closed** आहे, open नाही.
- **prompt injection / poisoned results ला प्रतिबंध.** संशयास्पद tool *results* वेगळ्या
  quarantine मध्ये ठेवले जातात, म्हणजे ते model च्या context मध्ये शिरूच शकत नाहीत —
  structure द्वारे, कोणत्याही classifier द्वारे नव्हे.

## 60-सेकंदांचा पुरावा (key नाही, model नाही, GPU नाही)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection थांबवले, task तरीही पूर्ण
```

## तुमच्या model सोबत

fak तुमचे model बदलत नाही — त्याला govern आणि cache करते. **Qwen2/Qwen3 आणि GLM-MoE**
in-kernel reference engine मध्ये bit-exact सिद्ध आहेत; बाकी सर्व (DeepSeek, Mistral,
कोणतेही open-weights model) OpenAI-compatible wire वर front होतात — Ollama / vLLM /
SGLang / llama.cpp / LM Studio किंवा कोणत्याही OpenAI-compatible API द्वारे.

## पुढे कुठे

- [README (पूर्ण आढावा)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 मिनिटांत local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — binary install करा](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — तुमचा एजंट जोडा](../../integrations/README.md)
- [डेटा residency आणि compliance — DPDP Act साठी](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — प्रत्येक आकड्याचा स्रोत](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — काय shipped/simulated/stub आहे](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
