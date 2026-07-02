---
title: "fak — फ़्यूज़्ड एजेंट कर्नेल (हिन्दी परिचय / Hindi introduction)"
description: "fak का हिन्दी प्रवेश-पृष्ठ: हर tool call को चलने से पहले जाँचने वाली Go बाइनरी — वही एजेंट loop ज़्यादा सुरक्षित, सस्ता, तेज़; DPDP-अनुकूल self-host।"
---

# fak — फ़्यूज़्ड एजेंट कर्नेल (हिन्दी परिचय)

> यह एक **स्थानीयकृत प्रवेश-पृष्ठ (entry point)** है, पूरी दस्तावेज़ का अनुवाद नहीं।
> पूरी दस्तावेज़ अंग्रेज़ी में है — यह पृष्ठ आपको fak का सार, 60-सेकंड का प्रमाण, और
> इंस्टॉल का रास्ता देकर आगे [अंग्रेज़ी डॉक्स](../../../README.md) तक पहुँचाता है।
> **सूचना:** यह अनुवाद मशीन द्वारा तैयार है और native समीक्षा बाकी है — सुधार के लिए
> issue/PR खोलें।
>
> **अन्य भारतीय भाषाओं में:** [தமிழ் (Tamil)](../ta/README.md) ·
> [తెలుగు (Telugu)](../te/README.md) · [বাংলা (Bengali)](../bn/README.md) ·
> [मराठी (Marathi)](../mr/README.md) — पूरी सूची [i18n hub](../README.md) पर।

## fak एक पंक्ति में

**fak एक Go बाइनरी है** जो आपके AI एजेंट और उसके tool calls के बीच बैठती है — हर tool
call को *चलने से पहले* जाँचती है, और लंबी session में दोहराए जाने वाले काम को दोबारा
इस्तेमाल करती है। नतीजा: वही एजेंट loop **ज़्यादा सुरक्षित, सस्ता और तेज़**, बिना कुछ
बदले।

आप अपना एजेंट दोबारा नहीं लिखते — बस एक base URL को `fak serve` की ओर मोड़ देते हैं, और
हर tool call पहले capability floor से गुज़रती है।

```bash
fak guard -- claude    # आपके मौजूदा एजेंट को एक ही कमांड में wrap कर देता है
```

## भारतीय स्टार्टअप्स के लिए यह क्यों मायने रखता है

- **लागत रुपये में चुभती है, टोकन बिल डॉलर में आता है।** fak लंबी sessions में साझा काम
  (system prompt, tool list का KV cache) दोबारा इस्तेमाल करता है — एक tuned warm-cache
  stack के मुक़ाबले 50×5 run पर **~4.1× कम काम** (naive re-send loop के मुक़ाबले ~60×,
  पर ईमानदार आँकड़ा 4.1× वाला है)। साथ ही per-aspect routing सस्ते हिस्सों को सस्ते model
  पर भेजता है। यह सीधे margin का लीवर है।
- **डेटा देश में ही रहे (DPDP Act, 2023)।** fak पहले self-host है: एक static बाइनरी जो
  किसी **local model** या घरेलू provider के आगे बैठती है, हर backend पर fail-closed
  residency, default-deny capability floor, और हर tool call का tamper-evident audit log
  देती है। डेटा आपकी मशीन से बाहर नहीं जाता।
- **कोई payment rail पार नहीं करना।** fak **Apache-2.0**, मुफ़्त, self-host है — न कार्ड,
  न cross-border invoice, न entity। `git clone` और `go install` ही पूरा रास्ता है।
- **एक static बाइनरी, शून्य बाहरी dependency।** छोटी टीम के लिए आसान ops — कोई sidecar,
  कोई अलग authorizer नहीं। laptop से fleet तक वही artifact; आप components नहीं, सिर्फ़
  flags जोड़ते हैं।

## fak किन समस्याओं को हल करता है

- **लंबी session महँगी होना बंद।** provider का prompt-cache छूट तभी टिकती है जब cached
  prefix byte-for-byte वही रहे; fak बीच के पुराने turns को हटाकर भी prefix को byte-identical
  रखता है, तो छूट टूटती नहीं।
- **default-deny सुरक्षा।** permission नीति kernel के *अंदर*, उसी call path पर चलती है।
  किसी irreversible action को रोकना attack "पकड़ने" पर निर्भर नहीं — वह lever कभी जुड़ा ही
  नहीं था। यह **fail-closed** है, open नहीं।
- **prompt injection / poisoned results का रोकथाम।** संदिग्ध tool *results* को एक अलग
  quarantine में रखा जाता है ताकि वे model के context में घुसें ही नहीं — structure से,
  किसी classifier से नहीं।

## 60-सेकंड का प्रमाण (कोई key नहीं, कोई model नहीं, कोई GPU नहीं)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection रोका, task फिर भी पूरा
```

## अपने model के साथ

fak आपका model नहीं बदलता — उसे govern और cache करता है। **Qwen2/Qwen3 और GLM-MoE**
in-kernel reference engine में bit-exact सिद्ध हैं; बाक़ी सब (DeepSeek, Mistral, कोई भी
open-weights model) OpenAI-compatible wire पर front होते हैं — Ollama / vLLM / SGLang /
llama.cpp / LM Studio या किसी भी OpenAI-compatible API के ज़रिए।

## आगे कहाँ जाएँ

- [README (पूरा अवलोकन)](../../../README.md)
- [START-HERE — 10 मिनट में local model](../../../START-HERE.md)
- [Getting Started — बाइनरी इंस्टॉल करें](../../../GETTING-STARTED.md)
- [Integrations — अपने एजेंट को जोड़ें](../../integrations/README.md)
- [डेटा residency और अनुपालन — DPDP Act के लिए](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — हर आँकड़े का स्रोत](../../../BENCHMARK-AUTHORITY.md)
- [CLAIMS — क्या shipped/simulated/stub है](../../../CLAIMS.md)

License: [Apache-2.0](../../../LICENSE)।
