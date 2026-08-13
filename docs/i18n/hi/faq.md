---
title: "fak — पहली मुलाक़ात के आम सवाल (FAQ, हिन्दी / Hindi FAQ)"
description: "fak के सबसे आम पहले सवालों के छोटे, ईमानदार जवाब: यह क्या है, model बदलना पड़ता है या नहीं, डेटा कहाँ रहता है (DPDP Act, 2023), लागत रुपये में, और prompt injection कैसे रुकता है।"
---

# fak — पहली मुलाक़ात के आम सवाल (FAQ)

> यह एक **स्थानीयकृत प्रवेश-पृष्ठ (entry point)** है, पूरी दस्तावेज़ का अनुवाद नहीं।
> पूरी दस्तावेज़ अंग्रेज़ी में है — यह पृष्ठ सबसे आम पहले सवालों के छोटे जवाब देकर आपको आगे
> [अंग्रेज़ी डॉक्स](https://github.com/anthony-chaudhary/fak/blob/main/README.md) तक पहुँचाता है।
> **सूचना:** यह अनुवाद मशीन द्वारा तैयार है और native समीक्षा बाकी है — सुधार के लिए
> issue/PR खोलें।
>
> इस भाषा में और: [हिन्दी परिचय (README)](./README.md) · पूरी सूची [i18n hub](../README.md) पर।

## Q1. fak है क्या?

एक ही static Go बाइनरी, जिसे आप अपने पहले से चल रहे AI एजेंट (Claude Code, Codex, Cursor,
कोई भी OpenAI/Anthropic/MCP client) के **आगे** रख देते हैं — बस एक base URL मोड़कर, बिना कुछ
दोबारा लिखे। यह लंबी sessions को सस्ता बनाती है (पुराने turns हटाती है पर provider का
prompt-cache prefix byte-identical रखती है), हर tool call को route करती है, ज़रूरत पड़े तो
local GGUF models in-process चला सकती है, और हर call का एक auditable verdict दर्ज करती है।

## Q2. क्या मुझे model बदलना या एजेंट दोबारा लिखना होगा?

नहीं। fak आपके पहले से इस्तेमाल हो रहे model को ही govern और cache करता है। इसे इस तरह
wrap करें:

```bash
fak manage claude
```

या एक base URL को `fak serve` की ओर मोड़ दें।

## Q3. मेरा डेटा कहाँ जाता है — क्या यह अनुपालक (compliant) है?

**Self-host पहले:** एक static बाइनरी किसी **local model** या अपने किसी provider के आगे बैठती है,
साथ में fail-closed residency, default-deny capability floor, और हर tool call का
tamper-evident audit log। आपका डेटा आपकी मशीन से बाहर नहीं जाता। यह भारत के **DPDP Act,
2023** के तहत data-residency ज़रूरतों से सीधे मेल खाता है — डेटा देश में, आपके अपने
नियंत्रण में रहता है।

## Q4. लागत कितनी है? क्या यह सचमुच मुफ़्त है?

**Apache-2.0, मुफ़्त, self-host।** न credit card, न cross-border invoice, न कोई legal
entity। `git clone` और `go install` ही पूरा रास्ता है — यानी कोई डॉलर-आधारित subscription
नहीं, आपकी लागत सिर्फ़ आपके अपने compute की (रुपये में, आपके अपने hardware पर)।

## Q5. यह कितना सस्ता या तेज़ है?

एक मापी गई **50-turn × 5-agent** session पर, एक **tuned warm-cache stack** के मुक़ाबले
**~4.1× कम काम**। यही ईमानदार headline आँकड़ा है। ~60× वाला आँकड़ा (लगभग 19 घंटे से लगभग
19 मिनट) **सिर्फ़ naive re-send-everything loop** के मुक़ाबले टिकता है — इसे कभी headline
न बनाएँ। reuse वाला यह फ़ायदा **सिर्फ़ self-host** पर, read-heavy fleets के लिए है।
रुपये के लिहाज़ से: कम दोहराया गया काम = कम token = कम बिल।

## Q6. कौन-से models चलते हैं?

**Qwen2/Qwen3 और GLM-MoE** in-kernel reference engine में bit-exact सिद्ध हैं। बाक़ी सब
(DeepSeek, Mistral, कोई भी open-weights model) OpenAI-compatible wire के ज़रिए जुड़ते हैं:
Ollama / vLLM / SGLang / llama.cpp / LM Studio, या कोई भी OpenAI-compatible API।

## Q7. यह prompt injection कैसे रोकता है?

दो **structural** gates, कोई classifier नहीं:

- **default-deny capability floor** — कोई ख़तरनाक tool allow-list पर होता ही नहीं, इसलिए
  model को चाहे जितना बहला लिया जाए, वह action चल नहीं सकता।
- **result quarantine** — संदिग्ध tool *results* को अलग रखा जाता है, वे model के context
  में घुसते ही नहीं।

Live tests में injection बिना-सुरक्षा वाले baseline तक **5/5** पहुँची; fak ने उसे **5/5**
रोक दिया।

## Q8. इसे इंस्टॉल कैसे करूँ?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

या clone से:

```bash
go build -o fak ./cmd/fak
```

(अगर आपके नेटवर्क पर Go का डिफ़ॉल्ट module proxy अटकता हो, तो `GOPROXY` को किसी पहुँच-योग्य
mirror पर सेट कर लें।)

## 60-सेकंड का प्रमाण (कोई key नहीं, कोई model नहीं, कोई GPU नहीं)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Q9. आगे कहाँ जाएँ

- [README (पूरा अवलोकन)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 मिनट में local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — बाइनरी इंस्टॉल करें](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — हर आँकड़े का स्रोत](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — क्या shipped/simulated/stub है](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [डेटा residency और अनुपालन — DPDP Act के लिए](../../explainers/data-residency-and-compliance.md)
- [Integrations — अपने एजेंट को जोड़ें](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)।
