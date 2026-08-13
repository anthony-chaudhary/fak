---
title: "fak क्विकस्टार्ट — 10 मिनट में एक local model (हिन्दी / Hindi quickstart)"
description: "शून्य से लेकर एक governed local AI तक ~10 मिनट में: offline, बिना key, बिना cloud बिल, डेटा आपकी मशीन पर। एक कमांड में मौजूदा एजेंट को local model के साथ wrap करें; DPDP-अनुकूल self-host।"
---

# fak क्विकस्टार्ट — 10 मिनट में एक local model

> यह एक **स्थानीयकृत प्रवेश-पृष्ठ (entry point)** है, पूरे दस्तावेज़ का अनुवाद नहीं।
> पूरा दस्तावेज़ अंग्रेज़ी में है — यह पृष्ठ आपको सबसे तेज़ रास्ता, 60-सेकंड का प्रमाण
> और एक ईमानदार गति-आँकड़ा देकर आगे
> [अंग्रेज़ी स्रोत](https://github.com/anthony-chaudhary/fak/blob/main/README.md) तक
> पहुँचाता है।
> **सूचना:** यह अनुवाद मशीन द्वारा तैयार किया गया है और इसकी native समीक्षा अभी बाकी है — सुधार के लिए
> issue/PR खोलें।
>
> हिन्दी परिचय के लिए देखें: [fak एक पंक्ति में (README)](./README.md) ·
> पूरी भाषा-सूची [i18n hub](../README.md) पर।

## वादा

शून्य से लेकर एक **governed local AI** तक — लगभग **10 मिनट** में। यह पूरी तरह
**offline** चलता है: कोई API key नहीं, कोई cloud बिल नहीं, आपका डेटा आपकी मशीन से बाहर
नहीं जाता (**DPDP Act, 2023** के अनुकूल self-host), और छोटे models के लिए **CPU ही
काफ़ी** है — GPU ज़रूरी नहीं।

fak आपका model नहीं बदलता — वह आपके AI एजेंट और उसके tool calls के बीच बैठकर हर call को
*चलने से पहले* जाँचता है और लंबी session में दोहराए जाने वाले साझा काम को दोबारा इस्तेमाल
करता है। वही एजेंट loop **ज़्यादा सुरक्षित, सस्ता और तेज़** हो जाता है — बिना कुछ दोबारा लिखे।

## सबसे तेज़ रास्ता (एक कमांड)

अपने मौजूदा एजेंट को एक ही कमांड में एक local model के साथ wrap कर दें:

```bash
fak manage --gguf qwen2.5:7b -- claude
```

यही काफ़ी है — एजेंट-साइड पर कोई code बदलाव नहीं। हर tool call अब kernel के अंदर बने
default-deny capability floor से गुज़रती है, और model आपकी मशीन पर local चलता है।

## 60-सेकंड का प्रमाण (कोई key नहीं, कोई model नहीं, कोई GPU नहीं)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

पहली दो कमांड दिखाती हैं कि allow-list में न होने वाली action **संरचनात्मक रूप से** रुकती
है — किसी attack को "पकड़ने" पर निर्भर नहीं, बल्कि **fail-closed** होकर। तीसरी कमांड एक
live injection को दीवार के पीछे रखते हुए भी task पूरा करती है। (live tests में
unprotected baseline तक injection 5/5 पहुँची; fak ने 5/5 रोकी।)

## fak क्या है

**fak एक static Go बाइनरी है** जो आपके AI एजेंट और उसके tools के बीच बैठती है। सब कुछ इसी
एक process के अंदर चलता है — gateway, permission जाँच, cache, quarantine, routing, metrics
— तो कोई sidecar या अलग authorizer नहीं। आप एजेंट दोबारा नहीं लिखते: या तो एक base URL को
`fak serve` की ओर मोड़ते हैं, या ऊपर की तरह `fak manage claude` से wrap कर देते हैं।

संदिग्ध tool *results* को एक अलग quarantine में रखा जाता है ताकि वे model के context में
घुसें ही नहीं — यह **structure** से होता है, किसी classifier से नहीं (जिस detector को
इन्हें flag करना है, उसे design से ~100% evadable माना गया है — वह बोनस है, floor नहीं)।

## कितना तेज़ (ईमानदार आँकड़ा)

एक मापी गई **50-turn × 5-agent** session पर, एक *tuned* warm-cache stack के मुक़ाबले
ईमानदार लाभ लगभग **~4.1× कम काम** है। यही headline आँकड़ा है।

वह चर्चित **~60×** (लगभग 19 घंटे से घटकर ~19 मिनट) आँकड़ा **सिर्फ़ naive
re-send-everything loop** के मुक़ाबले सच है — इसे कभी headline न मानें। यह reuse-लाभ
**self-host only** है और read-heavy fleets पर लागू होता है।

लागत रुपये (INR) में चुभती है जबकि token बिल आमतौर पर डॉलर में आता है — इसलिए साझा काम का
यह reuse सीधे **margin** का लीवर है। साथ ही: provider की prompt-cache छूट तभी बनी रहती है जब
cached prefix byte-for-byte वही रहे; fak बीच के पुराने turns हटाकर भी prefix को
**byte-identical** रखने की गारंटी देता है — provider उस cache को दोबारा इस्तेमाल करता है
या नहीं, यह provider का फ़ैसला है, जिसे fak दावा करने के बजाय बस relay कर देता है।

## आगे कहाँ जाएँ

- [README — पूरा अवलोकन](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 मिनट में local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — बाइनरी इंस्टॉल + पूरा feature set](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — हर आँकड़े का स्रोत](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — क्या shipped/simulated/stub है](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [डेटा residency और अनुपालन — DPDP Act, 2023 के लिए](../../explainers/data-residency-and-compliance.md)
- [Integrations — अपने एजेंट को जोड़ें](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE) — मुफ़्त,
self-host; न कार्ड, न cross-border invoice, न legal entity।
