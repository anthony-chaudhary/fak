---
title: "fak — पहिल्या भेटीतले प्रमुख प्रश्न (मराठी FAQ / Marathi FAQ)"
description: "fak बद्दल पहिल्या-संपर्कातील सर्वात सामान्य प्रश्नांची थोडक्यात उत्तरे — तो काय आहे, model बदलावा लागतो का, डेटा कुठे जातो (DPDP Act, 2023), खर्च, वेग, आणि install; self-host, Apache-2.0."
---

# fak — पहिल्या भेटीतले प्रमुख प्रश्न (मराठी FAQ)

> हे एक **स्थानिकीकृत प्रवेश-पान (entry point)** आहे — संपूर्ण दस्तऐवजांचे भाषांतर नाही.
> संपूर्ण दस्तऐवज इंग्रजीत आहेत; हे पान fak बद्दलच्या पहिल्या प्रश्नांची थोडक्यात उत्तरे
> देऊन तुम्हाला [इंग्रजी सत्य-स्रोताकडे](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
> पोहोचवते.
> **सूचना:** हे भाषांतर यंत्राने तयार केले असून native तपासणी बाकी आहे — दुरुस्तीसाठी
> issue/PR उघडा.
>
> इतर भाषा आणि पाने: [i18n hub](../README.md) · या भाषेतील [परिचय](./README.md) ·
> [quickstart](./quickstart.md).

---

## Q1. fak म्हणजे नक्की काय?

**fak ही एकच static Go binary आहे** जी तुम्ही आधीच वापरत असलेल्या AI एजंटच्या (Claude Code,
Codex, Cursor, कोणताही OpenAI/Anthropic/MCP client) *समोर* बसते — फक्त एक base URL वळवून,
काहीही पुन्हा न लिहिता. ती लांब session स्वस्त करते (जुने turns काढते, पण provider चा
prompt-cache prefix byte-identical ठेवते), प्रत्येक tool call चे routing करते, in-process
local models चालवू शकते, आणि प्रत्येक call साठी तपासणीयोग्य (auditable) निर्णय नोंदवते.

## Q2. मला model बदलावा लागतो का, किंवा एजंट पुन्हा लिहावा लागतो का?

नाही. fak तुम्ही आधीच वापरत असलेल्या model ला **govern आणि cache** करते. त्याला असे wrap करा:

```bash
fak manage claude
```

किंवा एक base URL `fak serve` कडे वळवा. बाकी काहीही बदलावे लागत नाही.

## Q3. माझा डेटा कुठे जातो — हे compliant आहे का?

**Self-host-first:** एकच static binary एका **local model** किंवा देशी provider समोर बसते —
fail-closed residency, default-deny capability floor, आणि प्रत्येक tool call चा
tamper-evident audit log सह. **तुमचा डेटा तुमच्या मशीनबाहेर जात नाही.** भारतातील
data-residency साठी हे थेट **DPDP Act, 2023** शी जुळते: डेटा देशातच, तुमच्या ताब्यात राहतो.

## Q4. याची किंमत किती? हे खरंच मोफत आहे का?

**Apache-2.0, मोफत, self-host.** कार्ड नाही, cross-border invoice नाही, entity नाही —
रुपयांत किंवा डॉलरमध्ये कोणतेही license बिल नाही. `git clone` आणि `go install` हाच पूर्ण मार्ग.

## Q5. हे किती स्वस्त किंवा वेगवान आहे?

मोजलेल्या **50-turn × 5-agent** session वर, एका **tuned warm-cache stack** च्या तुलनेत
सुमारे **~4.1× कमी काम**. रुपयांत हा थेट margin चा लिव्हर आहे — तेच काम कमी token खर्चात.
**~60× आकडा (सुमारे 19 तास → सुमारे 19 मिनिटे) फक्त naive re-send-everything loop च्या
तुलनेतच** खरा आहे — तो कधीही headline म्हणून घेऊ नका. हा पुनर्वापराचा फायदा **self-host-only**
आहे आणि read-heavy fleets साठी लागू होतो.

## Q6. कोणते models चालतात?

**Qwen2/Qwen3 आणि GLM-MoE** in-kernel reference engine मध्ये bit-exact सिद्ध आहेत. बाकी
सर्व (DeepSeek, Mistral, कोणतेही open-weights model) OpenAI-compatible wire वर front होतात —
Ollama / vLLM / SGLang / llama.cpp / LM Studio.

## Q7. prompt injection ला ते कसे थांबवते?

classifier ने नव्हे, तर **दोन structural gates** ने:

- **default-deny capability floor** — धोकादायक tool कधीच allow-list वर नसतो, म्हणून model
  ला कितीही भुलवले तरी तो चालू शकत नाही.
- **result quarantine** — विषारी tool results model च्या context मध्ये शिरतच नाहीत.

live tests मध्ये: injection ने असुरक्षित baseline ला **5/5** गाठले; fak ने ते **5/5** वेळा
भिंतीआड ठेवले.

## Q8. मी हे install कसे करू?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

किंवा clone मधून:

```bash
go build -o fak ./cmd/fak
```

(जिथे `proxy.golang.org` अडते तिथे `GOPROXY` एका पोहोचणाऱ्या module proxy कडे set करा.)

### 60-सेकंदांचा पुरावा (key नाही, model नाही, GPU नाही)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## Q9. पुढे कुठे जावे?

- [README (पूर्ण आढावा)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 मिनिटांत local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — binary install करा](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — प्रत्येक आकड्याचा स्रोत](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — काय shipped/simulated/stub आहे](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [डेटा residency आणि compliance — DPDP Act साठी](../../explainers/data-residency-and-compliance.md)
- [Integrations — तुमचा एजंट जोडा](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
