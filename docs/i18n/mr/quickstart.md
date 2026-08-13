---
title: "quickstart — 10 मिनिटांत local model (मराठी / Marathi quickstart)"
description: "fak चे मराठी quickstart पान: शून्यापासून सुमारे 10 मिनिटांत govern केलेला local AI — offline, key नाही, cloud bill नाही, डेटा तुमच्या मशीनवरच; small model साठी CPU पुरेसा."
---

# quickstart — 10 मिनिटांत local model

> हे एक **स्थानिकीकृत प्रवेश-पान (entry point)** आहे — संपूर्ण दस्तऐवजांचे भाषांतर नाही.
> संपूर्ण दस्तऐवज इंग्रजीत आहेत — हे पान तुम्हाला शून्यापासून एका govern केलेल्या local
> AI पर्यंत नेण्याचा सर्वात लहान मार्ग देऊन पुढे [इंग्रजी डॉक्स](https://github.com/anthony-chaudhary/fak/blob/main/README.md)कडे पोहोचवते.
> **सूचना:** हे भाषांतर यंत्राने तयार केले असून native तपासणी बाकी आहे — दुरुस्तीसाठी
> issue/PR उघडा.
>
> **या भाषेतील इतर पाने:** [परिचय (README)](./README.md) — पूर्ण यादी
> [i18n hub](../README.md) वर.

## आश्वासन

शून्यापासून एका **govern केलेल्या local AI** पर्यंत सुमारे **10 मिनिटांत** — offline,
कोणतीही key नाही, cloud bill नाही, आणि **डेटा तुमच्या मशीनवरच** राहतो. small model
साठी CPU पुरेसा आहे — GPU लागत नाही. भारतासाठी हे थेट **DPDP Act, 2023** शी जुळते:
डेटा देशातच, तुमच्या मशीनबाहेर जात नाही. खर्च रुपयांत बोचतो — इथे token bill शून्य आहे.

## सर्वात वेगवान मार्ग

तुमचा सध्याचा एजंट एका कमांडमध्ये wrap करा:

```bash
fak manage claude
```

तुम्ही तुमचा एजंट पुन्हा लिहीत नाही — तोच agent loop आता प्रत्येक tool call आधी fak च्या
capability floor मधून जाते.

## 60-सेकंदांचा पुरावा (key नाही, model नाही, GPU नाही)

model download न करता, तुमच्या मशीनवरच tool-call सीमा सिद्ध करा:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection थांबवले, task तरीही पूर्ण
```

पहिले दोन कमांड दाखवतात की allow-list वर नसलेली action चालूच शकत नाही — हे **fail-closed**
आहे, kernel च्या आत त्याच call path वर तपासले जाते. तिसरा कमांड दाखवतो की संशयास्पद tool
*results* वेगळ्या quarantine मध्ये ठेवले जातात (structure द्वारे, classifier द्वारे नव्हे),
तरीही task पूर्ण होते. थेट चाचणीत: unprotected baseline वर prompt injection 5/5 पोहोचले;
fak ने ते 5/5 अडवले.

## fak म्हणजे काय

**fak ही एक static Go binary आहे** जी तुमच्या AI एजंट आणि त्याच्या tool calls च्या मध्ये
बसते — प्रत्येक tool call *चालण्याआधीच* तपासते, आणि लांब session मध्ये पुन्हा-पुन्हा येणारे
सामायिक काम पुनर्वापरते. परिणाम: तोच agent loop **अधिक सुरक्षित, स्वस्त आणि वेगवान** —
काहीही न बदलता. fak तुमचे model बदलत नाही — त्याला govern आणि cache करते.

## किती वेगवान

एका मोजलेल्या **50-turn × 5-agent** session वर, एका *tuned* warm-cache stack च्या तुलनेत
प्रामाणिक फायदा **~4.1× कमी काम** आहे — हाच headline आकडा आहे. जो ~60× आकडा दिसतो (सुमारे
19 तासांवरून सुमारे 19 मिनिटे) तो **फक्त naive re-send-everything loop च्या तुलनेत** खरा
आहे; तो headline म्हणून कधीही सांगू नये. हा reuse फायदा **self-host only** आहे आणि
read-heavy fleets ना लागू होतो.

provider ची prompt-cache सूट टिकते: fak मधले जुने turns काढूनही cached prefix
byte-for-byte तोच ठेवते. fak prefix ची byte-identity **हमी देते**; provider ती cache
प्रत्यक्षात पुनर्वापरतो की नाही हा provider चा निर्णय — fak तो relay करते, त्याचा दावा
करत नाही.

## पुढे कुठे

- [README (पूर्ण आढावा)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 मिनिटांत local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — binary install करा](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — प्रत्येक आकड्याचा स्रोत](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — काय shipped/simulated/stub आहे](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [डेटा residency आणि compliance — DPDP Act साठी](../../explainers/data-residency-and-compliance.md)
- [Integrations — तुमचा एजंट जोडा](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE) — मोफत, self-host. कार्ड नाही, cross-border invoice नाही, entity नाही.
