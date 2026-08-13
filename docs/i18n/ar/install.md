---
title: "fak — التثبيت والبدء (مدخل عربي / Arabic getting started)"
description: "صفحة تثبيت وتشغيل fak بالعربية: ثنائية Go واحدة تراجع كل استدعاء أداة قبل تنفيذه وتعيد استخدام العمل المشترك عبر الجلسات الطويلة — استضافة ذاتية متوافقة مع نظام حماية البيانات الشخصية PDPL، ورخصة Apache-2.0."
---

# fak — التثبيت والبدء (مدخل عربي)

> هذه **صفحة مدخل مُوطَّنة (entry point)**، وليست ترجمة كاملة للتوثيق.
> التوثيق الكامل باللغة الإنجليزية — تمنحك هذه الصفحة مسار التثبيت والتشغيل، ثم
> تحيلك إلى [التوثيق الإنجليزي (مصدر الحقيقة)](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **ملاحظة:** هذه الصفحة مُولّدة آليًّا وبانتظار مراجعة متحدّث أصلي — لأي تصحيح، افتح
> issue أو PR.
> **مركز التوطين ولغات أخرى:** [i18n hub](../README.md).

هذه هي **واجهة التثبيت والتشغيل**. أمّا العرض المكثّف (لماذا fak؟) ففي
[README](https://github.com/anthony-chaudhary/fak/blob/main/README.md).

**fak ثنائية Go واحدة** تجلس بين وكيلك الذكي والأدوات التي يستدعيها: تراجع كل tool
call *قبل تنفيذه*، وتعيد استخدام العمل المتكرّر عبر الجلسة الطويلة. النتيجة: نفس حلقة
الوكيل تصبح **أكثر أمانًا وأقل كلفة وأسرع**، دون إعادة كتابة. لا تعيد كتابة وكيلك —
توجّه base URL واحدًا إلى `fak serve`، أو تغلّف وكيلك القائم بأمر واحد:

```bash
fak manage claude
```

**للسوق الخليجي ومنطقة الشرق الأوسط.** fak تُستضاف ذاتيًّا أولًا: ثنائية ساكنة تقف
أمام نموذج محلّي أو مزوّد داخلي، مع residency فاشلة-الإغلاق (fail-closed) على كل
backend، وسقف صلاحيات افتراضي-الرفض (default-deny)، وسجل تدقيق مقاوم للعبث لكل
استدعاء. تبقى البيانات داخل حدودك بما يخدم الامتثال لنظام حماية البيانات الشخصية
(PDPL) في السعودية والإمارات. وعلى صعيد الكلفة: إعادة استخدام العمل المشترك تعني
عملًا **أقل بنحو 4.1× مقارنةً بمكدّس warm-cache مضبوط** (مقيسة على جلسة من 50 دورة
× 5 وكلاء)، وهو ما يترجَم مباشرةً إلى هامش أعلى بالريال السعودي (SAR) والدرهم
الإماراتي (AED). هذه الميزة **للاستضافة الذاتية فقط** وتنطبق على الأساطيل كثيفة
القراءة. (الرقم ~60× صحيح **فقط** مقابل النمط الساذج re-send-everything، وليس
العنوان.)

---

## المتطلبات

- **Go 1.26+.** يعلن `go.mod` عن `go 1.26`؛ ومع `GOTOOLCHAIN=auto` (الافتراضي) تُنزَّل
  السلسلة الأداتية الصحيحة تلقائيًّا عند أول بناء (تحتاج شبكة مرّة واحدة). تحقّق بـ
  `go version`.
- **للمستوى Tier 0: لا شيء غير ذلك** — لا GPU، ولا مفتاح API، ولا شبكة.
- **Tier 1** يحتاج إضافةً أي خادم نموذج متوافق مع OpenAI (مثل Ollama).

---

## المستويات

أربعة أشياء يمكنك فعلها بالثنائية نفسها، بترتيب تصاعدي في كلفة الإعداد، **ولا يُثبَّت
شيء جديد بينها**:

| المستوى | ماذا تحصل عليه | الإعداد |
|---|---|---|
| **Tier 0 — جرّب النواة** | شغّل حدود المراجعة (adjudication) وقِسها بلا اتصال | `go build` |
| **Tier 1 — ضع fak أمام نموذج حقيقي** | نموذج تخدمه في مكان آخر (Ollama / vLLM / llama.cpp / مزوّد سحابي) | + خادم متوافق مع OpenAI |
| **Tier 1b — نموذج محلّي بأمر واحد** | نموذج GGUF محلّي داخل النواة مع وكيلك القائم — بلا مفتاح ولا شبكة | `fak guard --gguf qwen2.5:7b -- claude` |
| **Tier 2 — النموذج المدمَج داخل النواة** | تمريرة SmolLM2 الأمامية النقية بلغة Go التي تملكها النواة | + تصدير الأوزان (Python) |

إن كان همّك فقط **تشغيل نموذج مفيد وfak أمامه**، فاختر **Tier 1**. نموذج Tier 2 داخل
النواة هو *تمريرة أمامية مرجعية* مُثبَتة bit-for-bit مقابل HuggingFace، لا محرّك خدمة
بجودة محادثة.

---

## التثبيت

fak ثنائية واحدة مكتفية بذاتها، بلا اعتماديات خارجية. اختر المسار المناسب:

**البناء من النسخة المستنسخة (contributor):**

```bash
go build -o fak ./cmd/fak
```

(على Windows ابنِ بـ `go build -o fak.exe ./cmd/fak`.)

**التثبيت عبر Go.** مسار الوحدة `github.com/anthony-chaudhary/fak` هو جذر المستودع،
فيُثبَّت مباشرةً:

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

لا حاجة إلى بطاقة ائتمان، ولا فاتورة عابرة للحدود، ولا كيان قانوني — `git clone` ثم
`go install` هو المسار كاملًا. الرخصة **Apache-2.0**، مجّانية، استضافة ذاتية.

---

## إثبات في 60 ثانية (بلا مفتاح، بلا نموذج، بلا GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

هذا القرار **بنيوي لا مصنِّف**: أي فعل ليس على قائمة السماح لا يمكنه العمل مهما جرت
مخادعة النموذج (default-deny، fail-closed)، ويُفحَص داخل النواة على مسار الاستدعاء
نفسه. كما تُحتجَز *نتائج* الأدوات المشبوهة خارج سياق النموذج بالكامل (حجْر بنيوي، لا
كاشف). في اختبارات حيّة: اخترق حقن التعليمات الخطَّ الأساس غير المحمي 5/5، وصدّته fak
5/5.

---

## مع نموذجك

fak **تحكم نموذجك وتخزّنه مؤقتًا، ولا تستبدله.** نموذجا Qwen2/Qwen3 وGLM-MoE مُثبَتان
bit-exact في المحرك المرجعي داخل النواة؛ وكل ما عداهما (DeepSeek، Mistral، أي نموذج
مفتوح الأوزان) يُقدَّم عبر السلك المتوافق مع OpenAI: Ollama / vLLM / SGLang /
llama.cpp / LM Studio أو أي API متوافق مع OpenAI. وبخصوص الكلفة عبر الجلسات الطويلة:
تبقى خصومات prompt-cache لدى المزوّد سارية لأن fak يُبقي البادئة (prefix) مطابِقة
بايتًا-ببايت حتى مع إسقاط الأدوار الوسطى القديمة. تضمن fak تطابق البادئة؛ أمّا إعادة
استخدام المزوّد للتخزين فقرارٌ يعود إليه، تنقله fak ولا تدّعيه.

---

## الخطوة التالية

- [README — نظرة كاملة](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — نموذج محلّي في 10 دقائق](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — المرجع الكامل للتثبيت والمستويات](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — اربط وكيلك](../../integrations/README.md)
- [إقامة البيانات والامتثال — لنظام PDPL](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — مصدر كل رقم](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — ما هو shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
