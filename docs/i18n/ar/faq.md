---
title: "fak — الأسئلة الأكثر شيوعًا عند أول لقاء (مدخل عربي / Arabic FAQ)"
description: "صفحة دخول عربية إلى fak: ملف Go ثنائي واحد يجلس أمام وكيل الذكاء الاصطناعي الذي تشغّله بالفعل — يراجع كل tool call قبل تنفيذه، ويعيد استخدام العمل المشترك في الجلسات الطويلة. استضافة ذاتية، متوافق مع نظام حماية البيانات الشخصية (PDPL) في السعودية والإمارات، ورخصة Apache-2.0."
---

# fak — الأسئلة الأكثر شيوعًا عند أول لقاء (مدخل عربي)

> هذه **صفحة دخول مُوطَّنة (entry point)**، وليست ترجمة كاملة للوثائق.
> الوثائق الكاملة بالإنجليزية — تمنحك هذه الصفحة جوهر fak وإجاباتٍ سريعة، ثم تحيلك إلى
> [المصدر الإنجليزي الموثوق](https://github.com/anthony-chaudhary/fak/blob/main/README.md).
> **تنبيه:** هذه الترجمة مُولّدة آليًا وبانتظار مراجعة متحدّث أصلي — إن وجدت خطأً،
> افتح issue أو PR.
> **لغات ومداخل أخرى:** انظر [مركز التوطين (i18n hub)](../README.md).

## س1. ما هو fak؟

هو **ملف Go ثنائي (binary) واحد وساكن** تضعه أمام وكيل الذكاء الاصطناعي الذي تشغّله
بالفعل (Claude Code أو Codex أو Cursor أو أي عميل OpenAI / Anthropic / MCP) بمجرد إعادة
توجيه عنوان base URL واحد، ودون أي إعادة كتابة. يجعل الجلسات الطويلة أرخص (يُسقِط الأدوار
القديمة مع إبقاء بادئة (prefix) الـ prompt-cache لدى المزوّد متطابقة بايتًا ببايت)، ويوجّه
كل tool call، ويمكنه تشغيل النماذج محليًا داخل العملية نفسها عبر محرك المرجع داخل النواة
(in-kernel reference engine)، ويسجّل حُكمًا قابلًا للتدقيق لكل استدعاء.

## س2. هل عليّ تبديل النموذج أو إعادة كتابة الوكيل؟

لا. fak يحكم ويُخزّن (cache) النموذج الذي تستخدمه أصلًا، ولا يستبدله. لُفّه بأمر واحد:

```bash
fak manage claude
```

أو أعد توجيه عنوان base URL واحد إلى `fak serve`.

## س3. أين تذهب بياناتي — وهل هذا متوافق مع الأنظمة؟

الاستضافة الذاتية أولًا: ملف ثنائي ساكن واحد أمام نموذج محلي أو مزوّد داخل البلد، مع
إقامة بيانات (residency) تفشل مغلقةً (fail-closed)، وأرضية صلاحيات بمبدأ الرفض الافتراضي
(default-deny)، وسجلّ تدقيق يكشف أي عبث (tamper-evident) لكل tool call. بياناتك لا تغادر
جهازك. هذا يجعل مواءمة **نظام حماية البيانات الشخصية (PDPL) في المملكة العربية السعودية
والإمارات العربية المتحدة** — الذي يقيّد نقل البيانات عبر الحدود — أمرًا بنيويًا لا مجرد
وعد: فالبيانات ببساطة لا تخرج لتُنقل. للمزيد راجع
[شرح إقامة البيانات والامتثال](../../explainers/data-residency-and-compliance.md).

## س4. كم يكلّف؟ هل هو مجاني فعلًا؟

رخصة **Apache-2.0**، مجاني، باستضافة ذاتية. لا بطاقة ائتمان، ولا فاتورة عابرة للحدود، ولا
كيان قانوني — أي لا تحويل من الريال السعودي (SAR) أو الدرهم الإماراتي (AED) إلى عملة أخرى،
ولا رسوم صرف. `git clone` ثم `go install` هو كامل الطريق.

## س5. كم يوفّر من التكلفة أو الوقت؟

على جلسة مقيسة من **50 دورة × 5 وكلاء**، عمل أقل بنحو **4.1×** مقارنةً بمكدس warm-cache
مُحسَّن (tuned) — وهذا هو الرقم الأمين. أما رقم الـ **~60×** (من نحو 19 ساعة إلى نحو 19
دقيقة) فهو صحيح **فقط مقابل حلقة ساذجة تعيد إرسال كل شيء**، ولا يُقدَّم أبدًا كعنوان رئيسي.
مكسب إعادة الاستخدام للاستضافة الذاتية فقط، وللأساطيل كثيفة القراءة (read-heavy). بلغة
الأعمال: فاتورة الرموز (tokens) تُحسب بالدولار بينما هامشك بالريال والدرهم، وهذا رافعة
مباشرة على الهامش. كما يُبقي fak بادئة الـ prompt-cache متطابقة بايتًا ببايت أثناء إسقاط
الأدوار الوسطى القديمة، فيبقى خصم المزوّد قائمًا؛ لكن fak يضمن تطابق البادئة بايتيًا فقط،
أما إعادة استخدام الذاكرة المؤقتة فهي قرار المزوّد الذي ينقله fak ولا يدّعيه.

## س6. أي النماذج تعمل معه؟

**Qwen2/Qwen3 و GLM-MoE** مُثبَتة بدقة بِتّية تامة (bit-exact) في محرك المرجع داخل النواة
(in-kernel reference engine). أما كل ما عداها (DeepSeek و Mistral وأي نموذج مفتوح الأوزان)
فيتصدّر عبر واجهة OpenAI-compatible: Ollama / vLLM / SGLang / llama.cpp / LM Studio أو أي
واجهة API متوافقة مع OpenAI.

## س7. كيف يوقف حقن الأوامر (prompt injection)؟

ببوّابتين بنيويتين، لا بمصنِّف (classifier):

- **أرضية صلاحيات بمبدأ الرفض الافتراضي:** تُفحص داخل النواة على مسار الاستدعاء نفسه وتفشل
  مغلقةً. فعلٌ ليس على قائمة السماح لا يمكن أن يُنفَّذ مهما جرى التلاعب بالنموذج.
- **حجْر النتائج (result quarantine):** نتائج الأدوات المشبوهة تُحجب كليًا عن سياق النموذج —
  بنية لا كاشف. المصنِّف الذي يميّزها يُعامَل على أنه قابل للتحايل بنسبة ~100% بحكم التصميم؛
  مكافأة لا أرضية.

في اختبارات حية، وصل الحقن إلى الأساس غير المحمي 5/5، وحجزه fak 5/5.

## س8. كيف أثبّته؟

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

أو من نسخة مستنسخة (clone):

```bash
go build -o fak ./cmd/fak
```

إن تعذّر الوصول إلى `proxy.golang.org` من شبكتك، اضبط `GOPROXY` على مرآة قريبة قبل
`go install`.

### برهان الـ 60 ثانية (بلا مفتاح، بلا نموذج، بلا GPU)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

## س9. إلى أين أذهب بعد ذلك؟

- [README — النظرة الكاملة](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — نموذج محلي خلال 10 دقائق](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — تثبيت الملف الثنائي](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — مصدر كل رقم](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — ما هو shipped / simulated / stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [شرح إقامة البيانات والامتثال — لِـ PDPL](../../explainers/data-residency-and-compliance.md)
- [Integrations — اربط وكيلك](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
