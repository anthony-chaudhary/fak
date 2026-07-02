---
title: "fak — இணைந்த ஏஜென்ட் கெர்னல் (தமிழ் அறிமுகம் / Tamil introduction)"
description: "fak-இன் தமிழ் நுழைவுப் பக்கம்: ஒவ்வொரு tool call-ஐயும் இயங்கும் முன் சரிபார்க்கும் Go binary — அதே agent loop மேலும் பாதுகாப்பாக, மலிவாக, வேகமாக; self-host."
---

# fak — இணைந்த ஏஜென்ட் கெர்னல் (தமிழ் அறிமுகம்)

> இது ஒரு **உள்ளூர்மயமாக்கப்பட்ட நுழைவுப் பக்கம் (entry point)** — முழு ஆவணத்தொகுப்பின்
> மொழிபெயர்ப்பு அல்ல. முழு ஆவணங்கள் ஆங்கிலத்தில் உள்ளன — இந்தப் பக்கம் fak-இன் சாரம்,
> 60-வினாடி சான்று, நிறுவும் வழி ஆகியவற்றைத் தந்து, உங்களை
> [ஆங்கில டாக்ஸ்](https://github.com/anthony-chaudhary/fak/blob/main/README.md) நோக்கி அழைத்துச் செல்கிறது.
> **குறிப்பு:** இந்த மொழிபெயர்ப்பு இயந்திரத்தால் உருவாக்கப்பட்டது; native சரிபார்ப்பு
> நிலுவையில் — திருத்தங்களுக்கு issue/PR திறக்கவும்.
>
> **பிற இந்திய மொழிகளில்:** [हिन्दी (Hindi)](../hi/README.md) ·
> [తెలుగు (Telugu)](../te/README.md) · [বাংলা (Bengali)](../bn/README.md) ·
> [मराठी (Marathi)](../mr/README.md) — முழுப் பட்டியல் [i18n hub](../README.md)-இல்.

## fak ஒரே வரியில்

**fak ஒரு Go binary** — அது உங்கள் AI ஏஜென்டுக்கும் அதன் tool calls-க்கும் இடையில்
அமர்ந்து, ஒவ்வொரு tool call-ஐயும் *இயங்கும் முன்பே* சரிபார்க்கிறது; நீண்ட session-களில்
மீண்டும் மீண்டும் வரும் வேலையை மறுபயன்பாடு செய்கிறது. விளைவு: அதே agent loop
**மேலும் பாதுகாப்பாக, மலிவாக, வேகமாக** — வேறெதையும் மாற்றாமல்.

உங்கள் ஏஜென்டை மறுஎழுத்து செய்யத் தேவையில்லை — ஒரு base URL-ஐ `fak serve` நோக்கித்
திருப்பினால் போதும்; ஒவ்வொரு tool call-உம் முதலில் capability floor வழியாகச் செல்கிறது.

```bash
fak guard -- claude    # உங்கள் இருக்கும் ஏஜென்டை ஒரே கட்டளையில் wrap செய்கிறது
```

## இந்திய ஸ்டார்ட்அப்களுக்கு இது ஏன் முக்கியம்

- **செலவு ரூபாயில் உறைக்கிறது, token bill டாலரில் வருகிறது.** நீண்ட session-களில்
  பகிரப்பட்ட வேலையை (system prompt, tool list-இன் KV cache) fak மறுபயன்படுத்துகிறது —
  ஒரு tuned warm-cache stack-ஐ விட 50×5 run-இல் **~4.1× குறைவான வேலை** (naive re-send
  loop-ஐ விட ~60×; நேர்மையான எண் 4.1×தான்). கூடவே per-aspect routing மலிவான பகுதிகளை
  மலிவான model-க்கு அனுப்புகிறது. இது நேரடியாக margin-ஐ உயர்த்தும் நெம்புகோல்.
- **தரவு நாட்டுக்குள்ளேயே இருக்கட்டும் (DPDP Act, 2023).** fak self-host முதன்மை: ஒரு
  static binary எந்த **local model** அல்லது உள்நாட்டு provider-க்கும் முன் அமர்ந்து,
  ஒவ்வொரு backend-இலும் fail-closed residency, default-deny capability floor, ஒவ்வொரு
  tool call-இன் tamper-evident audit log ஆகியவற்றைத் தருகிறது. தரவு உங்கள்
  இயந்திரத்தை விட்டு வெளியேறாது.
- **எந்த payment rail-ஐயும் கடக்கத் தேவையில்லை.** fak **Apache-2.0**, இலவசம், self-host —
  கார்டு இல்லை, cross-border invoice இல்லை, entity இல்லை. `git clone` மற்றும்
  `go install` — இதுவே முழு வழி.
- **ஒற்றை static binary, பூஜ்ஜிய வெளிச் சார்புகள்.** சிறிய குழுவுக்கு எளிய ops —
  sidecar இல்லை, தனி authorizer இல்லை. laptop முதல் fleet வரை அதே artifact; நீங்கள்
  components அல்ல, flags மட்டுமே சேர்க்கிறீர்கள்.

## fak எந்தப் பிரச்சனைகளைத் தீர்க்கிறது

- **நீண்ட session-கள் விலை உயர்வது நிற்கிறது.** provider-இன் prompt-cache தள்ளுபடி,
  cached prefix byte-for-byte அப்படியே இருந்தால் மட்டுமே நிலைக்கும்; fak நடுவிலுள்ள
  பழைய turns-ஐ அகற்றியும் prefix-ஐ byte-identical ஆக வைத்திருக்கிறது — எனவே தள்ளுபடி
  உடையாது.
- **default-deny பாதுகாப்பு.** permission policy kernel-இன் *உள்ளே*, அதே call path-இல்
  இயங்குகிறது. ஒரு irreversible action-ஐத் தடுப்பது attack-ஐ "பிடிப்பதை" சார்ந்தில்லை —
  அந்த நெம்புகோல் ஒருபோதும் இணைக்கப்படவே இல்லை. இது **fail-closed**, open அல்ல.
- **prompt injection / poisoned results தடுப்பு.** சந்தேகத்திற்குரிய tool *results* தனி
  quarantine-இல் வைக்கப்படுகின்றன — அவை model-இன் context-க்குள் நுழையவே முடியாது;
  இது structure மூலம், எந்த classifier மூலமும் அல்ல.

## 60-வினாடி சான்று (key இல்லை, model இல்லை, GPU இல்லை)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection தடுக்கப்பட்டது, task எனினும் நிறைவு
```

## உங்கள் model-உடன்

fak உங்கள் model-ஐ மாற்றாது — அதை govern செய்து cache செய்கிறது. **Qwen2/Qwen3 மற்றும்
GLM-MoE** in-kernel reference engine-இல் bit-exact என நிரூபிக்கப்பட்டவை; மற்ற அனைத்தும்
(DeepSeek, Mistral, எந்த open-weights model-உம்) OpenAI-compatible wire வழியாக front
செய்யப்படுகின்றன — Ollama / vLLM / SGLang / llama.cpp / LM Studio அல்லது எந்த
OpenAI-compatible API வழியாகவும்.

## அடுத்து எங்கே

- [README (முழு கண்ணோட்டம்)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 நிமிடத்தில் local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — binary நிறுவவும்](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — உங்கள் ஏஜென்டை இணைக்கவும்](../../integrations/README.md)
- [தரவு residency மற்றும் compliance — DPDP Act-க்கு](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — ஒவ்வொரு எண்ணின் மூலம்](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — எது shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
