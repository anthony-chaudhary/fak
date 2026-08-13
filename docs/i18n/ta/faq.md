---
title: "fak — முதல்-தொடர்பு அடிக்கடி கேட்கப்படும் கேள்விகள் (தமிழ் FAQ / Tamil FAQ)"
description: "fak-ஐ முதல் முறை சந்திப்பவர்களுக்கான தமிழ் FAQ: fak என்றால் என்ன, model-ஐ மாற்ற வேண்டுமா, தரவு எங்கே செல்கிறது (DPDP Act, 2023), செலவு ரூபாயில், எவ்வளவு மலிவு/வேகம், prompt injection எப்படித் தடுக்கப்படுகிறது, நிறுவுதல் — self-host, Apache-2.0."
---

# fak — முதல்-தொடர்பு அடிக்கடி கேட்கப்படும் கேள்விகள் (FAQ)

> இது ஒரு **உள்ளூர்மயமாக்கப்பட்ட நுழைவுப் பக்கம் (entry point)** — முழு ஆவணத்தொகுப்பின்
> மொழிபெயர்ப்பு அல்ல. முழு ஆவணங்கள் ஆங்கிலத்தில் உள்ளன — இந்தப் பக்கம் முதல் முறை fak-ஐ
> சந்திப்பவர்களின் பொதுவான கேள்விகளுக்கு சுருக்கமான பதில்களைத் தந்து, உங்களை
> [ஆங்கில டாக்ஸ்](https://github.com/anthony-chaudhary/fak/blob/main/README.md) நோக்கி
> அழைத்துச் செல்கிறது.
> **குறிப்பு:** இந்த மொழிபெயர்ப்பு இயந்திரத்தால் உருவாக்கப்பட்டது; native சரிபார்ப்பு
> நிலுவையில் — திருத்தங்களுக்கு issue/PR திறக்கவும். மற்ற மொழிகள் மற்றும் முழுப் பட்டியல்
> [i18n hub](../README.md)-இல்.
>
> **இந்த மொழியில் மேலும்:** [தமிழ் அறிமுகம் (README)](./README.md)

---

## Q1. fak என்றால் என்ன?

நீங்கள் ஏற்கனவே இயக்கும் AI ஏஜென்டுக்கு (Claude Code, Codex, Cursor, அல்லது எந்த
OpenAI/Anthropic/MCP client) முன் நீங்கள் வைக்கும் **ஒற்றை static Go binary**. ஒரே ஒரு
base URL-ஐத் திருப்புவதன் மூலம் இதை இணைக்கிறீர்கள் — மறுஎழுத்து இல்லை. இது நீண்ட
session-களை மலிவாக்குகிறது (பழைய turns-ஐ அகற்றியும் provider-இன் prompt-cache prefix-ஐ
byte-identical ஆக வைத்திருக்கிறது), ஒவ்வொரு tool call-ஐயும் வழிநடத்துகிறது, local GGUF
model-களை in-process ஆக இயக்க முடியும், மற்றும் ஒவ்வொரு call-க்கும் தணிக்கை செய்யத்தக்க
(auditable) தீர்ப்பைப் பதிவு செய்கிறது.

## Q2. Model-ஐ மாற்றவோ, என் ஏஜென்டை மறுஎழுத்து செய்யவோ வேண்டுமா?

இல்லை. நீங்கள் ஏற்கனவே பயன்படுத்தும் model-ஐ fak govern செய்து cache செய்கிறது. அதை இப்படி
wrap செய்யுங்கள்:

```bash
fak manage claude
```

அல்லது ஒரு base URL-ஐ `fak serve` நோக்கித் திருப்புங்கள்.

## Q3. என் தரவு எங்கே செல்கிறது — இது compliant-ஆ?

**Self-host முதன்மை:** ஒரு static binary, ஒரு local model அல்லது உள்நாட்டு provider-க்கு
முன் அமர்கிறது; இதனுடன் fail-closed residency, default-deny capability floor, மற்றும்
ஒவ்வொரு tool call-க்கும் tamper-evident audit log. உங்கள் தரவு உங்கள் இயந்திரத்தை விட்டு
வெளியேறாது. இது இந்தியாவின் **DPDP Act, 2023** (Digital Personal Data Protection Act) போன்ற
உள்நாட்டு தரவு-பாதுகாப்புச் சட்டங்களுக்கு data-residency-ஐ நேரடியாக இணைக்க உதவுகிறது —
சிங்கப்பூர் மற்றும் இலங்கை fleet-களுக்கும் அதே self-host அணுகுமுறை பொருந்தும்.

## Q4. இதன் செலவு எவ்வளவு? உண்மையாகவே இலவசமா?

**Apache-2.0, இலவசம், self-host.** கார்டு இல்லை, cross-border invoice இல்லை, entity
இல்லை — எனவே டாலரில் வரும் எல்லைக்கு அப்பாற்பட்ட ரசீதுகள் இல்லை. `git clone` மற்றும்
`go install` — இதுவே முழு வழி. உங்கள் ரூபாய் செலவு உங்கள் சொந்த hardware/hosting மட்டுமே.

## Q5. இது எவ்வளவு மலிவு அல்லது வேகம்?

அளக்கப்பட்ட ஒரு **50-turn × 5-agent** session-இல், ஒரு **tuned warm-cache stack**-ஐ விட
**~4.1× குறைவான வேலை** — இதுதான் நேர்மையான தலைப்பு எண். ~60× (சுமார் 19 மணி நேரம் → சுமார்
19 நிமிடம்) என்ற எண் **நேவ் (naive) மறு-அனுப்பும் loop-ஐ விட மட்டுமே** உண்மை; அதை ஒருபோதும்
தலைப்பு எண்ணாகக் காட்ட வேண்டாம். token bill டாலரில் வந்தாலும் margin ரூபாயில் உறைப்பதால்,
இந்த மறுபயன்பாட்டு வெற்றி நேரடியாக margin-ஐ உயர்த்தும் நெம்புகோல் — ஆனால் அது **self-host
மட்டுமே**, read-heavy fleet-களுக்கு.

## Q6. எந்தெந்த model-கள் வேலை செய்கின்றன?

**Qwen2/Qwen3 மற்றும் GLM-MoE** in-kernel reference engine-இல் bit-exact என
நிரூபிக்கப்பட்டவை. மற்ற அனைத்தும் (DeepSeek, Mistral, எந்த open-weights model-உம்)
OpenAI-compatible wire வழியாக front செய்யப்படுகின்றன: Ollama / vLLM / SGLang / llama.cpp /
LM Studio அல்லது எந்த OpenAI-compatible API வழியாகவும். fak உங்கள் model-ஐ மாற்றாது —
அதை govern செய்து cache செய்கிறது.

## Q7. Prompt injection-ஐ இது எப்படி நிறுத்துகிறது?

Classifier அல்ல — **இரண்டு அமைப்புசார் (structural) வாயில்கள்:**

- **Default-deny capability floor** — kernel-இன் உள்ளே, அதே call path-இல் சரிபார்க்கப்படுகிறது
  (fail-closed). ஆபத்தான ஒரு tool ஒருபோதும் allow-list-இல் இருக்காது; model-ஐ எப்படி
  ஏமாற்றினாலும் அது இயங்க முடியாது.
- **Result quarantine** — சந்தேகத்திற்குரிய tool *results* model-இன் context-க்குள்
  நுழையவே முடியாதபடி தனியாக வைக்கப்படுகின்றன (structure, detector அல்ல). அவற்றைக்
  குறிக்கும் detector ~100% ஏமாற்றத்தக்கது என வடிவமைப்பில் கருதப்படுகிறது — அது ஒரு bonus,
  ஒருபோதும் floor அல்ல.

நேரடி சோதனைகளில்: prompt injection பாதுகாப்பில்லாத baseline-ஐ **5/5** எட்டியது; fak அதை
**5/5** தடுத்துச் சுவர் எழுப்பியது.

## Q8. இதை எப்படி நிறுவுவது?

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

அல்லது ஒரு clone-இலிருந்து:

```bash
go build -o fak ./cmd/fak
```

`proxy.golang.org` உங்கள் பிராந்தியத்தில் மெதுவாகவோ அணுக முடியாததாகவோ இருந்தால், முதலில்
ஒரு அருகிலுள்ள module proxy-ஐ `GOPROXY` வழியாக அமைத்துக்கொள்ளுங்கள்.

## 60-வினாடி சான்று (key இல்லை, model இல்லை, GPU இல்லை)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection தடுக்கப்பட்டது, task எனினும் நிறைவு
```

## Q9. அடுத்து எங்கே செல்வது?

- [README (முழு கண்ணோட்டம்)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 நிமிடத்தில் local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — binary நிறுவவும்](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — ஒவ்வொரு எண்ணின் மூலம்](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — எது shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [தரவு residency மற்றும் compliance — DPDP Act, 2023-க்கு](../../explainers/data-residency-and-compliance.md)
- [Integrations — உங்கள் ஏஜென்டை இணைக்கவும்](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
