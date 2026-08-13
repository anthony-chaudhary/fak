---
title: "fak — விரைவுத் தொடக்கம் (10 நிமிடத்தில் ஒரு local model)"
description: "fak-இன் தமிழ் விரைவுத் தொடக்கப் பக்கம்: பூஜ்ஜியத்திலிருந்து governed local AI வரை சுமார் 10 நிமிடத்தில் — offline, key இல்லை, cloud bill இல்லை, தரவு உங்கள் இயந்திரத்திலேயே; சிறிய model-களுக்கு CPU போதும்."
---

# fak — விரைவுத் தொடக்கம் (10 நிமிடத்தில் ஒரு local model)

> இது ஒரு **உள்ளூர்மயமாக்கப்பட்ட நுழைவுப் பக்கம் (entry point)** — முழு ஆவணத்தொகுப்பின்
> மொழிபெயர்ப்பு அல்ல. முழு ஆவணங்கள் ஆங்கிலத்தில் உள்ளன. இந்தப் பக்கம் இயந்திரத்தால்
> உருவாக்கப்பட்டது; native சரிபார்ப்பு நிலுவையில் — திருத்தங்களுக்கு issue/PR திறக்கவும்.
> முழுமையான, அதிகாரப்பூர்வ மூலத்திற்கு
> [ஆங்கில டாக்ஸ்](https://github.com/anthony-chaudhary/fak/blob/main/README.md) பார்க்கவும்;
> தமிழ் நுழைவுப் பக்கங்களின் மையம் [i18n hub](../README.md)-இல்.
>
> **இந்த மொழியில்:** [அறிமுகம் (README)](./README.md) · [விரைவுத் தொடக்கம் (இந்தப் பக்கம்)](./quickstart.md)

## வாக்குறுதி: பூஜ்ஜியத்திலிருந்து governed local AI வரை ~10 நிமிடம்

இந்தப் படிகளை முடித்தபின், உங்கள் சொந்த இயந்திரத்திலேயே இயங்கும் ஒரு AI உங்களிடம்
இருக்கும் — **offline** வேலை செய்யும், **key இல்லை, cloud bill இல்லை** (ரூபாய் bill
இல்லை), **தரவு உங்கள் இயந்திரத்தை விட்டு வெளியேறாது** (DPDP Act, 2023 அடிப்படையிலான
data-residency-க்கு உகந்தது), மற்றும் சிறிய model-களுக்கு **CPU போதும்** — GPU தேவையில்லை.

## வேகமான வழி: இருக்கும் ஏஜென்டை ஒரே கட்டளையில் wrap செய்யுங்கள்

உங்கள் ஏஜென்டை மறுஎழுத்து செய்யத் தேவையில்லை. ஒரே கட்டளையில், அதை ஒரு local model-க்குப்
பின்னால் wrap செய்யுங்கள்:

```bash
fak manage claude
```

இது உங்கள் இருக்கும் agent loop-ஐ அப்படியே வைத்து, ஒவ்வொரு tool call-ஐயும் *இயங்கும்
முன்பே* capability floor வழியாகச் செலுத்துகிறது — அதே loop மேலும் பாதுகாப்பாக, மலிவாக,
வேகமாக.

## 60-வினாடி சான்று (key இல்லை, model இல்லை, GPU இல்லை)

முதலில் எதையும் download செய்யாமல், boundary-ஐ இங்கேயே நிரூபிக்கலாம்:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

முதல் இரண்டு வரிகள் kernel-இன் *உள்ளே* இயங்கும் default-deny முடிவைக் காட்டுகின்றன:
allow-list-இல் இல்லாத action இயங்காது — model எப்படி ஏமாற்றப்பட்டாலும் (**fail-closed**).
கடைசி வரி offline-இல் prompt injection தடுக்கப்படுவதைக் காட்டுகிறது; நேரடிச் சோதனையில்
பாதுகாப்பற்ற baseline-ஐ injection 5/5 எட்டியது, fak அதை 5/5 தடுத்தது.

## fak என்றால் என்ன

**fak ஒரு static Go binary** — அது உங்கள் AI ஏஜென்டுக்கும் அது அழைக்கும் tool-களுக்கும்
இடையில் அமர்ந்து, ஒவ்வொரு tool call-ஐயும் இயங்கும் முன் சரிபார்க்கிறது; நீண்ட session-களில்
மீண்டும் வரும் பகிர்வு-வேலையை மறுபயன்பாடு செய்கிறது. விளைவு: *அதே* agent loop மேலும்
பாதுகாப்பாக, மலிவாக, வேகமாக — மறுஎழுத்து இல்லாமல். நீங்கள் ஒரு base URL-ஐ `fak serve`
நோக்கித் திருப்புகிறீர்கள், அல்லது `fak manage claude` எனும் ஒரே கட்டளையில் wrap
செய்கிறீர்கள். fak உங்கள் model-ஐ மாற்றாது — அதை govern செய்து cache செய்கிறது.

சந்தேகத்திற்குரிய tool *results* தனி quarantine-இல் வைக்கப்படுகின்றன; அவை model-இன்
context-க்குள் நுழையவே முடியாது — structure மூலம், எந்த detector-ஐயும் நம்பாமல்.

## எவ்வளவு வேகம் (நேர்மையான எண்)

50-turn × 5-agent session-இல் அளக்கப்பட்ட நேர்மையான தலைப்பு-எண்: ஒரு **tuned warm-cache
stack**-ஐ விட **~4.1× குறைவான வேலை**. இதுவே headline. நீண்ட read-heavy fleet-களுக்கு,
ரூபாய் token bill மற்றும் margin மீது இது நேரடி நெம்புகோல் — ஆனால் இந்த மறுபயன்பாட்டு
வெற்றி **self-host-க்கு மட்டுமே**.

~19 மணி நேரத்திலிருந்து ~19 நிமிடம் வரையிலான **~60×** எண், எல்லாவற்றையும் மீண்டும்
அனுப்பும் **naive re-send loop-ஐ விட மட்டுமே** உண்மை — headline அல்ல; அதைக் குறிப்பிட்டால்,
"naive pattern-ஐ விட மட்டுமே" என வெளிப்படையாகச் சொல்லுங்கள்.

provider-இன் prompt-cache தள்ளுபடி நிலைக்க, fak நடுவில் உள்ள பழைய turns-ஐ அகற்றியும்
cached prefix-ஐ **byte-for-byte** ஒரே மாதிரி வைத்திருக்கிறது. prefix-இன் byte-identity-ஐ
fak **உறுதி செய்கிறது**; provider அந்த cache-ஐ மறுபயன்படுத்துகிறதா என்பது provider-இன்
முடிவு — அதை fak claim செய்யாமல் relay செய்கிறது.

ஒவ்வொரு எண்ணும் அதன் commit மற்றும் artifact வரை கண்டறியப்படுவதை
[BENCHMARK-AUTHORITY.md](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)-இல் பார்க்கவும்.

## நிறுவல் (self-host, இலவசம்)

fak **Apache-2.0**, இலவசம், self-host — கார்டு இல்லை, cross-border invoice இல்லை,
entity இல்லை. `git clone` மற்றும் ஒரே `go install` — இதுவே முழு வழி:

```bash
git clone https://github.com/anthony-chaudhary/fak.git && cd fak
go install github.com/anthony-chaudhary/fak/cmd/fak@latest
```

fak governs மற்றும் caches செய்யும் model-கள்: **Qwen2/Qwen3 மற்றும் GLM-MoE**
in-kernel reference engine-இல் bit-exact என நிரூபிக்கப்பட்டவை; மற்ற அனைத்தும் (DeepSeek,
Mistral, எந்த open-weights model-உம்) OpenAI-compatible wire வழியாக front செய்யப்படுகின்றன
— Ollama / vLLM / SGLang / llama.cpp / LM Studio அல்லது எந்த OpenAI-compatible API வழியாகவும்.

## அடுத்து எங்கே

- [README (முழு கண்ணோட்டம்)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — 10 நிமிடத்தில் local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — முழு feature set](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — ஒவ்வொரு எண்ணின் மூலம்](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — எது shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [தரவு residency மற்றும் compliance — DPDP Act, 2023-க்கு](../../explainers/data-residency-and-compliance.md)
- [Integrations — உங்கள் ஏஜென்டை இணைக்கவும்](../../integrations/README.md)
- தமிழ் அறிமுகப் பக்கம்: [README (தமிழ்)](./README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE).
