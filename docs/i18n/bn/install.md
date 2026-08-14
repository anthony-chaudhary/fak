---
title: "fak — install ও শুরু করা (বাংলা প্রবেশ-পৃষ্ঠা / Bengali install guide)"
description: "fak install করে চালানোর বাংলা প্রবেশ-পৃষ্ঠা: go build দিয়ে kernel, Ollama/vLLM/llama.cpp দিয়ে সত্যিকারের model, বা fak manage দিয়ে এক কমান্ডে local model; DPDP-সহায়ক self-host, Apache-2.0।"
---

# fak — install ও শুরু করা (বাংলা)

> এটি একটি **স্থানীয়কৃত প্রবেশ-পৃষ্ঠা (entry point)** — পুরো ডকুমেন্টেশনের অনুবাদ নয়।
> পূর্ণ ডকুমেন্টেশন ইংরেজিতে — এই পৃষ্ঠা শুধু install ও প্রথম run-এর পথ দেখিয়ে আপনাকে
> [ইংরেজি ডকস](https://github.com/anthony-chaudhary/fak/blob/main/README.md)-এ পৌঁছে দেয়।
> **দ্রষ্টব্য:** এই অনুবাদ যন্ত্রে তৈরি; native পর্যালোচনা বাকি — সংশোধনের জন্য
> issue/PR খুলুন।
>
> **এই ভাষায় আরও:** [বাংলা README (ঘন পিচ)](./README.md) — পূর্ণ ভাষা-তালিকা
> [i18n hub](../README.md)-এ।

এটি **install-করে-চালানোর সদর দরজা**; fak-এর ঘন পিচ আছে [README](./README.md)-এ।
এই পৃষ্ঠা একটি পরিষ্কার checkout থেকে চালু kernel পর্যন্ত, আর তার পেছনে একটি model
বসানো পর্যন্ত নিয়ে যায় — copy-করে-চালানো যায় এমন কমান্ড দিয়ে।

> **কখনো `fak` চালাননি?** তার বদলে [গাইডেড প্রথম session](https://github.com/anthony-chaudhary/fak/blob/main/docs/fak/tutorial.md)
> নিন (~১৫ মিনিট): প্রতিটি কমান্ড তার আসল ধরা-পড়া output সহ, প্রথম দুই অংশ offline,
> কোনো key বা GPU ছাড়াই। এই পৃষ্ঠা হলো install reference আর tier-এর মানচিত্র।

## পূর্বশর্ত

- **Go 1.26+.** `fak/go.mod`-এ `go 1.26` ঘোষিত। Go-র default `GOTOOLCHAIN=auto`
  থাকলে পুরনো `go`-ও প্রথম build-এ সঠিক toolchain নিজে নামিয়ে নেয় (একবার network
  লাগে); নইলে <https://go.dev/dl/> থেকে Go 1.26 install করুন। `go version` দিয়ে যাচাই।
- **Tier 0-এর জন্য এটুকুই** — কোনো GPU, API key বা network লাগে না।
- **Tier 1**-এ অতিরিক্ত লাগে যেকোনো OpenAI-compatible model server (যেমন Ollama)।

## চারটি tier (setup-খরচের ক্রমে)

fak হলো **একটিই Go binary** — শূন্য বাহ্যিক dependency। এটি দিয়ে যা করা যায়, setup-এর
খরচ অনুসারে সাজানো (দুইয়ের মাঝে নতুন কিছু install হয় না):

| Tier | কী পাবেন | Setup | Downloads |
|---|---|---|---|
| **0 — kernel পরখ করুন** | adjudication boundary offline চালান/মাপুন | `go build` | কিছু না |
| **1 — সত্যিকারের model-এর সামনে** | অন্যত্র serve করা model-এর সামনে kernel বসান (Ollama / vLLM / llama.cpp / cloud) | + একটি চালু OpenAI-compatible server | একটি chat model |
| **1b — এক কমান্ডে local model** | বিদ্যমান এজেন্টের সঙ্গে in-kernel local GGUF — key নেই, network নেই | `fak manage --gguf qwen2.5:7b -- claude` | ~5 GB GGUF (cached) |
| **2 — fused in-kernel model** | kernel-এর নিজস্ব pure-Go SmolLM2 forward pass | + (real weights) Python export | ~135M params |

শুধু **fak-কে সামনে রেখে একটি কাজের model serve** করতে চাইলে **Tier 1** দরকার।
Tier 2-র in-kernel model একটি *reference forward pass* — HuggingFace-এর সঙ্গে
bit-for-bit প্রমাণিত, chat-quality serving engine নয়।

## binary নিন

**Contributor (clone থেকে build):**

```bash
git clone https://github.com/anthony-chaudhary/fak.git
cd fak
go build -o fak ./cmd/fak          # -> ./fak   (Windows: go build -o fak.exe ./cmd/fak)
./fak help
```

**Go দিয়ে install** (module path-ই repo root, তাই সরাসরি resolve হয়):

```bash
go install github.com/anthony-chaudhary/fak/cmd/fak@latest   # -> $(go env GOBIN) / $GOPATH/bin
```

> **Windows note.** `go build`/`go vet`/`go run` native ভাবে চলে; কিন্তু
> `-o fak.exe` দিয়ে build করুন — extension ছাড়া `fak` ফাইল cmd.exe / PowerShell
> নাম দিয়ে চালাতে পারে না। এই গাইডে যেখানে `./fak` লেখা, সেখানে `.\fak.exe` টাইপ করুন।
> test suite (`go test ./...`) OS Application-Control policy-তে আটকাতে পারে — দরকার হলে
> WSL-এ চালান; binary নিজে অক্ষত থাকে।

## ৬০-সেকেন্ডের প্রমাণ (key নেই, model নেই, GPU নেই)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection blocked, task still completes
```

default-deny capability floor kernel-এর *ভেতরে*, একই call path-এ চলে (fail-closed):
allow-list-এ না থাকা action চলতেই পারে না, model-কে যা-ই বোঝানো হোক। সন্দেহজনক tool
*results* আলাদা quarantine-এ থাকে, model-এর context-এ ঢোকে না — structure দিয়ে, কোনো
classifier দিয়ে নয়। লাইভ পরীক্ষায় prompt injection অরক্ষিত baseline-এ পৌঁছেছে ৫/৫;
fak আটকে দিয়েছে ৫/৫।

> **কেন self-host (ভারত/বাংলাদেশ):** খরচ লাগে টাকায় (INR), token bill আসে ডলারে।
> দীর্ঘ session-এ fak ভাগ-করা কাজ পুনর্ব্যবহার করে — একটি tuned warm-cache stack-এর
> তুলনায় ৫০-turn × ৫-agent run-এ **~4.1× কম কাজ** (এই reuse win কেবল self-host,
> read-heavy fleet-এ; naive re-send loop-এর তুলনায় ~60×, কিন্তু সেটি *শুধু* naive
> প্যাটার্নের বিপরীতে — headline নয়)। fak cached prefix byte-for-byte একই রাখে বলে
> provider-এর prompt-cache ছাড় টেকে; fak prefix byte-identity নিশ্চিত করে, provider
> সেই cache পুনর্ব্যবহার করবে কি না সেটি provider-এর সিদ্ধান্ত, যা fak relay করে।
> ডেটা মেশিন ছাড়ে না — যা **DPDP Act, 2023** অনুপালনে সহায়ক। বিস্তারিত
> [data residency ও compliance](../../explainers/data-residency-and-compliance.md)-এ।

## এরপর কোথায়

- [বাংলা README (ঘন পিচ)](./README.md) — fak এক লাইনে, পূর্ণ পরিচিতি।
- [Getting Started — পূর্ণ tier map ও install reference (ইংরেজি)](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [START-HERE — ১০ মিনিটে local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [README — পূর্ণ অবলোকন](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [Integrations — আপনার এজেন্ট যুক্ত করুন](../../integrations/README.md)
- [ডেটা residency ও compliance — DPDP Act, 2023-এর জন্য](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — প্রতিটি সংখ্যার উৎস](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — কোনটি shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)।
