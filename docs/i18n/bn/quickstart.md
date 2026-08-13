---
title: "fak — quickstart (১০ মিনিটে একটি local model)"
description: "fak-এর বাংলা quickstart: শূন্য থেকে একটি governed local AI মাত্র ~১০ মিনিটে — offline, key নেই, cloud bill নেই, ডেটা আপনার মেশিনেই থাকে; DPDP-সহায়ক self-host, Apache-2.0।"
---

# fak — quickstart (১০ মিনিটে একটি local model)

> এটি একটি **স্থানীয়কৃত প্রবেশ-পৃষ্ঠা (entry point)** — পুরো ডকুমেন্টেশনের অনুবাদ নয়।
> পূর্ণ ডকুমেন্টেশন ইংরেজিতে — এই পৃষ্ঠা শুধু সবচেয়ে দ্রুত পথ, ৬০-সেকেন্ডের প্রমাণ আর
> পরের ধাপ দিয়ে আপনাকে [ইংরেজি ডকস](https://github.com/anthony-chaudhary/fak/blob/main/README.md)-এ পৌঁছে দেয়।
> **দ্রষ্টব্য:** এই অনুবাদ যন্ত্রে তৈরি; native পর্যালোচনা বাকি — সংশোধনের জন্য
> issue/PR খুলুন।
>
> **আরও:** [বাংলা পরিচিতি (README)](./README.md) · পূর্ণ ভাষা-তালিকা [i18n hub](../README.md)-এ।

## প্রতিশ্রুতি

শূন্য থেকে একটি **governed local AI** — প্রায় **১০ মিনিটে**। এই পথে যা পাবেন:

- **Offline** — কোনো API key নেই, কোনো cloud bill নেই।
- **ডেটা আপনার মেশিনেই থাকে** — বাইরে যায় না (DPDP Act, 2023-এর অধীনে residency-সহায়ক)।
- **CPU-তেই চলে** — ছোট model-এর জন্য GPU লাগে না; খরচ টাকায় গোনার আগেই শূন্য।

## সবচেয়ে দ্রুত পথ

আপনার বিদ্যমান এজেন্টকে এক কমান্ডে একটি local model-এর পেছনে wrap করুন:

```bash
fak manage claude
```

এজেন্ট নতুন করে লিখতে হয় না — শুধু একটি base URL `fak serve`-এর দিকে ঘুরিয়ে দিন,
প্রতিটি tool call প্রথমে capability floor পেরিয়ে যায়।

## ৬০-সেকেন্ডের প্রমাণ (key নেই, model নেই, GPU নেই)

কিছু download না করেই boundary-টা নিজে দেখুন — একটি structural DENY, একটি ALLOW,
আর একটি offline agent যাতে injection আটকানো হয় কিন্তু task তবুও সম্পূর্ণ হয়:

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection আটকানো হলো, task তবুও সম্পূর্ণ
```

permission নীতি kernel-এর *ভেতরে*, একই call path-এ চলে — **fail-closed**। allow-list-এ
নেই এমন কোনো action চলতে পারে না, model-কে যত ভালোভাবেই ভোলানো হোক না কেন। সন্দেহজনক
tool *results* আলাদা quarantine-এ রাখা হয়, যাতে সেগুলো model-এর context-এ ঢুকতেই না
পারে — structure দিয়ে, কোনো classifier দিয়ে নয়। live পরীক্ষায় prompt injection
অরক্ষিত baseline-এ ৫/৫ পৌঁছেছিল; fak সেটি ৫/৫ দেয়ালবন্দি করেছে।

## fak কী

**fak একটি Go binary** যা আপনার AI এজেন্ট আর তার tool calls-এর মাঝে বসে — প্রতিটি
tool call *চলার আগেই* যাচাই করে, আর দীর্ঘ session-এ বারবার আসা ভাগ-করা কাজ পুনর্ব্যবহার
করে। ফলাফল: একই agent loop **আরও নিরাপদ, সস্তা ও দ্রুত** — কোনো rewrite ছাড়াই।

fak আপনার model বদলায় না — সেটিকে govern ও cache করে। **Qwen2/Qwen3 আর GLM-MoE**
in-kernel reference engine-এ bit-exact প্রমাণিত; বাকি সব (DeepSeek, Mistral, যেকোনো
open-weights model) OpenAI-compatible wire-এ front হয় — Ollama / vLLM / SGLang /
llama.cpp / LM Studio বা যেকোনো OpenAI-compatible API-এর মাধ্যমে।

## কতটা দ্রুত (সৎ হিসাব)

৫০-turn × ৫-agent-এর একটি মাপা session-এ, একটি **tuned warm-cache stack**-এর তুলনায়
সৎ লাভ প্রায় **~4.1× কম কাজ**। এটিই headline সংখ্যা।

**~60×** (প্রায় ১৯ ঘণ্টা থেকে প্রায় ১৯ মিনিট) সংখ্যাটি সত্য **কেবল একটি naive
re-send-everything loop-এর তুলনায়** — এটি কখনো headline নয়। এই পুনর্ব্যবহারের লাভ
**self-host only** এবং read-heavy fleet-এ প্রযোজ্য।

provider-এর prompt-cache ছাড় টেকে কেবল তখনই, যখন cached prefix byte-for-byte একই
থাকে; fak মাঝের পুরনো turns সরিয়েও prefix-কে byte-identical **রাখে** — এটি fak-এর
গ্যারান্টি। provider আসলে cache পুনর্ব্যবহার করবে কি না, সেটি provider-এর সিদ্ধান্ত,
যা fak দাবি করে না, শুধু relay করে।

License: **Apache-2.0**, বিনামূল্যে, self-host — কার্ড নেই, cross-border invoice নেই,
entity নেই। `git clone` আর `go install github.com/anthony-chaudhary/fak/cmd/fak@latest`-ই
পুরো পথ।

## এরপর কোথায়

- [README (পূর্ণ পরিদর্শন)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — ১০ মিনিটে local model (ইংরেজি উৎস)](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [GETTING-STARTED — control plane বসান](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [BENCHMARK-AUTHORITY — প্রতিটি সংখ্যার উৎস](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — কোনটি shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)
- [ডেটা residency ও compliance — DPDP Act-এর জন্য](../../explainers/data-residency-and-compliance.md)
- [Integrations — আপনার এজেন্ট যুক্ত করুন](../../integrations/README.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)।
