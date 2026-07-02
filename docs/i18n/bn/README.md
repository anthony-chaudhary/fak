---
title: "fak — ফিউজড এজেন্ট কার্নেল (বাংলা পরিচিতি / Bengali introduction)"
description: "fak-এর বাংলা প্রবেশ-পৃষ্ঠা: প্রতিটি tool call চলার আগে যাচাই করা Go binary — একই agent loop আরও নিরাপদ, সস্তা, দ্রুত; DPDP-সহায়ক self-host, Apache-2.0."
---

# fak — ফিউজড এজেন্ট কার্নেল (বাংলা পরিচিতি)

> এটি একটি **স্থানীয়কৃত প্রবেশ-পৃষ্ঠা (entry point)** — পুরো ডকুমেন্টেশনের অনুবাদ নয়।
> পূর্ণ ডকুমেন্টেশন ইংরেজিতে — এই পৃষ্ঠা fak-এর সারমর্ম, ৬০-সেকেন্ডের প্রমাণ আর install
> পথ দিয়ে আপনাকে [ইংরেজি ডকস](../../../README.md)-এ পৌঁছে দেয়।
> **দ্রষ্টব্য:** এই অনুবাদ যন্ত্রে তৈরি; native পর্যালোচনা বাকি — সংশোধনের জন্য
> issue/PR খুলুন।
>
> **অন্যান্য ভারতীয় ভাষায়:** [हिन्दी (Hindi)](../hi/README.md) ·
> [தமிழ் (Tamil)](../ta/README.md) · [తెలుగు (Telugu)](../te/README.md) ·
> [मराठी (Marathi)](../mr/README.md) — পূর্ণ তালিকা [i18n hub](../README.md)-এ।

## এক লাইনে fak

**fak একটি Go binary** যা আপনার AI এজেন্ট আর তার tool calls-এর মাঝে বসে — প্রতিটি
tool call *চলার আগেই* যাচাই করে, আর দীর্ঘ session-এ বারবার আসা কাজ পুনর্ব্যবহার করে।
ফলাফল: একই agent loop **আরও নিরাপদ, সস্তা ও দ্রুত** — আর কিছু না বদলে।

আপনার এজেন্ট নতুন করে লিখতে হয় না — শুধু একটি base URL `fak serve`-এর দিকে ঘুরিয়ে
দিন; প্রতিটি tool call প্রথমে capability floor পেরিয়ে যায়।

```bash
fak guard -- claude    # আপনার বিদ্যমান এজেন্টকে এক কমান্ডে wrap করে
```

## ভারতীয় স্টার্টআপদের জন্য এটি কেন গুরুত্বপূর্ণ

- **খরচ টাকায় লাগে, token bill আসে ডলারে।** দীর্ঘ session-এ ভাগ-করা কাজ (system
  prompt, tool list-এর KV cache) fak পুনর্ব্যবহার করে — একটি tuned warm-cache
  stack-এর তুলনায় 50×5 run-এ **~4.1× কম কাজ** (naive re-send loop-এর তুলনায় ~60×;
  সৎ সংখ্যাটি 4.1×)। সঙ্গে per-aspect routing সস্তা অংশগুলো সস্তা model-এ পাঠায়।
  এটি সরাসরি margin-এর লিভার।
- **ডেটা দেশের ভেতরেই থাকুক (DPDP Act, 2023)।** fak self-host-first: একটি static
  binary যেকোনো **local model** বা দেশীয় provider-এর সামনে বসে, প্রতিটি backend-এ
  fail-closed residency, default-deny capability floor, আর প্রতিটি tool call-এর
  tamper-evident audit log দেয়। ডেটা আপনার মেশিন ছেড়ে বাইরে যায় না।
- **কোনো payment rail পেরোতে হয় না।** fak **Apache-2.0**, বিনামূল্যে, self-host —
  কার্ড নেই, cross-border invoice নেই, entity নেই। `git clone` আর `go install`-ই
  পুরো পথ।
- **একটিই static binary, শূন্য বাহ্যিক dependency।** ছোট দলের জন্য সহজ ops — কোনো
  sidecar নেই, আলাদা authorizer নেই। laptop থেকে fleet পর্যন্ত একই artifact; আপনি
  components নয়, শুধু flags যোগ করেন।

## fak কোন সমস্যাগুলো সমাধান করে

- **দীর্ঘ session আর দামি হয় না।** provider-এর prompt-cache ছাড় টেকে কেবল তখনই,
  যখন cached prefix byte-for-byte একই থাকে; fak মাঝের পুরনো turns সরিয়েও prefix-কে
  byte-identical রাখে — তাই ছাড় ভাঙে না।
- **default-deny নিরাপত্তা।** permission নীতি kernel-এর *ভেতরে*, একই call path-এ
  চলে। কোনো irreversible action আটকানো attack "ধরার" ওপর নির্ভর করে না — সেই লিভার
  কখনো যুক্তই হয়নি। এটি **fail-closed**, open নয়।
- **prompt injection / poisoned results প্রতিরোধ।** সন্দেহজনক tool *results* আলাদা
  quarantine-এ রাখা হয়, যাতে সেগুলো model-এর context-এ ঢুকতেই না পারে — structure
  দিয়ে, কোনো classifier দিয়ে নয়।

## ৬০-সেকেন্ডের প্রমাণ (key নেই, model নেই, GPU নেই)

```bash
go build -o fak ./cmd/fak
./fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"   # -> DENY (POLICY_BLOCK)
./fak preflight --policy examples/customer-support-readonly-policy.json --tool search_kb     --args "{}"   # -> ALLOW
./fak agent --offline                                                                                       # injection আটকানো হলো, task তবুও সম্পূর্ণ
```

## আপনার model-এর সঙ্গে

fak আপনার model বদলায় না — সেটিকে govern ও cache করে। **Qwen2/Qwen3 আর GLM-MoE**
in-kernel reference engine-এ bit-exact প্রমাণিত; বাকি সব (DeepSeek, Mistral, যেকোনো
open-weights model) OpenAI-compatible wire-এ front হয় — Ollama / vLLM / SGLang /
llama.cpp / LM Studio বা যেকোনো OpenAI-compatible API-এর মাধ্যমে।

## এরপর কোথায়

- [README (পূর্ণ পরিদর্শন)](../../../README.md)
- [START-HERE — ১০ মিনিটে local model](../../../START-HERE.md)
- [Getting Started — binary install করুন](../../../GETTING-STARTED.md)
- [Integrations — আপনার এজেন্ট যুক্ত করুন](../../integrations/README.md)
- [ডেটা residency ও compliance — DPDP Act-এর জন্য](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — প্রতিটি সংখ্যার উৎস](../../../BENCHMARK-AUTHORITY.md)
- [CLAIMS — কোনটি shipped/simulated/stub](../../../CLAIMS.md)

License: [Apache-2.0](../../../LICENSE)।
