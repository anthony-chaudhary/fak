---
title: "fak — প্রথম-পরিচয়ের FAQ (বাংলা / Bengali first-contact FAQ)"
description: "fak নিয়ে প্রথম প্রশ্নগুলোর সংক্ষিপ্ত বাংলা উত্তর: এটি কী, model বদলাতে হয় কি না, ডেটা কোথায় থাকে (DPDP Act, 2023), খরচ কত, কতটা সস্তা/দ্রুত, কোন model চলে, prompt injection কীভাবে আটকায়, আর install পথ। self-host, Apache-2.0."
---

# fak — প্রথম-পরিচয়ের FAQ (বাংলা)

> এটি একটি **স্থানীয়কৃত প্রবেশ-পৃষ্ঠা (entry point)** — পুরো ডকুমেন্টেশনের অনুবাদ নয়।
> পূর্ণ ডকুমেন্টেশন ইংরেজিতে — এই পৃষ্ঠা প্রথম প্রশ্নগুলোর সংক্ষিপ্ত উত্তর দিয়ে আপনাকে
> [ইংরেজি ডকস](https://github.com/anthony-chaudhary/fak/blob/main/README.md)-এ পৌঁছে দেয়।
> **দ্রষ্টব্য:** এই অনুবাদ যন্ত্রে তৈরি; native পর্যালোচনা বাকি — সংশোধনের জন্য
> issue/PR খুলুন।
>
> **অন্যান্য পৃষ্ঠা:** [বাংলা প্রবেশ-পৃষ্ঠা](./README.md) · [দ্রুত-শুরু](./quickstart.md) ·
> [ইনস্টল](./install.md) · সম্পূর্ণ ভাষা-তালিকা [i18n hub](../README.md)-এ।

## Q1. fak কী?

একটিই static Go binary, যা আপনি ইতিমধ্যে চালানো AI এজেন্টের (Claude Code, Codex,
Cursor — যেকোনো OpenAI/Anthropic/MCP client) সামনে বসিয়ে দেন, শুধু একটি base URL
ঘুরিয়ে, কোনো rewrite ছাড়াই। এটি দীর্ঘ session সস্তা করে (পুরনো turns সরিয়ে দেয়, অথচ
provider-এর prompt-cache prefix byte-identical রাখে), প্রতিটি tool call রুট করে,
প্রয়োজনে local GGUF model in-process চালাতে পারে, আর প্রতিটি call-এর জন্য একটি
auditable verdict লিখে রাখে।

## Q2. আমাকে কি model বদলাতে বা এজেন্ট নতুন করে লিখতে হবে?

না। fak আপনার ইতিমধ্যে ব্যবহৃত model-কেই govern ও cache করে। এক কমান্ডে wrap করুন —
`fak manage -- claude` — অথবা একটি base URL `fak serve`-এর দিকে ঘুরিয়ে দিন।

## Q3. আমার ডেটা কোথায় যায় — এটি কি অনুবর্তী (compliant)?

self-host-first: একটি static binary যা একটি local model বা দেশীয় provider-এর সামনে
বসে, সঙ্গে fail-closed residency, default-deny capability floor, আর প্রতিটি tool
call-এর tamper-evident audit log। আপনার ডেটা আপনার মেশিন ছেড়ে বাইরে যায় না। এটি
সরাসরি **DPDP Act, 2023**-এর data-residency চাহিদার সঙ্গে মেলে — ডেটা দেশের ভেতরেই
থাকে, কোনো cross-border প্রবাহ নেই।

## Q4. খরচ কত? এটি কি সত্যিই বিনামূল্যে?

**Apache-2.0**, বিনামূল্যে, self-host। কার্ড নেই, cross-border invoice নেই, entity
নেই। খরচ টাকায় (₹) লাগে, অথচ কোনো ভেন্ডর-বিল নেই — `git clone` আর `go install`-ই
পুরো পথ।

## Q5. কতটা সস্তা বা দ্রুত?

মাপা একটি 50-turn × 5-agent session-এ, একটি **TUNED warm-cache stack**-এর তুলনায়
প্রায় **4.1× কম কাজ** — এটিই সৎ শিরোনাম-সংখ্যা, আর এটিই আপনার token-বিল টাকায় (₹)
কমানোর সরাসরি লিভার। **~60×** সংখ্যাটি (প্রায় ১৯ ঘণ্টা থেকে প্রায় ১৯ মিনিট) কেবল
একটি naive re-send-everything loop-এর তুলনায় সত্য — কখনো শিরোনাম হিসেবে নয়। এই
পুনর্ব্যবহারের সুবিধা কেবল self-host-এ, read-heavy fleet-এর জন্য।

## Q6. কোন model চলে?

**Qwen2/Qwen3 আর GLM-MoE** in-kernel reference engine-এ bit-exact প্রমাণিত। বাকি সব
(DeepSeek, Mistral, যেকোনো open-weights model) OpenAI-compatible wire-এ front হয় —
Ollama / vLLM / SGLang / llama.cpp / LM Studio।

## Q7. এটি prompt injection কীভাবে আটকায়?

দুটি structural gate, কোনো classifier নয়: একটি default-deny capability floor (বিপজ্জনক
tool কখনো allow-list-এ থাকে না) আর result quarantine (বিষাক্ত tool result কখনো model-এর
context-এ পৌঁছায় না)। live test-এ, injection অরক্ষিত baseline-এ পৌঁছেছিল **5/5**, আর
fak সেটি প্রাচীরবদ্ধ করেছে **5/5**।

## Q8. কীভাবে install করব?

`go install github.com/anthony-chaudhary/fak/cmd/fak@latest`, অথবা একটি clone থেকে
`go build -o fak ./cmd/fak`। যদি `proxy.golang.org` অপ্রাপ্য হয়, তবে install-এর আগে
একটি স্থানীয় Go module proxy সেট করুন (`GOPROXY`)।

## Q9. এরপর কোথায়?

- [README (পূর্ণ পরিদর্শন)](https://github.com/anthony-chaudhary/fak/blob/main/README.md)
- [START-HERE — ১০ মিনিটে local model](https://github.com/anthony-chaudhary/fak/blob/main/START-HERE.md)
- [Getting Started — binary install করুন](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md)
- [Integrations — আপনার এজেন্ট যুক্ত করুন](../../integrations/README.md)
- [ডেটা residency ও compliance — DPDP Act, 2023-এর জন্য](../../explainers/data-residency-and-compliance.md)
- [BENCHMARK-AUTHORITY — প্রতিটি সংখ্যার উৎস](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)
- [CLAIMS — কোনটি shipped/simulated/stub](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md)

License: [Apache-2.0](https://github.com/anthony-chaudhary/fak/blob/main/LICENSE)।
