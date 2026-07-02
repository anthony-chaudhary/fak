# Making fak attractive to India- and China-based startups (2026-06-30)

> Concept note / go-to-market strategy. Answers the `/goal`: *how do we make fak
> attractive to Indian-based startups — a wide range of things, including Hindi doc
> entry points, and the same for China.* It maps the full lever set, grades each as
> **shipped here / partial / proposed**, and names the honest gaps. The first lever —
> in-language entry points — ships alongside this note under
> [`docs/i18n/`](../i18n/README.md) as the concrete proof, not a promise.

## Why these two markets, and why fak fits

India and China are the two largest pools of startup engineering talent outside the US,
and both build agent products under constraints that happen to line up with fak's core
design — not with a hosted US SaaS:

1. **Cost is denominated in USD, revenue is not.** A token bill that is a rounding error
   for a US Series-B is a real margin line for an INR- or CNY-revenue seed startup.
   fak's whole value proposition is *the same agent loop, cheaper* — cache reuse across a
   fleet (~4.1× less work than a tuned warm-cache stack on the 50×5 run, ~60× vs. the
   naive re-send loop), fewer wasted turns, and per-aspect routing that sends the cheap
   aspects to a cheap model. Margin, not features, is the pitch.
2. **Data must often stay in-country.** India's DPDP Act (2023) and China's PIPL / DSL /
   CSL push personal and "important" data toward local processing and constrain
   cross-border transfer. fak is self-host-first: one static binary in front of a *local*
   model or a domestic provider, with fail-closed residency across backends, a
   default-deny capability floor, and a tamper-evident decision log. That is a compliance
   story a hosted API cannot tell.
3. **The models they already reach for are the ones fak already governs.** China's
   default open models — Qwen (Alibaba), GLM (Zhipu), DeepSeek, Yi, Baichuan, Kimi — are
   exactly the family fak fronts, and Qwen2/Qwen3 + GLM-MoE are proven **bit-exact in the
   in-kernel reference engine** (the rest are fronted over the OpenAI-compatible wire).
   fak is not asking a Chinese startup to switch models; it wraps the ones they run.
4. **Adoption has no payment rail to cross.** fak is Apache-2.0, free, self-host. There is
   no cross-border card-billing / entity / invoicing friction — the quiet tax that slows
   hosted-SaaS adoption in both markets. `git clone` and `go install` are the whole funnel.

## The lever set (wide range), graded honestly

| # | Lever | India | China | Status |
|---|---|---|---|---|
| 1 | **In-language doc entry points** | Hindi | Simplified Chinese | **Shipped here** — [`docs/i18n/`](../i18n/README.md) POC pages |
| 2 | **In-language answer-engine (AEO) discovery** | Hindi + English | Baidu/Zhihu/Juejin, zh terms | Proposed — extends `internal/marketing/aeo.go` |
| 3 | **Domestic-model fit** | open + IndiaAI models | Qwen/GLM/DeepSeek/Yi/Kimi | Partial — Qwen/GLM in-kernel; rest over the wire |
| 4 | **Data-residency / compliance framing** | DPDP Act 2023 | PIPL / DSL / CSL | Partial — capability exists; framing is new |
| 5 | **Cost framing in local unit-economics** | INR margin | CNY margin | Proposed — re-skin the benchmark story |
| 6 | **Domestic silicon via the compute HAL** | nascent (IndiaAI compute) | Ascend / Cambricon / Biren / Moore Threads | Partial — HAL ships CUDA/Vulkan; a domestic backend is a registration seam, not a shipped backend |
| 7 | **Access / distribution friction** | low (GitHub OK) | GOPROXY, ModelScope, Gitee mirror | Partial — GOPROXY tip is real & shipped in zh page; mirrors proposed |
| 8 | **Zero payment friction (Apache-2.0, self-host)** | ✓ | ✓ | **Already true** — no work needed, just say it |
| 9 | **Community channels** | dev communities, IndiaAI, campuses | WeChat, Zhihu, Bilibili, Juejin, Gitee | Proposed — GTM action, not code |
| 10 | **Per-market positioning line** | "cheaper agents on your stack, data in-country" | "govern + cache the domestic models you already run" | Proposed — copy, lands in the i18n pages |

### 1 — In-language documentation entry points (shipped here)

Native-language entry points do two things at once: they **lower the first-contact
barrier** for a non-native-English reader, and they **signal welcome** — a project that
put a Hindi and a 中文 front door up is visibly courting these developers. The honest scope
is an *entry point*, not a full doc-set translation: a compact, faithful page that carries
the one-line pitch, the 60-second proof, the install path, and the market-specific value
props, then hands off to the (English) deep docs. This is what ships with this note:

- [`docs/i18n/README.md`](../i18n/README.md) — the localization hub + contribution path.
- [`docs/i18n/hi/README.md`](../i18n/hi/README.md) — Hindi (हिन्दी) entry point.
- [`docs/i18n/zh/README.md`](../i18n/zh/README.md) — Simplified Chinese (简体中文) entry point.

**Fence:** these are machine-authored translations pending native review — the hub marks
them as such and asks for corrections via issue/PR. Do not claim "professionally
localized." Update 2026-07-01: the Indian-language follow-ons shipped — Tamil
([`ta/`](../i18n/ta/README.md)), Telugu ([`te/`](../i18n/te/README.md)), Bengali
([`bn/`](../i18n/bn/README.md)), Marathi ([`mr/`](../i18n/mr/README.md)) — each with the
same machine-authored fence. Remaining follow-ons: native review of all pages and
Traditional Chinese (TW/HK).

### 2 — In-language answer-engine discoverability

The repo already runs an AEO program (`internal/marketing/aeo.go`, `tools/seo_aeo_scorecard.py`,
`gen_structured_data.py`) so LLM answer engines surface fak for the right English terms.
Chinese developers ask Baidu / Zhihu / 掘金 (Juejin); many Indian developers search in
Hindi-English code-switch. The lever is to **emit in-language disambiguation terms and
structured data** so an answer engine responding in Hindi or Chinese names fak for
"agent 内核 / 工具调用防火墙" or "एजेंट कर्नेल". Gap: no localized terms are emitted today; this is
a bounded extension of the existing generator, not new infrastructure.

### 3 — Domestic-model fit (fak's strongest China hook)

China's agent startups overwhelmingly build on domestic open models under a
"self-host-first" posture. fak's position is not "use our model" — it is **"keep your
model; we govern and cache it."** Proven bit-exact in the in-kernel reference engine:
**Qwen2/Qwen3 and GLM-MoE**. Fronted unchanged over the OpenAI-compatible wire (Ollama /
vLLM / SGLang / llama.cpp / LM Studio, or a domestic API): **DeepSeek, Yi, Baichuan, Kimi,
and any open-weights model.** For India, the same wire fronts open models and the emerging
IndiaAI / Sarvam / Krutrim endpoints. Honest split: in-kernel decode is proven only for
the listed architectures; everything else gets the full capability / cache / audit / route
value over the wire, just not in-kernel decode.

### 4 — Data-residency & compliance framing

fak already has the *mechanism* (self-host, fail-closed residency, default-deny floor,
`X-Trace-Id`-correlated audit log). What is missing is the *framing* aimed at these two
regulatory regimes:

- **India — DPDP Act 2023:** run inference and tool execution on infrastructure you
  control, keep personal data on-box, and hand an auditor a tamper-evident decision log
  for every tool call. fak is the enforcement boundary, not another data processor.
- **China — PIPL / DSL / CSL:** the same self-host boundary keeps "important data" and
  personal information in-country; the capability floor is *structural* (an effect the
  model cannot request is one it cannot leak), which is a stronger claim than a classifier.

This ships here as the [data-residency & compliance explainer](../explainers/data-residency-and-compliance.md)
— the mechanism mapped to both regimes, with the honest "not legal advice, not a
certification" fence. No new code; the boundary already had the properties.

### 5 — Cost framing in local unit-economics

Re-tell the existing (honest) benchmark story in the language a cost-sensitive founder
reads: cost per 1,000 agent turns, margin per seat, the self-host fleet reuse win. Keep
the net-true-value discipline — quote the tuned baseline (~4.1×), not the naive 60×, as
the headline. No new measurements; a re-skin of `BENCHMARK-AUTHORITY.md` numbers.

### 6 — Domestic silicon via the compute HAL

Under export controls, Chinese startups increasingly target **Huawei Ascend, Cambricon,
Biren, Moore Threads**; India's domestic compute (IndiaAI) is nascent and mostly NVIDIA
today. fak's `internal/compute` HAL + [neo-silicon onboarding](../vendor/neo-silicon-onboarding.md)
is the seam to add a `compute.Backend` by registration rather than re-forking the forward
pass — a genuine "bring-your-accelerator" story. **Fence:** the HAL ships CUDA and Vulkan
backends; a domestic-accelerator backend is a *registration path we support*, not a
backend we have shipped or benchmarked. Say "the seam is ready," never "we run on Ascend."

### 7 — Access & distribution friction (China-specific, partly real today)

- **`go install` in China needs a module proxy.** The accurate, shipped tip (in the zh
  page): `GOPROXY=https://goproxy.cn,direct`. This is a true, high-value onboarding detail
  — the default `proxy.golang.org` is unreliable from the mainland.
- **Model downloads:** Hugging Face is frequently unreachable; **ModelScope (魔搭)** mirrors
  the same GGUF/weights. Proposed: list ModelScope URLs beside the HF ones in the zh page.
- **Repo mirror:** a **Gitee** mirror of the repo lowers clone friction. Proposed (GTM), not
  built — do not claim a mirror exists until one does.

### 8 — Zero payment friction (already true — just say it)

Apache-2.0 + self-host means **no card, no cross-border invoice, no entity** stands between
a developer and adoption. This removes the single biggest hosted-SaaS adoption tax in both
markets. It costs nothing to ship because it is already the license; the lever is to
*state it prominently* in the localized pages (both do).

### 9 & 10 — Community channels and per-market positioning

Go-to-market actions, named for completeness, not built here: seed the China-facing
channels (WeChat 公众号, Zhihu, Bilibili, 掘金, a Gitee mirror) and the India-facing ones
(dev communities, IndiaAI, campus/hackathon ecosystems), each with the per-market
positioning line from the table. These belong to a human GTM owner; this note's job is to
make the docs and the product ready for them.

## What ships with this note vs. what is the next checkable step

**Ships now (docs-only, no code, green under the doc gates):**
- This strategy note.
- The [`docs/i18n/`](../i18n/README.md) hub + Hindi + Simplified-Chinese entry points
  (lever 1), including the real `GOPROXY=https://goproxy.cn` onboarding tip (lever 7) and
  the zero-payment-friction + residency framing (levers 4, 8) in-language.
- The [data-residency & compliance explainer](../explainers/data-residency-and-compliance.md)
  (lever 4) — the DPDP / PIPL mapping as a real page, linked from both localized front doors.

**Not yet — the honest follow-ons, each a bounded next step:**
- Localized AEO terms + structured data (lever 2) — extend `internal/marketing/aeo.go`.
- ModelScope download URLs + a Gitee mirror (lever 7) — needs a real mirror before the claim.
- Native review of the machine-authored translations (lever 1 fence).
- GTM channel seeding + per-market landing copy (levers 9, 10) — human-owned.

Reported as `not yet` where unproven, per the repo's honesty contract: the mechanism for
every lever above already exists in fak; what is new here is the framing, the in-language
front doors, and one accurate onboarding fix — not a claim that any market has adopted it.
