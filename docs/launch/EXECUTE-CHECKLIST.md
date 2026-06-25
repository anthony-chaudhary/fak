---
title: "fak launch — the fire-when-ready execution checklist"
description: "One page to run the human-gated launch (HN / Reddit / X / Lobsters) from the existing paste-ready kit. Pre-flight checks, the proven sequence, and exactly which file to paste from each step."
---

# Execute checklist — the human-gated launch

> This is the **do-it** companion to [`README.md`](README.md) (the kit index). The posts
> themselves are already written, fact-checked, and fenced in the files below. This page is
> the order to fire them in, the checks to run first, and the one rule that governs all of it.
>
> **Why a human runs these:** Reddit / Hacker News / X / Bluesky / Lobsters forbid undisclosed
> automation and zero out manipulated votes. Posting must come from your account, with your
> judgment on timing and live replies. Everything here is paste-ready so that takes one sitting.

## The one rule (don't skip)

**Lead with the fence — it's the hook, not the caveat.** Every target audience is allergic to
AI launch-speak and rewards self-skepticism. Make your own first comment the prosecution: name
the weaknesses before the top commenter does. The five fences that *are* the credibility:

- the injection detector is **~100% evadable by design** — explicitly not the floor;
- the perf headline is **~1.5–4.1× vs a tuned warm-cache stack**, never the naive 8.8–9.7× alone;
- the prior-art audit scored **0/29 novel** — the contribution is the assembly;
- power/energy/$ figures are **simulated**; the ~60× / "agent city" numbers are **design targets**;
- fak is **not a faster token engine** — the contrast is operational surface, not tok/s.

## Pre-flight (run all four before posting anything)

- [ ] **The 60-second proof reproduces on a clean checkout:**
      `go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"`
      → `DENY (POLICY_BLOCK)`; same with `--tool search_kb` → `ALLOW`.
- [ ] **Live demos load:** <https://anthony-chaudhary.github.io/fak/demos.html> (not a 404).
- [ ] **Colab opens:** the quickstart notebook link from the README renders + runs.
- [ ] **Social card resolves:** paste the repo URL into any link-preview tool — the OG image
      (`visuals/social-preview.png`) must render, not 404.

## The sequence that works (upstream-first)

Creators and newsletters mine the Show HN front page and r/LocalLLaMA for topics, so land the
technical primary first, then let the rest follow the signal.

| # | When | Channel | Paste from | The move |
|---|---|---|---|---|
| 1 | Day 0, morning | **r/LocalLLaMA** (PRIMARY) | [`reddit-localllama.md`](reddit-localllama.md) | Mechanism-titled post on mid-run KV eviction (`max\|Δ\|=0`), demo embedded, Colab one click away. You first-comment the oracle-parity table **and** the tuned 4.1× number. |
| 2 | Day 0 same day / Day 1 | **Show HN** | [`show-hn.md`](show-hn.md) | Syscall-intuition title (not "security"). Post the prosecution-first author comment immediately; answer every technical reply fast for ~2 hours. |
| 3 | After the primary has signal | **X / Bluesky** | [`x-thread.md`](x-thread.md) | Tag the *vocabulary* (lethal-trifecta / capability lock / result quarantine), not the product. Each post paired with an existing visual. |
| 4 | After the primary has signal | **Lobsters** | [`lobsters-and-blog.md`](lobsters-and-blog.md) | Idea-first essay (the bit-exact KV eviction + ed25519 cert writeup). **Needs an aged account** — see [`landscape-research.md`](landscape-research.md). |
| 5 | Opportunistic | **Other subs** | [`reddit-other-subs.md`](reddit-other-subs.md) | r/golang, r/selfhosted, r/netsec, r/LLMDevs, r/AI_Agents — each tuned to that sub's self-promo rule. r/selfhosted repo link goes in the **weekly thread**, not standalone. |
| 6 | Hand-off | **YouTube creators** | [`youtube-demo-script.md`](youtube-demo-script.md) · [`untrusted-program-talk.md`](untrusted-program-talk.md) | Pre-cut script for a short, or the per-creator angle table in [`positioning-brief.md`](positioning-brief.md) (Fireship / LiveOverflow / Primeagen / Cole Medin). |

## Universal rules (every channel)

- **Disclose authorship in the first line** — "disclosure: I built this" *raises* trust here.
- **Never ask for upvotes.** Vote-ring detection silently zeroes them and flags the post.
- **Strip the Provenance & fact-check appendix** from each file before pasting — it's your audit
  trail, not part of the post.
- **Lead the comment, not the post, with the parity tables and the honest baseline** so the post
  stays about the mechanism.

## What's already handled for you (no action needed)

These durable surfaces are being driven separately from this human-gated list:

- **GitHub discoverability** — topics (20 set), description, custom OG image, CITATION.cff: live.
- **Docs-site SEO** — sitemap.xml (275 URLs), robots.txt, jekyll-seo-tag (title/description/OG/
  Twitter/4× JSON-LD), pkg.go.dev indexing: all verified live.
- **Awesome-list PRs + registry/directory submissions** — researched and staged for review (the
  external-backlink half of SEO). See the distribution working notes.

The durable surfaces compound for months; this human-gated list is the one-day spike that seeds them.
