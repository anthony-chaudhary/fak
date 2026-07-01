# Localized entry points for fak (i18n)

fak's canonical documentation is English. This directory holds **localized entry-point
pages** — a compact, faithful front door in another language that carries the one-line
pitch, the 60-second proof, the install path, and the value props that matter most in that
market, then hands off to the full (English) docs.

The scope is deliberate: an **entry point, not a full translation** of the ~hundreds of
docs. The goal is to lower first-contact friction for a non-native-English reader and to
signal that the project welcomes them — then get them to the working proof fast.

## Available languages

| Language | Page | Market focus |
|---|---|---|
| हिन्दी (Hindi) | [`hi/README.md`](hi/README.md) | India — cost in INR terms, DPDP-Act data residency, self-host, one binary |
| 简体中文 (Simplified Chinese) | [`zh/README.md`](zh/README.md) | China — domestic models (Qwen/GLM/DeepSeek), PIPL residency, `GOPROXY` onboarding |

## Status & honesty fence

These pages are **machine-authored translations pending native review.** They are meant to
be correct in substance and idiomatic enough to be useful, but they have not been reviewed
by a native technical writer. If you spot a mistranslation, an awkward phrasing, or a
stale link, please open an issue or PR — corrections are welcome and credited.

Every technical claim in a localized page must match the English source of truth
([`README.md`](../../README.md), [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md),
[`CLAIMS.md`](../../CLAIMS.md)). Localization changes the *language*, never the *claim* —
the same net-true-value discipline applies, so a localized page quotes the tuned baseline
(~4.1×), never the naive headline, exactly as the English docs do.

## Adding a language

1. Create `docs/i18n/<code>/README.md` using a two-letter (or `zh-Hant`-style) code.
2. Translate the **structure** of an existing entry-point page — do not invent new claims;
   mirror the English source and keep all commands, flags, and code blocks in English.
3. Add a row to the table above and a line in [`INDEX.md`](../../INDEX.md).
4. Keep it compact: entry point + hand-off, not a doc-set fork.

The broader go-to-market rationale (why these markets, the full lever set, what is shipped
vs. proposed) lives in
[the emerging-market adoption note](../notes/CONCEPT-EMERGING-MARKET-ADOPTION-2026-06-30.md).
