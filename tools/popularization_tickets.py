#!/usr/bin/env python3
"""Generate the 50 concept-popularization tickets as self-contained work units.

Each ticket is a dict: dim (dimension letter), title (Conventional-Commits style),
concepts (which of the 5 core concepts it serves), body (markdown, self-contained
with a Deliverable / Why-it-popularizes / Acceptance / Lane triplet). Writing them
here keeps the 50 bodies consistent and reviewable; `--emit-files` drops one .md per
ticket for filing, `--list` prints a table.

This is authoring scaffold, not runtime code — it never ships in the binary.
"""
from __future__ import annotations
import argparse
import json
import os

# Dimension -> human name (mirrors the epic doc table).
DIMS = {
    "A": "Explainer content",
    "B": "Visual & diagram assets",
    "C": "Interactive & runnable demos",
    "D": "Positioning & comparison",
    "E": "Social proof & community",
    "F": "Developer experience & onramp",
    "G": "Integration recipes",
    "H": "Benchmark-as-story",
    "I": "Memorable framing & naming",
    "J": "Distribution & channels",
    "K": "Adoption measurement",
}

# The five core concepts (short keys used in each ticket's `concepts` field).
CONCEPTS = {
    "syscall": "Treat the tool call like a syscall",
    "dos": "Verify, don't trust (DOS)",
    "kvcache": "Addressable bit-exact KV cache",
    "capgate": "Default-deny capability gate + quarantine",
    "binary": "One static Go binary, drop-in",
}


def T(dim, title, concepts, deliverable, why, acceptance, lane):
    return {
        "dim": dim,
        "title": title,
        "concepts": concepts,
        "deliverable": deliverable.strip(),
        "why": why.strip(),
        "acceptance": acceptance.strip(),
        "lane": lane,
    }


TICKETS = [
    # ---------------- A. Explainer content (8) ----------------
    T("A", "docs(explainer): \"the tool call is a syscall\" — the one-page mental model",
      ["syscall", "capgate"],
      "A single, tight explainer at `docs/explainers/tool-call-is-a-syscall.md` that teaches the "
      "core analogy from scratch: an OS kernel does not trust a user program's word that a write "
      "is safe — the syscall crosses a boundary the program does not control. fak does the same to "
      "the LLM. Open with the analogy, show the proposed-call -> verdict path, end with the one "
      "sentence a reader can repeat. ~600-900 words, one diagram reference (link to the B-dimension "
      "asset when it lands), zero jargon before it is defined.",
      "This is the keystone explainer. Everyone who 'gets' fak got it through this analogy; right "
      "now it is scattered across the README and llms.txt. A dedicated, linkable page is the thing "
      "people share when they explain fak to someone else.",
      "File exists with SEO front-matter (`title:`/`description:`); linked from `docs/explainers/` "
      "index and `INDEX.md`; a non-expert can restate 'model proposes, kernel disposes' after "
      "reading; `python tools/seo_aeo_scorecard.py` does not regress.",
      "docs-explainers"),
    T("A", "docs(explainer): \"why default-deny beats a classifier\" for prompt injection",
      ["capgate"],
      "`docs/explainers/default-deny-vs-classifier.md`: why a capability lock that was never wired "
      "up cannot be argued past, whereas a recognizer that must *catch* an attack fails open. Use a "
      "concrete injected-prompt walk-through: the classifier approach vs the fak approach, same "
      "attack, different outcome. Tie to OWASP Agentic Top-10 / MCP Top-10 by name.",
      "The single most common objection is 'isn't this just a guardrail/classifier?'. A crisp "
      "explainer that draws the structural line converts skeptics and is citable in comparison "
      "threads.",
      "File exists with front-matter; contains a before/after attack walk-through; names the OWASP "
      "and MCP top-10 items it addresses; linked from FAQ and explainers index.",
      "docs-explainers"),
    T("A", "docs(explainer): \"how a long agent session stops getting expensive\"",
      ["kvcache"],
      "`docs/explainers/long-session-economics.md`: plain-language account of why a growing "
      "transcript re-sends everything each turn, why the provider discount only survives while the "
      "prefix is byte-identical, and how splicing on original bytes (a memcpy, never a re-marshal) "
      "keeps the discount alive. Include the honest boundary: fak guarantees byte-identical prefix; "
      "the provider decides reuse.",
      "Cost is the emotional hook for the largest audience (anyone paying for long agent runs). A "
      "readable economics explainer turns an abstract cache claim into felt money saved.",
      "File exists with front-matter; the honest 'fak relays the provider's number' fence is "
      "present; one worked cost example; linked from explainers index and `INDEX.md`.",
      "docs-explainers"),
    T("A", "docs(explainer): \"what DOS actually checks\" — the verify-don't-trust primer",
      ["dos"],
      "`docs/explainers/verify-dont-trust.md`: the DOS primer for a newcomer — a commit subject is "
      "forgeable, the diff is not; a 'done' claim is checked from git evidence; a refusal carries a "
      "reason from a closed vocabulary; a recalled memory is re-verified at read time. Ground each "
      "in the real verb (`dos verify`, `dos commit-audit`, `dos recall`).",
      "DOS is the most transferable idea (it works with zero fak internals) but the least explained "
      "to outsiders. A standalone primer makes the trust-substrate pitch portable to any "
      "agent-fleet audience.",
      "File exists with front-matter; every claim maps to a real `dos` verb; a reader can name the "
      "three things DOS re-checks; linked from explainers index.",
      "docs-explainers"),
    T("A", "docs(explainer): \"the addressable KV cache in 5 minutes\" (concept, not internals)",
      ["kvcache"],
      "`docs/explainers/addressable-kv-cache-in-5-min.md`: a gentler on-ramp than the existing "
      "internals page — analogy first (a cache you can reach into and surgically edit vs a cache "
      "you can only append to), then the one property that matters (evict a poisoned span, cache "
      "stays bit-exact at `max|Δ|=0`), then a link to the deep page for the mechanics.",
      "The existing addressable-KV page is correct but dense. A 5-minute version widens the funnel: "
      "the concept is genuinely novel-feeling to most readers and deserves an accessible door.",
      "File exists with front-matter; strictly gentler than the existing page (defined terms, one "
      "analogy, no code); cross-links to the deep page; linked from explainers index.",
      "docs-explainers"),
    T("A", "docs(explainer): \"fak is not a serving engine\" — the honest boundary",
      ["binary"],
      "`docs/explainers/what-fak-is-not.md`: a short, disarming page that says what fak does NOT do "
      "— it does not replace vLLM/SGLang/llama.cpp for raw tokens/sec; it fronts them for the agent "
      "boundary. State the 0/29-novel prior-art audit plainly and reframe it as the point (the "
      "contribution is the assembly). Detectors are evadable by design; the floor is the cap lock.",
      "Honesty is attractive and rare in this space. A page that volunteers the limits builds more "
      "trust than any hype page, and preempts the 'they're overclaiming' takedown.",
      "File exists with front-matter; states the serving-engine boundary and the 0/29 audit without "
      "hedging; frames scope as a feature; linked from FAQ and explainers index.",
      "docs-explainers"),
    T("A", "docs(explainer): \"engineering is building loops\" — the worldview page, tightened",
      ["syscall"],
      "Tighten and elevate the existing `docs/explainers/engineering-is-building-loops.md` into a "
      "shareable manifesto-style piece: modern engineering is increasingly the act of building "
      "agentic loops (observe->orient->decide->act->verify), and fak is the kernel they run on. Cut "
      "to essentials, add a memorable close line, ensure it reads standalone.",
      "The worldview framing is the emotional/intellectual hook that makes fak feel inevitable "
      "rather than optional. A punchy version is the piece people quote in talks.",
      "Edited page reads standalone (no prerequisite jargon), has a quotable closing line, and does "
      "not lose any existing accurate claim; SEO scorecard does not regress.",
      "docs-explainers"),
    T("A", "docs(explainer): a 12-term glossary of the fak/DOS vocabulary",
      ["syscall", "dos", "kvcache", "capgate"],
      "`docs/explainers/glossary.md`: one-line, plain definitions for the vocabulary a newcomer "
      "trips on — verdict, capability floor, quarantine, addressable KV cache, bit-exact eviction, "
      "witness, lease/arbitration, refusal reason, recall re-verification, syscall boundary, "
      "prefix-cache discount, fail-closed. Each term links to its explainer.",
      "A shared vocabulary is how a concept spreads — people can only repeat what they can name. A "
      "glossary is also the highest-leverage SEO/AEO artifact (each term is a query).",
      "File exists with front-matter; >=12 terms, each one line + a link; every linked target "
      "exists; linked from `INDEX.md` and explainers index.",
      "docs-explainers"),

    # ---------------- B. Visual & diagram assets (5) ----------------
    T("B", "docs(visual): the \"proposed call -> kernel -> verdict\" flow diagram (SVG)",
      ["syscall", "capgate"],
      "A clean, brand-neutral SVG at `docs/adoption/diagrams/syscall-flow.svg` (+ a source "
      "`.md`/mermaid or the dataviz-skill palette) showing: agent proposes a tool call -> fak "
      "kernel adjudicates (ALLOW / DENY / TRANSFORM / REQUIRE_WITNESS) -> allowed call executes / "
      "denied call never runs / result passes quarantine before entering context. Light+dark safe.",
      "The syscall analogy is a picture in people's heads; giving them the actual picture makes it "
      "shareable in a slide, a README, a tweet. This is the single most reused asset.",
      "SVG renders in light and dark; four verdict types shown; used in the syscall explainer and "
      "README; follows the dataviz skill's palette/accessibility rules.",
      "docs-adoption-visuals"),
    T("B", "docs(visual): before/after cost curve — long session with vs without fak",
      ["kvcache"],
      "A chart asset at `docs/adoption/diagrams/cost-curve.svg` built from the existing "
      "`tools/cache_curve.py` output (calibrated to the measured 96.6% ceiling): billed prompt "
      "tokens per turn, naive re-send vs fak prefix-preserving, over a 50-turn session. Label the "
      "witnessed numbers; do NOT invent data.",
      "A cost curve is the most persuasive single image for the paying audience — it makes the "
      "KV-cache concept felt, not argued.",
      "SVG generated from real `cache_curve.py` data (not hand-drawn numbers); axes/units labeled; "
      "witnessed-vs-modeled distinction visible; used in the long-session explainer.",
      "docs-adoption-visuals"),
    T("B", "docs(visual): the DOS \"forgeable vs witnessed\" two-column diagram",
      ["dos"],
      "`docs/adoption/diagrams/forgeable-vs-witnessed.svg`: two columns — LEFT 'what the agent "
      "says' (commit subject, 'tests pass', a recalled memory) marked forgeable; RIGHT 'what git "
      "proves' (the diff, the file set, ancestry) marked witnessed. A single arrow: DOS trusts the "
      "right, re-checks the left.",
      "The forgeable/witnessed split is the entire DOS thesis in one image. It is the asset that "
      "makes the verify-don't-trust idea instantly graspable.",
      "SVG renders light+dark; the two columns and the trust arrow are unambiguous; used in the "
      "verify-don't-trust explainer.",
      "docs-adoption-visuals"),
    T("B", "docs(visual): the single-binary vs multi-process-stack contrast diagram",
      ["binary"],
      "`docs/adoption/diagrams/single-binary.svg`: LEFT the usual governed-serving stack (reverse "
      "proxy + policy layer + audit sidecar + quarantine service, multi-process) vs RIGHT one fak "
      "box containing gateway + capability floor + quarantine + audit. Same responsibilities, one "
      "process.",
      "The operational-surface pitch ('you add flags, not components') is abstract until you see "
      "the box count collapse from five to one. This visual sells the ops story.",
      "SVG renders light+dark; responsibilities are labeled identically on both sides so the "
      "collapse is the only difference; used in the README and the 'what fak is not' explainer.",
      "docs-adoption-visuals"),
    T("B", "docs(visual): a printable one-page \"fak concept card\" (PDF-ready)",
      ["syscall", "dos", "kvcache", "capgate", "binary"],
      "`docs/adoption/concept-card.md` designed to render to a single page: the five concepts, each "
      "one sentence + a tiny icon/diagram reference, plus the install one-liner and the 60-second "
      "proof command. The thing you hand someone at a meetup.",
      "A one-pager is the physical/portable form of the pitch — conference tables, internal "
      "channels, onboarding decks. It forces the whole story to fit on one page, which sharpens it.",
      "File renders cleanly to one page; all five concepts present in one sentence each; install "
      "one-liner and proof command are copy-pasteable and correct; linked from `INDEX.md`.",
      "docs-adoption-visuals"),

    # ---------------- C. Interactive & runnable demos (6) ----------------
    T("C", "examples(demo): 60-second \"deny an irreversible call\" runnable proof (no key, no GPU)",
      ["capgate", "binary"],
      "`examples/deny-in-60s/`: a self-contained script + README that, with only the fak binary and "
      "no provider credential or GPU, proposes an irreversible tool call, shows the kernel DENY verdict, and "
      "shows the same call ALLOWed under a permissive policy — proving the floor is structural. Uses "
      "the offline adjudication path.",
      "The fastest possible 'I saw it work' moment. A reader who runs one command and watches a "
      "dangerous call get refused is converted in under a minute — the highest-conversion asset.",
      "Runs offline with no key/GPU; one command; prints the DENY then the ALLOW; README states the "
      "expected output; listed in the examples index.",
      "examples-deny-in-60s"),
    T("C", "examples(demo): \"catch an over-claiming commit\" — dos commit-audit in 60 seconds",
      ["dos"],
      "`examples/commit-audit-in-60s/`: a script that creates a throwaway git repo, makes a commit "
      "whose subject claims 'fix: handle nulls' but whose diff only edits a README, then runs "
      "`dos commit-audit` and shows the CLAIM_UNWITNESSED verdict — vs a commit that actually does "
      "what it says returning OK.",
      "This makes the verify-don't-trust idea concrete and self-serving: the reader watches a lie "
      "get caught from evidence. It also demonstrates DOS works on any git repo, zero fak internals.",
      "Runs standalone (creates its own temp repo); shows both the caught claim and the clean "
      "commit; no network needed; README documents expected verdicts; in examples index.",
      "examples-commit-audit-60s"),
    T("C", "examples(demo): \"evict a poisoned span, cache stays bit-exact\" walkthrough",
      ["kvcache"],
      "`examples/addressable-evict/`: a runnable demonstration (or a faithful recorded transcript "
      "if it needs a model, clearly labeled) that evicts one span from a kept run and shows the "
      "post-eviction cache is bit-for-bit identical to a run that never saw it (`max|Δ| = 0`).",
      "The addressable-KV claim is the most 'no one else does this' concept; a runnable proof of "
      "`max|Δ|=0` is the demo that makes people believe it instead of nodding politely.",
      "Either runs end-to-end, or ships a clearly-labeled recorded transcript with the exact command "
      "to reproduce; the `max|Δ|=0` line is shown; README states requirements honestly; in index.",
      "examples-addressable-evict"),
    T("C", "examples(demo): \"quarantine a poisoned tool result\" before it enters context",
      ["capgate"],
      "`examples/quarantine-demo/`: feed a tool result shaped like a prompt-injection / "
      "secret-exfil into the result-admission path, show it comes back QUARANTINE with the bytes "
      "held out of context, and show the taint high-water mark rise so a later egress is gated.",
      "Prompt-injection is the fear that drives the most searching. A demo where the poison is "
      "structurally paged out — not classified — is the memorable proof of the containment story.",
      "Runs offline; shows the QUARANTINE verdict and the withheld bytes; README explains what was "
      "injected and why structure (not a classifier) caught it; in examples index.",
      "examples-quarantine-demo"),
    T("C", "examples(demo): a browser-free interactive `fak` playground script",
      ["syscall", "binary"],
      "`examples/playground/`: a single interactive shell/Go script that walks a first-timer through "
      "proposing several tool calls and reading the verdicts, with prompts ('now try an rm -rf — "
      "watch it get denied'). A guided REPL, not a wall of docs.",
      "Guided interactivity beats reading. A playground that narrates each verdict as the user pokes "
      "at it turns a passive reader into an active one, which is how concepts stick.",
      "Runs offline with just the binary; guides through >=3 distinct verdicts; each step explains "
      "what happened; README lists it as 'try this first'; in examples index.",
      "examples-playground"),
    T("C", "docs(demo): an asciinema-style recorded terminal cast of the 60-second proof",
      ["binary", "capgate"],
      "Record (as a checked-in cast file + a still frame, or a faithful annotated transcript) the "
      "install-to-first-verdict flow so a reader can watch it without running anything. Store under "
      "`docs/adoption/casts/`. Link from README's 'Start here'.",
      "A recorded cast is the zero-friction version of a demo — it plays in a browser or a README "
      "and gets the 'oh, that's it?' reaction that written steps cannot. High shareability.",
      "Cast/transcript checked in; shows install -> first DENY verdict; annotated so it reads "
      "without audio; linked from README; honest about being a recording.",
      "docs-adoption-casts"),

    # ---------------- D. Positioning & comparison (5) ----------------
    T("D", "docs(compare): \"fak vs a guardrails library\" honest side-by-side",
      ["capgate"],
      "`docs/adoption/compare/vs-guardrails.md`: a fair, sourced comparison against the guardrails "
      "class (Guardrails AI, NeMo Guardrails, LlamaGuard). Draw the line: they recognize/classify "
      "content (fail-open); fak is a default-deny capability gate on the call path (fail-closed). "
      "Give them their real strengths; state where each fits.",
      "The '#1 question' audiences ask is 'how is this different from X'. A fair comparison that "
      "concedes the other tool's strengths is more persuasive than a strawman and gets cited.",
      "File exists with front-matter; each compared tool's real strength is stated; the fail-open "
      "vs fail-closed distinction is the spine; every factual claim about a competitor is sourced or "
      "hedged; in the compare index.",
      "docs-adoption-compare"),
    T("D", "docs(compare): \"fak vs vLLM/SGLang\" — different boundary, not a race",
      ["binary"],
      "`docs/adoption/compare/vs-serving-engines.md`: the operational-surface positioning — fak "
      "does not compete on tokens/sec; it owns the governance band those engines leave open. Use the "
      "single-binary vs multi-process contrast. Explicitly recommend using vLLM/SGLang for raw "
      "throughput and fronting them with `fak serve`.",
      "Preempts the most damaging misread ('they claim to beat vLLM'). Positioning fak as a "
      "complement that recommends the incumbent is disarming and correct, and it widens the "
      "addressable audience to every vLLM user.",
      "File exists with front-matter; explicitly says 'use vLLM for throughput, front it with fak'; "
      "no tokens/sec claim vs a serving engine; links the single-binary diagram; in compare index.",
      "docs-adoption-compare"),
    T("D", "docs(compare): \"fak vs an API gateway / LLM router\" (OpenRouter, Portkey, LiteLLM)",
      ["syscall", "binary"],
      "`docs/adoption/compare/vs-routers.md`: routers pick *which model* per request; fak governs "
      "*which effects* per tool call and preserves cache legality. Show they compose (fak in front, "
      "route through). Reuse the honest categorical positioning already in "
      "`docs/integrations/routers.md`.",
      "Router users are a large, adjacent audience who will ask 'don't I already have this?'. "
      "Showing the layers are orthogonal (and composable) turns a perceived competitor into a "
      "recommended pairing.",
      "File exists with front-matter; states the which-model vs which-effect distinction; shows the "
      "compose topology; consistent with `docs/integrations/routers.md`; in compare index.",
      "docs-adoption-compare"),
    T("D", "docs(compare): a one-glance capability matrix across the whole category",
      ["syscall", "dos", "kvcache", "capgate"],
      "`docs/adoption/compare/matrix.md`: a single table — rows are capabilities (default-deny "
      "on-path, result quarantine, addressable KV eviction, commit-level verify, structured "
      "refusal, single-binary), columns are fak + the honest set of alternatives. Cells are "
      "yes/no/partial with a footnote/source each. No cell without a citation.",
      "A capability matrix is the artifact people screenshot and paste into buying discussions. It "
      "must be scrupulously fair or it backfires — done right, it is the highest-authority "
      "positioning asset.",
      "File exists with front-matter; every 'no'/'partial' about a competitor has a source or an "
      "explicit 'unverified' tag; fak's own rows match CLAIMS.md honesty tags; in compare index.",
      "docs-adoption-compare"),
    T("D", "docs(compare): \"is this just a firewall?\" — the boundary FAQ, expanded",
      ["capgate", "syscall"],
      "`docs/adoption/compare/vs-firewall.md`: expand the FAQ answer into a standalone page — a "
      "network firewall filters packets by rules on traffic it inspects; fak gates *effects* by "
      "capability on a path the model does not control, and it understands tool-call semantics a "
      "packet filter cannot. Same instinct, different layer.",
      "'Firewall' is the metaphor people reach for, and it is close enough to mislead. Owning the "
      "comparison (rather than fighting the word) lets fak borrow the firewall's credibility while "
      "drawing the real line.",
      "File exists with front-matter; concedes the firewall analogy's usefulness before drawing the "
      "line; consistent with the FAQ; in compare index.",
      "docs-adoption-compare"),

    # ---------------- E. Social proof & community (5) ----------------
    T("E", "docs(community): a CONTRIBUTING-friendly \"good first popularization tasks\" board",
      ["dos"],
      "`docs/adoption/good-first-tasks.md`: a curated list of small, well-scoped contributions a "
      "newcomer can make in an afternoon — fix a doc link, add a language to an i18n front door, "
      "write one comparison row, add an integration recipe for a harness we don't cover. Each entry "
      "has a difficulty tag and the file to touch.",
      "Popularity compounds through contributors, not just readers. A clear on-ramp for small wins "
      "is how an outside person becomes an invested one; it is the seed of a community.",
      "File exists with front-matter; >=10 concrete tasks each naming the file and difficulty; "
      "linked from CONTRIBUTING.md and `INDEX.md`; every named file exists.",
      "docs-adoption-community"),
    T("E", "docs(community): a \"who is this for\" persona gallery with a quote-ready pitch each",
      ["syscall", "dos", "kvcache", "capgate", "binary"],
      "`docs/adoption/personas.md`: for each top persona (the indie dev running one agent, the "
      "platform team hardening a fleet, the security reviewer, the researcher), a two-line 'this is "
      "why you specifically care' pitch and the one door to walk through first. Reuse the "
      "persona-readiness scorecard's roster.",
      "People adopt when they see themselves in the pitch. A persona gallery lets each visitor "
      "self-select the value prop that lands for them instead of reading a generic pitch.",
      "File exists with front-matter; covers >=4 personas from the persona scorecard; each has a "
      "distinct pitch + a first-door link; consistent with `tools/persona_readiness_scorecard.py`; "
      "in `INDEX.md`.",
      "docs-adoption-community"),
    T("E", "docs(community): a testimonial/quote scaffold ready for real adopters (honest, empty-safe)",
      ["binary"],
      "`docs/adoption/voices.md`: a structured, clearly-labeled scaffold for adopter quotes and "
      "case notes — with a template and an explicit 'no fabricated testimonials' rule and a "
      "placeholder that reads honestly while empty ('be the first — here's how to share your "
      "story'). NOT invented quotes.",
      "Social proof is decisive, but only if real. A disciplined scaffold that is honest while empty "
      "and ready to fill turns the first real adopter's story into an asset the moment it exists — "
      "without ever tempting fabrication.",
      "File exists with front-matter; contains a submission template and an explicit anti-"
      "fabrication note; the empty state reads as an invitation, not a fake quote; in `INDEX.md`.",
      "docs-adoption-community"),
    T("E", "docs(community): a public roadmap / \"what we're building next\" reader-facing page",
      ["dos"],
      "`docs/adoption/roadmap.md`: a reader-facing (not internal) view of the near-term direction, "
      "derived honestly from the open epics (native harness, durable sessions, neo-silicon binding, "
      "cache default-on) with the now/next/future taxonomy from `docs/generation.md`. Label what is "
      "shipped vs planned.",
      "A visible roadmap signals momentum and invites people to bet on the project. It also gives "
      "the community a place to see themselves in the future of the tool, which drives stars/watch.",
      "File exists with front-matter; every 'shipped' item is git-witnessed; 'planned' is labeled "
      "planned; consistent with `docs/generation.md` taxonomy; in `INDEX.md`.",
      "docs-adoption-community"),
    T("E", "docs(community): a code-of-conduct + welcoming issue/PR templates pass",
      ["dos"],
      "Add or refine `.github/ISSUE_TEMPLATE/` and `PULL_REQUEST_TEMPLATE.md` and a "
      "`CODE_OF_CONDUCT.md` so a first-time contributor lands in a structured, welcoming flow "
      "(bug/feature/adoption-story templates; a PR checklist that references the ship discipline). "
      "Keep it light, not bureaucratic.",
      "The GitHub-native welcome surface is the first thing a would-be contributor touches. A warm, "
      "structured intake is table-stakes social proof — its absence quietly signals 'not for "
      "outsiders'.",
      "Templates exist and render on new-issue/new-PR; a code of conduct is present; the PR template "
      "references the commit/ship discipline; nothing blocks a trivial contribution behind heavy "
      "process.",
      "github-community-templates"),

    # ---------------- F. Developer experience & onramp (5) ----------------
    T("F", "docs(onramp): a \"copy one line, see one verdict\" hero quickstart above the fold",
      ["binary", "syscall"],
      "Refine the top of `START-HERE.md` (and mirror the block into README) so the very first thing "
      "a reader sees is a single copy-pasteable command that installs and produces one adjudicated "
      "verdict — no key, no GPU, no config. Everything else moves below it.",
      "The first 30 seconds decide adoption. A hero command that yields a visible result before any "
      "reading is the highest-leverage DX change; every other doc improvement is downstream of this.",
      "The first code block in START-HERE is a single working command that needs no key/GPU; running "
      "it prints a verdict; README mirrors it; the command is tested to exit 0 in CI or a smoke "
      "script.",
      "docs-onramp"),
    T("F", "feat(cli): `fak demo` — one verb that runs the 60-second proof self-contained",
      ["syscall", "capgate", "binary"],
      "A `fak demo` subcommand (pure logic in `internal/demo/`, thin shell in `cmd/fak/demo.go` per "
      "the Go-not-Python rule) that runs the canonical offline proof end to end: propose a safe "
      "call (ALLOW), an irreversible call (DENY), a poisoned result (QUARANTINE), printing a "
      "narrated verdict for each. Zero flags needed.",
      "Turning the demo into a first-class verb (not a script to find) removes all discovery "
      "friction — 'just run `fak demo`' is the most repeatable onboarding instruction possible and "
      "belongs in every doc and talk.",
      "`fak demo` runs offline with no args, no key, no GPU; prints ALLOW/DENY/QUARANTINE with a "
      "one-line narration each; has a table test; documented in START-HERE and CLI reference.",
      "cmd-fak-demo"),
    T("F", "docs(onramp): a troubleshooting/\"it didn't work\" page for the first five failures",
      ["binary"],
      "`docs/adoption/troubleshooting-first-run.md`: the five most likely first-run failures "
      "(binary not on PATH, wrong base URL, missing key env for a keyed provider, policy denied the "
      "call you expected, port in use) each with the symptom, the cause, and the fix.",
      "The gap between 'ran the quickstart' and 'it worked' is where adopters silently churn. A "
      "targeted first-run troubleshooting page recovers the people who hit a wall and would "
      "otherwise leave without saying anything.",
      "File exists with front-matter; covers >=5 concrete first-run failures with symptom/cause/fix; "
      "linked from START-HERE and GETTING-STARTED; in `INDEX.md`.",
      "docs-onramp"),
    T("F", "docs(onramp): a copy-paste \"put fak in front of Claude Code\" 3-step recipe up top",
      ["binary", "capgate"],
      "Promote the Claude Code integration to a 3-step, copy-paste hero recipe (set base URL, "
      "launch, watch a verdict) at the top of `docs/integrations/claude.md`, with the full detail "
      "below. Because Claude Code is the most likely first harness, this is the highest-traffic "
      "recipe.",
      "The single most common adopter already runs Claude Code. A frictionless three-step recipe "
      "for that exact path converts the largest identifiable audience with the least reading.",
      "The top of claude.md is 3 copy-paste steps that end in a visible verdict; the commands are "
      "correct against current flags; detail preserved below; linked from the integration index.",
      "docs-integrations-claude"),
    T("F", "docs(onramp): a \"install paths\" decision table (binary / go install / source / MCP)",
      ["binary"],
      "`docs/adoption/install-paths.md`: a small decision table — 'I just want to try it' -> "
      "prebuilt binary; 'I have Go' -> `go install`; 'I want to hack on it' -> from source; 'I use "
      "an MCP client' -> the one-paste `.mcp.json`. One correct command per row.",
      "Install ambiguity is silent friction. A decision table that routes each reader to their one "
      "correct command in a glance removes the 'which of these do I run' hesitation.",
      "File exists with front-matter; >=4 rows each with a verified command; consistent with "
      "GETTING-STARTED and the MCP example; in `INDEX.md`.",
      "docs-onramp"),

    # ---------------- G. Integration recipes (4) ----------------
    T("G", "docs(integration): a recipe for a harness we don't yet cover (Aider or Cline)",
      ["binary", "capgate"],
      "Add `docs/integrations/aider.md` (or cline.md — pick the one with the clearer base-URL "
      "repoint) following the house recipe shape: which wire it speaks, the exact repoint key, a "
      "3-step setup, and the honest interop grade. Add its row to the compatibility matrix.",
      "Each new first-party recipe converts that harness's entire user base from 'unsupported?' to "
      "'drop-in'. Breadth of covered harnesses is directly the breadth of the addressable audience.",
      "File exists with front-matter; states the wire + repoint key + 3 steps; a matrix row added "
      "with a source; consistent with `docs/integrations/compatibility-matrix.md`; in the "
      "integration index.",
      "docs-integrations-newharness"),
    T("G", "examples(integration): a runnable OpenAI-SDK \"set base URL\" minimal repo",
      ["binary", "syscall"],
      "`examples/openai-sdk-minimal/`: the smallest possible working repo (one file + README) that "
      "points the official OpenAI SDK at `fak serve` and makes one governed call, showing the "
      "verdict in the trace. The universal 'set the base URL' recipe as running code.",
      "The 'set one base URL' claim is central and needs a runnable proof, not just prose. A minimal "
      "copy-me repo is what developers fork to start, making adoption a git-clone away.",
      "Repo runs against a local `fak serve` (offline path documented); one file, clear README; "
      "shows the trace/verdict; listed in the examples index and the integration index.",
      "examples-openai-sdk-minimal"),
    T("G", "docs(integration): a docker-compose \"fak in front of Ollama\" copy-paste stack",
      ["binary"],
      "`examples/compose-ollama/`: a `docker-compose.yml` + README that brings up Ollama and "
      "`fak serve` in front of it with an allow-list and audit, so a reader gets a governed local "
      "model with one `docker compose up`. Reuse the deployment guide's compose patterns.",
      "Compose is how most developers try a multi-service stack. A one-command governed-local-model "
      "stack is a very high-conversion demo for the self-host audience and needs no cloud key.",
      "`docker compose up` brings up both services; a governed call succeeds and an out-of-policy "
      "call is denied; README documents both; consistent with the deployment guide; in examples "
      "index.",
      "examples-compose-ollama"),
    T("G", "docs(integration): an \"adopt fak in an existing repo\" 10-minute migration checklist",
      ["binary", "dos"],
      "`docs/adoption/adopt-in-your-repo.md`: a checklist for taking an existing agent project and "
      "putting fak in front in 10 minutes — repoint the base URL, choose a starter policy, wire the "
      "DOS trust gate into the runtime's hooks (`dos init --hooks auto .`), verify a verdict "
      "appears. Reuse the migration guide + the DOS one-command wire.",
      "The migration audience already has a working agent and the most to gain; a concrete 'adopt in "
      "your repo' checklist lowers the switching cost from 'a project' to 'an afternoon'.",
      "File exists with front-matter; a numbered checklist ending in a verified verdict + a DOS gate "
      "firing; consistent with `docs/fak/migration-guide.md` and the DOS wire answer; in `INDEX.md`.",
      "docs-adoption-migrate"),

    # ---------------- H. Benchmark-as-story (4) ----------------
    T("H", "docs(bench-story): \"the 4x that's real\" — the tuned warm-cache result, told honestly",
      ["kvcache"],
      "`docs/adoption/stories/the-real-4x.md`: narrate the witnessed multi-agent reuse result "
      "(~4.1x fewer tokens vs a tuned warm-cache stack on the 50-turn x 5-agent run) as a story — "
      "the setup, what was measured, what the tuned baseline was, and why the honest number beats "
      "the naive 60x. Quote only witnessed figures.",
      "A benchmark becomes persuasive when it is a story with stakes and an honest baseline. Telling "
      "the *tuned* number (not the flattering naive one) is itself the credibility move that makes "
      "the audience trust every other number.",
      "File exists with front-matter; every figure is witnessed and matches the bench docs; the "
      "tuned baseline is described; the naive vs tuned distinction is explicit; in the stories "
      "index.",
      "docs-adoption-stories"),
    T("H", "docs(bench-story): \"max|Δ| = 0\" — the bit-exact eviction result as a claim you can check",
      ["kvcache"],
      "`docs/adoption/stories/bit-exact.md`: tell the addressable-KV eviction result as a "
      "falsifiable claim — here is what bit-exact means, here is the exact assertion (`max|Δ| = 0`), "
      "here is the command/test that checks it, here is what would falsify it. Link the runnable "
      "demo (C-dimension).",
      "'No shipped serving engine offers this' is a strong claim; wrapping it in a checkable "
      "assertion and a repro command converts the strlength into credibility instead of "
      "skepticism.",
      "File exists with front-matter; states the assertion, the check command, and the falsifier; "
      "links the addressable-evict example; no claim beyond what the test proves; in stories index.",
      "docs-adoption-stories"),
    T("H", "docs(bench-story): \"the cache cliff\" — why public prompt-cache hit rates mislead",
      ["kvcache"],
      "`docs/adoption/stories/cache-cliff.md`: a reader-facing retelling of the frozen-trajectory "
      "cache-cliff explainer as a story — the 96.6% hit rate is real but only because the "
      "trajectory is frozen; it decays toward 0% along editing, tool-call density, and cross-agent "
      "fan-out. Use the `cache_curve.py` demonstrator.",
      "This reframes a 'weakness' (fak's whole reason to exist) into an insight the reader did not "
      "have. Teaching someone why a headline number they trusted is misleading earns durable "
      "attention.",
      "File exists with front-matter; the three decay axes are named; the 96.6% calibration is "
      "cited; consistent with the existing cache-cliff explainer; in stories index.",
      "docs-adoption-stories"),
    T("H", "docs(bench-story): \"~100% evadable, on purpose\" — the detector honesty story",
      ["capgate"],
      "`docs/adoption/stories/evadable-on-purpose.md`: tell the counterintuitive story that the "
      "result *detector* is ~100% evadable by design and that this is a strength — the floor is the "
      "capability lock plus containment, which do not depend on detection. Explain why a security "
      "story that concedes the detector is honest and stronger.",
      "Volunteering that a component is evadable is a striking honesty move that flips the usual "
      "security-marketing script. It makes the structural floor (which is not evadable) far more "
      "believable by contrast.",
      "File exists with front-matter; clearly separates the evadable detector (bonus) from the "
      "non-evadable floor; consistent with the 'what fak is not' explainer; in stories index.",
      "docs-adoption-stories"),

    # ---------------- I. Memorable framing & naming (3) ----------------
    T("I", "docs(framing): a canonical \"pitch ladder\" — 1 sentence / 1 paragraph / 1 page",
      ["syscall", "dos", "kvcache", "capgate", "binary"],
      "`docs/adoption/pitch-ladder.md`: the same pitch at three zoom levels — one sentence (the "
      "tweet), one paragraph (the HN comment), one page (the blog intro) — each self-consistent and "
      "quotable. Anchor on 'treat the tool call like a syscall: the model proposes, the kernel "
      "disposes.'",
      "A concept spreads at the speed of its shortest correct expression. Giving advocates a "
      "ready-made pitch at each length means they repeat *our* framing instead of inventing a weaker "
      "one.",
      "File exists with front-matter; three lengths present and mutually consistent; the one-"
      "sentence version is <=30 words and memorable; used as the source for README's lead; in "
      "`INDEX.md`.",
      "docs-adoption-framing"),
    T("I", "docs(framing): an \"objections & one-line answers\" card for advocates",
      ["capgate", "binary"],
      "`docs/adoption/objections.md`: the eight most common objections ('isn't this a "
      "classifier?', 'doesn't this slow the agent down?', 'why not just use vLLM?', 'is the "
      "security real if the detector is evadable?', 'another gateway?') each with a crisp one-to-"
      "two-line answer that links to the deeper page.",
      "Advocates win or lose the concept in comment threads. A pocket set of tight, correct rebuttals "
      "lets a supporter defend the idea in real time without misstating it — the difference between "
      "a concept that survives scrutiny and one that dies in a thread.",
      "File exists with front-matter; >=8 objections each with a short answer + a deeper link; every "
      "answer is consistent with the explainers; in `INDEX.md`.",
      "docs-adoption-framing"),
    T("I", "docs(framing): the search-disambiguation \"how to find and name this\" note, reader-facing",
      ["syscall", "binary"],
      "`docs/adoption/naming.md`: a short reader-facing note that the bare word 'fak' is dominated "
      "by homophone/acronym noise, so the concept travels under 'agent kernel', 'agent tool "
      "firewall', 'treat the tool call like a syscall'. Give people the terms to use when they "
      "search for or refer to it.",
      "A concept people cannot find is a concept that cannot spread. Teaching advocates the "
      "disambiguated terms (which also feed AEO) means word-of-mouth lands on findable language.",
      "File exists with front-matter; lists the disambiguated terms from llms.txt; consistent with "
      "`internal/marketing/aeo.go`; in `INDEX.md`; SEO scorecard does not regress.",
      "docs-adoption-framing"),

    # ---------------- J. Distribution & channels (3) ----------------
    T("J", "docs(distribution): a \"Show HN / launch post\" draft kit (honest, ready-to-post)",
      ["syscall", "capgate", "binary"],
      "`docs/adoption/launch-kit.md`: a ready-to-adapt launch-post draft (title options, the "
      "opening hook, the honest what-it-is/what-it-isn't, the 60-second proof, the ask) plus a "
      "checklist of what must be true before posting (green CI, working quickstart, honest scope). "
      "Draft only — posting is human-owned.",
      "Launch moments are high-leverage but easy to fumble. A pre-written, honest launch kit means "
      "the moment a human decides to post, the strongest version is ready — and the honesty "
      "checklist prevents an over-claim that would backfire.",
      "File exists with front-matter; contains a full draft + a pre-post checklist; every claim in "
      "the draft is witnessed/honest-scoped; explicitly marked human-owned to post; in `INDEX.md`.",
      "docs-adoption-distribution"),
    T("J", "docs(distribution): a social-thread / carousel storyboard for the five concepts",
      ["syscall", "dos", "kvcache", "capgate", "binary"],
      "`docs/adoption/social-storyboard.md`: a storyboard for a short thread/carousel — one card "
      "per concept, each with a one-line hook + which diagram (B-dimension) illustrates it + a "
      "link. The reusable spine for any social push, referencing the visual assets so it is not "
      "text-only.",
      "Concepts spread on social in cards, not paragraphs. A storyboard that pairs each concept with "
      "its diagram gives an advocate a drop-in visual thread — far more shareable than a link to "
      "docs.",
      "File exists with front-matter; five cards each with a hook + a diagram reference that exists "
      "(or is stubbed to the B ticket); no unwitnessed claim; in `INDEX.md`.",
      "docs-adoption-distribution"),
    T("J", "docs(distribution): a curated \"where to submit\" list of relevant awesome-lists & directories",
      ["binary"],
      "`docs/adoption/directories.md`: a maintained list of relevant awesome-lists, tool "
      "directories, and registries (LLM-tooling, agent-infra, security, MCP) with the current "
      "submission status for each (submitted / merged / not-yet / declined) — a checklist that "
      "prevents duplicate submissions and tracks reach. Reuse the awesome-list campaign state.",
      "Directory presence is durable, compounding discovery. A single tracked checklist prevents the "
      "duplicate-submission waste noted in the campaign memory and turns scattered outreach into a "
      "measurable surface.",
      "File exists with front-matter; each directory has a status; consistent with the existing "
      "awesome-list campaign tracking; no duplicate-submission of an already-merged list; in "
      "`INDEX.md`.",
      "docs-adoption-distribution"),

    # ---------------- K. Adoption measurement (2) ----------------
    T("K", "feat(scorecard): a popularization-readiness scorecard (does the front door convert?)",
      ["binary", "syscall"],
      "`tools/popularization_scorecard.py` following the scorecard doctrine: deterministic KPIs over "
      "the git-tracked tree measuring whether the popularization surface exists and is wired — a "
      "hero quickstart command present, `fak demo` documented, >=N explainers, the compare/matrix "
      "present, the concept card, the pitch ladder, every adoption doc reachable from `INDEX.md`, "
      "every claim honest-tagged. Emits the control-pane payload and a committed snapshot.",
      "You cannot improve popularity you cannot measure. A scorecard turns 'is the front door good?' "
      "into a debt integer that ratchets down over time and folds into the existing control pane — "
      "making this whole epic self-verifying and repeatable on a /loop.",
      "Scorecard runs deterministically; emits `popularization_debt` + grade + control-pane payload; "
      "regenerates a snapshot with no diff on a clean tree; a missing hero command or an "
      "unreachable adoption doc raises debt; wired into `scorecard_control_pane.py`.",
      "tools-popularization-scorecard"),
    T("K", "docs(measurement): an adoption-signals dashboard spec (stars/forks/mentions, honest)",
      ["dos"],
      "`docs/adoption/signals.md`: define the honest, external adoption signals worth watching "
      "(stars/forks/watchers, directory listings merged, integration recipes shipped, distinct "
      "harnesses covered, docs reachable) and how each is collected — labeling each as a proxy, not "
      "proof of adoption, per the conflation discipline (witnessed vs observed).",
      "Vanity metrics mislead; a disciplined signals spec that labels each as a proxy keeps the team "
      "honest about whether popularity is actually rising and prevents a good-looking-number from "
      "masquerading as adoption.",
      "File exists with front-matter; each signal labeled proxy/observed vs witnessed per the "
      "conflation discipline; a collection method per signal; consistent with "
      "`tools/conflation_scorecard.py`; in `INDEX.md`.",
      "docs-adoption-measurement"),
]


def _likely_files(t):
    """Derive the concrete path(s) a worker will touch from the deliverable text.

    The contract gate reads backtick code-spans out of the `## Likely files`
    section, so every path is rendered inside backticks.
    """
    import re
    # Any path-looking backtick span already in the deliverable is authoritative.
    spans = re.findall(r"`([^`]+)`", t["deliverable"])
    paths = [s for s in spans if ("/" in s or s.endswith(".py") or s.endswith(".md"))
             and " " not in s.strip()]
    if not paths:
        # Fall back to the lane -> a representative path.
        lane = t["lane"]
        if lane.startswith("examples-"):
            paths = [f"`examples/{lane[len('examples-'):]}/`"]
        elif lane.startswith("cmd-"):
            paths = [f"`cmd/fak/{lane.split('-')[-1]}.go`"]
        elif lane.startswith("tools-"):
            paths = [f"`tools/{lane[len('tools-'):].replace('-', '_')}.py`"]
        else:
            paths = ["`docs/`"]
    # de-dup, keep order, always add INDEX.md for doc lanes
    seen, out = set(), []
    for p in paths:
        pb = p if p.startswith("`") else f"`{p}`"
        if pb not in seen:
            seen.add(pb)
            out.append(pb)
    if any("docs/" in p for p in out) and "`INDEX.md`" not in out:
        out.append("`INDEX.md`")
    return ", ".join(out)


def render_body(t, epic_ref):
    dim_name = DIMS[t["dim"]]
    concepts = ", ".join(CONCEPTS[c] for c in t["concepts"])
    files = _likely_files(t)
    lane = t["lane"]
    return f"""**Dimension {t['dim']} — {dim_name}** · part of the concept-popularization epic ({epic_ref}).

**Concepts served:** {concepts}

## Parent context
Concept-popularization epic — `{epic_ref}`. This is one of 50 self-contained tickets under the `popularization` label making the fak/DOS concepts broadly known and attractive.

## Current state
The concept exists in the code/docs but its human-facing popularization artifact for this dimension does not: {t['deliverable'].split(':')[0].strip()} is not yet present as a standalone, shareable unit. Today a reader has no single artifact for this angle.

## Why now
The AEO/SEO surface (machine-facing discovery) is maintained, but the human-facing half — the artifacts that make a person want and remember the concept — is the current gap. This dimension has no dedicated owner-artifact yet.

## Working spine
{t['deliverable']}

## In scope
The single deliverable named above, plus its wiring (index link / front-matter / cross-links) so it is reachable and honest.

## Out of scope
Any other popularization ticket's deliverable; kernel/engineering changes; market-adoption claims; any benchmark not already run; renaming or restructuring existing docs beyond what this artifact needs.

## Why it popularizes the concept
{t['why']}

## Done condition
{t['acceptance']}

## Witness
The acceptance artifact exists and is checkable: the named file/command is present and correct; `python tools/seo_aeo_scorecard.py` does not regress for any new `docs/*.md`; the ship commit passes `dos commit-audit`.

## Acceptance gate
{t['acceptance']}

## Work unit
One doc/example/tool artifact a single worker owns end to end in one sitting; no dependency on another popularization ticket landing first.

## Expected steps
3

## Assumptions
- The five core concepts and honest-scope fences in the epic doc are authoritative.
- Witnessed numbers only (the tuned ~4.1×, not the naive 60×); simulated is labeled simulated.

## Confusion risks
- Do not overclaim market adoption or a novelty the 0/29 prior-art audit refutes.
- Keep this lane disjoint from sibling popularization tickets — touch only the files below.

## Coordination
- One worker per lane; lane `{lane}` is disjoint from the other 49 tickets.
- Verify lane disjointness via `dos_arbitrate` before writing if the trunk is busy.

## Trigger
Filed as part of the 2026-07-02 concept-popularization epic; dispatched via the account-switching headless resolver.

## Batch policy
One issue per popularization dimension-slot; deduped by title; update the existing issue rather than re-filing. Capped at the 50-ticket epic set.

## Likely files
{files}

## Lane
`{lane}` (disjoint from other popularization tickets — one worker can own it end to end).

## Closure binding
Closed by the ship commit that creates the accepted artifact, stamped `(fak <leaf>)` and referencing this issue number; the commit's `dos commit-audit` verdict is the binding witness.

## Ship discipline
- Trunk only; commit by explicit path; Conventional-Commits subject + a `(fak <leaf>)` stamp.
- New `docs/*.md` need SEO front-matter (`title:`/`description:`) and an `INDEX.md` line.
- Honest-scope fence: no market-adoption claim, no unrun benchmark, no novelty the 0/29 prior-art audit refutes; quote witnessed numbers, label simulated as simulated.

_This is one self-contained unit of work. It does not depend on any other popularization ticket landing first (it may cross-link to them)._
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-files", metavar="DIR", help="write one .md body per ticket into DIR")
    ap.add_argument("--epic-ref", default="the concept-popularization epic", help="epic reference string")
    ap.add_argument("--list", action="store_true", help="print a dim/title table")
    ap.add_argument("--json", action="store_true", help="dump tickets as JSON")
    args = ap.parse_args()

    assert len(TICKETS) == 50, f"expected 50 tickets, have {len(TICKETS)}"

    if args.list:
        by_dim = {}
        for i, t in enumerate(TICKETS, 1):
            by_dim.setdefault(t["dim"], 0)
            by_dim[t["dim"]] += 1
            print(f"{i:2d}  [{t['dim']}] {t['title']}")
        print("\nper-dimension:", {k: by_dim[k] for k in sorted(by_dim)})
        return
    if args.json:
        print(json.dumps(TICKETS, indent=2))
        return
    if args.emit_files:
        os.makedirs(args.emit_files, exist_ok=True)
        for i, t in enumerate(TICKETS, 1):
            path = os.path.join(args.emit_files, f"ticket-{i:02d}.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(render_body(t, args.epic_ref))
            # title on its own sidecar line for the filer
            with open(os.path.join(args.emit_files, f"ticket-{i:02d}.title"), "w", encoding="utf-8") as fh:
                fh.write(t["title"])
        print(f"wrote {len(TICKETS)} ticket bodies to {args.emit_files}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
