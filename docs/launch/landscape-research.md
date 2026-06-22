# Organic-discovery landscape research (2026)

> Where AI-agent-infra / LLM-security / Go projects actually get organically discovered, by channel, with the rules that get you removed and the one highest-ROI move per channel. Source research for the launch assets in this directory.

### reddit
# fak subreddit traction map (2026)

The thing that governs every channel below: fak's credibility *is* its honesty ledger (the "~100% evadable BY DESIGN" detector, the "0/29 novel — the contribution is the assembly" audit, the tuned-baseline-not-naive number). That self-skepticism is the single biggest organic asset you have, because every one of these communities is reflexively hostile to launch-speak and will reward a post that pre-empts its own takedown. Lead technical, disclose authorship in the first line, and never let the naive 8.8–9.7x number appear without the tuned 1.5–4.1x next to it — a skeptic finding that gap themselves is how you get buried.

## The eight, ranked by ROI

**r/LocalLLaMA — PRIMARY TARGET. This is the one.** ~700k+ members who self-host, care about KV cache and prefill mechanics, and will actually click a Colab. They are the only audience that natively understands *both* halves of your thesis (the self-host reuse win and the mid-run KV eviction) without a glossary. Norms: text/self-post, not a bare GitHub link; show the artifact in-feed. Upvoted: real benchmarks with reproducible commands, novel cache mechanics, "I built X, here's the receipts." Removed/buried: marketing tone, vague "introducing," anything that smells like a startup. Self-promo is tolerated (~10% of activity) *if the post is substance-first*. Title style: lowercase-ish, claim-forward, mechanism-named — `Mid-run KV eviction: cut a poisoned tool result out of a live model run, KV cache stays bit-identical (max|Δ|=0). Pure-Go, one binary.` **Highest-ROI move: post the mid-run-eviction / bit-exact-KV result with the live demo GIF embedded and the Colab linked, and be the first commenter with the per-layer cos=1.000000 oracle table.** The KV story is your most differentiated claim and this is the only sub that will grade it on the merits.

**r/AI_Agents — STRONG SECONDARY, best fit for the security thesis.** Smaller but precisely your buyer: people wiring tool-calling agents who feel the prompt-injection problem viscerally. Self-promo is welcomed here when it's an agent-building artifact. Lead with the *syscall* framing — "treat the tool call like a syscall, default-deny capability gate the model can't talk past." Upvoted: working agent infra, threat demos. Removed: low-effort "check out my SaaS." **Highest-ROI move: the 60-second `preflight refund_payment → DENY` proof as a copy-pasteable demo, framed as "the lever was never wired up, so the model can't talk past it."** This audience converts on the security half, not the perf half.

**r/selfhosted — STRONG, but only the operational angle.** Operators who hate multi-GB Python/CUDA stacks. Your "one ~13MB static Go binary, zero deps, no go.sum, drops in as an OpenAI/Anthropic gateway" line is tailor-made. Critical rule: **self-promotion is restricted to the weekly thread** — posting your project as a standalone link outside it gets removed. The exception that works: a genuinely useful standalone post (a guide, a "how I replaced X") with the tool incidental. Do NOT lead with benchmarks here; lead with operational surface. **Highest-ROI move: a standalone post framed as the operational contrast ("one static binary as your agent gateway vs a Python/CUDA multi-process stack") with deployment specifics — and drop the repo link in the weekly self-promo thread separately.**

**r/golang — GOOD, narrow framing.** Gophers reward the *engineering*, not the AI. The hook is "pure-Go SmolLM2 forward pass proven against a HF oracle; CUDA decode at llama.cpp parity; one static binary, no go.sum." Removed/downvoted: AI hype, anything where Go is incidental. They are skeptical of "I rewrote X in Go" but love proven-correct numerics in Go. Title style: understated, engineering-first — `A pure-Go transformer forward pass, verified per-layer against HuggingFace (cos=1.000000)`. **Highest-ROI move: post the pure-Go inference + oracle-parity work as a Go engineering story; mention the security thesis only in comments.** The no-go.sum/single-binary detail is catnip here.

**r/netsec — HIGH BAR, high payoff if you clear it.** Heavily moderated, allowlisted domains, ruthless on "blogspam" and vendor pitches; a self-promotional GitHub link often won't clear the mod queue. What passes: original technical security research with a real writeup, not a product. Your defensible netsec post is the *architecture argument* — "result quarantine + capability lock as two independent gates; the detector is intentionally evadable and is NOT the floor." Frame it as a defense pattern against tool-result prompt injection, with the threat model explicit. **Highest-ROI move: a technical writeup (hosted on your docs site, not a bare repo) on the two-gate containment model and why detection-as-floor is the wrong abstraction — submitted as research, with "disclosure: I wrote this."** Expect it to need to be genuinely good to survive the queue.

**r/MachineLearning — MODERATE, timing-gated.** 3M+ but rules are strict: use the `[P]` (Project) flair, and **non-arxiv/promotional links are weekend-only** in practice — weekday self-promo gets removed or routed to the periodic self-promotion thread. The audience respects rigor and is allergic to overclaiming, so your honest-prior-art-audit framing plays well. **Highest-ROI move: a weekend `[P]` post centered on the reproducibility story (byte-identical results across 4 GPU backends / 4 OSes, every number traced to commit+artifact).** Don't lead with the headline speedup here; lead with method and reproducibility.

**r/programming — LOW-MODERATE, easy to get buried.** 5.8M but deeply skeptical of marketing and quick to downvote self-promo; a launch-flavored post dies fast. It *can* work as an engineering essay, not a project announcement. The winning shape is a "show, don't tell" article that teaches something (the syscall analogy, or "the safety boundary and the reuse boundary are the same boundary" as a genuine systems insight) where fak is the worked example. **Highest-ROI move: an essay-style post on the one-binary-as-syscall-gate idea with the GIF inline and "full disclosure, I built this" up top — not a "introducing fak" post.** Treat this as a reach channel, not a primary.

**r/LLMDevs — DO NOT POST A LAUNCH. Self-promotion is banned.** A standalone project post will be removed. The only legitimate play is participation: answer real questions about tool-call security or self-host caching, and mention fak only when it's a direct, disclosed answer to someone's problem. **Highest-ROI move: build comment karma by answering injection/tool-gating questions; never post the repo as a thread.**

## The cross-channel rules that get you removed

- **Bare GitHub link as the whole post** → buried or removed nearly everywhere except dedicated "show" subs. Always a text post with the substance in-feed.
- **Posting outside the designated thread** where one exists → instant removal on r/selfhosted, r/LLMDevs, often r/MachineLearning (weekday). Know each sub's thread before you post.
- **Naive-baseline number without the tuned baseline** → not a mod removal, but a community kill on r/LocalLLaMA and r/MachineLearning the moment someone does the math. Self-inflicted.
- **No authorship disclosure** → on r/netsec and r/programming, undisclosed self-promo reads as astroturf and tanks the post; "disclosure: I built this" *raises* trust.
- **Marketing register** ("introducing," "excited to share," "game-changing") → downvote trigger in every technical sub here.

## The single sharpest move overall

One r/LocalLLaMA post, mechanism-titled around the bit-exact mid-run KV eviction (the claim no shipped engine offers), with the live demo GIF embedded and the Colab one click away, and you as first commenter posting the oracle-parity table and the *honest* tuned-baseline numbers. That community is the only one that grades all three of your differentiators (KV mechanics, self-host reuse, injection containment) on technical merit, and a top post there is what every other channel's post can cite as social proof.

Sources: [Tereza Tizkova, "Best subreddits for sharing your project"](https://tereza-tizkova.medium.com/best-subreddits-for-sharing-your-project-517c433442f9); [Reddit self-promotion rules 2026 (replyagent)](https://www.replyagent.ai/blog/reddit-self-promotion-rules-naturally-mention-product); [The complete guide to Reddit self-promotion rules in 2026 (redship)](https://redship.io/blog/reddit-self-promotion-rules); [r/MachineLearning community profile (AIX)](https://aiexpert.network/r-machinelearning/); [Top subreddits for cybersecurity/infosec incl. r/netsec (SentinelOne)](https://www.sentinelone.com/blog/top-50-subreddits-for-cybersecurity-and-infosec/); [Reddit self-promotion across 50 subreddits (OneUp)](https://oneup.today/blogs/reddit-self-promotion-rules-saas).

### hn_lobsters
That 70-day restriction for new Lobsters users (no `show` tag, no new domains) is the load-bearing constraint for this author. I have everything I need.

---

# Brief: Landing fak on Show HN / Lobsters front pages (2026)

## The one thing that decides everything
In 2026 both communities are in an active anti-AI-slop crackdown. Lobsters added a working "AI generated" flag campaign and HN's mods kill posts that read like LLM-expanded marketing copy. **A submission that smells machine-written dies before votes matter.** Your CLAIMS.md honesty ledger and the "0/29 novel, the contribution is the assembly" framing are your single biggest asset — they are unfakeable human signals. Lead with them, never hide them.

---

## Channel 1 — Show HN

**Named target:** `news.ycombinator.com/show` → the `/front` is the goal. You need ~5–10 upvotes in the first ~30–60 min from organic visitors (no decay yet) to clear into the main pool.

**Title.** HN titles are shared discussion labels, not ad copy. The winning Show HN form is `Show HN: <name> – <concrete thing it does>`, active voice, zero adjectives. For fak the highest-leverage hook is the *mechanism surprise*, not the word "security" (the AI-security category is saturated and triggers "yet another agent framework" fatigue — see [the recurring HN agent-fatigue thread](https://news.ycombinator.com/item?id=44276830)).
- Strong: `Show HN: Fak – treat the LLM like an untrusted program and each tool call like a syscall`
- Strong: `Show HN: An agent kernel where the safety boundary and the cache-reuse boundary are the same boundary`
- Avoid: anything with "secure," "powerful," "revolutionary," "9.7x faster" in the title. A throughput number in a *security* title reads as bait and invites the "your benchmark is naive" pile-on immediately.

**Show, don't tell.** This is literal on HN: the post must let them *run* it. Your 60-second `go run ./cmd/fak preflight ... refund_payment → DENY` with no key/model/GPU is the perfect Show HN artifact — one static binary, zero deps, no go.sum is itself the demo. Put the copy-pasteable command in the post body and the [live in-browser demos](https://anthony-chaudhary.github.io/fak/demos.html) as the second link. The single greatest Show HN failure mode is "looks interesting" with nothing to touch.

**What the top comment will attack (rank order):**
1. **"Prompt injection is unsolvable — a 2026 paper proves it"** (the [Abdelnabi/Bagdasarian impossibility result](https://www.csoonline.com/article/4184455/prompt-injection-breaks-todays-ai-agents-study-warns.html) is now the reflexive top reply to any AI-security launch). **Pre-empt it in your own first comment** by agreeing: "Right — that's why the detector is ~100% evadable by design and is *not* the floor. The floor is the capability lock (the lever doesn't exist) + result quarantine. We don't claim to detect injection; we claim the model can't act on it." This converts the attack into your thesis.
2. **"Your 9.7x is vs a naive baseline — vLLM/SGLang already do prefix caching."** **Never let this be discovered.** Lead the perf claim with the *tuned-baseline* number (4.1x conservative, ~1.5–4x honest) and state the self-host-only / read-heavy-only / write-rate-flips-it-negative fences in the post itself.
3. **"What's actually novel here?"** Disarm with the 0/29 audit up front.
4. **"Why Go / why not just a Python sidecar?"** Answer: in-process, same call path, no IPC — one 13MB static binary vs a multi-GB CUDA stack. That's the operational-surface argument, which HN respects.

**Author engagement.** Post a substantive first comment *as the author* immediately (HN norm; it's where you place the honest fences and the run command). Then answer every technical reply within minutes for the first 2 hours — fast, specific, conceding real limits. The honesty ledger means you can say "yes, that's a known weakness, here's the fence" instead of getting defensive, which is exactly the tone that wins HN. Do **not** thank people, do **not** post marketing language in replies.

**What gets you flagged/killed:** asking anyone to upvote (HN's voting-ring detection is genuinely good and silently zeroes those votes — see [Lucas Costa's launch guide](https://www.lucasfcosta.com/blog/hn-launch)); reposting after a flop within days; any reply that reads as PR; sock-puppet "great tool!" comments (instant flag magnet).

**Highest-ROI move for HN:** Put the *adversarial honesty* in the title-adjacent first comment before anyone else can — "the detector is evadable by design, here's why that's the point." On a 2026 AI-security launch, being the person who already named your own weakness is what flips the top comment from prosecutor to advocate.

---

## Channel 2 — Lobsters

**The disqualifying constraint first:** new Lobsters accounts cannot use the `show` tag, cannot submit from a never-seen domain, and cannot resubmit **for 70 days** ([content-marketing mitigation thread](https://lobste.rs/s/utbyws/)). And you need an **invite** to exist at all. So:
- If you don't have an aged, in-good-standing Lobsters account, **do not try to self-submit fak** — you structurally can't `show`-tag it, and a new-domain GitHub-adjacent self-post from a fresh account is the textbook flag target.
- The realistic play: get a *third party* who already reads Lobsters to submit it (organic, not coordinated — coordination is itself flaggable), or earn standing first by commenting substantively for weeks.

**If you can submit:** tag `ai`/`security`/`go`/`practices`; never `show` unless aged. The [front-page formula](https://blog.nilenso.com/blog/2026/01/20/lobsters-front-page/) gives self-authored links only a +0.25 nudge and applies log-scaled vote weight with a 22h decay — Lobsters is small and merit-sorted, so 8–15 thoughtful upvotes can front-page you, but a single early flag (only takes 50-karma users) zeroes comment points if base goes negative. Quality of the *first technical comment* matters more than on HN.

**What gets you killed on Lobsters specifically:** the "AI generated" / "slop" flag (your copy must read as a human engineer who built this — the CLAIMS.md voice is your defense); `spam` flag for "content designed to promote a commercial service"; exceeding ~25% self-promotion across your history; the 4-link domain-majority rule.

**Highest-ROI move for Lobsters:** Don't lead with the product — lead with the *idea*. A Lobsters-native submission is a writeup titled around the thesis ("The safety boundary and the reuse boundary are the same boundary") on your docs/blog, where fak is the worked example, not the pitch. Lobsters rewards the essay; it punishes the launch.

---

## Cross-channel summary

| | Show HN | Lobsters |
|---|---|---|
| **Title** | `Show HN: Fak – treat the model like an untrusted program, the tool call like a syscall` | Idea-first essay title; fak as the example |
| **Lead asset** | 60s `preflight → DENY` command + live demos | The "same boundary" thesis writeup |
| **#1 kill risk** | Reads as LLM marketing / asking for votes | `show` tag or new domain inside the 70-day wall; AI-slop flag |
| **Top-comment defense** | Pre-empt "injection is unsolvable" by agreeing — the lock is the floor | Same, in the first comment |
| **Perf number** | Tuned-baseline 4.1x only, with fences, never in the title | Same, framed as operational surface not tok/s |

**The single highest-ROI move overall:** make your *own first comment the prosecution*. Both communities in 2026 are primed to attack AI-security launches and sniff out AI-written hype. The author who opens with "the detector is evadable by design, the perf win is self-host-read-heavy-only, and we scored 0/29 on novelty" disarms all three reflexes at once — and that honesty is the one thing a competing slop post structurally cannot copy.

Sources: [Lobsters front-page mechanics](https://blog.nilenso.com/blog/2026/01/20/lobsters-front-page/) · [Lobsters content-marketing policy](https://lobste.rs/s/utbyws/) · [Lobsters "AI generated" flag proposal](https://lobste.rs/s/rkjpob/proposal_add_ai_generated_as_flag_reason) · [HN launch guide](https://www.lucasfcosta.com/blog/hn-launch) · [Show HN title conventions](https://syften.com/blog/hacker-news-marketing/) · [prompt-injection-unsolvable study](https://www.csoonline.com/article/4184455/prompt-injection-breaks-todays-ai-agents-study-warns.html) · [HN AI-agent fatigue thread](https://news.ycombinator.com/item?id=44276830)

### youtube
# fak — YouTube creator-pickup brief (2026)

The single biggest lever: fak is a **structural answer to Simon Willison's "lethal trifecta,"** the dominant AI-security vocabulary of 2026 (private data + untrusted content + exfiltration path = guaranteed exploit). Every security creator is already using that frame. fak's pitch — "the model is an untrusted program; two gates the attacker must beat, and one is a lever that was *never wired up*" — is the rare project that *closes* the trifecta by structure instead of detection. Lead there, not with performance.

## Named targets, by archetype

**A. AI-security / red-team researchers (highest fit, lowest reach, highest credibility transfer)**
- **Simon Willison** (blog + talks, not heavy YT, but he is the *node* — one mention propagates everywhere). He coined prompt injection and the lethal trifecta, gives Bay Area AI Security Meetup talks, and actively amplifies terms. He rewards *honest* containment claims and despises detector-as-floor hype — fak's "the detector is ~100% evadable BY DESIGN, the floor is the lock" is calibrated *exactly* to his taste.
- **LiveOverflow, John Hammond, IppSec** — binary-exploitation / CTF audience. They cover novel *boundaries*, not tutorials. The "lever was never wired up — refuses by structure" is a CTF-shaped claim; the demo is "attacker controls the tool result and *still* can't fire refund_payment."
- **Promptfoo / Garak / PyRIT** orbit — the red-team tooling channels. Angle: run their attacks *against* fak's quarantine gate on camera.

**B. AI-agent builders (highest reach for "I'll try it tonight")**
- **Cole Medin** (most consistent "ship a real agent" channel; LangGraph / Agents SDK / n8n), **AI Jason** (LLM app engineering), **Matthew Berman** (#1 local-LLM teaching), **Matt Wolfe** (AI news reach). These cover *what drops into Cursor/Claude Code today*. fak's OpenAI- *and* Anthropic-compatible gateway + MCP, no agent-side changes, is the hook for them: "add one binary, your agent can't get prompt-injected into a destructive call."

**C. Generalist dev-tech (mass reach, clip machines)**
- **Fireship** (4.1M, now heavy on Cursor/Claude Code/Copilot) — the 100-second-format king. fak is *perfectly* Fireship-shaped: one binary, one surprising thesis ("safety boundary == reuse boundary"), a number. **ThePrimeagen** (systems/Rust/perf, allergic to hype) — the contrarian fit: "one 13MB Go static binary vs a multi-GB Python/CUDA stack" is his exact aesthetic, *if* you keep the honest fences. **Theo (t3.gg)** — reacts to spicy infra theses.

**D. Self-host / homelab (perfect for the reuse-savings story)**
- **Jeff Geerling** (951K), **Wolfgang's Channel** (313K), **Techno Tim** (295K, already shipping local-AI content). Angle is NOT security — it's "self-host a read-heavy agent fleet, reuse setup work once instead of every turn." But note the honest fence is load-bearing here: the reuse win is **self-host + read-heavy only**, and even ~1% writes can flip it negative. Geerling especially will check your math; lead with the *tuned-baseline* 4.1x, never the naive 8.8x.

## What gets you removed / dismissed (the rules)

1. **Leading with the naive 8.8–9.7x.** A perf-literate creator (Prime, Geerling, anyone who's read the vLLM/SGLang papers) instantly recognizes a re-prefill strawman and writes you off as dishonest. Always lead with the **~1.5–4.1x vs tuned warm-cache**.
2. **Letting the ~60x / "agent city" projections read as measured.** Label them DESIGN TARGET on the same line, every time. One unlabeled projection = the whole repo is "vaporware" in the comments.
3. **Claiming fak is faster than vLLM/llama.cpp.** It isn't and doesn't try to be. The contrast is *operational surface*, not tok/s. Say it first, before they catch it.
4. **Presenting the detector as the defense.** The honesty ("~100% evadable by design") is the *hook* — burying it makes the lock claim look naive instead of deliberate.
5. **Simulated power/$/kWh shown as measured.** No power meter on the box. Any energy number must carry "simulated."
6. Security creators specifically: **don't claim "unbreakable."** Claim "attacker must beat two independent gates, here's the unsigned-field fail-closed receipt" and let them try to break it on camera. The invitation *is* the content.

## Demo-ability / clippability — what's already a short

The repo is unusually clippable because the proof needs **no key, no model, no GPU** and resolves in one terminal frame:

- **The 60-second preflight clip** is the money shot: `go run ./cmd/fak preflight ... refund_payment` → **DENY (POLICY_BLOCK)** in red; `search_kb` → **ALLOW** in green. Two words, two colors, one screen. That is a complete Fireship/Short with zero editing.
- **`fak agent --offline`**: "injection-in-context YES→no, destructive-op YES→no, task still completed." Three before/after toggles = three hard cuts.
- **The live browser demos** (the turn-tax race, model-reuse race vs *tuned* baseline) are a side-by-side race — inherently a Short, no terminal literacy required.
- **max|Δ|=0** is a perfect on-screen artifact: "cut a poisoned span out of the *middle* of a live KV cache, and it's bit-for-bit identical to a run that never saw it — not one number differs." No shipped engine (vLLM/SGLang/OpenAI/Anthropic caches) does mid-run eviction. That's a genuine "wait, what?" frame.

## Hook density: the first 15 seconds (2026 reality)

The bar moved: the swipe-gate is at **2.5–3 seconds**, with a **secondary hook at ~14–15s**, and you need visual + on-screen-text + voice all saying the *same* thing in frame one. For fak:

- **Frame 1 (0–3s):** the DENY in red, full screen, text overlay "the AI tried to refund. it couldn't." — show the payoff, not the setup. No logo, no intro, no "in this video."
- **0–8s:** the thesis as a contrast, not a topic — "every agent guide adds a *detector*. this deletes the *lever*." Contrast + concrete object beats explanation.
- **~14s (second hook):** the surprise — "and the same boundary that blocks the attack makes the agent cheaper to run." That's the line that earns the rest of the video and is unique to fak.

Practical density target for these niches: **one verifiable claim + one on-screen artifact every ~3–4 seconds**, because this audience is hostile to hand-waving. fak can sustain that because every claim has a terminal frame behind it (CLAIMS.md / BENCHMARK-AUTHORITY.md traceability is itself a flex — show the "every number → commit" line for the skeptic crowd).

## The ONE highest-ROI move per channel

- **Simon Willison:** a tight blog-comment / email framing fak as a *structural* lethal-trifecta closer with the honest "detector is evadable by design" fence. He amplifies vocabulary that's both new and honest. One mention from him reaches every creator in archetype A.
- **LiveOverflow / Hammond / IppSec:** ship a **"break my quarantine gate" challenge repo** — attacker fully controls the tool result, win condition is firing the destructive call. CTF creators cover *challenges*, not READMEs.
- **Cole Medin / AI Jason / Berman:** a 90-sec **"drop fak in front of your existing Claude Code / Cursor agent, no code changes"** demo. Their audience converts on "works with my current stack tonight."
- **Fireship:** a pre-cut **100-second script** built around the one thesis ("the safety boundary and the reuse boundary are the same boundary") + the DENY frame + the 4.1x. Hand him the angle, not the docs.
- **ThePrimeagen:** a single honest framing — **"13MB Go static binary, no go.sum, vs a multi-GB Python/CUDA serving stack"** with the fences intact. He reacts to *theses with receipts*; the prior-art "0/29 novel, the contribution is the assembly" honesty is catnip for him.
- **Geerling / Techno Tim / Wolfgang:** a **self-host read-heavy fleet** demo leading with the *tuned* 4.1x and the explicit "writes flip it negative" caveat — they trust creators who pre-state the limits.

**Distribution path that actually feeds these channels (no paid):** **Show HN** is the upstream — front-page Show HN (~2.3% make it; 50 pts = top 6%) drives 5–50k visits and ~1.4 stars/upvote in 48h, and creators in A/B/C openly mine the HN front page and r/LocalLLaMA for video topics. So the real sequence is: **Show HN + r/LocalLLaMA post with the DENY clip embedded → creators pick it up from the trending signal**, plus direct, *specific* outreach (the per-channel move above, never a generic blast) to the 3–4 highest-fit names. Get listed on the relevant **awesome-llm-security / awesome-ai-agents** lists too — that's where the security-research creators source.

Sources:
- [10 Best YouTube Channels for AI Engineering & LLMs in 2026 — learnwithpath](https://learnwithpath.com/blog/best-youtube-channels-for-ai-engineering-2026)
- [Best YouTube Creators for Learning AI Agents & Automation 2026 — AIAttention](https://aiattention.ai/blog/best-ai-agents-educators-youtube-2026)
- [9 Best YouTube Channels for Cybersecurity in 2026 — learnwithpath](https://learnwithpath.com/blog/best-youtube-channels-for-cybersecurity-2026)
- [My Lethal Trifecta talk at the Bay Area AI Security Meetup — Simon Willison](https://simonwillison.net/2025/Aug/9/bay-area-ai/)
- [AI Security in 2026: Prompt Injection, the Lethal Trifecta — Airia](https://airia.com/ai-security-in-2026-prompt-injection-the-lethal-trifecta-and-how-to-defend/)
- [LLM Red Teaming Guide 2026 — appsecsanta](https://appsecsanta.com/ai-security-tools/llm-red-teaming)
- [How Fireship became YouTube's favorite programmer — Engineer's Codex](https://read.engineerscodex.com/p/how-fireship-became-youtubes-favorite)
- [Cole Medin — YouTube](https://www.youtube.com/@ColeMedin)
- [Ollama vs vLLM vs llama.cpp vs MLX 2026 — codersera](https://codersera.com/blog/ollama-vs-lm-studio-vs-vllm-vs-llama-cpp-vs-mlx-2026/)
- [Best Home Lab Servers / homelab creators 2026 — wemustbegeeks](https://www.wemustbegeeks.com/best-home-lab-servers-for-beginners-in-2026/) and [Techno Tim homelab](https://technotim.com/categories/homelab/)
- [Viral Hooks for YouTube Shorts 2026 — vidIQ](https://vidiq.com/blog/post/viral-video-hooks-youtube-shorts/)
- [Viral YouTube Shorts 2026: Hooks, Trends & Tips — Praper Media](https://prapermedia.com/blog/make-viral-youtube-shorts/)
- [Hacker News Marketing for Developer Tools: Show HN — daily.dev](https://business.daily.dev/resources/hacker-news-marketing-developer-tools-show-hn-launch-day-sustained-coverage/)

### x_other
I have enough to write the brief. The standout finding: fak's "treat the model as untrusted, the tool call as a syscall" framing sits at the exact intersection (agent security + systems) that two specific micro-communities reward — the prompt-injection/lethal-trifecta crowd around Simon Willison, and the systems-aggregator audience (Lobsters). That's the ranking spine.

---

# Where AI-agent-infra gets discovered in 2026 — ranked by ROI for fak

The thesis that wins you distribution: **"treat the model as untrusted, the tool call as a syscall."** That sentence is a security claim AND a systems claim, which is exactly why it travels in two communities that most agent-infra launches can't reach: the prompt-injection/lethal-trifecta crowd, and the OS-people-on-aggregators crowd. Rank below is for *that* artifact, not generic dev-tool marketing.

## Tier 1 — highest ROI

**1. X/Twitter — the AI-agent-security cluster (not "AI Twitter" broadly).**
This is your single best channel because the audience that gets your one-liner already lives here and reposts each other. Concrete targets: **Simon Willison** (@simonw — coined "prompt injection," owns "the lethal trifecta," posts daily, links liberally to things that *honestly* fence their claims). Your CLAIMS.md honesty ledger — especially "the detector is ~100% evadable BY DESIGN; the floor is the lock + containment" — is *engineered* to be the kind of thing he quote-links. Also: the **OWASP LLM Top-10 / agentic-security** people (prompt injection is still their #1 in 2026), the **MCP-security** voices reacting to the CVE-2025-6514 RCE and the malicious-MCP-server-in-the-wild stories, and **swyx** (@swyx, Latent Space) for the systems/AI-engineer overlap.
- *Removes you:* pure self-promo with no claim to chew on; leading with the naive 8.8–9.7x number (the security crowd will smell it and you lose them permanently). Lead with the tuned-baseline ~1.5–4x or with the safety boundary, never the naive multiple.
- *Highest-ROI move:* one thread built entirely on the **lethal trifecta** — show fak structurally breaking it (capability lock = the lever doesn't exist; result quarantine = poison bytes never reach context; "attacker must beat both"), with the 60-second `preflight … refund_payment → DENY` proof as the first reply. Tag the framing to Willison's vocabulary, not your product name. This is the post most likely to get an organic repost from a high-signal account.

**2. Lobsters (lobste.rs).**
Smaller than HN but *denser* with exactly your reader — OS, systems, security people who will appreciate "in-process gate, no IPC/sidecar/second model" and "max|Δ|=0 bit-exact KV eviction." Conversion-to-real-eyeballs per upvote is higher than HN here.
- *Rules that remove you:* self-promo must be **under ~25% of your stories+comments** — they ban write-only accounts. You need a real account with prior non-fak participation, and you must check the **"I am the author"** box (it's enforced socially, and authored content gets a small boost). Tag correctly (`security`, `ai`, or `compsci`).
- *Highest-ROI move:* submit the **technical write-up of the bit-exact mid-run KV eviction + ed25519 deletion certificate** (not the product page) — "no shipped engine offers mid-run eviction, proven against an HF oracle at max|Δ|=0." That's a systems result, not an ad, and it's the most Lobsters-shaped thing you have. Be in the thread to defend the `max|Δ|=0` methodology.

**3. awesome-lists — but the *right* two.**
Generic "awesome-ai-agents" lists are noise. Two are precise category matches: **ai-boost/awesome-harness-engineering** (literally "permissions, MCP, observability, orchestration" — fak is a textbook entry) and the **awesome-mcp** lists (korchasa, Rodert) since you ship an MCP + OpenAI/Anthropic gateway.
- *Removes you:* PRs that don't follow the list's category/format/one-line convention get closed; "framework" lists reject infra that isn't a framework — pick the permission/harness category explicitly.
- *Highest-ROI move:* one PR to **awesome-harness-engineering** under a permissions/capability-gate heading. It's low-effort, durable (keeps delivering discovery for months), and the maintainer audience is your exact ICP.

## Tier 2 — real but slower

**4. Dev newsletters — TLDR over Bytes for you.**
**TLDR AI** (~1.1–1.25M technical readers, 2026) and **TLDR Web Dev** (within the 7.2M network) are the realistic targets; **Latent Space / AINews** (200k+ AI-engineer subs, swyx) is the *best-fit* audience even if smaller. **Bytes.dev** (216k) is JS-flavored — wrong crowd for a Go security kernel; skip it. **Pragmatic Engineer** is org/hiring-focused; also a weaker fit.
- *Reality:* you mostly don't submit — you get *picked up* after Tier 1 traction. AINews/Latent Space curates from what's already moving on X/HN.
- *Highest-ROI move:* don't pitch cold. Make the X thread land first; TLDR/AINews scrape the resulting signal. If you pitch one, pitch **Latent Space** with the systems angle (one binary vs multi-GB Python/CUDA stack), not throughput.

**5. dev.to / Hashnode.**
Low gatekeeping, decent SEO half-life, but low per-post discovery — these are *canonical-content homes*, not discovery engines. Worth it as the place your write-up lives so newsletters/lists have something to link.
- *Removes you:* dev.to flags overt product spam; tag-stuffing hurts.
- *Highest-ROI move:* cross-post the same KV-eviction or lethal-trifecta write-up here as the canonical URL, with the **live in-browser demo** (demos.html) embedded — it's the rare post where a reader can run the proof without installing anything.

## Tier 3 — low ROI for this product; spend little

**6. Bluesky.** Real and growing dev presence in 2026 (600+ software-eng starter packs, infra/distributed-systems/AI-tooling packs), and the systems crowd that left X partly lives here. But reach-per-post still trails X for a *cold* account, and the agent-security cluster's center of gravity is still X. *One move:* get added to the **infrastructure / distributed-systems / AI-tooling starter packs** (via stevendborrelli/bluesky-tech-starter-packs) and mirror your best X threads — near-zero cost, optional upside.

**7. LinkedIn.** Wrong register entirely — fak's credibility *is* its honest fences, which read as weakness in LinkedIn's tone. Skip unless you're recruiting or chasing enterprise buyers, which is a different motion.

**8. Discord/Slack (LangChain Slack, MCP servers).** These are *support/relationship* channels, not discovery channels — you don't get organically *found* here, you nurture people already aware of you. Don't drop links cold (fastest ban in any of these). Worth joining the **MCP** and **LangChain** communities to answer questions where fak is genuinely the answer, but rank it last for *discovery*.

## The one cross-channel rule that protects all of this
Your moat is the **honesty ledger** — naive-vs-tuned fenced, simulated-power labeled, "0/29 novel, the contribution is the assembly," detector "evadable by design." Every one of these audiences (Willison, Lobsters, AINews) is allergic to over-claiming and rewards exactly this. The instant a post leads with the unqualified 8.8x or an unlabeled "60x," you lose the high-signal reposters permanently. **Lead with the fence; it's not a caveat, it's the hook.**

**Sources:** [Lobsters self-promo guideline](https://lobste.rs/about) · [How the Lobsters front page works](https://blog.nilenso.com/blog/2026/01/20/lobsters-front-page/) · [Simon Willison — lethal trifecta / prompt-injection papers](https://simonwillison.net/2025/Nov/2/new-prompt-injection-papers/) · [OWASP: prompt injection still #1 in 2026](https://www.helpnetsecurity.com/2026/06/11/owasp-prompt-injection-ai-security-failures/) · [MCP CVE-2025-6514 / malicious-server risks](https://blog.cyberdesserts.com/ai-agent-security-risks/) · [awesome-harness-engineering](https://github.com/ai-boost/awesome-harness-engineering) · [awesome-mcp (korchasa)](https://github.com/korchasa/awesome-mcp) · [TLDR / Bytes reach](https://www.readless.app/blog/best-developer-newsletters-2026) · [TLDR AI reach](https://datanorth.ai/blog/top-10-ai-newsletters-to-follow-in-2026) · [Latent Space 200k+](https://www.latent.space/about) · [Bluesky tech starter packs](https://github.com/stevendborrelli/bluesky-tech-starter-packs) · [Engineering Bluesky starter packs](https://blueskystarterpack.com/engineering) · [LangChain community Slack](https://www.langchain.com/join-community)
