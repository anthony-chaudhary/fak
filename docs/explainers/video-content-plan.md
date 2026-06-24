---
title: "Video explainer plan — getting the initial fak/fleet points across"
description: "How to turn the existing fak explainers into short video content using the video tooling across the sibling repos (openclaw video_generate, hermes-agent ascii-video for brand/narration/captions, deer-flow structured prompts, sglang self-hosted backend, ffmpeg), without letting generated b-roll overstate measured results."
slug: video-content-plan
date: 2026-06-19
status: plan
---

# Video explainer plan — getting the initial points across

This plan maps the **video tooling that already exists across the sibling repos**
onto the **initial points fak/fleet needs to land** (the three written explainers +
the share-kit), and defines a repeatable production pipeline so each explainer becomes
a short video "generated in that way." The primary stack is `openclaw`'s
`video_generate`, but two other repos add capabilities that close real gaps —
`hermes-agent`'s deterministic **ascii-video** renderer (brand layer + TTS narration +
captions, $0 local) and `deer-flow`'s structured-prompt video skill — and `sglang`'s
multimodal_gen gives a self-hostable video-model backend (see §1.0).

The governing constraint comes straight from the share-kit: **be serious, concrete,
and reproducible; do not turn the research thesis into hype.** Generated video is
powerful for attention and metaphor and useless — actively harmful — for proving a
measured number. So the whole plan is built around one rule:

> **Generated video carries the *feeling* and the *framing*. Deterministic rendered
> diagrams and real screen-capture carry the *claims*.** Never let a Wan/Veo clip be
> the thing on screen when a measured figure is spoken.

---

## 1. What the tooling actually is

### 1.0 The tooling landscape across the repos

Four distinct video tools exist; they are complementary, not redundant. Pick by what
a shot needs.

| Tool | Repo | What it is | Cost / keys | Best for |
|---|---|---|---|---|
| **`video_generate`** | `openclaw` | hosted **generative** T2V/I2V/V2V (Wan/Veo/fal/minimax) + local `comfy` | hosted = per-clip API; `comfy` = $0 local | photoreal/cinematic **atmosphere** b-roll |
| **`ascii-video`** | `hermes-agent` | **deterministic** procedural ASCII-art video; also TTS narration, SRT/karaoke captions, audio-reactive, and **video-to-ASCII restyle** | **$0 local** (Python+ffmpeg; ElevenLabs key optional for TTS) | **brand/title layer, narration, captions, unifying restyle** — and *claim-safe* (renders exactly what you feed it) |
| **`video-generation`** | `deer-flow` | structured-**JSON** prompt → `generate.py` CLI (reference image as first/last frame) | depends on backend | a cleaner **authoring model** for storyboard shots |
| **multimodal_gen** | `cama-complete/…/sglang` | **self-hosted** video DiTs (Wan/Hunyuan) behind an OpenAI-compatible `video_api.py` | $0 marginal on our GPU | a **self-hosted backend** to slot behind `openclaw`'s provider abstraction at scale |

The pivotal one is **`ascii-video`**: it is deterministic (math/procedural, no model
hallucination), terminal-native (on-brand for a CLI / agent-kernel product), runs $0
locally, and **also supplies the narration (TTS) and caption (SRT) layers** the
generative tools don't. Because it renders exactly the text/data you give it, it is
the one generative-feeling tool that is also **safe to carry a claim**. The other tools
are folded into the pipeline in §4.

The rest of §1 details **`openclaw`'s hosted generative stack**. It lives in
`openclaw/src/video-generation/` and is surfaced to agents as a single tool,
`video_generate` (`openclaw/src/agents/tools/video-generate-tool.ts`). It is a real,
fallback-chained, multi-provider capability — not a toy.

### 1.1 The `video_generate` tool surface

| Param | What it does | Notes for us |
|---|---|---|
| `action` | `generate` (default), `status`, `list` | `list` prints providers/models + which `providerOptions` each accepts. Run it first. |
| `prompt` | text prompt (required) | the shot description |
| `image` / `images` (≤9) + `imageRoles` | reference images → **image-to-video** | roles: `first_frame`, `last_frame`, `reference_image`. **This is our main lane** — seed from our 175 diagrams. |
| `video` / `videos` (≤4) + `videoRoles` | reference videos → **video-to-video** | `reference_video`; restyle/extend an existing clip |
| `audioRef` / `audioRefs` (≤3) + `audioRoles` | reference audio (e.g. music bed) | `reference_audio` |
| `model` | `provider/model` override | e.g. `alibaba/wan2.6-i2v`, `google/veo-*` |
| `size` / `aspectRatio` / `resolution` | `1280x720`…; `16:9`,`9:16`,`1:1`,`21:9`,`adaptive`; `480P/720P/768P/1080P` | **16:9 for YouTube/landing, 9:16 for shorts, 1:1 for social** |
| `durationSeconds` | target length; **rounded to nearest provider-supported** | Wan caps at **10s/clip** — so videos are *assembled from clips*, not generated whole |
| `audio` | toggle generated audio | keep **off** for b-roll; we add narration/music in post |
| `watermark` | provider watermark toggle | off |
| `providerOptions` | provider-specific (e.g. `{"seed": 42}`) | **pin a seed** for reproducible re-renders |

The tool runs **async** (background task + `action=status` polling) and saves outputs
to openclaw-managed media storage, delivered as attachments.

### 1.2 Modes (auto-selected from inputs)

- **T2V** (`generate`) — prompt only. Good for abstract/atmospheric establishing shots.
- **I2V** (`imageToVideo`) — one+ reference images. **Our default**: animate a static
  diagram/title card so it has motion without re-drawing it.
- **V2V** (`videoToVideo`) — restyle/continue an existing clip.

### 1.3 Providers (configured via `agents.defaults.videoGenerationModel`)

`alibaba` (Wan 2.6/2.7 T2V/I2V/R2V/V2V, ≤10s, audio+watermark, via DashScope/Model
Studio key) · `fal` (hosted image+video) · `minimax` (video + music) · `google`
(Veo / Gemini media) · `comfy` (local ComfyUI workflows — no API cost, runs on our
own GPU). Each provider needs its own API key except `comfy`. The runtime falls back
across `primary → fallbacks` and **skips** any candidate that can't honor the request
(duration/audio/option mismatch) rather than silently dropping inputs.

### 1.4 Companion tools we'll lean on

- **`image_generate`** (`openclaw/src/image-generation/`) — make seed frames / title
  cards / character-consistent motifs to feed I2V.
- **`music_generate`** (minimax/comfy) — original, license-clean music beds.
- **`video-frames` skill** (`openclaw/skills/video-frames/`, ffmpeg) — extract frames
  for thumbnails / QA / picking I2V seed frames. **Local, no key.**

### 1.5 What `video_generate` does *not* give us (and what fills each gap)

`video_generate` alone has no timeline editor, no captioning, no narration/TTS, no
concatenation. The gaps are filled by the other tools, not by hand-rolling everything:

- **Narration (TTS)** → `ascii-video`'s TTS mode (ElevenLabs) — or any external TTS.
- **Captions / karaoke text** → `ascii-video`'s lyrics/SRT mode, or ffmpeg subtitle burn.
- **Concatenation / mux / crossfade / aspect exports** → **`ffmpeg`** (the assembler;
  the `video-frames` skill already proves ffmpeg is on the box).
- **A consistent house style across mixed footage** → `ascii-video`'s video-to-ASCII
  restyle pass over the generative clips.

---

## 2. The initial points to get across (and where they're written)

Source of truth = `docs/explainers/*` + `docs/share-kit.md` +
`docs/explain-executive-summary-2026-06-18.md`. The points, in adoption order:

| # | Point | One-liner (share-kit approved) | Source |
|---|---|---|---|
| **P1** | **The agent kernel** / syscall spine (the lead hook) | "Agents are programs. Give them permissions, quarantine, and proof." | `share-kit.md` |
| **P2** | **Local + kernel ≈ frontier on safety & cost, now** | "Frontier-grade safety and ~$0 cost on a 1.5B local model; size buys capability." | `explainers/local-vs-frontier-parity.md` |
| **P3** | **KV cache is the agent bill** | "Agents re-send the transcript every turn — ~239:1 input:output — so the cache deletes most of the spend." | `explainers/kv-cache-agentic-context.md` |
| **P4** | **Hardware-portable forward pass** | "One forward-pass loop, many backends — CPU, CUDA, Vulkan, all witnessed on real silicon." | `explainers/hardware-portability.md` |
| **P5** | **Fleet reuse / value** (expansion, *not* the first hook) | "Share a witnessed world; repeated reads collapse across agents." | executive summary |

The share-kit is explicit about sequencing: **lead with the boundary (P1), never lead
with the 45× chart.** P5 is the expansion story you earn *after* trust.

---

## 3. The hybrid production model (the honesty rule, made concrete)

Every video is **three layers**, and which tool owns which layer is fixed:

| Layer | Carries | Tool | Why |
|---|---|---|---|
| **Atmosphere / metaphor** (establishing shots, transitions, title stings, "feel") | emotion, attention | **`video_generate`** (photoreal) **or `ascii-video`** (terminal-native, $0) | generative is good at photoreal; ascii is on-brand + free for the rest |
| **Claims** (the boundary diagram, the A/B table, benchmark charts, code/CLI) | the actual argument + every number | **deterministic only**: `visuals/*.svg/png` (via `render-mermaid`), real `fak` terminal capture, on-screen text, **or `ascii-video`** (it renders exactly what you feed it) | a number must come from a tool that cannot invent it — never a `video_generate` clip |
| **Voice + music** | narration, pacing, music bed | **`ascii-video` TTS** (or external TTS) + `music_generate`; muxed by **ffmpeg** | keep generated-audio **off** in `video_generate` so the claim layer stays clean |

This is the literal implementation of the share-kit's "one clean diagram before any
benchmark chart" and "What Not To Say." The honesty line is **deterministic vs
hallucination-capable**, not text vs video: a rendered diagram, a real capture, *and*
an `ascii-video` data field are all deterministic, so any of them may carry a number.
A `video_generate` (Wan/Veo) clip never may. If a frame shows a measured figure, it
came from a tool that physically cannot have invented it.

---

## 4. The repeatable pipeline ("generated in that way")

Same five steps for every video; only the script and seeds change.

```
1. SCRIPT      Write a tight VO script (≤150 wpm). Mark each line CLAIM | ATMOSPHERE.
               CLAIM lines pin to a deterministic visual; ATMOSPHERE lines get a clip.
2. SEED        For each shot:
                 - claim shot      -> visuals/NN-*.svg|png (or render-mermaid a new one),
                                      real `fak` CLI capture, or an ascii-video data field
                 - atmosphere shot -> image_generate a seed frame (photoreal, pin style+seed),
                                      or design an ascii-video scene (terminal-native, $0)
                 - title/brand     -> ascii-video title scene (one reusable house style)
3a. NARRATE    ascii-video TTS mode renders the VO track from the script (or external TTS);
               music_generate makes a license-clean bed.
3b. ANIMATE    video_generate per photoreal atmosphere shot:
                 action=generate           (T2V) for pure abstract motion, OR
                 image=<seed> imageRoles=first_frame  (I2V) to animate a seed/diagram
                 durationSeconds<=8, aspectRatio=16:9, audio=false, providerOptions={seed:N}
                 (Wan caps ~10s -> generate in clips; never one long render)
               ascii-video renders the brand/title + any ASCII atmosphere scenes ($0 local).
4. ASSEMBLE    ffmpeg: concat clips + diagram stills in script order; burn captions
               (ascii-video SRT mode or ffmpeg subtitle burn); mux VO + music bed;
               crossfades; export 16:9 master. OPTIONAL: ascii-video video-to-ASCII pass
               over the generative clips to unify everything into one house style.
5. DERIVE      ffmpeg re-crops/re-exports the master to 9:16 (shorts) and 1:1 (social);
               video-frames skill grabs the thumbnail. One render, many cuts.
```

Reproducibility: **pin a `seed` in `providerOptions` and pin the `provider/model`** for
every clip, and keep the script + seed list in-repo next to the output. A re-render
then reproduces the same footage — the same discipline the rest of the repo applies to
benchmarks.

Cost control: most of the video can be **$0**. `ascii-video` (brand, titles, captions,
narration, and any ASCII atmosphere) runs locally for free; `comfy` (local GPU) drafts
the photoreal shots; a hosted provider (Wan/Veo) is reserved for the *final* polish pass
on the few photoreal beats that need it. We already run models on this box's RTX 4070 —
local renders cost electricity, not API spend. A fully-$0 first cut (all-ASCII house
style) is a legitimate shippable option, not just a draft.

---

## 5. The video slate

Tiered so we ship the highest-leverage asset first and reuse footage downward.

### Tier 0 — the one that must exist: "The agent kernel" (P1), ~75s, 16:9
The trust hook. Storyboard in §6. Everything else reuses its title system and motifs.
For a CLI/kernel product, a **fully terminal-native all-`ascii-video` cut** is a strong
(and $0) house style — the boundary/quarantine flow reads naturally as animated ASCII;
the storyboard's `[A]` shots simply become ASCII scenes instead of Wan clips.

### Tier 1 — one explainer per written point (60–90s each, 16:9)
Each is a faithful video cut of an existing `docs/explainers/*` doc:

| Video | Point | Deterministic claim visuals to reuse | Atmosphere beats |
|---|---|---|---|
| **Local vs Frontier** | P2 | the parity table (render from `local-vs-frontier-parity.md`); `02-boundaries`, `10-compute-tiers` | "small model on a desk-class GPU" establishing shot; safety-shield motif |
| **The KV cache is the bill** | P3 | the input:output ratio table + `07-kv-hierarchy`, `09-shared-prefix` | append-only stream vs head-mutation "cliff" metaphor |
| **One loop, many chips** | P4 | `10-compute-tiers`, the HAL seam table; real CUDA/Vulkan parity output capture | CPU→GPU→NPU→wafer "same loop lighting up different silicon" montage |

### Tier 2 — shorts (15–30s, 9:16) derived from Tier 0/1 masters
One claim each, for social. e.g. "The support agent can search docs but can't refund
money." (the P1 demo card), "239:1 — why caching is the agent bill," "$0, fully local."
These are **cut from existing masters**, not generated fresh.

### Tier 3 — expansion: "Fleet reuse" (P5)
Only after P1 has landed externally. Reuses `15-net-gain-frontier`,
`18-turns-saved-sweep`, the cumulative-impact PNG. Explicitly labeled *projection*.

Ship order: **Tier 0 → Tier 1 (P2, then P3, then P4) → Tier 2 shorts → Tier 3.**

---

## 6. Deep storyboard — Video #0 "The agent kernel" (~75s)

Aspect 16:9, target 75s. `[C]` = deterministic claim visual, `[A]` = generated
atmosphere clip. VO is the on-screen-truth layer.

| t | Shot | Type | On screen | VO (CLAIM/ATMOSPHERE) |
|---:|---|---|---|---|
| 0–5 | Title sting | `[A]` I2V from an `image_generate` title card | "Agents are programs." text resolves | *(music only)* |
| 5–16 | The problem | `[A]` T2V, abstract: an agent firing tool calls at "production" | dimmed icons: shell, refund, delete | "We're giving AI agents real tools — shells, payments, prod. Prompt text is not a permission system." `ATMOSPHERE` |
| 16–34 | **The boundary** | `[C]` animate `visuals/39-agent-kernel-card.svg` (I2V, subtle push-in) | `model proposes → fak policy boundary → admitted call → result quarantine → context` | "fak is an agent kernel — the model proposes, the kernel disposes." `CLAIM` |
| 34–50 | **Denied destructive call** | `[C]` real `fak preflight` terminal capture | `refund_payment → POLICY_BLOCK`; `search_kb → allow` | "The support agent can search the docs. It cannot refund money. Policy-denied tools don't run." `CLAIM` |
| 50–66 | **Poisoned-policy A/B** | `[C]` the offline A/B table (from `fak agent --offline`) | baseline reads the poison; protected arm quarantines it | "Same task, same poison. The baseline ingests it; the protected arm never sees it." `CLAIM` |
| 66–75 | Close + repro | `[C]` title + command | `go run ./cmd/fak agent --offline` | "Permissions first. Filters second. Bring a trace — the verdict replays." `CLAIM` |

**Generated shots: 2** (title sting + problem) — small, cheap, claim-free. Everything
load-bearing is a real capture or a rendered diagram. Caption every claim shot with the
caveat (e.g. "offline deterministic harness, not live-model recall") per the share-kit.

### Example tool calls for the two generated shots
```
# Title card seed (image_generate), then animate it (I2V):
video_generate action=list                       # confirm provider + seed support
video_generate prompt="slow cinematic push-in on the title card, subtle particle drift, dark serious palette" \
  image="<title-card.png>" imageRoles=first_frame \
  aspectRatio=16:9 resolution=1080P durationSeconds=5 audio=false \
  model="alibaba/wan2.6-i2v" providerOptions={"seed":4070} filename="00-title-sting"

# Problem b-roll (T2V, abstract, no readable text -> no claim risk):
video_generate prompt="abstract: an autonomous agent firing streams of tool calls toward a glowing 'production' core, ominous, no text" \
  aspectRatio=16:9 resolution=1080P durationSeconds=8 audio=false \
  model="alibaba/wan2.6-t2v" providerOptions={"seed":1207} filename="01-problem-broll"
```
ffmpeg then concatenates these two clips with the four diagram/capture stills (each
shown 12–16s with a slow Ken-Burns), burns the SRT captions, and muxes VO + music.

---

## 7. Reusable assets already in-repo

- **`visuals/` — 175 assets** (`NN-*.mmd` + rendered `.svg`/`.png`). These ARE the
  claim layer; most claim shots need zero new art. Key ones: `02-boundaries`,
  `05-preflight`, `06-context-mmu`, `07-kv-hierarchy`, `09-shared-prefix`,
  `10-compute-tiers`, `39-agent-kernel-card` (the share/hero card).
- **`render-mermaid` skill** — to render any new diagram (e.g. a parity-table card) to
  PNG for animation.
- **`docs/explain-impact-cumulative-2026-06-18.png`** — the P5 expansion chart.
- **`video-frames` skill** — frame extraction for thumbnails + seed-frame picking.

---

## 8. Costs, constraints, risks

- **Clip length cap.** Wan ≈ 10s/clip; design in 5–8s beats and assemble. (Veo/others
  vary — `action=list` shows caps; runtime rounds `durationSeconds` to the nearest
  supported value and reports the normalization.)
- **Keys.** Hosted `video_generate` providers need their own API key in config;
  **`comfy` and `ascii-video` are the $0 local paths** on our own GPU/CPU (ascii-video
  needs only Python+ffmpeg; ElevenLabs key optional for its TTS). No key → the
  `video_generate` runtime tries auth-backed defaults.
- **Determinism.** A `video_generate` (Wan/Veo) clip is *not* bit-reproducible across
  provider versions even with a seed — that's exactly why claims never live in one.
  `ascii-video` and rendered diagrams *are* deterministic, which is why they may.
- **Honesty / "What Not To Say"** (carried verbatim from `share-kit.md`): no
  "uninjectable," no "we solved prompt injection," no "45× faster" without the gate and
  scope, no leading with the 45× chart. Every claim shot carries its caveat caption.
- **Text-in-video.** Generative models mangle text; **never** render a number/label
  inside a `video_generate` clip — overlay real text in ffmpeg, use a rendered diagram,
  or use `ascii-video` (which draws text deterministically and is the on-brand choice).
- **Brand consistency.** Pin one `image_generate` style + a fixed title system and reuse
  it as the I2V seed across all videos so the slate looks like one series.

---

## 9. Phased rollout + checklist

**Phase A — pipeline proof (no spend):** wire the 5-step pipeline end-to-end on the
Tier-0 script using `comfy` local drafts; confirm ffmpeg concat + caption-burn + VO mux
produces a 16:9 master and a 9:16 cut. Deliverable: a rough-cut "The agent kernel."

**Phase B — Tier 0 polish:** swap the 2 atmosphere drafts for a hosted final pass
(Wan/Veo), capture the real `fak preflight` / `fak agent --offline` output for the claim
shots, finalize captions + music. Deliverable: shippable Video #0 + thumbnail.

**Phase C — Tier 1:** repeat the pipeline for P2/P3/P4, reusing the title system and the
`visuals/` diagrams as claim shots.

**Phase D — Tier 2 shorts** cut from the masters; **Phase E — Tier 3** expansion video.

Per-video checklist:
- [ ] Script written; every line tagged `CLAIM` or `ATMOSPHERE`.
- [ ] Every `CLAIM` shot is a rendered diagram or real capture (zero generated text).
- [ ] Seeds + `provider/model` pinned and recorded next to the output.
- [ ] Caveat caption present on every claim shot.
- [ ] Passes the `share-kit.md` "What Not To Say" check.
- [ ] 16:9 master + 9:16/1:1 derivatives + thumbnail exported.

---

## 10. Files / pointers

- Tooling: `openclaw/src/video-generation/` · `openclaw/src/agents/tools/video-generate-tool.ts`
  · `openclaw/src/image-generation/` · `openclaw/skills/video-frames/SKILL.md`
- Brand/narration/captions: `hermes-agent/skills/creative/ascii-video/` (README + `references/`)
- Structured-prompt authoring: `deer-flow/skills/public/video-generation/SKILL.md`
- Self-hosted video backend (optional): `cama-complete/.../sglang/multimodal_gen/` (`video_api.py`, `dits/wanvideo.py`, `dits/hunyuanvideo.py`)
- Provider setup: `openclaw/docs/providers/{alibaba,fal,minimax,google,comfy}.md`
- Message source: `docs/share-kit.md` · `docs/explainers/*` ·
  `docs/explain-executive-summary-2026-06-18.md`
- Claim-layer art: `visuals/*` (+ `render-mermaid` skill)
