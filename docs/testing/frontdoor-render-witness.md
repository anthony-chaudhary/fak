# Capture a comparable first-screen witness

**Primary audience:** page workers changing a landing page or public front door.
**Lifecycle:** current testing process. **Generation:** `gen/now`.
**Authority:** this page defines documentation capture evidence; the [front-door clarity scorecard](../quality/frontdoor-clarity-scorecard.md) defines the clarity verdict, and runtime/UI defects follow the repository's captured-render rule in [`AGENTS.md`](../../AGENTS.md#proof-by-default-every-issue-fix-ships-its-evidence).
**Next action:** choose the capture mode that matches the changed surface, freeze its profile, and save the first screen plus metadata before scoring or claiming clarity.

A comparable witness shows exactly what an unprimed reader receives before scrolling. It binds the page revision, rendering conditions, first-screen boundary, and artifact hash so another maintainer can reproduce the same observation.

## Choose the capture mode

| Changed surface | Required capture | Use when | Completion signal |
|---|---|---|---|
| Rendered web or documentation layout | **Image capture** (`.png`) | Typography, wrapping, hierarchy, clipping, spacing, responsive layout, or visual order matters. | The image has the full viewport, metadata, and hash; no crop hides first-screen content. |
| Markdown information architecture | **Structured text capture** (`.txt` or fenced witness) | Audience, job, choices, proof links, lifecycle, or next action changed and pixel layout is not the claim. | Ordered visible blocks and fold boundary can be traced to the source revision. |
| Terminal/TUI landing surface | **Byte/render capture** plus screenshot when live display matters | ANSI behavior, pane ownership, terminal wrapping, corruption, or interaction is the claim. | Captured bytes assert the property; screenshot records the live viewport when applicable. |

**Default:** use structured text for prose-only front-door changes. Upgrade to an image whenever visual placement or wrapping affects the claim. A structured capture cannot prove a visual defect is fixed; an image alone cannot prove hidden semantics or a command works.

## Freeze the profile

Record these fields before capture:

| Field | Required value |
|---|---|
| Source | Route path, full commit, and content hash |
| Mode | image, structured text, or terminal bytes |
| Renderer | Product and exact version; use `source-order` for a structured Markdown capture |
| Viewport | Width × height in CSS pixels and device-pixel ratio, or terminal columns × rows; use `source-blocks` for structured text |
| Presentation | Zoom, theme, font/size, locale, and authentication state when they can change output; `not applicable` only with a reason |
| Entry state | Direct URL/path, redirects, expanded/collapsed controls, banners, and clean-session state |
| Capture command | Exact command or manual procedure another maintainer can repeat |
| Boundary | Last fully visible element before scrolling; include partially visible content rather than cropping it away |
| Artifact | Repository path or durable issue/PR attachment, SHA-256, capture time in UTC, and author/tool |

Use the same profile for before/after comparisons. If a renderer upgrade or viewport change is intentional, capture both profiles and label the comparison rather than treating unlike images as a regression.

## Capture procedure

1. **Start from the declared revision.** Confirm the route content and direct assets come from the recorded commit.
2. **Open a clean entry state.** Use a fresh browser profile or equivalent deterministic state; record unavoidable banners or authentication.
3. **Apply the frozen profile.** Set viewport, zoom, theme, locale, and font before loading the route.
4. **Capture without editorial cropping.** Preserve the whole viewport from its top edge through the fold. Exclude browser chrome only when viewport dimensions are recorded independently.
5. **Hash the artifact.** Compute SHA-256 over the saved bytes. A pasted transcript without durable bytes is not a capture.
6. **Read back the first screen.** Without consulting deeper context, state the audience, page job, visible choice/default, current lifecycle/generation/support boundary, proof cue, and immediate action.
7. **Compare and classify.** Use the same profile for the baseline and candidate, then score the candidate with the [clarity scorecard](../quality/frontdoor-clarity-scorecard.md).
8. **Disposition every ambiguity.** Revise the route or file/link a deduplicated issue before claiming the witness passes.

## Structured text format

A structured capture preserves source order and the fold decision without pretending to reproduce pixels. Copy the visible heading, metadata, paragraphs, choice rows, commands, and link labels exactly; do not paraphrase inside `visible_blocks`.

```text
schema: fak-first-screen-witness/1
route: <path>
revision: <full commit>
content_sha256: <hex>
mode: structured-text
renderer: source-order
viewport: source-blocks
entry_state: direct route, no hidden context
capture_rule: <for example, title through first complete next-action block>
fold_after: <exact final included block>
visible_blocks:
  - <exact text in source order>
artifact_sha256: <hex>
captured_at_utc: <RFC3339>
captured_by: <person or tool>
read_back:
  audience: <what the capture states>
  page_job: <what the capture states>
  applicable_choice: <default or selection rule>
  lifecycle_generation_support: <visible boundary>
  proof_cue: <visible evidence route>
  next_action: <one checkable action>
ambiguities: []
```

For image mode, keep the same metadata in a sidecar `<image>.witness.txt`. For terminal mode, record columns, rows, `$TERM`, color capability, emitted-byte artifact, and screenshot sidecar where the live display matters.

## Pass contract

The witness passes only when:

- source revision, profile, boundary, durable artifact, and SHA-256 are present;
- the capture contains the complete first screen without hidden coaching or selective crop;
- a reader who did not author the route independently reaches the intended audience, job, applicable choice, current context, and next action;
- baseline and candidate use comparable profiles, or an intentional profile change is explicitly paired;
- every ambiguity is resolved or linked to an open issue;
- links and named actions are separately exercised—the capture proves presentation, not destination or runtime behavior.

Missing evidence is `not yet`, not a pass. Keep the artifact with the issue/PR or in a stable repository evidence path; do not rely on a local screenshot or worker narration.

## Witness this route

The companion [`frontdoor-render-witness.capture.txt`](frontdoor-render-witness.capture.txt) is the exact source-block artifact, verified by its [`SHA-256 sidecar`](frontdoor-render-witness.capture.sha256). Its [`witness metadata`](frontdoor-render-witness.capture.witness.txt) records the profile, exact artifact hash, source-content hash, fold, and read-back. The source hash binds the pre-publication page bytes; issue #4938 records the resulting landing commit without requiring an impossible self-referential commit hash inside that commit. Regenerate it after any change to the title, metadata, first action, or capture-mode choice table. Its independent reader must restate the page worker audience, comparable-first-screen job, structured-text default, visual-upgrade rule, and first action with `ambiguities: []`.

After a landing-page capture passes, attach its metadata and artifact link to the route's issue. Use the [front-door navigation witness](../quality/frontdoor-link-witness.md) for destinations and action shape, and the [generation drift check](../quality/generation-drift-check.md) when current authority changes.
