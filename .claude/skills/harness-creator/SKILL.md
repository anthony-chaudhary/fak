---
name: harness-creator
description: Create or customize a fak-native agent harness from a user's needs. Use when someone asks to make their own harness, agent product, branded/local UI, provider or tool profile, or a "10-minute" / "weekend" harness build. Drives the shipped external generator, preserves generated/user ownership boundaries, rebuilds and selfchecks the result, and routes deeper UI/extension work through public harnesskit contracts.
---

# /harness-creator — user need to rebuilt fak-native harness

Create a product around `pkg/harnesskit`; do not fork `cmd/fak`, import `internal/`, or parse terminal output. The minimum successful run is an external directory that builds and passes offline selfcheck after one user-owned customization.

## Intake: decide only what changes the product

Write this five-line frame before editing:

- **For:** the person who will use the harness.
- **Problem:** the job or failure they face now.
- **Today:** the real next-best alternative, including its setup and maintenance cost.
- **Better because:** the smallest observable improvement this harness should make.
- **Witness:** the command and artifact that will prove it.

Classify centrality as `Core`, `Enabling`, `Stewardship`, or `Peripheral`, then check all four project problems:

| Check | Harness question |
|---|---|
| P1 managed context | What repeated setup, history, or memory should the harness own? |
| P2 net-true efficiency | Against which real alternative will time/cost/quality be measured? |
| P3 bounded adaptation | What may skills, models, and tools change, and what remains fail-closed? |
| P4 integrated operations | How will the user run, inspect, rebuild, upgrade, and recover it? |

Ask follow-ups only when an answer changes the profile, policy boundary, interface, or witness. Prefer a runnable default over a questionnaire.

## Choose a track

### Ten-minute harness

Use this when the need fits product identity, profile/config, policy, instructions, or a thin interface projection. Promise ten minutes only when a timed clean-room witness supports it.

1. Pick an external target directory and Go module path.
2. Run:

   ```text
   fak harness init --dir <target> --module <module>
   ```

3. Read `<target>/harness.lock.json` before editing. Change only files marked `user`; the default customization seam is `<target>/product/config.go`.
4. Express the smallest user need by changing field values returned by `product.DefaultConfig` and, when needed, the body of `product.OfflineReply`. Preserve the `Config`, `DefaultConfig`, and `OfflineReply` signatures consumed by generated runtime code. Do not edit `generated/` or `cmd/product/main.go`.
5. Rebuild and selfcheck from the target:

   ```text
   go build -o <scratch-or-target-bin> ./cmd/product
   go run ./cmd/product --selfcheck
   ```

6. Capture: elapsed time, generator version, changed user-owned paths, build result, selfcheck semantic events, and the remaining gap to the stated witness.

If the timed witness is missing or red, say `not yet`; do not turn the slogan into a claim.

### Weekend harness

Use this when the user needs a new provider/model adapter, lifecycle middleware, skill pack, durable state, tool/policy adapter, channel, or custom UI. Start with the ten-minute spine, then add exactly one public seam at a time:

- composition and compatibility: #6792 and #6793;
- headless backend and extension loop: `pkg/harnesskit` plus #6803;
- skill/instruction packs: #6796;
- custom local web UI: headless protocol `fak.harness.run/v1` plus #6790;
- packaging and reproducible rebuilds: #6806.

A custom UI consumes semantic events and inputs. It must not scrape CLI/TUI prose or depend on provider wire objects. Keep branding/layout outside the kernel so UI iteration does not require rebuilding fak.

## Rebuild and upgrade contract

- Pin fak with `--fak-version`; record the immutable version in `harness.lock.json`.
- Rerun `fak harness init` to refresh recognized generated files. It must preserve user-owned files byte-for-byte.
- Rebuild the external product after every contract or config change.
- Run offline selfcheck before any live model/provider test.
- For a shipped binary, record OS/arch, Go version, module pin, build command, and output hash. Never imply that rebuilding the product rebuilds the fak kernel.
- A protocol-major or conformance change requires migration evidence, not a blind pin bump.

## Local UI path

For a usable local fak-native interface, keep three boundaries visible:

1. **Kernel:** stock fak execution, policy, context, and lifecycle behavior.
2. **Protocol:** `pkg/harnesskit` semantic events/inputs, cursor, resume, approvals, and redaction.
3. **Product UI:** locally served assets, branding, layout, and interaction state.

The first UI witness is: start locally, complete one offline run, render streamed text and tool/approval state from semantic events, reconnect by cursor, and capture browser renders. Until #6790 lands that witness, report the local web UI as `not yet`; the existing TUI is not evidence for a separately built UI.

## Current-interest research before promotion

When asked what people are discussing on X or elsewhere, use dated primary sources or a reproducible export/search. Inventory at least:

- requests for harness ownership versus IDE/plugin customization;
- local-first/private execution;
- model/provider routing and bring-your-own-model;
- skills, memory, tool/MCP, policy/approval, and multi-agent customization;
- custom UI, generative UI, replay/evals, observability, packaging, and upgrades.

For each theme record date, source URL/export, representative statement, engagement only if visible, and the product decision it could change. Separate **observed discussion** from **product inference**. Social popularity is discovery evidence, never proof of usability or value; #6809 owns the independent adoption benchmark.

## Promotional frame

Use witnessed language:

- **Fast path:** “Create a working fak-native harness in 10 minutes” only when init → one owned customization → rebuild → selfcheck is timed on a clean machine.
- **Deep path:** “Create your own harness this weekend” only when the named extension/UI path has a captured end-to-end result and maintenance boundary.
- **Always include:** audience, included outcome, prerequisites, measured environment/date, and the command to reproduce it.

A safe pre-witness formulation is: “Target: a rebuilt offline harness in ten minutes; custom providers, skills, and UI are the weekend track.”

## Completion receipt

Close with scannable bullets:

- verdict first: working / not yet;
- `For / Problem / Today / Better because / Witness`;
- external directory and pinned `module@version`;
- user-owned paths changed;
- exact build and selfcheck evidence;
- UI state: stock TUI or separately built protocol client;
- filed, deduplicated follow-ups for every remaining gap.

## References

- `docs/harness-init.md`
- `docs/harness-protocol.md`
- `docs/notes/HARNESS-CREATOR-SPINE-WITNESS-2026-08-15.md`
- `docs/spine-first-defaults.md`
- `docs/standards/net-true-value.md`
- epic #6777; local web UI #6790; skill packs #6796; gallery #6808; independent study #6809
