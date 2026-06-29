package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/loopindex"
)

// loopindex.go — the impure shell over internal/loopindex: the agentic-coding
// LOOP-INDEX scorecard (epic #1148, spine child #1152). It DERIVES each loop
// stage's witness probes from the TRACKED tree — file/verb/default presence, never
// a live runtime metric — and folds them into one loop-index + loopindex_debt via
// the pure loopindex.Score. Deterministic and re-runnable from a clean clone: the
// same tree scores the same number, so a peer reproduces it and a regression reds
// the gate.
//
// The loop is Orient → Plan → Act → Verify → Ship → Learn. Each stage's signal is
// witnessed by a small set of structural probes; a stage is WIRED when its keystone
// probes pass (the mechanism exists AND it is the default), and in DEBT when it is
// unwired or below its floor. This is the SPINE: every other dev-ex child (#1153 the
// tool map, #1154 priced fan-out, #1155 the green-gate budget, #1157 false-done
// refused, #1158 recall re-verify, #1161 the session→outcome loop) flips one of these
// probes, and its before/after delta is the move in this number.
//
//	fak loop-index-scorecard            human headline + per-stage work-list
//	fak loop-index-scorecard --json     the control-pane Report (corpus.loopindex_debt)
//	fak loop-index-scorecard --markdown regenerate docs/fak/loop-index-scorecard.md

const loopIndexSnapshotRel = "docs/fak/loop-index-scorecard.md"

func cmdLoopIndexScorecard(argv []string) {
	fs := flag.NewFlagSet("fak loop-index-scorecard", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "emit the machine-readable control-pane Report JSON")
	asMarkdown := fs.Bool("markdown", false, "regenerate the committed docs/fak/loop-index-scorecard.md snapshot")
	fs.SetOutput(io.Discard)
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	rep := loopindex.Score(collectLoopIndex(repoRoot()))
	switch {
	case *asJSON:
		_ = writeIndentedJSONNoEscape(os.Stdout, rep)
	case *asMarkdown:
		fmt.Print(renderLoopIndexMarkdown(rep))
	default:
		loopindex.Render(os.Stdout, rep)
	}
}

// liProbe is a small builder so each stage reads as a table of (name, keystone,
// pass) decisions against the tracked tree.
func liProbe(name, detail string, keystone, pass bool) loopindex.Probe {
	return loopindex.Probe{Name: name, Detail: detail, Keystone: keystone, Pass: pass}
}

// collectLoopIndex derives the six canonical loop stages from the tracked tree. Each
// probe is a concrete, deterministic check (a file/dir exists, a default flag is
// wired in both front doors, a not-yet-default marker is present). The "keystone +
// default" pairs encode the epic's honest snapshot: a mechanism may exist while its
// DEFAULT wiring does not, which leaves the stage witnessable-but-not-load-bearing —
// unwired, in debt, with a clear flip target for its child issue.
func collectLoopIndex(root string) loopindex.Loop {
	exists := func(rel string) bool {
		_, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
		return err == nil
	}
	read := func(rel string) string {
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return ""
		}
		return string(b)
	}
	has := func(rel, needle string) bool { return strings.Contains(read(rel), needle) }
	hasFold := func(rel string, needles ...string) bool {
		body := strings.ToLower(read(rel))
		for _, n := range needles {
			if strings.Contains(body, strings.ToLower(n)) {
				return true
			}
		}
		return false
	}
	hasAllFold := func(rel string, needles ...string) bool {
		body := strings.ToLower(read(rel))
		for _, n := range needles {
			if !strings.Contains(body, strings.ToLower(n)) {
				return false
			}
		}
		return true
	}
	bothDoors := func(needle string) bool { return has("cmd/fak/serve.go", needle) && has("cmd/fak/guard.go", needle) }
	mainCase := func(verb string) bool { return has("cmd/fak/main.go", `case "`+verb+`"`) }

	// --- orient: recalled memory re-checked before trust; context ref-counted/GC'd ---
	orient := loopindex.Stage{
		Name: loopindex.StageOrient, Signal: "recall-staleness / context-thrash", Floor: 0.6,
		Probes: []loopindex.Probe{
			liProbe("recall_mechanism", "the `fak recall` verb + internal/recall exist (the session-recall image gate)",
				true, mainCase("recall") && exists("internal/recall")),
			liProbe("recall_reverify_default", "recalled memory is re-verified at read time by DEFAULT before it is trusted (#1158)",
				true, hasFold("cmd/fak/recall.go", "read-time", "reverify", "re-verify")),
			liProbe("context_gc", "context is ref-counted / GC'd (internal/contextq + internal/ctxmmu)",
				false, exists("internal/contextq") && exists("internal/ctxmmu")),
			liProbe("ctxview_budget_default", "the ctxplan planned view is default-on in both front doors (one-read orient under a budget)",
				false, bothDoors(`fs.Int("ctx-view-budget", 8000`)),
		},
	}

	// --- plan: N agents launch only once proven they cannot collide (priced fan-out) ---
	plan := loopindex.Stage{
		Name: loopindex.StagePlan, Signal: "collision-priced fan-out coverage", Floor: 0.6,
		Probes: []loopindex.Probe{
			liProbe("arbitrate_substrate", "the collision substrate exists (internal/dispatchorder + the dos.toml per-leaf lane taxonomy)",
				true, exists("internal/dispatchorder") && has("dos.toml", "[lanes]")),
			liProbe("priced_fanout_default", "collision-priced arbitrate runs as the DEFAULT before any multi-agent dispatch (#1154)",
				true, hasFold("internal/dispatchorder/dispatchorder.go", "arbitrate", "collision-priced")),
			liProbe("dispatch_worker", "a dispatch worker exists to fan out across disjoint lanes (cmd/dispatchworker)",
				false, exists("cmd/dispatchworker")),
			liProbe("lane_trees_declared", "leaf trees are declared so the arbiter can prove disjointness ([lanes.trees])",
				false, has("dos.toml", "[lanes.trees]")),
		},
	}

	// --- act: a malformed tool call is repaired-or-refused, never a lost turn ---
	act := loopindex.Stage{
		Name: loopindex.StageAct, Signal: "malformed-call repair rate", Floor: 0.6,
		Probes: []loopindex.Probe{
			liProbe("toolfloor_default", "tool-floor pruning is default-on in both front doors (drop provably-unreachable tool defs)",
				true, bothDoors("ToolFloorDenies:")),
			liProbe("repair_mechanism", "a tool-call grammar/repair mechanism exists (internal/grammar)",
				true, exists("internal/grammar")),
			liProbe("vdso_dedup_default", "vDSO call-dedup is default-on (an identical call never costs a turn)",
				false, has("cmd/fak/serve.go", `fs.Bool("vdso", true`)),
			liProbe("warm_next_turn", "the next turn is warmed during tool latency by DEFAULT (#809)",
				false, bothDoors("vcachewarm") || bothDoors("warm-next-turn")),
		},
	}

	// --- verify: a false "done" refused at the STOP seam, not caught in review ---
	verify := loopindex.Stage{
		Name: loopindex.StageVerify, Signal: "unwitnessed-done rate", Floor: 0.6,
		Probes: []loopindex.Probe{
			liProbe("stop_seam_mechanism", "the STOP gate exists (internal/loopgate) and the STOP-failure marker planner/settler exists (internal/stopfailure)",
				true, exists("internal/loopgate") && exists("internal/stopfailure")),
			liProbe("false_done_refused_default", "a false \"done\" is refused at the STOP seam by DEFAULT via commit-audit / unwitnessed-done detection (#1157)",
				true,
				hasAllFold("internal/loopgate/loopgate.go", "ReasonDoneUnwitnessed", "CriterionCommitAudit", "OutcomeNotYet") &&
					hasAllFold("cmd/fak/loop_drive.go", "StatusWitnessRefused", "turn landed no new commit", "ReasonDoneUnwitnessed") &&
					hasAllFold("tools/githooks/pre-push", "FLEET_REVIEW_GUARD:-block", "dos review", "RESIDUAL")),
			liProbe("commit_preview", "`fak commit --preview` cross-checks the claim before it lands (lane/stamp pre-check)",
				false, has("cmd/fak/commit.go", "preview")),
			liProbe("review_residual_surface", "`dos review` is the default ship-review surface: residual first, cleared commits cost near-zero attention",
				false, hasAllFold("tools/githooks/pre-push", "dos review", "RESIDUAL", "CLEARED", "attention")),
			liProbe("stop_settle_verb", "the STOP-failure markers are settleable from the CLI (cmd/fak/stopfailure.go)",
				false, exists("cmd/fak/stopfailure.go")),
		},
	}

	// --- ship: the guard is a verb; the green gate is fast, incremental, a budget ---
	ship := loopindex.Stage{
		Name: loopindex.StageShip, Signal: "green-gate latency budget", Floor: 0.6,
		Probes: []loopindex.Probe{
			liProbe("commit_verb", "`fak commit` mechanizes commit-by-path + the green gate (cmd/fak/commit.go)",
				true, mainCase("commit") && exists("cmd/fak/commit.go")),
			liProbe("green_gate_budget", "the green gate is a TRACKED, fast & incremental latency budget, not a velocity tax (#1155)",
				true, hasFold("cmd/fak/commit.go", "green-gate", "latency budget", "gate budget") || exists("docs/fak/green-gate-budget.md")),
			liProbe("commit_preview_lane", "`fak commit --preview` resolves lane + ship-stamp before commit",
				false, has("cmd/fak/commit.go", "preview")),
			liProbe("trunk_guard", "trunk discipline is mechanized as a structured refusal (OFF_TRUNK)",
				false, has("dos.toml", "OFF_TRUNK")),
		},
	}

	// --- learn: every session's cost AND outcome captured + consumed by a loop ---
	learn := loopindex.Stage{
		Name: loopindex.StageLearn, Signal: "session→outcome link coverage", Floor: 0.6,
		Probes: []loopindex.Probe{
			liProbe("sessionobs_scorer", "the session→outcome scorer exists (internal/sessionobs grades the capture→link→learn ladder)",
				true, exists("internal/sessionobs")),
			liProbe("consuming_loop", "a registered RSI loop READS the committed corpus (internal/rsiloop + the gardenbundle sessions_learn member)",
				true, exists("internal/rsiloop/sessionobs.go") && has("internal/gardenbundle/gardenbundle.go", "sessions_learn")),
			liProbe("sessions_verb", "`fak sessions` ingests + scores this host's transcripts (cmd/fak/sessions.go)",
				false, mainCase("sessions") && exists("cmd/fak/sessions.go")),
			liProbe("outcome_link", "the value-vs-waste outcome link is classified (sessionobs.ClassifyOutcome)",
				false, has("internal/sessionobs/sessionobs.go", "ClassifyOutcome")),
		},
	}

	return loopindex.Loop{Stages: []loopindex.Stage{orient, plan, act, verify, ship, learn}}
}

// renderLoopIndexMarkdown re-renders the committed snapshot at
// docs/fak/loop-index-scorecard.md deterministically from the scored Report, so the
// doc is the binary's real behavior — and the per-stage ledger every dev-ex child
// updates when it moves the index. TestLoopIndexSnapshotFresh enforces byte-equality.
func renderLoopIndexMarkdown(rep loopindex.Report) string {
	c := rep.Corpus
	var b strings.Builder
	b.WriteString(`---
title: "fak loop-index scorecard — one witnessed number for the agentic-coding loop"
description: "fak's deterministic loop-index scorecard: it folds the six stages of the agentic-coding loop (orient → plan → act → verify → ship → learn) into one witnessed loop-index + a loopindex_debt count, derived from the tracked tree so a peer reproduces the number and a regression reds the gate. The SPINE of the 10x dev-experience epic (#1148): every lever reports its before/after delta here."
---

# Loop-index scorecard — is the agentic-coding loop getting 10x?

<!-- loop-index-scorecard · process: fak loop-index-scorecard --markdown -->

The agentic-coding loop is **Orient → Plan → Act → Verify → Ship → Learn**. "The loop got faster / more self-correcting" is a vibe until it is a number you can move and prove you moved. This scorecard folds each stage's **witnessed** signal into one **loop-index** (0–100) plus **loopindex_debt** (the count of stages not yet witnessed at their floor) — every value re-derived from the tracked tree by ` + "`fak loop-index-scorecard`" + `, so a clean clone reproduces it. It is the **spine** of the 10x dev-experience epic ([#1148](../../)): every other lever (the tool map, priced fan-out, the green-gate budget, false-done-refused, recall re-verify, the session→outcome loop) flips one of the probes below, and its before/after delta is the move in this number.

> Regenerate: ` + "`go run ./cmd/fak loop-index-scorecard --markdown > docs/fak/loop-index-scorecard.md`" + `

## Headline

| Metric | Value |
|---|---|
`)
	fmt.Fprintf(&b, "| **loopindex_debt (stages not yet witnessed at floor)** | **%d** |\n", c.LoopIndexDebt)
	fmt.Fprintf(&b, "| Loop-index (over all 6 stages; unwired = 0) | %d/100 (grade %s) |\n", c.LoopIndex, c.Grade)
	fmt.Fprintf(&b, "| Witnessed-index (health of the wired stages only) | %d/100 |\n", c.WitnessedIndex)
	fmt.Fprintf(&b, "| Stages wired (load-bearing witness) | %d/%d |\n", c.WiredStages, c.Stages)

	b.WriteString("\n## Per-stage rungs — where each loop stage stands\n\n")
	b.WriteString("A stage is **wired** when its keystone probes pass (the mechanism exists AND it is the default); its **health** is the fraction of its probes that pass. A stage is in **debt** when it is unwired, or wired but below its floor.\n\n")
	b.WriteString("| Stage | Signal | Wired | Score | Floor | Status |\n")
	b.WriteString("|---|---|:--:|---:|---:|---|\n")
	for i, k := range rep.KPIs {
		st := rep.StageDetail[i]
		status := "ok"
		if k.Debt > 0 {
			status = "**DEBT**"
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %d | %.0f%% | %s |\n",
			k.Name, k.Signal, tickWire(k.Wired), k.Score, 100*st.Floor, status)
	}

	b.WriteString("\n## Witness probes — the flip targets for each child\n\n")
	for i, st := range rep.StageDetail {
		fmt.Fprintf(&b, "### %s — %s\n\n", st.Name, st.Signal)
		_ = i
		b.WriteString("| Probe | Keystone | Pass | What it witnesses |\n|---|:--:|:--:|---|\n")
		for _, p := range st.Probes {
			fmt.Fprintf(&b, "| `%s` | %s | %s | %s |\n", p.Name, tickBool(p.Keystone), tickBool(p.Pass), p.Detail)
		}
		b.WriteString("\n")
	}

	b.WriteString("## What moves it\n\n")
	if c.LoopIndexDebt == 0 {
		b.WriteString("No loopindex_debt: every loop stage is witnessed at its floor. Raise a floor or deepen a stage's probes to push toward 10x. 🎉\n")
	} else {
		b.WriteString("Each **DEBT** stage above names the keystone witness still missing. The child issue that wires it (and regenerates this snapshot) is how the before/after delta lands:\n\n")
		for i, k := range rep.KPIs {
			if k.Debt == 0 {
				continue
			}
			fmt.Fprintf(&b, "- **%s** — %s\n", k.Name, missingKeystones(rep.StageDetail[i]))
		}
	}
	return b.String()
}

func tickWire(b bool) string {
	if b {
		return "wired"
	}
	return "vibe"
}

func tickBool(b bool) string {
	if b {
		return "✓"
	}
	return "·"
}

// missingKeystones names the unbuilt keystone probes of a stage — the concrete flip
// target(s) for the child that owns it.
func missingKeystones(st loopindex.Stage) string {
	var missing []string
	for _, p := range st.Probes {
		if p.Keystone && !p.Pass {
			missing = append(missing, "`"+p.Name+"`")
		}
	}
	if len(missing) == 0 {
		return "deepen its supporting probes to reach the floor"
	}
	return "build " + strings.Join(missing, ", ")
}
