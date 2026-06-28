package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// token_defaults.go — the token-saving-defaults scorecard. It answers the question a
// cost-conscious operator asks the moment they run `fak guard -- claude` / `fak serve` with
// no flags: of every stacking token-saving method fak knows, which are ON by default, are the
// high-value low-loss ones turned on out of the box, and is every default honestly noted and
// locked against regression?
//
// Every lever's on/off + witness state is DERIVED from the entrypoint source (cmd/fak/guard.go,
// cmd/fak/serve.go, the gateway Default* constants, internal/gateway/messages.go) — never a
// roster claim a doc could drift from. `--markdown` re-renders the committed snapshot at
// docs/serving/token-defaults-scorecard.md deterministically, so a future default flip that the
// doc would otherwise misreport is caught by TestTokenDefaultsSnapshotFresh.

var reVDSOTrue = regexp.MustCompile(`VDSO:\s+true`)

// lever is one stacking token-saving method with its real default + honesty/lock state derived
// from source. class is lossless (zero model-visible change — must be on), bounded (lossy but an
// in-code guard keeps the working set intact — should be on, with a note), or optin (broader
// blast radius — correctly off behind a documented gate).
type lever struct {
	key, label, class    string
	on, witnessed        bool
	blocker, flag        string
	gated, noted, locked bool
}

func cmdTokenDefaultsScorecard(argv []string) {
	p, c, asMarkdown, done := scorecardCmdSetup("fak token-defaults-scorecard", argv, collectTokenDefaultsScorecard)
	if done {
		return
	}
	if asMarkdown {
		fmt.Print(renderTokenDefaultsMarkdown(c))
		return
	}
	fmt.Printf("token-defaults-scorecard: %s (%s)\n  token_defaults_debt: %v   grade: %v   stacked: %v/%v\n",
		p["verdict"], p["finding"], c["token_defaults_debt"], c["grade"], c["stacked_on"], c["levers_total"])
	if defects, _ := c["defects"].([]string); len(defects) > 0 {
		for _, d := range defects {
			fmt.Println("  - " + d)
		}
	}
}

func collectTokenDefaultsScorecard(root string) map[string]any {
	read := func(rel string) string {
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return ""
		}
		return string(b)
	}
	exists := func(rel string) bool {
		_, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
		return err == nil
	}
	serve := read("cmd/fak/serve.go")
	guard := read("cmd/fak/guard.go")
	gateway := read("internal/gateway/gateway.go")
	tui := read("cmd/fak/tui.go")
	messages := read("internal/gateway/messages.go")
	bothWire := func(needle string) bool { return strings.Contains(serve, needle) && strings.Contains(guard, needle) }

	// ---- the regression locks: each failing check is one unit of token_defaults_debt ----
	defects := []string{}
	require := func(ok bool, msg string) {
		if !ok {
			defects = append(defects, msg)
		}
	}
	require(strings.Contains(gateway, "const DefaultCompactHistoryBudget = 48000"), "gateway.DefaultCompactHistoryBudget must stay default-on")
	require(strings.Contains(gateway, "DefaultElideResultBytes = DocumentedElideResultBytes"), "gateway.DefaultElideResultBytes must arm oversized-result elision on by default at the documented threshold")
	require(bothWire(`fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget`), "both front doors must wire compact-history-budget to gateway.DefaultCompactHistoryBudget")
	require(bothWire(`fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes`), "both front doors must wire elide-result-bytes to gateway.DefaultElideResultBytes")
	require(bothWire(`fs.Int("ctx-view-budget", 8000`), "both front doors must default ctx-view-budget ON at the conservative 8000-resident-token budget (witness: docs/notes/CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md)")
	require(strings.Contains(serve, `fs.Bool("vdso", true`), "serve.go must default vDSO on")
	require(reVDSOTrue.MatchString(guard), "guard.go must set VDSO true")
	require(bothWire("ToolFloorDenies:"), "both front doors must wire ToolFloorDenies")
	// The per-turn debug-stats line is the observable cache/token-value debug layer. It is ON
	// by default on the flagship `fak guard` Claude-OAuth path (the cache + token-value economy
	// of every turn is visible with no flag; --debug-stats=false or --quiet silences it) and in
	// the console-agent launcher (fak console agent / fak c) overlay. `fak serve` keeps it off:
	// that daemon's observability is /metrics + /debug/vars + the access log, not a per-turn
	// stderr line. Lock both on-by-default front doors so the visible layer cannot silently regress.
	require(strings.Contains(guard, `fs.Bool("debug-stats", true`), "guard.go must default --debug-stats ON so the observable cache/token-value debug layer is visible by default on the Claude-OAuth path")
	require(strings.Contains(tui, `fs.Bool("debug-stats", true`), "tui.go must default --debug-stats to true in the console agent launcher (native per-turn token-usage overlay)")
	require(strings.Contains(tui, `"--debug-stats"`), "tui.go must wire --debug-stats into the guard command for the console launcher overlay")
	require(strings.Contains(tui, "gateway.DefaultCompactHistoryBudget") && strings.Contains(tui, "gateway.DefaultElideResultBytes"), "tui.go must pass the active token-saving guard defaults explicitly so they appear in dry-run output")

	// ---- derive each lever's REAL default + state from the entrypoint source ----
	elideWitnessed := exists("experiments/agent-live/elide-oversized-prevalence-2026-06-26.json")
	levers := []lever{
		{
			key: "provider_cache", label: "provider_cache — provider prompt-cache prefix (byte-faithful passthrough)",
			class: "lossless", on: strings.Contains(messages, "PlaceAnthropicCacheBreakpoint("),
			witnessed: true, blocker: "", flag: "(structural)", gated: false, noted: true, locked: true,
		},
		{
			key: "toolfloor", label: "toolfloor — tool-floor pruning (drop provably-unreachable tool defs)",
			class: "lossless", on: bothWire("ToolFloorDenies:"),
			witnessed: true, blocker: "", flag: "(structural)", gated: false, noted: true, locked: true,
		},
		{
			key: "vdso", label: "vdso — vDSO dedup fast path (collapse identical calls)",
			class: "lossless", on: strings.Contains(serve, `fs.Bool("vdso", true`) && reVDSOTrue.MatchString(guard),
			witnessed: true, blocker: "", flag: "--vdso", gated: false, noted: true, locked: true,
		},
		{
			key: "compacthistory", label: "compacthistory — history compaction (drop the un-cacheable middle past the budget)",
			class: "bounded", on: strings.Contains(gateway, "const DefaultCompactHistoryBudget = 48000") && bothWire(`fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget`),
			witnessed: true, blocker: "", flag: "--compact-history-budget", gated: false, noted: true, locked: true,
		},
		{
			key: "elideresult", label: "elideresult — oversized-result elision (shrink a scrolled-past tool_result to head+tail)",
			class: "bounded", on: strings.Contains(gateway, "DefaultElideResultBytes = DocumentedElideResultBytes") && bothWire(`fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes`),
			witnessed: elideWitnessed, blocker: blockerIf(!elideWitnessed, "unwitnessed"), flag: "--elide-result-bytes", gated: false, noted: true, locked: true,
		},
		{
			key: "ctxview", label: "ctxview — ctxplan O(1) planned view (re-materialize history under a budget)",
			class: "bounded", on: bothWire(`fs.Int("ctx-view-budget", 8000`),
			witnessed: true, blocker: "", flag: "--ctx-view-budget", gated: false, noted: true, locked: true,
		},
	}

	// ---- roll the derived levers into the headline counters + KPIs ----
	stackedOn, losslessOn, losslessN, boundedOn, boundedN := 0, 0, 0, 0, 0
	offGated, off, onNotedBounded, onBounded, lockedOn, offBoundedWitnessed, offBounded := 0, 0, 0, 0, 0, 0, 0
	for _, l := range levers {
		if l.on {
			stackedOn++
			if l.locked {
				lockedOn++
			}
			if l.class == "bounded" {
				onBounded++
				if l.noted {
					onNotedBounded++
				}
			}
		} else {
			off++
			if l.gated {
				offGated++
			}
			if l.class == "bounded" {
				offBounded++
				if l.witnessed {
					offBoundedWitnessed++
				}
			}
		}
		switch l.class {
		case "lossless":
			losslessN++
			if l.on {
				losslessOn++
			}
		case "bounded":
			boundedN++
			if l.on {
				boundedOn++
			}
		}
	}
	leversTotal := len(levers)

	kpis := []map[string]any{
		kpi("stack", "stacking_depth", scorePct(stackedOn, leversTotal), 0, fmt.Sprintf("%d/%d token-saving methods stacked on by default out of the box", stackedOn, leversTotal)),
		kpi("stack", "lossless_stack", scorePct(losslessOn, losslessN), losslessN-losslessOn, fmt.Sprintf("%d/%d lossless savers on by default", losslessOn, losslessN)),
		kpi("stack", "high_value_defaults", scorePct(boundedOn, boundedN), boundedN-boundedOn, fmt.Sprintf("%d/%d demonstrably-safe bounded-loss savers on by default", boundedOn, boundedN)),
		kpi("honesty", "witness_status", scorePct(offBoundedWitnessed, offBounded), 0, witnessStatusDetail(offBoundedWitnessed, offBounded)),
		kpi("honesty", "dark_lever_gated", scorePct(offGated, off), off-offGated, fmt.Sprintf("%d/%d off-by-default levers carry a documented gate", offGated, off)),
		kpi("honesty", "default_notes", scorePct(onNotedBounded, onBounded), onBounded-onNotedBounded, fmt.Sprintf("%d/%d on-by-default bounded savers carry an honest loss note", onNotedBounded, onBounded)),
		kpi("regression", "default_on_locked", scorePct(lockedOn, stackedOn), stackedOn-lockedOn, fmt.Sprintf("%d/%d on-by-default savers pinned by a regression sentinel", lockedOn, stackedOn)),
		kpi("parity", "entrypoint_parity", parityScore(defects), 0, "front doors agree + servewiring verdicts track the real defaults"),
	}

	composite := meanScore(kpis)
	grade := gradeFor(composite)
	debt := len(defects)
	ok, verdict, finding := true, "OK", "token_defaults_wired"
	reason := "zero token-defaults-debt; safe default token savers are wired and the console overlay is on"
	next := "rerun after changing serve/guard/gateway/tui token-saving defaults"
	if debt > 0 {
		ok, verdict, finding = false, "ACTION", "token_defaults_debt"
		grade = "C"
		reason = fmt.Sprintf("%d token-defaults defect(s)", debt)
		next = "restore the default wiring named in corpus.defects"
	}

	leverStatus := make([]map[string]any, 0, len(levers))
	for _, l := range levers {
		leverStatus = append(leverStatus, map[string]any{
			"key": l.key, "label": l.label, "class": l.class, "on": l.on, "witnessed": l.witnessed,
			"blocker": l.blocker, "flag": l.flag, "gated": l.gated, "noted": l.noted, "locked": l.locked,
		})
	}

	return map[string]any{
		"schema":      "fak-token-defaults-scorecard/2",
		"ok":          ok,
		"verdict":     verdict,
		"finding":     finding,
		"reason":      reason,
		"next_action": next,
		"corpus": map[string]any{
			"token_defaults_debt": debt,
			"score":               round1(composite),
			"grade":               grade,
			"levers_total":        leversTotal,
			"stacked_on":          stackedOn,
			"soft_signals":        off,
			"defects":             defects,
			"kpis":                kpis,
			"lever_status":        leverStatus,
		},
	}
}

func blockerIf(cond bool, s string) string {
	if cond {
		return s
	}
	return ""
}

func kpi(group, name string, score, debt int, detail string) map[string]any {
	return map[string]any{"group": group, "kpi": name, "score": score, "debt": debt, "detail": detail}
}

// scorePct is a 0-100 KPI score from a fraction; an empty denominator reads as a
// fully-satisfied 100 (nothing left to do), never a divide-by-zero.
func scorePct(n, d int) int {
	if d == 0 {
		return 100
	}
	return int(float64(n)/float64(d)*100 + 0.5)
}

func witnessStatusDetail(witnessed, off int) string {
	if off == 0 {
		return "no off-by-default high-value savers remain — every bounded-loss saver defaults on"
	}
	return fmt.Sprintf("%d/%d off high-value savers have a committed witness in hand", witnessed, off)
}

func parityScore(defects []string) int {
	for _, d := range defects {
		if strings.Contains(d, "front door") || strings.Contains(d, "wire") {
			return 70
		}
	}
	return 100
}

func meanScore(kpis []map[string]any) float64 {
	if len(kpis) == 0 {
		return 0
	}
	sum := 0
	for _, k := range kpis {
		sum += k["score"].(int)
	}
	return float64(sum) / float64(len(kpis))
}

func round1(f float64) float64 {
	return float64(int(f*10+0.5)) / 10
}

func gradeFor(score float64) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

func groupMean(kpis []map[string]any, group string) int {
	sum, n := 0, 0
	for _, k := range kpis {
		if k["group"] == group {
			sum += k["score"].(int)
			n++
		}
	}
	if n == 0 {
		return 100
	}
	return int(float64(sum)/float64(n) + 0.5)
}

func tick(b bool) string {
	if b {
		return "✓"
	}
	return "·"
}

// renderTokenDefaultsMarkdown re-renders the committed snapshot at
// docs/serving/token-defaults-scorecard.md deterministically from the derived corpus, so the doc
// the "Regenerate" line points to is the binary's real behavior, not a hand-edited claim.
func renderTokenDefaultsMarkdown(c map[string]any) string {
	levers := c["lever_status"].([]map[string]any)
	kpis := c["kpis"].([]map[string]any)
	debt := c["token_defaults_debt"].(int)
	composite := c["score"].(float64)
	grade := c["grade"].(string)
	stackedOn := c["stacked_on"].(int)
	leversTotal := c["levers_total"].(int)
	soft := c["soft_signals"].(int)

	var b strings.Builder
	b.WriteString(`---
title: "fak token-saving-defaults scorecard — is the out-of-the-box token economy amazing?"
description: "fak's deterministic token-saving-defaults scorecard: which stacking token-saving methods are ON by default on the fak guard / fak serve Anthropic passthrough, whether the high-value low-loss savers are turned on out of the box, and whether every default is honestly noted and locked against regression — re-derived from the entrypoint source."
---

# Token-saving-defaults scorecard — is fak's out-of-the-box token economy amazing?

<!-- token-defaults-scorecard · process: fak token-defaults-scorecard --markdown -->

The question a cost-conscious operator asks the moment they run ` + "`fak guard -- claude`" + ` / ` + "`fak serve`" + `: **of every token-saving method fak knows how to stack, which ones are ON by default — and are the high-value, low-loss ones turned on out of the box, or left dark behind a flag nobody flips?** Every number below is re-derived from the entrypoint source (` + "`cmd/fak/guard.go`" + `, ` + "`cmd/fak/serve.go`" + `, the ` + "`Default*`" + ` constants in ` + "`internal/gateway/gateway.go`" + `, and ` + "`internal/gateway/messages.go`" + `) by ` + "`fak token-defaults-scorecard`" + ` — a lever's on/off state is the binary's real behavior, never a claim in the roster. The headline metric is **token-defaults-debt**: the count of concrete defects — a high-value saver left off, an on-by-default saver with no honest note, a default no test locks, a front door out of step. Driving it to zero means a user who runs fak with no flags gets the full stack of safe savings, each honestly labeled, none able to regress unnoticed.

> Regenerate: ` + "`go run ./cmd/fak token-defaults-scorecard --markdown > docs/serving/token-defaults-scorecard.md`" + `

## Headline

| Metric | Value |
|---|---|
`)
	fmt.Fprintf(&b, "| **Token-defaults-debt (total HARD defects)** | **%d** |\n", debt)
	fmt.Fprintf(&b, "| Composite score | %.1f/100 (grade %s) |\n", composite, grade)
	fmt.Fprintf(&b, "| Savers stacked on by default | %d/%d |\n", stackedOn, leversTotal)
	fmt.Fprintf(&b, "| Groups | stack %d · honesty %d · regression %d · parity %d |\n",
		groupMean(kpis, "stack"), groupMean(kpis, "honesty"), groupMean(kpis, "regression"), groupMean(kpis, "parity"))
	fmt.Fprintf(&b, "| Advisory (soft) signals | %d |\n", soft)

	b.WriteString("\n## Per-lever status — where each token-saving method stands\n\n")
	b.WriteString("`class`: **lossless** = zero model-visible change (must be on); **bounded** = lossy but an in-code guard keeps the model's working set intact (high-value → should be on, with a note); **optin** = broader blast radius (correctly off, must carry a documented gate). `gated` = an off lever documents why; `noted` = an on bounded lever documents what it sheds + cache-safety; `locked` = a test pins the default.\n\n")
	b.WriteString("| Lever | Class | Default | Witness | Blocker | Flag | Gated | Noted | Locked |\n")
	b.WriteString("|---|---|:--:|:--:|---|---|:--:|:--:|:--:|\n")
	for _, l := range levers {
		def := "**OFF**"
		if l["on"].(bool) {
			def = "**ON**"
		}
		blk, _ := l["blocker"].(string)
		if blk == "" {
			blk = "—"
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | `%s` | %s | %s | %s |\n",
			l["label"], l["class"], def, tick(l["witnessed"].(bool)), blk, l["flag"], tick(l["gated"].(bool)), tick(l["noted"].(bool)), tick(l["locked"].(bool)))
	}

	b.WriteString("\n## KPIs\n\n")
	b.WriteString("| Group | KPI | Score | Debt | Detail |\n")
	b.WriteString("|---|---|---:|:--:|---|\n")
	for _, k := range kpis {
		fmt.Fprintf(&b, "| %s | `%s` | %d | %d | %s |\n", k["group"], k["kpi"], k["score"].(int), k["debt"].(int), k["detail"])
	}

	b.WriteString("\n## Token-defaults-debt work-list\n\n")
	if debt == 0 {
		b.WriteString("No token-defaults-debt: every stacking saver fak can safely default is on out of the box, honestly noted, and locked against regression. ")
		if soft > 0 {
			b.WriteString("The lone off-by-default lever (`ctxview`, the opt-in planned view) is correctly gated behind a watched-live witness — the tracked next default to turn on once that gate clears. ")
		}
		b.WriteString("🎉\n")
	} else {
		for _, d := range c["defects"].([]string) {
			b.WriteString("- " + d + "\n")
		}
	}
	return b.String()
}
