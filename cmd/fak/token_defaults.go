package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/headroom"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
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
	selected             string
	variants             []string
	provenance           map[string]string
}

type tokenDefaultSources struct {
	root                       string
	serve, guard, gateway, tui string
	messages, agentSeam        string
	overrides                  map[string]string
}

func loadTokenDefaultSources(root string) tokenDefaultSources {
	s := tokenDefaultSources{root: root}
	s.serve = s.read("cmd/fak/serve.go")
	s.guard = s.read("cmd/fak/guard.go")
	// The gateway's configuration and server-state declarations live in cohesive
	// package modules. Inspect the complete split source surface so moving a default
	// out of gateway.go cannot make the scorecard mistake an enabled lever for OFF.
	s.gateway = strings.Join([]string{
		s.read("internal/gateway/gateway.go"),
		s.read("internal/gateway/config.go"),
		s.read("internal/gateway/server_state.go"),
	}, "\n")
	// The console agent launcher was split from the TUI coordinator in #10049.
	// Inspect both concern-aligned files so a source move cannot be mistaken for
	// disabled token-saving behavior.
	s.tui = strings.Join([]string{
		s.read("cmd/fak/tui.go"),
		s.read("cmd/fak/tui_agent.go"),
	}, "\n")
	s.messages = s.read("internal/gateway/messages.go")
	s.agentSeam = s.read("internal/agent/ctxplan_seam.go")
	return s
}

func (s tokenDefaultSources) read(rel string) string {
	if text, ok := s.overrides[rel]; ok {
		return text
	}
	b, err := os.ReadFile(filepath.Join(s.root, filepath.FromSlash(rel)))
	if err != nil {
		return ""
	}
	return string(b)
}

func (s tokenDefaultSources) withOverride(rel, text string) tokenDefaultSources {
	copy := make(map[string]string, len(s.overrides)+1)
	for path, contents := range s.overrides {
		copy[path] = contents
	}
	copy[rel] = text
	s.overrides = copy
	return s
}

func (s tokenDefaultSources) exists(rel string) bool {
	_, err := os.Stat(filepath.Join(s.root, filepath.FromSlash(rel)))
	return err == nil
}

func (s tokenDefaultSources) bothWire(needle string) bool {
	return strings.Contains(s.serve, needle) && strings.Contains(s.guard, needle)
}

func cmdTokenDefaultsScorecard(argv []string) {
	os.Exit(runTokenDefaultsScorecard(os.Stdout, os.Stderr, argv))
}

func runTokenDefaultsScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak token-defaults-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable scorecard JSON")
	asMarkdown := fs.Bool("markdown", false, "emit markdown")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	effectiveness := fs.Bool("effectiveness", false, "audit effectiveness-witness coverage for every default saver")
	if !parseFlags(fs, argv) {
		return 2
	}
	p := collectTokenDefaultsScorecard(repoRoot())
	c := p["corpus"].(map[string]any)

	if *effectiveness {
		return writeTokenEffectivenessReport(stdout, stderr, p, *asJSON)
	}
	if code, done := emitScorecardComparison(stdout, stderr, "fak token-defaults-scorecard", *comparePath, okExit(p["ok"].(bool)), func(base map[string]any) string {
		return compareTokenDefaults(c, base)
	}); done {
		return code
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, p); err != nil {
			fmt.Fprintf(stderr, "fak token-defaults-scorecard: encode json: %v\n", err)
			return 1
		}
		return okExit(p["ok"].(bool))
	}
	if *asMarkdown {
		fmt.Fprint(stdout, renderTokenDefaultsMarkdown(c))
		return okExit(p["ok"].(bool))
	}
	fmt.Fprintf(stdout, "token-defaults-scorecard: %s (%s)\n  token_defaults_debt: %v   grade: %v   stacked: %v/%v\n",
		p["verdict"], p["finding"], c["token_defaults_debt"], c["grade"], c["stacked_on"], c["levers_total"])
	if defects, _ := c["defects"].([]string); len(defects) > 0 {
		for _, d := range defects {
			fmt.Fprintln(stdout, "  - "+d)
		}
	}
	return okExit(p["ok"].(bool))
}

// compareTokenDefaults prints the token-defaults-debt delta against a prior --json
// payload, mirroring the shared scorecard-family --compare convention (see
// pkg/scorecard.Compare / internal/conceptusage.Compare / internal/dogfoodscore.Compare):
// a compact "debt P -> C" line plus the composite/grade movement. This card's corpus is a
// plain map (not a scorecard.Payload), so the delta is computed directly off the corpus
// maps rather than via pkg/scorecard.Compare.
func compareTokenDefaults(current, baseline map[string]any) string {
	bc, _ := baseline["corpus"].(map[string]any)
	if bc == nil {
		bc = baseline
	}
	bDebt := anyIntTokenDefaults(bc["token_defaults_debt"])
	cDebt := anyIntTokenDefaults(current["token_defaults_debt"])
	delta := cDebt - bDebt
	verdict := "flat"
	if delta < 0 {
		verdict = "improved"
	} else if delta > 0 {
		verdict = "regressed"
	}
	abs := delta
	if abs < 0 {
		abs = -abs
	}
	lines := []string{
		fmt.Sprintf("token-defaults-scorecard: %v (score %v, token_defaults_debt %v)", current["grade"], current["score"], cDebt),
		fmt.Sprintf("  compare: token_defaults_debt %d -> %d (%s by %d)", bDebt, cDebt, verdict, abs),
		fmt.Sprintf("  composite: %v -> %v  grade %v -> %v", bc["score"], current["score"], bc["grade"], current["grade"]),
	}
	return strings.Join(lines, "\n")
}

// anyIntTokenDefaults coerces a JSON-decoded debt count (int from a fresh build, float64
// from a --compare baseline decoded via encoding/json) to int.
func anyIntTokenDefaults(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	default:
		return 0
	}
}

func collectTokenDefaultsScorecard(root string) map[string]any {
	return collectTokenDefaultsScorecardWithInputs(root, loadTokenDefaultSources(root), headroom.Names(), headroom.Selected().Name())
}

var headroomVariantEvidence = map[string]string{
	"distill": "internal/headroom/distill.go", "headroom": "internal/headroom/bridge.go",
	"lingua": "internal/headroom/lingua.go", "native": "internal/headroom/native.go",
	"none": "internal/headroom/compare.go", "noop": "internal/headroom/noop.go",
}

func checkTokenDefaultsRegressions(sources tokenDefaultSources, registered []string) ([]string, int, []string) {
	serve, guard, gateway, tui := sources.serve, sources.guard, sources.gateway, sources.tui
	agentSeam := sources.agentSeam
	bothWire := sources.bothWire

	var defects []string
	require := func(ok bool, msg string) {
		if !ok {
			defects = append(defects, msg)
		}
	}
	require(strings.Contains(gateway, "const DefaultCompactHistoryBudget = 48000"), "gateway.DefaultCompactHistoryBudget must stay default-on")
	require(strings.Contains(gateway, "DefaultElideResultBytes = DocumentedElideResultBytes"), "gateway.DefaultElideResultBytes must arm oversized-result elision on by default at the documented threshold")
	require(bothWire(`fs.Int("compact-history-budget", gateway.DefaultCompactHistoryBudget`), "both front doors must wire compact-history-budget to gateway.DefaultCompactHistoryBudget")
	require(bothWire(`fs.Int("elide-result-bytes", gateway.DefaultElideResultBytes`), "both front doors must wire elide-result-bytes to gateway.DefaultElideResultBytes")
	require(strings.Contains(gateway, "const DefaultElideStaleReads = true"), "gateway.DefaultElideStaleReads must arm read-lifecycle STALE elision on by default (the restorable sibling of oversized elision)")
	require(bothWire(`fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads`), "both front doors must wire elide-stale-reads to gateway.DefaultElideStaleReads")
	require(strings.Contains(agentSeam, "const DefaultCtxViewBudget = 8000"), "agent.DefaultCtxViewBudget must stay default-on at the witnessed 8000-resident-token budget")
	require(bothWire(`fs.Int("ctx-view-budget", agent.DefaultCtxViewBudget`), "both front doors must default ctx-view-budget ON by wiring it to agent.DefaultCtxViewBudget (the witnessed 8000-resident-token budget: docs/notes/CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md)")
	require(strings.Contains(serve, `fs.Bool("vdso", true`), "serve.go must default vDSO on")
	require(reVDSOTrue.MatchString(guard), "guard.go must set VDSO true")
	require(bothWire("ToolFloorDenies:"), "both front doors must wire ToolFloorDenies")
	require(strings.Contains(guard, `fs.Bool("debug-stats", true`), "guard.go must default --debug-stats ON so the observable cache/token-value debug layer is visible by default on the Claude-OAuth path")
	require(strings.Contains(tui, `fs.Bool("debug-stats", true`), "console agent source must default --debug-stats to true (native per-turn token-usage overlay)")
	require(strings.Contains(tui, `"--debug-stats"`), "console agent source must wire --debug-stats into the guard command for the launcher overlay")
	require(strings.Contains(tui, "gateway.DefaultCompactHistoryBudget") && strings.Contains(tui, "gateway.DefaultElideResultBytes"), "console agent source must pass the active token-saving guard defaults explicitly so they appear in dry-run output")

	rawWindowTargetDebt := 0
	for _, e := range ctxplan.DefaultEnvelopes() {
		if e.RawWindowTarget() {
			rawWindowTargetDebt++
		}
	}
	require(rawWindowTargetDebt == 0, fmt.Sprintf("%d default context envelope(s) derive the resident target from the raw provider window with no same-task witness (see docs/long-context-defaults.md)", rawWindowTargetDebt))

	var unboundVariants []string
	for _, name := range registered {
		path, ok := headroomVariantEvidence[name]
		if !ok || !sources.registeredCompressorProof(path, name) {
			unboundVariants = append(unboundVariants, name)
			require(false, fmt.Sprintf("registered headroom compressor %q has no source-bound token-defaults evidence binding", name))
		}
	}
	return defects, rawWindowTargetDebt, unboundVariants
}

func buildTokenDefaultsLevers(sources tokenDefaultSources, registered []string, selected string) []lever {
	serve, guard, gateway := sources.serve, sources.guard, sources.gateway
	messages, agentSeam := sources.messages, sources.agentSeam
	read, exists, bothWire := sources.read, sources.exists, sources.bothWire

	elideWitnessed := exists("experiments/agent-live/elide-oversized-prevalence-2026-06-26.json")
	elideStaleWitnessed := exists("experiments/agent-live/elide-stale-read-prevalence-2026-07-09.json")
	levers := []lever{
		{
			key: "provider_cache", label: "provider_cache — provider prompt-cache prefix (byte-faithful passthrough)",
			class: "lossless", on: strings.Contains(messages, "PlaceAnthropicCacheBreakpoint"),
			witnessed: true, blocker: "", flag: "(structural)", gated: false, noted: true, locked: true,
		},
		{
			key: "toolfloor", label: "toolfloor — tool-floor pruning (drop provably-unreachable tool defs)",
			class: "lossless", on: bothWire("ToolFloorDenies:"),
			witnessed: true, blocker: "", flag: "(structural)", gated: false, noted: true, locked: true,
		},
		{
			key: "mcptoolfilter", label: "mcptoolfilter — native MCP tools/list cold-schema filtering",
			class: "bounded", on: strings.Contains(read("internal/gateway/mcp_defer.go"), "Native filtering is default-on"),
			witnessed: exists("internal/gateway/mcp_filter_ab_test.go"), blocker: blockerIf(!exists("internal/gateway/mcp_filter_ab_test.go"), "unwitnessed"), flag: "FAK_ABLATE_MCP_TOOL_FILTER=1", gated: false, noted: true, locked: true,
		},
		{
			key: "defercoldtools", label: "defercoldtools — outbound Anthropic cold-tool schema deferral",
			class: "bounded", on: strings.Contains(gateway, "const DefaultDeferColdTools = true") && bothWire(`fs.Bool("defer-cold-tools", gateway.DefaultDeferColdTools`),
			witnessed: exists("internal/gateway/tooldefer_default_on_test.go"), blocker: blockerIf(!exists("internal/gateway/tooldefer_default_on_test.go"), "unwitnessed"), flag: "--defer-cold-tools", gated: false, noted: true, locked: true,
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
			key: "elidestale", label: "elidestale — stale-read elision (replace a Read superseded by a later same-file edit with a restorable marker)",
			class: "bounded", on: strings.Contains(gateway, "const DefaultElideStaleReads = true") && bothWire(`fs.Bool("elide-stale-reads", gateway.DefaultElideStaleReads`),
			witnessed: elideStaleWitnessed, blocker: blockerIf(!elideStaleWitnessed, "unwitnessed"), flag: "--elide-stale-reads", gated: false, noted: true, locked: true,
		},
		{
			key: "ctxview", label: "ctxview — ctxplan O(1) planned view (re-materialize history under a budget)",
			class: "bounded", on: strings.Contains(agentSeam, "const DefaultCtxViewBudget = 8000") && bothWire(`fs.Int("ctx-view-budget", agent.DefaultCtxViewBudget`),
			witnessed: true, blocker: "", flag: "--ctx-view-budget", gated: false, noted: true, locked: true,
		},
		{
			key: "headroomcompressor", label: "headroomcompressor - registered result-compressor family",
			class: "optin", on: selected != "" && selected != "noop" && selected != "none",
			blocker: "matched-quality live-wire witness missing", flag: "--compress / FAK_COMPRESSOR", gated: true,
			selected: selected, variants: append([]string(nil), registered...),
		},
	}

	for i := range levers {
		l := &levers[i]
		binding, mapped := tokenProofCatalog[l.key]
		l.provenance = map[string]string{}
		if !mapped {
			l.witnessed, l.locked, l.noted = false, false, false
			l.blocker = "no typed evidence binding"
			continue
		}
		l.witnessed = sources.executableProof(binding.Effect)
		l.locked = sources.executableProof(binding.Lock)
		l.noted = l.class == "lossless" || sources.literalProof(binding.Note)
		if binding.Effect.Path != "" {
			l.provenance["witness"] = binding.Effect.Path + "#" + binding.Effect.Function
		}
		if binding.Lock.Path != "" {
			l.provenance["lock"] = binding.Lock.Path + "#" + binding.Lock.Function
		}
		if binding.Note.Path != "" {
			l.provenance["note"] = binding.Note.Path + "#" + binding.Note.Function
		}
		if gateBlocker := tokenGateBlocker(sources, binding.Gate); gateBlocker != "" {
			l.witnessed = false
			l.blocker = gateBlocker
			l.provenance["gate"] = binding.Gate.Schema + ":" + binding.Gate.ID
			l.provenance["pass"] = binding.Gate.PassArtifact
		} else if l.key != "headroomcompressor" {
			l.blocker = blockerIf(!l.witnessed, "executable effect/control sentinel missing or vacuous")
		}
	}
	return levers
}

func collectTokenDefaultsScorecardWithInputs(root string, sources tokenDefaultSources, registered []string, selected string) map[string]any {
	defects, rawWindowTargetDebt, unboundVariants := checkTokenDefaultsRegressions(sources, registered)
	require := func(ok bool, msg string) {
		if !ok {
			defects = append(defects, msg)
		}
	}
	envelopes := ctxplanEnvelopeRows()
	levers := buildTokenDefaultsLevers(sources, registered, selected)

	for _, l := range levers {
		if l.on && !l.locked {
			require(false, l.key+": default-on state has no executable regression sentinel")
		}
		if l.on && l.class == "bounded" && !l.noted {
			require(false, l.key+": default-on bounded saver has no executable operator loss note")
		}
		if l.on && (!l.witnessed || l.blocker != "") {
			require(false, l.key+": default-on saver is not witnessed-safe: "+l.blocker)
		}
		if !l.on && !l.gated {
			require(false, l.key+": default-off saver has no explicit gate")
		}
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
			"selected": l.selected, "registered_variants": l.variants, "provenance": l.provenance,
		})
	}

	return map[string]any{
		"schema":      "fak-token-defaults-scorecard/3",
		"ok":          ok,
		"verdict":     verdict,
		"finding":     finding,
		"reason":      reason,
		"next_action": next,
		"corpus": map[string]any{
			"detector_schema":           "fak-token-defaults-scorecard/3",
			"token_defaults_debt":       debt,
			"score":                     round1(composite),
			"grade":                     grade,
			"levers_total":              leversTotal,
			"stacked_on":                stackedOn,
			"soft_signals":              off,
			"defects":                   defects,
			"kpis":                      kpis,
			"lever_status":              leverStatus,
			"context_envelope":          envelopes,
			"raw_window_target_debt":    rawWindowTargetDebt,
			"unbound_registered_savers": unboundVariants,
		},
	}
}

// ctxplanEnvelopeRows renders the internal/ctxplan default effective-context envelopes into the
// scorecard JSON: each row exposes the four quantities the long-context doctrine keeps distinct
// (hard cap, min viable evidence floor, target resident budget, effective ceiling) plus the derived
// safe cap, the provenance label (WITNESSED / OBSERVED / MODELED / FALLBACK), and whether the row
// is a raw-window-target defect. A cost-conscious operator reading `--json` can now see that the
// no-flag default target is held below the provider's advertised window and honestly labeled.
func ctxplanEnvelopeRows() []map[string]any {
	rows := make([]map[string]any, 0)
	for _, e := range ctxplan.DefaultEnvelopes() {
		rows = append(rows, map[string]any{
			"model_pattern":              e.ModelPattern,
			"task_class":                 e.TaskClass,
			"hard_context_cap":           e.HardContextCap,
			"output_reserve":             e.OutputReserve,
			"min_viable_evidence_tokens": e.MinViableEvidenceTokens,
			"target_resident_tokens":     e.TargetResidentTokens,
			"max_effective_tokens":       e.MaxEffectiveTokens,
			"safe_cap":                   e.SafeCap(),
			"provenance":                 e.Provenance,
			"witness":                    e.Witness,
			"raw_window_target":          e.RawWindowTarget(),
		})
	}
	return rows
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
	return scorecard.GradeStd(score)
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

The question a cost-conscious operator asks the moment they run ` + "`fak guard -- claude`" + ` / ` + "`fak serve`" + `: **of every default-stack token-saving method fak knows how to stack, which ones are ON by default — and are the high-value, low-loss ones turned on out of the box, or left dark behind a flag nobody flips?** Every number below is re-derived from the entrypoint source (` + "`cmd/fak/guard.go`" + `, ` + "`cmd/fak/serve.go`" + `, the ` + "`Default*`" + ` constants in ` + "`internal/gateway/gateway.go`" + `, and ` + "`internal/gateway/messages.go`" + `) by ` + "`fak token-defaults-scorecard`" + ` — a lever's on/off state is the binary's real behavior, never a claim in the roster. The headline metric is **token-defaults-debt**: the count of concrete defects — a high-value saver left off, an on-by-default saver with no honest note, a default no test locks, a front door out of step. Driving it to zero means a user who runs fak with no flags gets the full stack of safe savings, each honestly labeled, none able to regress unnoticed.

Budget size is governed separately by the [long-context defaults doctrine](../long-context-defaults.md): the advertised context window is a hard cap, not a target, and resident budgets should be labeled as witnessed, observed, modeled, or fallback.

> Regenerate: ` + "`go run ./cmd/fak token-defaults-scorecard --markdown > docs/serving/token-defaults-scorecard.md`" + `

## Headline

| Metric | Value |
|---|---|
`)
	fmt.Fprintf(&b, "| **Token-defaults-debt (total HARD defects)** | **%d** |\n", debt)
	fmt.Fprintf(&b, "| Detector schema (same corpus as `--json`) | `%s` |\n", c["detector_schema"])
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
