// Package defaultvaluescore is the default-value scorecard -- the RECURRING GUARD for
// epic #1089's finding: fak value features that ship NOT-fully-enabled (compaction was
// illegible, amplification dead-on-proxy, vcache modeled, kvmmu unwired). Each of those
// regressed silently; this card makes "a value feature shipped off / vacuous / modeled"
// a tracked debt integer that a CI floor can pin, so it can never silently regrow.
//
// Like internal/conflationscore it is a TREE-READING scorecard (no data dir): the flag
// definitions in cmd/fak/guard.go + cmd/fak/serve.go and the exit-summary / score
// surfaces ARE the data, parsed from the Go source, so the score cannot be gamed by
// editing a JSON file -- only by fixing the real wiring. The allow-list of value-flags
// that may legitimately ship OFF lives here WITH a reason; an OFF value-flag not on the
// list is debt. The scoring fold / grade / markdown machinery is the shared kernel in
// pkg/scorecard; this package holds only the default-value-specific tables, parsing, and
// the four KPIs that map worst-first onto the child issues #1090-#1095.
//
// The four checks (issue #1096):
//  1. VALUE_FLAG_OFF      -- a value-flag (a cost/cache/amplification lever) parsed from
//     guard.go/serve.go that ships default-OFF, unless allow-listed with a reason.
//  2. VALUE_FLAG_CONTEXT_DRIFT -- a value saver offered in BOTH serving surfaces (guard.go
//     AND serve.go) that ships default-ON in one but default-OFF in the other, so the same
//     cost/cache/context win is silently disabled on one serving path. Scoped to SHARED
//     flags, so a role-specific saver present in only one surface is never a false positive.
//  3. VACUOUS_ON_GUARD    -- an exit-summary line that folds kernel.Counters
//     (VDSOHits/Transforms/Denies/EngineCalls) WITHOUT a proxy-aware path-split, because
//     those counters are structurally 0 on the Decide-only `fak guard -- claude` proxy.
//  4. C_MODELED_NOT_OBSERVED -- a score/report surface whose DEFAULT headline source is
//     "planned"/modeled rather than observed live telemetry.
package defaultvaluescore

import (
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id the fold and any consumer key on.
const Schema = "fak-default-value-scorecard/2"

// DebtKey is the headline integer the control-pane folds (scorecard_control_pane reads
// corpus.default_value_debt).
const DebtKey = "default_value_debt"

// CleanFloor is the disciplined tree's expected debt; the live-tree smoke pins the
// scorecard against silent regrowth. It is ZERO: epic #1089's last tracked default-value
// defect -- the vcache score surface defaulting its headline source to "planned"
// (modeled), the C_MODELED_NOT_OBSERVED gap -- is retired. internal/vcachescore/score.go
// now defaults its headline to the OBSERVED cache value (ActiveSourceObserved) and labels
// the witness-free fallback honestly as a FORECAST (ActiveSourceForecast), never a
// modeled number dressed as measured. The floor is 0 so the gate fails the instant ANY
// default-value debt reappears.
const CleanFloor = 0

// FlagSources are the two surfaces whose value-flags this card classifies. guard.go is
// the flagship `fak guard -- claude` proxy; serve.go is the in-kernel serve loop. Both
// define their knobs with the same flag.FlagSet idiom, so one parser reads both.
var FlagSources = []string{
	"cmd/fak/guard.go",
	"cmd/fak/serve.go",
}

// AmplificationSurface is the exit-summary source whose kernel.Counters fold check
// grades. It is the file that renders the `fak guard:` exit lines.
const AmplificationSurface = "cmd/fak/guard.go"

// ScoreSurfaces are the score/report surfaces graded for a default "planned"/
// modeled headline source.
var ScoreSurfaces = []string{
	"internal/vcachescore/score.go",
}

// kernelCounterFields are the kernel.Counters axes that are structurally 0 on the
// Decide-only proxy path. An exit-summary site reading these into a `fak guard:` line
// without a proxy-aware split prints a vacuous number on the dominant guard path.
var kernelCounterFields = []string{
	"VDSOHits", "Transforms", "Denies", "EngineCalls",
}

// valueFlagTokens identify a flag as a VALUE flag -- a cost/cache/amplification/context
// economy lever, the class epic #1089 found shipping not-fully-enabled. A flag whose
// NAME matches one of these is in scope for the default-on rule. Matching is on the flag
// NAME only, deliberately NOT the help text: nearly every flag's help mentions "cache" or
// "token economy" in passing (--session-id, --tokenizer, --replay-trace all do), and a
// help-text mention does not make a flag a value lever. The name is the precise signal --
// a value lever is named for the lever it pulls (compact-history-budget, elide-result-
// bytes, vdso, engine-cache-engine), a transport/identity flag is not. Substring, case-
// insensitive (scorecard.HasAny).
var valueFlagTokens = []string{
	"compact", "elide", "ctx-view", "context-budget", "amplif",
	"vdso", "prune", "view-budget", "engine-cache", "cuda-graph",
	"gpudirect",
}

// offWithReason is the allow-list of value-flags that may legitimately ship default-OFF,
// each with the documented reason it is gated. An OFF value-flag NOT in this map is debt
// (VALUE_FLAG_OFF). The reason is what a reviewer reads to confirm the gate is genuine,
// not an excuse for a dead feature -- the same discipline conflationscore's qualifier
// tables apply to provenance prose. Keyed by the flag name as it appears in fs.<T>("..").
type reviewedDefaultDecision struct {
	reason   string
	reviewBy string
}

// offWithReason is the reviewed gate registry for value flags that legitimately ship
// default-OFF. A reason alone is not permanent absolution: reviewBy makes omission harm
// observable as the agentic default window moves. Dates are UTC YYYY-MM-DD.
var offWithReason = map[string]reviewedDefaultDecision{
	"context-budget-tokens": {
		reason:   "seeds a hard context budget that returns reset directives -- an operator policy, not a silent default (0 = off keeps today's path)",
		reviewBy: "2026-11-01",
	},
	"cuda-graph": {
		reason:   "a measured no-win on small-model/L4 decode (launch overhead already small) -- requires a per-node tok/s witness before it pays",
		reviewBy: "2026-11-01",
	},
	"vdso-proxy-fill": {
		reason:   "changes cross-turn cache residency and requires a named principal plus writes routed through fak",
		reviewBy: "2026-10-01",
	},
	"compact-anchor-head": {
		reason:   "can burst the recent provider-cache breakpoint once -- needs a live session-turn payback witness",
		reviewBy: "2026-10-01",
	},
	"compact-solvency-floor": {
		reason:   "only the launcher knows model window minus output reserve; a gateway guess is harmful in both directions",
		reviewBy: "2026-11-01",
	},
	"engine-cache-engine": {
		reason:   "needs a configured self-hosted serving engine and admin credential",
		reviewBy: "2026-11-01",
	},
	"engine-cache-base-url": {
		reason:   "the serving-engine control URL is undefined until an engine is configured",
		reviewBy: "2026-11-01",
	},
	"engine-cache-admin-key-env": {
		reason:   "a deployment credential is meaningless until an engine is configured",
		reviewBy: "2026-11-01",
	},
	"engine-cache-idle-timeout": {
		reason:   "a per-engine tuning knob paired with engine configuration",
		reviewBy: "2026-11-01",
	},
	"engine-cache-require-exact-span": {
		reason:   "deployment fail-closed policy paired with engine configuration",
		reviewBy: "2026-11-01",
	},
}

// valueFlag is one parsed value-flag: its name, kind, default literal, whether the parse
// judged it default-on, the source file, and the help text (for the reason check).
// onWithReason is the reviewed registry for value flags that ship default-ON. ON is
// still a product decision: each entry records why automatic exposure is net-beneficial
// and when that conclusion must be revisited. Shared guard/serve flags use one rationale.
var onWithReason = map[string]reviewedDefaultDecision{
	"precompact-hook":        {reason: "shadow-mode lifecycle handoff preserves provider-cache continuity without changing the user task", reviewBy: "2026-11-01"},
	"ctx-view-budget":        {reason: "bounded context views reduce repeated prompt tokens while retaining the configured working set", reviewBy: "2026-11-01"},
	"compact-history-budget": {reason: "bounded history compaction reduces stale context cost while preserving recent turns and anchors", reviewBy: "2026-11-01"},
	"compact-anchor-head":    {reason: "retaining the stable head protects provider-cache reuse across compacted turns", reviewBy: "2026-11-01"},
	"elide-result-bytes":     {reason: "result elision removes repeated payload bytes while preserving references needed for recall", reviewBy: "2026-11-01"},
	"elide-stale-reads":      {reason: "stale-read elision suppresses superseded observations while preserving the latest value", reviewBy: "2026-11-01"},
	"vdso":                   {reason: "the in-process fast path avoids redundant engine calls and retains the policy checkpoint", reviewBy: "2026-11-01"},
	"gpudirect-overflow":     {reason: "direct P2P DMA NVMe/host overflow bypasses CPU bounce buffering on VRAM saturation", reviewBy: "2026-11-01"},
}

type valueFlag struct {
	name      string
	kind      string // Int|Bool|String|Int64|Float64|Duration
	defLit    string // the default-value literal as written in source
	defaultOn bool
	source    string
	help      string
}

// fsFlag matches a flag.FlagSet definition: fs.<Kind>("name", <default>, "help"). The
// default group is non-greedy up to the comma before the help string; help is the final
// double-quoted argument body (escapes tolerated). One regex per source line covers the
// guard.go/serve.go idiom (every flag there is a single-line fs.<Kind>(...) call).
var reFlag = regexp.MustCompile(
	`fs\.(Int|Bool|String|Int64|Float64|Duration)\(\s*"([^"]+)"\s*,\s*(.+?)\s*,\s*"((?:[^"\\]|\\.)*)"\s*\)`)

// ParseFlags pulls every fs.<Kind>(...) value-flag out of a flag-source file and returns
// the ones that are VALUE flags (in scope for the default-on rule). The default-on
// judgment is per kind: a Bool is on iff its default literal is true; an Int/Float/
// Duration is on iff its default is a NON-zero literal (0 / 0s is the documented "off"
// sentinel across guard.go/serve.go); a String is on iff its default is a non-empty
// literal. A default that references a named constant (e.g. gateway.DefaultCompact-
// HistoryBudget) is treated as ON -- a named default-budget constant is the codebase's
// idiom for "shipped enabled", and conflating it with 0 would false-positive the very
// flags #1089 fixed to be default-on.
func ParseFlags(text, source string) []valueFlag {
	var out []valueFlag
	for _, m := range reFlag.FindAllStringSubmatch(text, -1) {
		kind, name, def, help := m[1], m[2], strings.TrimSpace(m[3]), unescape(m[4])
		if !isValueFlag(name, help) {
			continue
		}
		out = append(out, valueFlag{
			name:      name,
			kind:      kind,
			defLit:    def,
			defaultOn: judgeDefaultOn(kind, def),
			source:    source,
			help:      help,
		})
	}
	return out
}

// isValueFlag reports whether a flag is a value lever (in scope). Only the flag NAME is
// matched against valueFlagTokens (see the table comment): a value lever is named for its
// lever; everything else (transport, identity, credential, logging path) is out of scope
// and never debt. help is accepted for signature symmetry but intentionally not matched.
func isValueFlag(name, help string) bool {
	_ = help
	return scorecard.HasAny(name, valueFlagTokens)
}

// judgeDefaultOn applies the per-kind default-on rule documented on ParseFlags.
func judgeDefaultOn(kind, def string) bool {
	switch kind {
	case "Bool":
		// A Bool is ON unless its default is a literal false sentinel. A NAMED-CONSTANT
		// default (e.g. gateway.DefaultElideStaleReads) is ON -- the same "a named default
		// constant is the codebase's idiom for 'shipped enabled'" rule ParseFlags already
		// documents for the Int/String branches. The Bool branch previously matched only
		// the bare literal `true`, so a Bool value-flag defaulted from a constant was
		// false-positived as default-OFF (the --elide-stale-reads regression the live-tree
		// floor caught). Only a bare `false` literal is the off sentinel.
		return !isFalseLiteral(def)
	case "String":
		// A non-empty string literal default is ON. Anything that is not a bare ""
		// literal (a named constant, a non-empty literal) counts as configured-on.
		return def != `""` && def != `"" `
	default: // Int, Int64, Float64, Duration
		// 0 / 0.0 / 0s is the off sentinel; a named constant or a non-zero literal is on.
		return !isZeroLiteral(def)
	}
}

var reZeroNum = regexp.MustCompile(`^0(?:\.0+)?$`)

// isZeroLiteral reports whether a numeric default literal is the off sentinel (0, 0.0,
// 0s, 0 * time.Second). A named constant (contains a letter) is never zero here.
func isZeroLiteral(def string) bool {
	d := strings.TrimSpace(def)
	if reZeroNum.MatchString(d) {
		return true
	}
	// Duration off sentinels written as a literal.
	if d == "0s" || d == "0 * time.Second" || d == "time.Duration(0)" {
		return true
	}
	return false
}

// isFalseLiteral reports whether a Bool default literal is the off sentinel. In an
// fs.Bool("name", <default>, ...) call the default is a Go bool expression, so the only
// source-level off sentinel is the bare `false` literal; a `true` literal or a named
// constant (gateway.DefaultElideStaleReads) is ON -- the "named constant == shipped
// enabled" idiom the numeric branch already applies via isZeroLiteral.
func isFalseLiteral(def string) bool {
	return strings.TrimSpace(def) == "false"
}

func unescape(s string) string {
	s = strings.ReplaceAll(s, `\"`, `"`)
	s = strings.ReplaceAll(s, `\n`, " ")
	s = strings.ReplaceAll(s, `\t`, " ")
	return s
}

func validReviewDate(value string, asOf time.Time) bool {
	review, err := time.Parse("2006-01-02", value)
	if err != nil {
		return false
	}
	return !review.Before(asOf.UTC().Truncate(24 * time.Hour))
}

func nextGateReview(asOf time.Time) string {
	var next time.Time
	registries := []map[string]reviewedDefaultDecision{offWithReason, onWithReason}
	for _, registry := range registries {
		for _, gate := range registry {
			review, err := time.Parse("2006-01-02", gate.reviewBy)
			if err != nil || review.Before(asOf.UTC().Truncate(24*time.Hour)) {
				continue
			}
			if next.IsZero() || review.Before(next) {
				next = review
			}
		}
	}
	if next.IsZero() {
		return "due"
	}
	return next.Format("2006-01-02")
}

// kpiValueFlagDefaultOn (HARD): every VALUE flag ships default-ON unless it is on the
// offWithReason allow-list. An off value-flag with no documented gating reason is debt --
// the exact "shipped not-fully-enabled" failure #1089 found. Maps worst-first onto the
// child issues #1090-#1095 (the off-by-default value features to retire).
func kpiValueFlagDefaultOn(flags []valueFlag, asOf time.Time) scorecard.KPI {
	var defects []string
	total := len(flags)
	on := 0
	for _, f := range flags {
		if f.defaultOn {
			on++
			gate, ok := onWithReason[f.name]
			if ok && gate.reason != "" && validReviewDate(gate.reviewBy, asOf) {
				continue
			}
			class := "DEFAULT_ON_UNREVIEWED"
			detail := "no reviewed benefit/harm gate"
			if ok {
				class = "DEFAULT_ON_REVIEW_DUE"
				detail = "gate review " + gate.reviewBy + " is missing, invalid, or due"
			}
			defects = append(defects, fmt.Sprintf(
				"%s: value-flag --%s ships default-ON (default %s); %s -- exposure harm is unreviewed (%s)",
				lastSegment(f.source), f.name, f.defLit, detail, class))
			continue
		}
		if gate, ok := offWithReason[f.name]; ok && gate.reason != "" && validReviewDate(gate.reviewBy, asOf) {
			continue
		}
		class := "VALUE_FLAG_OFF"
		detail := "no reviewed gate"
		if gate, ok := offWithReason[f.name]; ok {
			class = "OPT_IN_REVIEW_DUE"
			detail = "gate review " + gate.reviewBy + " is missing, invalid, or due"
		}
		defects = append(defects, fmt.Sprintf(
			"%s: value-flag --%s ships default-OFF (default %s); %s -- omission harm may now exceed exposure harm (%s)",
			lastSegment(f.source), f.name, f.defLit, detail, class))
	}
	score := 100.0
	if total > 0 {
		score = 100.0 * float64(total-len(defects)) / float64(total)
	}
	return scorecard.KPI{
		Key: "value_flag_default_on", Group: "value",
		Score:   score,
		Detail:  fmt.Sprintf("%d/%d value-flags carry a current reason+review date for their ON/OFF default", total-len(defects), total),
		Defects: defects,
	}
}

// kpiValueFlagContextParity is epic #1089's CROSS-CONTEXT rung. kpiValueFlagDefaultOn asks
// "is each value-flag on in its OWN surface"; this asks "is each value saver offered in MORE
// THAN ONE serving surface on CONSISTENTLY across them". A saver that ships default-ON under
// `fak guard` but default-OFF under `fak serve` (or vice versa) is the same cost/cache/context
// win silently disabled on ONE serving path -- the purest "enabled in one context, not another"
// defect this epic hunts, and the one kpiValueFlagDefaultOn structurally cannot see because it
// judges each surface in isolation. It is scoped to flags present in >=2 surfaces, so a
// legitimately role-specific saver (a serve-only inference-engine knob like --engine-cache-*,
// a guard-only session-lifecycle lever like --precompact-hook) is NEVER a false positive: it
// is simply not shared, so no parity is claimed for it.
func kpiValueFlagContextParity(flags []valueFlag) scorecard.KPI {
	// name -> (surface segment -> defaultOn), preserving first-seen order for stable output.
	postures := map[string]map[string]bool{}
	defLits := map[string]map[string]string{}
	var order []string
	for _, f := range flags {
		if _, seen := postures[f.name]; !seen {
			postures[f.name] = map[string]bool{}
			defLits[f.name] = map[string]string{}
			order = append(order, f.name)
		}
		seg := lastSegment(f.source)
		postures[f.name][seg] = f.defaultOn
		defLits[f.name][seg] = f.defLit
	}
	var defects []string
	shared := 0
	for _, name := range order {
		bySurface := postures[name]
		if len(bySurface) < 2 {
			continue // present in a single serving surface -- role-specific, no parity claim
		}
		shared++
		var anyOn, anyOff bool
		for _, on := range bySurface {
			if on {
				anyOn = true
			} else {
				anyOff = true
			}
		}
		if anyOn && anyOff {
			defects = append(defects, fmt.Sprintf(
				"value-flag --%s ships default-ON in one serving context but default-OFF in another (%s) -- the same win is silently disabled on one serving path (VALUE_FLAG_CONTEXT_DRIFT)",
				name, renderPostures(bySurface, defLits[name])))
		}
	}
	score := 100.0
	if shared > 0 {
		score = 100.0 * float64(shared-len(defects)) / float64(shared)
	}
	return scorecard.KPI{
		Key: "value_flag_context_parity", Group: "value",
		Score:   score,
		Detail:  fmt.Sprintf("%d/%d shared value-flags ship the same default across serving contexts", shared-len(defects), shared),
		Defects: defects,
	}
}

// renderPostures renders a flag's per-surface default posture deterministically, e.g.
// "guard.go=on[true] serve.go=off[false]", sorted by surface so the witness never reorders.
func renderPostures(bySurface map[string]bool, defLits map[string]string) string {
	segs := make([]string, 0, len(bySurface))
	for s := range bySurface {
		segs = append(segs, s)
	}
	sort.Strings(segs)
	parts := make([]string, 0, len(segs))
	for _, s := range segs {
		state := "off"
		if bySurface[s] {
			state = "on"
		}
		parts = append(parts, fmt.Sprintf("%s=%s[%s]", s, state, defLits[s]))
	}
	return strings.Join(parts, " ")
}

// counterReadRe matches a kernel.Counters field read on a guard.go line: kc.VDSOHits,
// kc.Transforms, etc. The receiver name is not pinned (any ident.<Field>), so a renamed
// receiver still trips the check.
var counterReadRe = regexp.MustCompile(`\b\w+\.(VDSOHits|Transforms|Denies|EngineCalls)\b`)

// proxyAwareMarkers prove an exit-summary site that folds kernel.Counters KNOWS the
// counters are 0 on the Decide proxy and splits the path / frames the line honestly.
// formatAmplification carries exactly these (the "(proxy path: ... Decide ...)" line and
// the comment naming the Decide-only axis), so the canonical correct site passes.
var proxyAwareMarkers = []string{
	"decide proxy", "decide-only", "decide increments none", "proxy path:",
	"adjudicates with decide", "does not apply", "structurally 0", "kc is empty",
}

// kpiNoVacuousCounterFold (HARD): an exit-summary function that reads kernel.Counters
// into a `fak guard:` line must carry a proxy-aware marker, because those counters are
// structurally 0 on the dominant Decide-only `fak guard -- claude` proxy. A fold with no
// such marker would print a vacuous 0/1.0x on the dominant path (VACUOUS_ON_GUARD). The
// check is per FUNCTION block: it scans each func that both reads a counter field and
// renders a `fak guard:` line, and requires a marker somewhere in that block.
func kpiNoVacuousCounterFold(text, source string) scorecard.KPI {
	var defects []string
	folds := 0
	for _, fn := range splitFuncs(text) {
		low := strings.ToLower(fn.body)
		readsCounter := counterReadRe.MatchString(fn.body)
		rendersGuardLine := strings.Contains(low, "fak guard:")
		if !readsCounter || !rendersGuardLine {
			continue
		}
		folds++
		if !scorecard.HasAny(fn.body, proxyAwareMarkers) {
			defects = append(defects, fmt.Sprintf(
				"%s: func %s folds kernel.Counters (%s) into a `fak guard:` exit line with no proxy-aware path-split -- those counters are 0 on the Decide-only guard proxy, so the line is vacuous there (VACUOUS_ON_GUARD)",
				lastSegment(source), fn.name, namedCounters(fn.body)))
		}
	}
	score := 100.0
	if folds > 0 {
		score = 100.0 * float64(folds-len(defects)) / float64(folds)
	}
	return scorecard.KPI{
		Key: "no_vacuous_counter_fold", Group: "value",
		Score:   score,
		Detail:  fmt.Sprintf("%d/%d kernel.Counters exit-folds are proxy-aware", folds-len(defects), folds),
		Defects: defects,
	}
}

// kpiObservedNotModeledDefault (HARD): a score/report surface must not default its
// headline source to "planned"/modeled. A surface whose default ActiveSource is "planned"
// (upgrading to observed telemetry only when live data is present) presents a modeled
// number as its headline (C_MODELED_NOT_OBSERVED) -- the vcache-was-modeled failure
// #1089 found. The check reads each score surface for a default-"planned" assignment that
// is not paired, in the same file, with an honest observed/telemetry upgrade AND a
// modeled-vs-observed disambiguation. (Today the surface DOES carry the upgrade but still
// ships "planned" as the default headline, which is the tracked debt.)
func kpiObservedNotModeledDefault(surfaces map[string]string) scorecard.KPI {
	var defects []string
	checked := 0
	for _, path := range sortedKeys(surfaces) {
		text := surfaces[path]
		if text == "" {
			continue
		}
		checked++
		low := strings.ToLower(text)
		defaultsPlanned := strings.Contains(low, `= "planned"`) || strings.Contains(low, `:= "planned"`)
		if !defaultsPlanned {
			continue
		}
		defects = append(defects, fmt.Sprintf(
			"%s: score surface defaults its headline source to \"planned\" (modeled), upgraded to observed only when live telemetry is present -- the default report is a modeled number (C_MODELED_NOT_OBSERVED)",
			lastSegment(path)))
	}
	score := 100.0
	if checked > 0 {
		score = 100.0 * float64(checked-len(defects)) / float64(checked)
	}
	return scorecard.KPI{
		Key: "observed_not_modeled_default", Group: "value",
		Score:   score,
		Detail:  fmt.Sprintf("%d/%d score surfaces default to an OBSERVED headline", checked-len(defects), checked),
		Defects: defects,
	}
}

// Build reads the flag, exit-summary, and score surfaces, runs the four KPIs, and folds
// them into the control-pane payload via the shared kernel. root is the repo root.
func Build(root string) scorecard.Payload {
	return BuildAsOf(root, time.Now())
}

// BuildAsOf evaluates the default-value scorecard as of a given timestamp, providing a
// deterministic scoring seam for tests and historical audits. root is the repo root.
func BuildAsOf(root string, asOf time.Time) scorecard.Payload {
	var flags []valueFlag
	for _, rel := range FlagSources {
		text := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(rel)))
		flags = append(flags, ParseFlags(text, rel)...)
	}
	ampText := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(AmplificationSurface)))
	scoreSurfaces := map[string]string{}
	for _, rel := range ScoreSurfaces {
		scoreSurfaces[rel] = scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(rel)))
	}

	kpis := []scorecard.KPI{
		kpiValueFlagDefaultOn(flags, asOf),
		kpiValueFlagContextParity(flags),
		kpiNoVacuousCounterFold(ampText, AmplificationSurface),
		kpiObservedNotModeledDefault(scoreSurfaces),
	}
	debt := 0
	for _, k := range kpis {
		debt += len(k.Defects)
	}
	offFlags := 0
	for _, f := range flags {
		if !f.defaultOn {
			offFlags++
		}
	}

	finding := "the agentic default window is balanced: every value lever has a current reviewed reason for its ON/OFF default, context defaults agree, and reports use observed evidence"
	next := "hold -- re-run when a value flag lands or any default review date arrives"
	if debt > 0 {
		finding = scorecard.CountNoun(debt, "default-window defect") + ": unreviewed exposure, stale ON/OFF gating, context drift, or modeled evidence is outside the reviewed window"
		next = "review worst-first (" + worstKPI(kpis) + "): choose ON/OFF from evidence, renew the witnessed gate, or exclude the lever"
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStrict,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		ExtraCorpus: map[string]any{
			"value_flags_seen":          len(flags),
			"value_flags_off":           offFlags,
			"reviewed_opt_in_flags":     len(offWithReason),
			"reviewed_default_on_flags": len(onWithReason),
			"reviewed_default_flags":    len(offWithReason) + len(onWithReason),
			"next_default_review":       nextGateReview(asOf),
			"flag_surfaces":             len(FlagSources),
			"score_surfaces":            len(ScoreSurfaces),
		},
	})
	p.Workspace = root
	return p
}

// --- func-block splitter (so the counter-fold check is per-function) ---------------------

type funcBlock struct {
	name string
	body string
}

var reFuncHead = regexp.MustCompile(`(?m)^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(`)

// splitFuncs breaks Go source into top-level function blocks (name + body up to the next
// top-level func). It is a lightweight slicer, not a parser: it keys on the column-0
// `func ` header the gofmt'd tree guarantees, which is all the per-function counter-fold
// check needs. A trailing block (after the last header) is included.
func splitFuncs(text string) []funcBlock {
	idx := reFuncHead.FindAllStringSubmatchIndex(text, -1)
	var out []funcBlock
	for i, m := range idx {
		name := text[m[2]:m[3]]
		start := m[0]
		end := len(text)
		if i+1 < len(idx) {
			end = idx[i+1][0]
		}
		out = append(out, funcBlock{name: name, body: text[start:end]})
	}
	return out
}

// namedCounters lists which kernel.Counters fields a block reads, for the defect message.
func namedCounters(body string) string {
	seen := map[string]bool{}
	var names []string
	for _, f := range kernelCounterFields {
		if regexp.MustCompile(`\b\w+\.` + f + `\b`).MatchString(body) {
			if !seen[f] {
				seen[f] = true
				names = append(names, f)
			}
		}
	}
	return strings.Join(names, ", ")
}

// --- small local helpers ------------------------------------------------------------------

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func lastSegment(path string) string {
	if i := strings.LastIndexAny(path, "/\\"); i >= 0 {
		return path[i+1:]
	}
	return path
}

// plural renders "N noun" with a trailing "(s)" when N != 1.

func worstKPI(kpis []scorecard.KPI) string {
	worst := kpis[0]
	for _, k := range kpis[1:] {
		if len(k.Defects) > len(worst.Defects) {
			worst = k
		}
	}
	return worst.Key
}
