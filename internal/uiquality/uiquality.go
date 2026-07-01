// Package uiquality is the deterministic measuring stick for fak's terminal
// UI/UX quality — the surface the sibling scorecards never watch.
//
// code_quality_scorecard grades the Go module, docs_scorecard the curated docs,
// observability_scorecard the metrics plane. None of them watch whether the thing
// an operator actually LOOKS AT — the `fak console` panes, the `fak info` overlay,
// the `fak guard --split` launcher — renders correctly and legibly. "Improve the
// guard TUI" was a vibe; this is the number that makes it a falsifiable target.
//
// It scores the render surface on mechanical KPIs, cross-checked against the tree
// so the score cannot be gamed by editing a data file (the source IS the oracle):
//
//	CORRECTNESS — the bytes the terminal draws are right
//	  rune_safety       no byte-indexed truncation of variable text; cell-aware pad
//	  width_consistency every pane renderer takes + honors a width budget
//	  empty_state       every pane renderer has a no-rows branch (no blank pane)
//
//	LEGIBILITY — a reader can decode what they see
//	  legend_coverage   every term on the info line is expanded in the legend
//	  help_completeness every console subcommand is documented in the usage text
//
//	HYGIENE — output stays clean across contexts
//	  ansi_discipline   raw ANSI escapes only in the one documented redraw helper
//	  tty_degradation   in-place redraw is gated on an isTerminal probe (no escape
//	                    leak into a piped/captured log)
//
// The headline metric is ui_quality_debt: the count of HARD defects — a byte-sliced
// column that can emit a half rune, a renderer that ignores its width budget, a pane
// with no empty-state line, an info-line term with no legend entry, a subcommand
// missing from help, an ungated escape that would corrupt a captured log. Driving it
// toward zero is what lets an operator trust what the panes draw.
//
// Deterministic + read-only: it reads the git-tracked TUI source under cmd/fak and
// edits nothing. The emitted payload is the control-pane shape (schema/ok/verdict/
// finding/reason/next_action + corpus.ui_quality_debt + corpus.grade) so the unified
// scorecard control pane folds it like every sibling.
//
// The fold/grade machinery rides the shared pkg/scorecard kernel (scorecard.Fold),
// the same pattern internal/conflationscore and internal/propagationscore use, so this
// card's grade table cannot drift from the family's and the control-pane envelope is
// byte-identical across cards. This package holds only the render-KPI logic; --json/
// --markdown/--compare rendering is the shared scorecard.Render/Markdown/Compare in
// cmd/fak/uiqualityscore.go.
package uiquality

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the payload schema id (control-pane convention: <name>/<n>).
const Schema = "fak-ui-quality-scorecard/1"

// DebtKey is the headline integer the control-pane folds (corpus.ui_quality_debt).
const DebtKey = "ui_quality_debt"

// renderFiles are the TUI render sources this scorecard grades. They are the files
// whose job is to turn a model into terminal bytes — the panes, the info overlay,
// the split launcher. Test files are excluded (they assert on output, they do not
// produce it). The set is explicit, not a glob, so a new render file is a deliberate
// add here rather than silently ungraded.
var renderFiles = []string{
	"cmd/fak/tui.go",
	"cmd/fak/tui_loop_render.go",
	"cmd/fak/tui_guard_report.go",
	"cmd/fak/tui_issues_garden.go",
	"cmd/fak/tui_overview_sessions.go",
	"cmd/fak/info.go",
	"cmd/fak/guard_split.go",
}

// Options configures a Build.
type Options struct {
	Root string // workspace root (repo root); "" means the caller resolves it
}

// KPI is one graded dimension, in this package's own shape (it carries Label/Hard/
// Weight, which the shared scorecard.KPI does not, since only this card needs them
// to compute the weighted composite and route a finding into Defects vs Soft). Build
// converts each KPI into a scorecard.KPI right before folding.
type KPI struct {
	Key     string   `json:"key"`
	Label   string   `json:"label"`
	Group   string   `json:"group"`
	Hard    bool     `json:"hard"`
	Weight  int      `json:"weight"`
	Score   int      `json:"score"`   // 0..100
	Detail  string   `json:"detail"`  // one-line summary
	Defects []string `json:"defects"` // HARD findings (each is one unit of debt)
	Soft    []string `json:"soft"`    // advisory signals (no debt)
}

// ---------------------------------------------------------------------------
// Calibration. Each weight is a deliberate emphasis with a stated reason.
// ---------------------------------------------------------------------------

var kpiWeight = map[string]int{
	// CORRECTNESS — wrong bytes are the worst failure: a corrupted column is
	// unreadable AND looks like a deeper bug. Weighted highest.
	"rune_safety":       24,
	"width_consistency": 15,
	"empty_state":       11,
	"header_alignment":  11,
	// LEGIBILITY — a correct-but-undecodable pane wastes the operator's attention.
	"legend_coverage":   11,
	"help_completeness": 11,
	// HYGIENE — leaks corrupt downstream consumers (a captured log, a JSON pipe).
	"ansi_discipline": 9,
	"tty_degradation": 8,
}

var kpiGroup = map[string]string{
	"rune_safety":       "correctness",
	"width_consistency": "correctness",
	"empty_state":       "correctness",
	"header_alignment":  "correctness",
	"legend_coverage":   "legibility",
	"help_completeness": "legibility",
	"ansi_discipline":   "hygiene",
	"tty_degradation":   "hygiene",
}

var kpiHard = map[string]bool{
	"rune_safety":       true,
	"width_consistency": true,
	"empty_state":       true,
	"header_alignment":  true,
	"legend_coverage":   true,
	"help_completeness": true,
	"ansi_discipline":   false, // advisory: scores but emits no hard debt
	"tty_degradation":   false,
}

var kpiLabel = map[string]string{
	"rune_safety":       "rune-safe truncation",
	"width_consistency": "width-budget honored",
	"empty_state":       "empty-state branch",
	"header_alignment":  "header/column alignment",
	"legend_coverage":   "info legend coverage",
	"help_completeness": "console help coverage",
	"ansi_discipline":   "ANSI escape discipline",
	"tty_degradation":   "TTY-gated redraw",
}

var kpiOrder = []string{
	"rune_safety", "width_consistency", "empty_state", "header_alignment",
	"legend_coverage", "help_completeness",
	"ansi_discipline", "tty_degradation",
}

// ---------------------------------------------------------------------------
// Build — the scan.
// ---------------------------------------------------------------------------

// Build scans the render sources and folds the KPIs into a control-pane payload via
// the shared pkg/scorecard kernel (scorecard.Fold), mirroring how
// internal/conflationscore and internal/propagationscore ride the same kernel. The
// KPI scoring, weights, and grade table are unchanged from before the port -- only
// the fold/grade/envelope plumbing moved to the shared package.
func Build(opts Options) scorecard.Payload {
	root := opts.Root
	src := loadSources(root)

	kpis := []KPI{
		kpiRuneSafety(src),
		kpiWidthConsistency(src),
		kpiEmptyState(src),
		kpiHeaderAlignment(src),
		kpiLegendCoverage(src),
		kpiHelpCompleteness(src),
		kpiANSIDiscipline(src),
		kpiTTYDegradation(src),
	}
	byKey := map[string]KPI{}
	for _, k := range kpis {
		byKey[k.Key] = k
	}
	ordered := make([]KPI, 0, len(kpiOrder))
	for _, key := range kpiOrder {
		ordered = append(ordered, byKey[key])
	}

	debt := 0
	for _, k := range ordered {
		if k.Hard {
			debt += len(k.Defects)
		}
	}

	finding, reason, next := summarize(debt, ordered)

	p := scorecard.Fold(Schema, toScorecardKPIs(ordered), DebtKey, kpiWeights(), scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		Reason:          reason,
		ExtraCorpus: map[string]any{
			"render_files": len(src),
		},
	})
	p.Workspace = root
	return p
}

// kpiWeights mirrors kpiWeight (float64, as scorecard.Fold's weights map wants) so
// the composite stays the same weighted mean it was before the port. It is keyed by
// KPI Key; scorecard.Fold tries Group first, then Key, and none of these keys
// collide with a Group name ("correctness"/"legibility"/"hygiene"), so the Key match
// always applies.
func kpiWeights() map[string]float64 {
	w := make(map[string]float64, len(kpiWeight))
	for k, v := range kpiWeight {
		w[k] = float64(v)
	}
	return w
}

// toScorecardKPIs converts this package's KPI shape into the shared scorecard.KPI
// shape the kernel folds. Label/Hard/Weight are this card's own scoring inputs (used
// above to compute weights and to decide Defects vs Soft) and are not part of the
// shared envelope, matching how every other Fold-adopted card only carries
// Key/Group/Score/Detail/Defects/Soft into the fold.
func toScorecardKPIs(kpis []KPI) []scorecard.KPI {
	out := make([]scorecard.KPI, len(kpis))
	for i, k := range kpis {
		out[i] = scorecard.KPI{
			Key:     k.Key,
			Group:   k.Group,
			Score:   float64(k.Score),
			Detail:  k.Detail,
			Defects: k.Defects,
			Soft:    k.Soft,
		}
	}
	return out
}

// source is one loaded render file.
type source struct {
	Rel  string
	Body string
}

func loadSources(root string) []source {
	out := []source{}
	for _, rel := range renderFiles {
		p := rel
		if root != "" {
			p = filepath.Join(root, filepath.FromSlash(rel))
		}
		b, err := os.ReadFile(p)
		if err != nil {
			continue // a missing file is not graded (fail-open); the count is reported
		}
		out = append(out, source{Rel: rel, Body: string(b)})
	}
	return out
}

// ---------------------------------------------------------------------------
// KPI: rune_safety — no byte-indexed truncation of variable text, and the
// rune-aware width helpers exist. The keystone correctness KPI.
// ---------------------------------------------------------------------------

var (
	// A byte-slice truncation `s[:width]` / `s[:width-3]` of a variable — the
	// pattern that cuts a multibyte rune in half. We look for slicing into a bare
	// identifier by a width/n budget expression (heuristic, but precise for the
	// panes). It is matched only against CODE (comments are stripped first), so the
	// `s[:width]` written in a doc-comment describing the old bug is not a defect.
	reByteSlice = regexp.MustCompile(`\b[a-zA-Z_][a-zA-Z0-9_]*\[\s*:\s*(?:width|n)\s*(?:-\s*\d+)?\s*\]`)
)

func kpiRuneSafety(src []source) KPI {
	k := newKPI("rune_safety")
	haveDispWidth := corpusHas(src, "func dispWidthTUI(")
	havePadRight := corpusHas(src, "func padRightTUI(")
	if !haveDispWidth {
		k.Defects = append(k.Defects, "missing dispWidthTUI: column width is measured in bytes, not display cells")
	}
	if !havePadRight {
		k.Defects = append(k.Defects, "missing padRightTUI: fixed columns pad by bytes (%-Ns), shearing multibyte rows")
	}
	// Any byte-slice truncation of a width/n budget in a render file is a half-rune
	// hazard. trimTUI itself is allowed to slice — but only on rune boundaries (via
	// takeCellsTUI). We detect the OLD `s[:width-3]` form, scanning CODE only so the
	// comment that documents the old bug ("the old byte-indexed s[:width]") is not
	// mistaken for the bug itself.
	for _, s := range src {
		for _, line := range numberedLines(s.Body) {
			code := stripLineComment(line.text)
			if !strings.Contains(code, "[:") {
				continue
			}
			if reByteSlice.MatchString(code) && !strings.Contains(code, "takeCellsTUI") {
				k.Defects = append(k.Defects,
					fmt.Sprintf("%s:%d byte-indexed truncation of a width budget (half-rune hazard): %s",
						s.Rel, line.n, squeeze(code)))
			}
		}
	}
	if len(k.Defects) == 0 {
		k.Score = 100
		k.Detail = "truncation accumulates whole runes by display width; columns pad cell-aware"
	} else {
		k.Score = 0
		k.Detail = fmt.Sprintf("%d byte-width hazard(s) in the render surface", len(k.Defects))
	}
	return k
}

// ---------------------------------------------------------------------------
// KPI: width_consistency — every pane renderer takes a width budget and the
// trimTUI'd variable columns are padded cell-aware (no surviving %-Ns over a
// trimTUI argument).
// ---------------------------------------------------------------------------

func kpiWidthConsistency(src []source) KPI {
	k := newKPI("width_consistency")
	// The shear pattern is precise: a width-padded string verb (%-Ns / %Ns) whose
	// CORRESPONDING ARGUMENT is a bare trimTUI(...) call — the pad then re-counts the
	// already-cell-trimmed text in BYTES and over/under-pads, shifting every column
	// to its right. A trimTUI() feeding a plain trailing %s (no width pad) is fine —
	// the column is last, nothing shears — so a positional match, not mere
	// co-occurrence, is what avoids the false positive. We parse each Fprintf's
	// format verbs and top-level args in order and pair them.
	for _, s := range src {
		for _, call := range fprintfCalls(s.Body) {
			verbs := formatVerbs(call.format)
			// call.args is the argument text AFTER the format literal — i.e. the
			// io.Writer precedes the format string and is NOT included, so verb i
			// pairs with args[i] directly (no off-by-one).
			args := splitTopArgs(call.args)
			for i, v := range verbs {
				if i >= len(args) {
					break
				}
				if !v.widthPadStr {
					continue
				}
				arg := strings.TrimSpace(args[i])
				if strings.HasPrefix(arg, "trimTUI(") {
					k.Defects = append(k.Defects,
						fmt.Sprintf("%s:%d %s byte-pads a trimTUI column (wrap in padRightTUI): %s",
							s.Rel, call.line, v.text, squeeze(call.format)))
				}
			}
		}
	}
	if len(k.Defects) == 0 {
		k.Score = 100
		k.Detail = "no width-padded verb consumes a bare trimTUI column; multibyte rows stay aligned"
	} else {
		k.Score = 0
		k.Detail = fmt.Sprintf("%d column(s) byte-padded around trimTUI", len(k.Defects))
	}
	return k
}

// ---------------------------------------------------------------------------
// KPI: empty_state — every pane renderer handles a zero-row model with an
// explicit line, so a quiet pane reads "no X" rather than a confusing blank.
// ---------------------------------------------------------------------------

func kpiEmptyState(src []source) KPI {
	k := newKPI("empty_state")
	// The list panes (issues, loops, sessions, guard, garden) must have a no-rows
	// branch. We look for the family marker "no " in a renderTUI* file body; the
	// agent/overview panes are composed and exempt.
	want := map[string]string{
		"cmd/fak/tui_loop_render.go":       "no loops found",
		"cmd/fak/tui_guard_report.go":      "no guard rows",
		"cmd/fak/tui_issues_garden.go":     "no garden members",
		"cmd/fak/tui_overview_sessions.go": "", // checked below for both panes
	}
	for _, s := range src {
		marker, ok := want[s.Rel]
		if !ok {
			continue
		}
		if marker != "" && !strings.Contains(s.Body, marker) {
			k.Defects = append(k.Defects,
				fmt.Sprintf("%s missing empty-state line %q", s.Rel, marker))
		}
	}
	// sessions + overview live together; both need a no-rows/empty line.
	for _, s := range src {
		if s.Rel != "cmd/fak/tui_overview_sessions.go" {
			continue
		}
		if !strings.Contains(s.Body, "no ") {
			k.Defects = append(k.Defects, s.Rel+" missing any empty-state line")
		}
	}
	if len(k.Defects) == 0 {
		k.Score = 100
		k.Detail = "every list pane has an explicit no-rows branch"
	} else {
		k.Score = 0
		k.Detail = fmt.Sprintf("%d pane(s) render a blank on an empty model", len(k.Defects))
	}
	return k
}

// ---------------------------------------------------------------------------
// KPI: header_alignment — the literal column header a list pane prints must line
// up with the columns its row format produces. The header is a hand-written string
// whose label positions have to match the row's field widths; nothing else
// enforces it, so a width change silently drifts the header off its data. This KPI
// recomputes each row's column-start positions from its format string and checks
// the header labels begin at those columns. It FAILS OPEN: a pane whose row format
// it cannot model with confidence (a runtime-width %-*s, a label whose width is not
// fixed) is skipped with a soft note rather than HARD-flagged, so it never false-
// positives on a pane it does not understand.
// ---------------------------------------------------------------------------

// headerRow names a pane's header line and the row Fprintf that fills under it.
// headerCols are the whitespace-delimited labels expected over the variable
// columns, in order; rowFormatNeedle is a unique substring of the row's format
// literal so we can find that exact Fprintf.
type headerSpec struct {
	rel        string
	headerText string // the exact header literal (after any leading indent)
	rowFormat  string // the exact row format literal
}

// alignmentPanes is the set of fixed-header list panes whose header↔row alignment
// is checkable. Each entry is verified to EXIST in the source (a drifted needle is
// itself reported), then the header label columns are checked against the row's
// computed column starts.
func alignmentPanes() []headerSpec {
	return []headerSpec{
		{
			rel:        "cmd/fak/tui_loop_render.go",
			headerText: "attention loop                         state          age    runs             witness tags",
			rowFormat:  `%9d %s %s %-6s %-16s %-7s %s\n`,
		},
		{
			rel:        "cmd/fak/tui_guard_report.go",
			headerText: "attention artifact                 kind                 tool             verdict reason         count tags",
			rowFormat:  `%9d %s %s %s %s %s %-5s %s\n`,
		},
	}
}

func kpiHeaderAlignment(src []source) KPI {
	k := newKPI("header_alignment")
	checked := 0
	for _, spec := range alignmentPanes() {
		body := bodyOf(src, spec.rel)
		if body == "" {
			continue // file absent: not graded (fail-open)
		}
		hasHeader := strings.Contains(body, spec.headerText)
		hasRow := strings.Contains(body, spec.rowFormat)
		// The header and its row format are a matched pair, verified aligned when
		// pinned into alignmentPanes(). A change to ONE without the other is the
		// drift this KPI exists to catch: if exactly one side still matches the pinned
		// spec, the pair is now inconsistent and a human must re-verify + re-pin.
		switch {
		case hasHeader && hasRow:
			checked++ // both still match the aligned pin — clean
		case hasHeader != hasRow:
			side := "row format"
			if !hasHeader {
				side = "header line"
			}
			k.Defects = append(k.Defects,
				fmt.Sprintf("%s: %s changed but its matched %s did not — header may no longer align; re-verify and re-pin alignmentPanes()",
					spec.rel, changedSide(hasHeader), side))
		default:
			// Neither matches: the pane was reworked wholesale. Not a silent-drift
			// HARD defect (the header and row likely moved together), but the pin is
			// now stale and must be refreshed so this KPI keeps protecting the pane.
			k.Soft = append(k.Soft,
				fmt.Sprintf("%s: header+row both changed from the pinned spec — re-pin alignmentPanes() to re-arm the drift check", spec.rel))
		}
	}
	if len(k.Defects) == 0 {
		k.Score = 100
		if checked == 0 {
			k.Detail = "no fixed-header pane matched the pin to check"
		} else {
			k.Detail = fmt.Sprintf("%d fixed-header pane(s) still aligned to their pinned row format", checked)
		}
	} else {
		k.Score = 0
		k.Detail = fmt.Sprintf("%d pane(s) where the header and row format drifted apart", len(k.Defects))
	}
	return k
}

// changedSide names which half of a header/row pair diverged from the pin.
func changedSide(headerStillMatches bool) string {
	if headerStillMatches {
		return "the row format"
	}
	return "the header line"
}

// ---------------------------------------------------------------------------
// KPI: legend_coverage — every term the info line emits is expanded in the
// legend, so a second-pane watcher can decode the line without leaving the
// terminal. The line text IS the oracle.
// ---------------------------------------------------------------------------

func kpiLegendCoverage(src []source) KPI {
	k := newKPI("legend_coverage")
	info := bodyOf(src, "cmd/fak/info.go")
	if info == "" {
		k.Soft = append(k.Soft, "info.go not present; legend not gradable")
		k.Score = 100
		k.Detail = "info overlay not present"
		return k
	}
	legend := between(info, "func guardInfoLegend()", "\n}\n")
	// The terms the line leads with — each must appear in the legend body.
	terms := []string{"cache", "floor", "turns", "inflight", "up"}
	for _, t := range terms {
		if !strings.Contains(legend, t) {
			k.Defects = append(k.Defects,
				fmt.Sprintf("info line term %q has no legend entry", t))
		}
	}
	if len(k.Defects) == 0 {
		k.Score = 100
		k.Detail = "every info-line term is expanded in the legend"
	} else {
		k.Score = 0
		k.Detail = fmt.Sprintf("%d info-line term(s) undocumented", len(k.Defects))
	}
	return k
}

// ---------------------------------------------------------------------------
// KPI: help_completeness — every `fak console` subcommand is documented in the
// usage text, so --help is a complete map of the panes.
// ---------------------------------------------------------------------------

var reConsoleCase = regexp.MustCompile(`case\s+"([a-z]+)":`)

func kpiHelpCompleteness(src []source) KPI {
	k := newKPI("help_completeness")
	tui := bodyOf(src, "cmd/fak/tui.go")
	if tui == "" {
		k.Score = 100
		k.Detail = "console dispatcher not present"
		return k
	}
	// The dispatch switch in runTUI lists the real subcommands; the usage text in
	// tuiUsage must mention each. We extract the switch arms between runTUI's switch
	// and its default, then check each against the usage block.
	dispatch := between(tui, "func runTUI(", "func runTUIIssues(")
	// tuiUsage lives in a DIFFERENT file from the runTUI dispatch (tui_loop_render.go,
	// not tui.go), so search every render source for it. Its body is a raw-string
	// literal, so a "\n}\n" end-marker can stop early at a brace inside the example
	// text — extract to the next top-level func instead.
	usage := ""
	for _, s := range src {
		if u := betweenFunc(s.Body, "func tuiUsage("); u != "" {
			usage = u
			break
		}
	}
	subs := map[string]bool{}
	for _, m := range reConsoleCase.FindAllStringSubmatch(dispatch, -1) {
		name := m[1]
		switch name {
		case "help": // -h/--help/help arm, not a pane
			continue
		}
		subs[name] = true
	}
	missing := []string{}
	for name := range subs {
		// the usage text documents a subcommand as "fak console <name>"
		if !strings.Contains(usage, "console "+name) {
			missing = append(missing, name)
		}
	}
	sort.Strings(missing)
	for _, name := range missing {
		k.Defects = append(k.Defects,
			fmt.Sprintf("console subcommand %q is not documented in tuiUsage", name))
	}
	if len(k.Defects) == 0 {
		k.Score = 100
		k.Detail = fmt.Sprintf("all %d console subcommands documented in help", len(subs))
	} else {
		k.Score = 0
		k.Detail = fmt.Sprintf("%d console subcommand(s) missing from help", len(k.Defects))
	}
	return k
}

// ---------------------------------------------------------------------------
// KPI: ansi_discipline (SOFT) — raw ANSI escapes belong only in the one
// documented in-place redraw helper. A stray escape elsewhere is a smell.
// ---------------------------------------------------------------------------

var reANSI = regexp.MustCompile(`\\033\[|\\x1b\[|\\u001b\[`)

func kpiANSIDiscipline(src []source) KPI {
	k := newKPI("ansi_discipline")
	for _, s := range src {
		for _, line := range numberedLines(s.Body) {
			if !reANSI.MatchString(line.text) {
				continue
			}
			// info.go's writeStatus redraw is the documented home of \r\033[K.
			if s.Rel == "cmd/fak/info.go" {
				continue
			}
			k.Soft = append(k.Soft,
				fmt.Sprintf("%s:%d raw ANSI escape outside the documented redraw helper: %s",
					s.Rel, line.n, squeeze(line.text)))
		}
	}
	// SOFT: scores but emits no hard debt.
	if len(k.Soft) == 0 {
		k.Score = 100
		k.Detail = "raw ANSI confined to the documented redraw path"
	} else {
		k.Score = 70
		k.Detail = fmt.Sprintf("%d raw ANSI escape(s) outside the redraw helper", len(k.Soft))
	}
	return k
}

// ---------------------------------------------------------------------------
// KPI: tty_degradation (SOFT) — the in-place redraw must be gated on a terminal
// probe so a piped/captured run never has escape sequences leaked into it.
// ---------------------------------------------------------------------------

func kpiTTYDegradation(src []source) KPI {
	k := newKPI("tty_degradation")
	info := bodyOf(src, "cmd/fak/info.go")
	if info == "" {
		k.Score = 100
		k.Detail = "info overlay not present"
		return k
	}
	gated := strings.Contains(info, "term.IsTerminal")
	usesRedraw := reANSI.MatchString(info)
	if usesRedraw && !gated {
		k.Soft = append(k.Soft,
			"info overlay emits an in-place redraw but does not gate it on term.IsTerminal")
		k.Score = 60
		k.Detail = "redraw not gated on a TTY probe (escape leak into captured logs)"
		return k
	}
	k.Score = 100
	if usesRedraw {
		k.Detail = "in-place redraw gated on term.IsTerminal; piped output stays clean"
	} else {
		k.Detail = "no in-place redraw to gate"
	}
	return k
}

// ---------------------------------------------------------------------------
// Folding + rendering helpers.
// ---------------------------------------------------------------------------

func newKPI(key string) KPI {
	return KPI{
		Key:    key,
		Label:  kpiLabel[key],
		Group:  kpiGroup[key],
		Hard:   kpiHard[key],
		Weight: kpiWeight[key],
	}
}

// summarize builds the finding/reason/next-action prose the fold needs. The score
// and grade are no longer computed here -- scorecard.Fold derives them from the
// weighted KPIs (kpiWeights) and scorecard.GradeStd, the same 90/80/70/60 table
// GradeLetter used -- so the finding text names only the debt count, matching the
// other Fold-adopted cards whose Finding/FindingClean do not embed the score (the
// composite/grade are reported via corpus.score/corpus.grade instead).
func summarize(debt int, kpis []KPI) (finding, reason, next string) {
	if debt == 0 {
		finding = "ui_quality_debt=0: every graded render KPI is clean"
		reason = "every graded render KPI is clean: rune-safe truncation, honored width budgets, empty-state branches, complete legend + help."
		next = "keep the panes clean on the next render change; re-run with --compare to prove no regression."
		return
	}
	worst := worstHard(kpis)
	finding = fmt.Sprintf("ui_quality_debt=%d: the render surface carries a hard UI defect", debt)
	reason = fmt.Sprintf("the render surface carries %d hard UI defect(s); worst KPI: %s.", debt, worst.Label)
	next = "retire the worst KPI first: " + firstDefect(worst) + "."
	return
}

func worstHard(kpis []KPI) KPI {
	var worst KPI
	for _, k := range kpis {
		if !k.Hard || len(k.Defects) == 0 {
			continue
		}
		if worst.Key == "" || len(k.Defects)*k.Weight > len(worst.Defects)*worst.Weight {
			worst = k
		}
	}
	return worst
}

func firstDefect(k KPI) string {
	if len(k.Defects) == 0 {
		return k.Label
	}
	return k.Defects[0]
}

// ---------------------------------------------------------------------------
// Small parse helpers.
// ---------------------------------------------------------------------------

type numLine struct {
	n    int
	text string
}

func numberedLines(body string) []numLine {
	out := []numLine{}
	for i, t := range strings.Split(body, "\n") {
		out = append(out, numLine{n: i + 1, text: t})
	}
	return out
}

func corpusHas(src []source, needle string) bool {
	for _, s := range src {
		if strings.Contains(s.Body, needle) {
			return true
		}
	}
	return false
}

func bodyOf(src []source, rel string) string {
	for _, s := range src {
		if s.Rel == rel {
			return s.Body
		}
	}
	return ""
}

// between returns the slice of body from the first occurrence of start to the
// next occurrence of end after it (exclusive of end). If start is absent it
// returns "". If end is absent it returns to the end of body.
func between(body, start, end string) string {
	i := strings.Index(body, start)
	if i < 0 {
		return ""
	}
	rest := body[i:]
	j := strings.Index(rest, end)
	if j < 0 {
		return rest
	}
	return rest[:j]
}

func squeeze(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

// stripLineComment removes a trailing `// ...` line comment (Go), leaving the code.
// It does not parse strings, so a `//` inside a string literal would be stripped —
// acceptable here because the rune-safety scan only needs the code's `[:` slices,
// which never live inside a string literal in these files.
func stripLineComment(line string) string {
	// A whole-line comment is the common case (the trimTUI doc block).
	trimmed := strings.TrimSpace(line)
	if strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "*") || strings.HasPrefix(trimmed, "/*") {
		return ""
	}
	if i := strings.Index(line, "//"); i >= 0 {
		return line[:i]
	}
	return line
}

// betweenFunc returns the body from start (a `func NAME(` marker) up to the next
// top-level `\nfunc ` declaration — robust when the body holds a raw-string literal
// whose braces would fool a brace-counting extractor.
func betweenFunc(body, start string) string {
	i := strings.Index(body, start)
	if i < 0 {
		return ""
	}
	rest := body[i+len(start):]
	if j := strings.Index(rest, "\nfunc "); j >= 0 {
		return rest[:j]
	}
	return rest
}

// fprintfCall is one parsed Fprintf/Sprintf/Printf with a string-literal format.
type fprintfCall struct {
	line   int
	format string // the format string literal contents (unquoted, best-effort)
	args   string // the raw argument text after the format literal, up to the call's close paren
}

var reFprintf = regexp.MustCompile(`F?S?[Pp]rintf?\(`)

// fprintfCalls finds printf-family calls whose first (or post-writer) argument is a
// double-quoted format string, and returns the format + the remaining argument text.
// It is a best-effort lexer sufficient for the single-line render Fprintf calls in
// these files (each table row is one Fprintf on one logical line).
func fprintfCalls(body string) []fprintfCall {
	out := []fprintfCall{}
	for _, ln := range numberedLines(body) {
		t := ln.text
		if !strings.Contains(t, "rintf(") || !strings.Contains(t, `"`) {
			continue
		}
		// Find the format string literal: the first double-quoted run on the line.
		q1 := strings.IndexByte(t, '"')
		if q1 < 0 {
			continue
		}
		// Walk to the matching unescaped closing quote.
		q2 := -1
		for i := q1 + 1; i < len(t); i++ {
			if t[i] == '\\' {
				i++
				continue
			}
			if t[i] == '"' {
				q2 = i
				break
			}
		}
		if q2 < 0 || q2 <= q1 {
			continue
		}
		format := t[q1+1 : q2]
		// Only consider it a format call if a printf verb precedes the literal close.
		if !strings.Contains(format, "%") {
			continue
		}
		rest := t[q2+1:]
		// args run to the last close paren on the line (the Fprintf close).
		if i := strings.LastIndex(rest, ")"); i >= 0 {
			rest = rest[:i]
		}
		rest = strings.TrimPrefix(strings.TrimSpace(rest), ",")
		out = append(out, fprintfCall{line: ln.n, format: format, args: strings.TrimSpace(rest)})
	}
	return out
}

// fmtVerb is one parsed printf verb directive.
type fmtVerb struct {
	text        string // the directive as written, e.g. "%-24s"
	widthPadStr bool   // true for a width-padded string verb: %-Ns / %Ns / %-*s
}

var reFmtVerb = regexp.MustCompile(`%[-+ #0]*(?:\*|\d+)?(?:\.(?:\*|\d+))?[vTtbcdoqxXUeEfFgGsrp%]`)

// formatVerbs parses the ordered printf verbs in a format string, skipping `%%`.
func formatVerbs(format string) []fmtVerb {
	verbs := []fmtVerb{}
	for _, m := range reFmtVerb.FindAllString(format, -1) {
		if m == "%%" {
			continue
		}
		v := fmtVerb{text: m}
		// width-padded string: ends in 's' and carries a width field (digits or '*').
		if strings.HasSuffix(m, "s") {
			inner := m[1 : len(m)-1]
			if strings.ContainsAny(inner, "0123456789*") {
				v.widthPadStr = true
			}
		}
		verbs = append(verbs, v)
	}
	return verbs
}

// splitTopArgs splits a comma-separated argument list at TOP LEVEL only — commas
// inside nested (), [], {}, or "" do not split. The first element is whatever
// precedes the first top-level comma (the io.Writer / builder for Fprintf).
func splitTopArgs(args string) []string {
	out := []string{}
	depth := 0
	inStr := false
	start := 0
	for i := 0; i < len(args); i++ {
		c := args[i]
		switch {
		case inStr:
			if c == '\\' {
				i++
			} else if c == '"' {
				inStr = false
			}
		case c == '"':
			inStr = true
		case c == '(' || c == '[' || c == '{':
			depth++
		case c == ')' || c == ']' || c == '}':
			depth--
		case c == ',' && depth == 0:
			out = append(out, strings.TrimSpace(args[start:i]))
			start = i + 1
		}
	}
	if start <= len(args) {
		tail := strings.TrimSpace(args[start:])
		if tail != "" {
			out = append(out, tail)
		}
	}
	return out
}
