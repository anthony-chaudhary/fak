package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/bench"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/tuiplugin"
)

// fak console ablate — the LIVE VISUAL twin of the `fak ablate` table.
//
// `internal/ablate` already computes how each managed-context CACHING CONCEPT pays off
// (replay one frozen trace under N feature configs; record per-arm token-equivalent
// savings split provider-vs-fak, plus each concept's owner/plane/fidelity/evidence/
// status). `printAblation` renders that as a monochrome fixed-width table that crushes
// the whole honesty model into one cramped `feature=status/fidelity/owner` string. This
// pane restores it as a visual: one row per concept with a savings BAR (provider vs fak
// split), fidelity/evidence badges color-coded by trust, a status dot, and a Δ-vs-
// baseline column — redrawing in place under --follow.
//
//	fak console ablate                         # in-process vdso sweep, rendered
//	fak console ablate --sweep vdso,radix,...   # more concepts (env-gated arms re-exec)
//	fak console ablate --report ab.json         # render a `fak ablate --out` artifact
//	fak console ablate --report ab.json --follow # tail a running sweep's --out, live
//
// It reuses the existing TUI primitives (guardInfoRuleTUI / writeGuardInfoFrame /
// the cell-safe width helpers / tuiColorEnabled) — no framework, no new data logic.

const tuiAblateSchema = "fak.tui.ablate.v1"

func init() {
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "ablate",
		Summary: "visualize the managed-context caching-concept ablation: per-concept savings bars, fidelity/evidence badges, Δ-vs-baseline",
		Usage:   "fak console ablate [--sweep vdso,...] [--report FILE] [--follow] [--width N] [--color auto|always|never] [--json]",
		Schema:  tuiAblateSchema,
		BuiltIn: true,
		Controls: []tuiplugin.Control{
			{ID: "baseline", Label: "Baseline", Kind: "input", Flag: "--baseline", Default: "all-off", Detail: "arm id used as the delta reference"},
			{ID: "color", Label: "Color", Kind: "flag", Flag: "--color", Default: "auto", Options: []string{"auto", "always", "never"}, Detail: "auto, always, or never; NO_COLOR disables color"},
			{ID: "follow", Label: "Follow", Kind: "toggle", Flag: "--follow", Detail: "redraw in place every 500ms (Ctrl-C to stop); pairs with --report to tail a running sweep"},
			{ID: "json", Label: "JSON", Kind: "toggle", Flag: "--json", Detail: "emit the typed ablate pane model"},
			{ID: "report", Label: "Report", Kind: "input", Flag: "--report", Detail: "render a saved AblationReport JSON (a fak ablate --out artifact)"},
			{ID: "sweep", Label: "Sweep", Kind: "input", Flag: "--sweep", Default: "vdso", Detail: "comma list of features to ablate when --report is not given"},
			{ID: "suite", Label: "Suite", Kind: "input", Flag: "--suite", Default: "tau2-smoke", Detail: "trace suite for the in-process sweep"},
		},
		Run: runTUIAblate,
	})
}

func runTUIAblate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tui ablate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	report := fs.String("report", "", "render a saved AblationReport JSON (a `fak ablate --out` artifact) instead of running a sweep")
	suite := fs.String("suite", "tau2-smoke", "trace suite for the in-process sweep (when --report is not given)")
	tracePath := fs.String("trace", "", "explicit trace path for the sweep (overrides --suite)")
	sweep := fs.String("sweep", "vdso", "comma list of features to sweep (known: "+strings.Join(ablate.KnownFeatures(), ",")+"); env-gated arms re-exec, so keep the interactive default light")
	baseline := fs.String("baseline", "all-off", "arm id used as the delta reference")
	width := fs.Int("width", 120, "target terminal width for human rendering")
	colorMode := fs.String("color", "auto", "colorize human output: auto, always, or never (NO_COLOR disables color)")
	asJSON := fs.Bool("json", false, "emit the typed ablate pane model as JSON")
	follow := fs.Bool("follow", false, "redraw the pane in place every 500ms (Ctrl-C to stop); with --report it re-reads the file, tailing a running sweep's --out")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak console ablate: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *width < 80 {
		*width = 80
	}
	color, err := tuiColorEnabled(stdout, *colorMode)
	if err != nil {
		fmt.Fprintf(stderr, "fak console ablate: %v\n", err)
		return 2
	}

	var load func() (*ablate.Report, error)
	if *report != "" {
		path := *report
		load = func() (*ablate.Report, error) { return loadAblateReport(path) }
	} else {
		features := splitCommaList(*sweep)
		load = func() (*ablate.Report, error) {
			rep, dropped, err := sweepAblate(features, *tracePath, *suite, *baseline)
			for _, d := range dropped {
				fmt.Fprintf(stderr, "fak console ablate: arm %q dropped: %s\n", d.ArmID, droppedArmSummary(d))
			}
			return rep, err
		}
	}

	if *asJSON {
		rep, err := load()
		if err != nil {
			fmt.Fprintln(stderr, "fak console ablate:", err)
			return 1
		}
		return encodeJSONOrFail(stdout, stderr, buildAblateView(rep), "fak console ablate")
	}
	if *follow {
		return followAblate(stdout, stderr, load, *width, color)
	}
	rep, err := load()
	if err != nil {
		fmt.Fprintln(stderr, "fak console ablate:", err)
		return 1
	}
	fmt.Fprint(stdout, renderAblateView(buildAblateView(rep), *width, color))
	return 0
}

// loadAblateReport reads a `fak ablate --out` artifact and re-checks the identical-
// workload guard, so a hand-edited or truncated report is refused rather than rendered.
func loadAblateReport(path string) (*ablate.Report, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rep ablate.Report
	if err := json.Unmarshal(data, &rep); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	if err := rep.Validate(); err != nil {
		return nil, err
	}
	return &rep, nil
}

// sweepAblate runs the ablation sweep and returns its report, mirroring the rung routing
// of the `fak ablate` verb: a vdso-only (in-process) sweep flips the runtime knob with no
// spawn; any env-gated feature re-execs one child per arm (rung 2). Returns the dropped
// arms so the caller can surface each subprocess hole rather than silently gapping.
func sweepAblate(features []string, tracePath, suite, baseline string) (*ablate.Report, []ablate.DroppedArm, error) {
	if len(features) == 0 {
		return nil, nil, errors.New("no features to sweep (try --sweep " + strings.Join(ablate.KnownFeatures(), ",") + ")")
	}
	configs, err := ablate.BuildSweep(features)
	if err != nil {
		return nil, nil, err
	}
	path := tracePath
	if path == "" {
		path = resolveSuite(traceDir(), suite)
	}
	t, err := bench.LoadTrace(path)
	if err != nil {
		return nil, nil, err
	}
	const engineID, engineModel = "mock", "mock-offline"
	if anyEnvGated(features) {
		bin, err := os.Executable()
		if err != nil {
			return nil, nil, fmt.Errorf("resolve fak binary for arm re-exec: %w", err)
		}
		return ablate.SweepViaSubprocess(ctx(), bin, t, engineID, engineModel, configs, baseline, ablateArmRunner)
	}
	rep, err := ablate.Sweep(ctx(), t, engineID, engineModel, configs, baseline)
	return rep, nil, err
}

// ---------------------------------------------------------------------------
// view model
// ---------------------------------------------------------------------------

// ablateView is the typed, JSON-safe projection of an ablate.Report the pane renders and
// --json emits, so rendering is a pure function of the view. RefreshedAt is set only by
// the follow loop (empty for a one-shot render, keeping the golden deterministic).
type ablateView struct {
	Schema       string          `json:"schema"`
	SliceID      string          `json:"slice_id"`
	EngineModel  string          `json:"engine_model"`
	WorkloadHash string          `json:"workload_hash"`
	Baseline     string          `json:"baseline_arm"`
	MaxTokEq     float64         `json:"max_token_equiv"`
	Rows         []ablateViewRow `json:"rows"`
	Caveats      []string        `json:"caveats,omitempty"`
	RefreshedAt  string          `json:"refreshed_at,omitempty"`
}

type ablateViewRow struct {
	Concept       string  `json:"concept"`
	Features      string  `json:"features"`
	IsBaseline    bool    `json:"is_baseline"`
	Fidelity      string  `json:"fidelity,omitempty"`
	Evidence      string  `json:"evidence,omitempty"`
	Owner         string  `json:"owner,omitempty"`
	Status        string  `json:"status,omitempty"`
	ProviderTokEq float64 `json:"provider_token_equiv"`
	FakTokEq      float64 `json:"fak_token_equiv"`
	TotalTokEq    float64 `json:"total_token_equiv"`
	DeltaTokens   int64   `json:"delta_tokens"`
	VDSOHits      int64   `json:"vdso_hits"`
}

// buildAblateView projects the report into the view. It is pure: no time, no I/O. The
// per-concept badges come from the CacheEffect that matches the arm's toggled feature; the
// baseline row (all-off) carries none, so it renders as "—". MaxTokEq floors at 1 so a
// tiny saving draws a tiny bar rather than dividing by zero.
func buildAblateView(rep *ablate.Report) ablateView {
	v := ablateView{
		Schema:       tuiAblateSchema,
		SliceID:      rep.Provenance.SliceID,
		EngineModel:  rep.Provenance.EngineModel,
		WorkloadHash: rep.WorkloadHash,
		Baseline:     rep.Baseline,
		Caveats:      rep.Caveats,
		MaxTokEq:     1,
	}
	var baseTokens int64
	if base := rep.ArmByID(rep.Baseline); base != nil {
		baseTokens = base.Tokens()
	}
	for i := range rep.Runs {
		r := &rep.Runs[i]
		row := ablateViewRow{
			Concept:       r.ArmID,
			Features:      featStr(r.Features),
			IsBaseline:    r.ArmID == rep.Baseline,
			ProviderTokEq: r.ProviderTokenEquiv(),
			FakTokEq:      r.FakTokenEquiv(),
			TotalTokEq:    r.TotalTokenEquiv(),
			DeltaTokens:   r.Tokens() - baseTokens,
			VDSOHits:      r.Arm.VDSOHits,
		}
		if !row.IsBaseline {
			if e, ok := primaryAblateEffect(*r); ok {
				row.Fidelity, row.Evidence, row.Owner, row.Status = e.Fidelity, e.Evidence, e.Owner, e.Status
			}
			if t := r.TotalTokenEquiv(); t > v.MaxTokEq {
				v.MaxTokEq = t
			}
		}
		v.Rows = append(v.Rows, row)
	}
	return v
}

// primaryAblateEffect picks the CacheEffect that represents an arm's concept: the effect
// whose feature/component matches the arm id (BuildSweep names each single-feature arm
// after its feature), else the first active effect, else the first effect.
func primaryAblateEffect(run ablate.AblationRun) (ablate.CacheEffect, bool) {
	if len(run.CacheEffects) == 0 {
		return ablate.CacheEffect{}, false
	}
	for _, e := range run.CacheEffects {
		if strings.EqualFold(e.Feature, run.ArmID) || strings.EqualFold(e.Component, run.ArmID) {
			return e, true
		}
	}
	for _, e := range run.CacheEffects {
		if strings.EqualFold(e.Status, "active") {
			return e, true
		}
	}
	return run.CacheEffects[0], true
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

// renderAblateView renders the view as the terminal dashboard. Every line is cell-width
// capped (takeCellsTUI) so nothing wraps; color is applied only to already-padded
// segments (never re-measured), so ANSI escapes can never desync the column math. With
// color off (NO_COLOR / non-tty) the same layout renders in plain ASCII/block glyphs.
func renderAblateView(v ablateView, width int, color bool) string {
	barW := width - 70
	if barW < 0 {
		barW = 0
	}
	if barW > 30 {
		barW = 30
	}
	var b strings.Builder
	for _, h := range ablateHeaderLines(v, width) {
		b.WriteString(h)
		b.WriteByte('\n')
	}
	b.WriteString(ablatePaint(guardInfoRuleTUI("concepts", width), tuiSGRDim, color))
	b.WriteByte('\n')
	b.WriteString(ablatePaint(ablateColHeader(barW, width), tuiSGRDim, color))
	b.WriteByte('\n')
	for _, row := range v.Rows {
		emitAblateLine(&b, ablateRowSegs(row, v.MaxTokEq, barW), width, color)
	}
	if v.MaxTokEq <= 1 {
		// No arm produced a token-equiv saving, so every bar is empty — most often a
		// vdso-only sweep (whose win is avoided CALLS, in Δtokens, not prompt-cache $).
		// Point the reader at the sweep/report that fills the bars rather than leaving a
		// blank chart that looks broken.
		b.WriteString(ablatePaint(trimTUI("· no token-equiv savings in this sweep — vdso saves avoided CALLS (Δtokens), not prompt-cache $; try --sweep vdso,compressor,ttl_1h or --report a guarded-session sweep to fill the bars", width), tuiSGRDim, color))
		b.WriteByte('\n')
	}
	b.WriteString(ablatePaint(guardInfoRuleTUI("legend", width), tuiSGRDim, color))
	b.WriteByte('\n')
	b.WriteString(ablatePaint(ablateLegend(width), tuiSGRDim, color))
	b.WriteByte('\n')
	for _, c := range v.Caveats {
		b.WriteString(ablatePaint(trimTUI("! "+c, width), tuiSGRYellowBold, color))
		b.WriteByte('\n')
	}
	b.WriteString(ablatePaint(ablateFooter(v, width), tuiSGRDim, color))
	b.WriteByte('\n')
	return b.String()
}

func ablateHeaderLines(v ablateView, width int) []string {
	title := "fak · managed context — caching-concept ablation"
	l1 := ablateJustify(title, "engine "+ablateOrQ(v.EngineModel), width)
	hash := v.WorkloadHash
	if len(hash) > 8 {
		hash = hash[:8]
	}
	l2 := fmt.Sprintf("trace %s   workload#%s   arms %d   baseline %s   (same trace each arm → deltas apples-to-apples)",
		ablateOrQ(v.SliceID), ablateOrQ(hash), len(v.Rows), ablateOrQ(v.Baseline))
	return []string{trimTUI(l1, width), trimTUI(l2, width)}
}

// ablateColHeader labels the concept columns, matching the field widths of ablateRowSegs.
func ablateColHeader(barW, width int) string {
	line := padRightTUI("concept", 12) + " " +
		padRightTUI("fidelity", 12) + " " +
		padRightTUI("evidence", 11) + " "
	if barW > 0 {
		line += padRightTUI(trimTUI("saved · provider+fak", barW), barW) + " "
	}
	line += ablatePadLeft("tok-eq", 10) + " " + ablatePadLeft("Δtokens", 9) + " status"
	return takeCellsTUI(line, width)
}

func ablateLegend(width int) string {
	return trimTUI("█ provider-cache save   ▓ fak-authored save   ░ headroom   "+
		"fidelity lossless>recoverable>lossy>passive   evidence witnessed>observed>configured>simulated", width)
}

func ablateFooter(v ablateView, width int) string {
	if v.RefreshedAt != "" {
		return trimTUI("live · refreshed "+v.RefreshedAt+" · every 500ms · Ctrl-C to stop", width)
	}
	return trimTUI("q quit · --json model · --report FILE renders a saved sweep · --follow live", width)
}

// ablateSeg is one already-padded (plain-width-known) render cell plus the SGR to paint it.
type ablateSeg struct{ text, sgr string }

// emitAblateLine writes one row. It measures the PLAIN concatenation first; if it exceeds
// width it truncates plain (dropping color for that line), otherwise it paints each segment
// — so a colored line's visible width always equals the plain width, never over-budget.
func emitAblateLine(b *strings.Builder, segs []ablateSeg, width int, color bool) {
	var plain strings.Builder
	for _, s := range segs {
		plain.WriteString(s.text)
	}
	p := plain.String()
	if dispWidthTUI(p) > width {
		b.WriteString(takeCellsTUI(p, width))
		b.WriteByte('\n')
		return
	}
	if !color {
		b.WriteString(p)
		b.WriteByte('\n')
		return
	}
	for _, s := range segs {
		b.WriteString(ablatePaint(s.text, s.sgr, true))
	}
	b.WriteByte('\n')
}

func ablateRowSegs(row ablateViewRow, maxEq float64, barW int) []ablateSeg {
	concept := padRightTUI(trimTUI(row.Concept, 12), 12)
	fid := padRightTUI(trimTUI(ablateBadge(row.Fidelity), 12), 12)
	ev := padRightTUI(trimTUI(ablateBadge(row.Evidence), 11), 11)
	tokeq := ablatePadLeft(ablateSignedTokEq(row.TotalTokEq), 10)
	delta := ablatePadLeft(ablateSignedDelta(row.DeltaTokens), 9)

	var status string
	if strings.TrimSpace(row.Status) == "" {
		status = padRightTUI("—", 10)
	} else {
		status = padRightTUI(trimTUI(ablateStatusGlyph(row.Status)+" "+row.Status, 10), 10)
	}

	segs := []ablateSeg{
		{concept, ""},
		{" ", ""},
		{fid, ablateFidelitySGR(row.Fidelity)},
		{" ", ""},
		{ev, ablateEvidenceSGR(row.Evidence)},
		{" ", ""},
	}
	if barW > 0 {
		prov, fak, empty := ablateBarCells(row.ProviderTokEq, row.FakTokEq, maxEq, barW)
		segs = append(segs,
			ablateSeg{strings.Repeat("█", prov), tuiSGRCyanBold},
			ablateSeg{strings.Repeat("▓", fak), tuiSGRGreen},
			ablateSeg{strings.Repeat("░", empty), tuiSGRDim},
			ablateSeg{" ", ""},
		)
	}
	segs = append(segs,
		ablateSeg{tokeq, ablateTokEqSGR(row.TotalTokEq)},
		ablateSeg{" ", ""},
		ablateSeg{delta, ablateDeltaSGR(row.DeltaTokens)},
		ablateSeg{" ", ""},
		ablateSeg{status, ablateStatusSGR(row.Status)},
	)
	return segs
}

// ablateBarCells splits a barW-cell bar into provider / fak / empty cells. The fill is the
// arm's total token-equiv saving scaled to the sweep's max; within it, provider and fak get
// cells in proportion to their POSITIVE contribution (a negative provider effect — a cold
// write costing more than it saves — contributes no cells rather than a phantom bar).
func ablateBarCells(provEq, fakEq, maxEq float64, barW int) (prov, fak, empty int) {
	if barW <= 0 {
		return 0, 0, 0
	}
	total := provEq + fakEq
	fill := 0
	if maxEq > 0 && total > 0 {
		fill = int(total/maxEq*float64(barW) + 0.5)
	}
	if fill > barW {
		fill = barW
	}
	provPos, fakPos := provEq, fakEq
	if provPos < 0 {
		provPos = 0
	}
	if fakPos < 0 {
		fakPos = 0
	}
	sum := provPos + fakPos
	if sum > 0 && fill > 0 {
		prov = int(float64(fill)*provPos/sum + 0.5)
	}
	if prov > fill {
		prov = fill
	}
	return prov, fill - prov, barW - fill
}

// ---------------------------------------------------------------------------
// follow (live in-place redraw)
// ---------------------------------------------------------------------------

// followAblate redraws the pane in place on a 500ms ticker until Ctrl-C, reusing the
// guard --follow shape (signal.NotifyContext + writeGuardInfoFrame). The core loop takes
// an explicit context + interval so it is deterministically testable without a signal.
func followAblate(stdout, stderr io.Writer, load func() (*ablate.Report, error), width int, color bool) int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	return followAblateLoop(ctx, stdout, stderr, load, width, color, 500*time.Millisecond)
}

func followAblateLoop(ctx context.Context, stdout, stderr io.Writer, load func() (*ablate.Report, error), width int, color bool, interval time.Duration) int {
	fmt.Fprintf(stdout, "fak console ablate · live caching-concept ablation · every %s · Ctrl-C to stop\n", interval)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	prevRows := 0
	paint := func() {
		var block string
		if rep, err := load(); err != nil {
			block = "fak console ablate: " + err.Error()
		} else {
			v := buildAblateView(rep)
			v.RefreshedAt = time.Now().Format("15:04:05")
			block = strings.TrimRight(renderAblateView(v, width, color), "\n")
		}
		prevRows = writeGuardInfoFrame(stdout, block, prevRows)
	}
	paint()
	for {
		select {
		case <-ctx.Done():
			fmt.Fprintln(stdout) // release the parked cursor with a trailing newline
			return 0
		case <-ticker.C:
			paint()
		}
	}
}

// ---------------------------------------------------------------------------
// small formatting + color helpers
// ---------------------------------------------------------------------------

func ablatePaint(s, sgr string, color bool) string {
	if !color || sgr == "" || strings.TrimSpace(s) == "" {
		return s
	}
	return sgr + s + tuiSGRReset
}

func ablateBadge(s string) string {
	if strings.TrimSpace(s) == "" {
		return "—"
	}
	return s
}

func ablateOrQ(s string) string {
	if strings.TrimSpace(s) == "" {
		return "?"
	}
	return s
}

func ablateJustify(left, right string, width int) string {
	lw, rw := dispWidthTUI(left), dispWidthTUI(right)
	if lw+1+rw > width {
		return trimTUI(left, width)
	}
	return left + strings.Repeat(" ", width-lw-rw) + right
}

// ablatePadLeft right-justifies s into a field of width display cells (rune-aware, the
// mirror of padRightTUI). Callers keep s within width, so an over-wide s is returned as-is.
func ablatePadLeft(s string, width int) string {
	gap := width - dispWidthTUI(s)
	if gap <= 0 {
		return s
	}
	return strings.Repeat(" ", gap) + s
}

func ablateSignedTokEq(v float64) string {
	if v > 0 {
		return "+" + gateway.HumanTokenEquiv(v)
	}
	return gateway.HumanTokenEquiv(v)
}

func ablateSignedDelta(n int64) string {
	if n > 0 {
		return "+" + ablateComma(n)
	}
	return ablateComma(n)
}

// ablateComma formats an int64 with thousands separators (Go has no built-in).
func ablateComma(n int64) string {
	neg := n < 0
	if neg {
		n = -n
	}
	s := strconv.FormatInt(n, 10)
	var parts []string
	for len(s) > 3 {
		parts = append([]string{s[len(s)-3:]}, parts...)
		s = s[:len(s)-3]
	}
	parts = append([]string{s}, parts...)
	out := strings.Join(parts, ",")
	if neg {
		return "-" + out
	}
	return out
}

func ablateStatusGlyph(status string) string {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "active":
		return "●"
	case "no-op":
		return "·"
	case "inactive":
		return "○"
	case "unavailable":
		return "⊘"
	default:
		return "•"
	}
}

func ablateFidelitySGR(fidelity string) string {
	switch strings.ToLower(strings.TrimSpace(fidelity)) {
	case "lossless":
		return tuiSGRGreen
	case "recoverable":
		return tuiSGRCyanBold
	case "lossy":
		return tuiSGRYellowBold
	case "passive":
		return tuiSGRBlueBold
	default:
		return tuiSGRDim
	}
}

func ablateEvidenceSGR(evidence string) string {
	switch strings.ToLower(strings.TrimSpace(evidence)) {
	case "witnessed":
		return tuiSGRGreenBold
	case "observed":
		return "" // relayed, but real — plain
	default:
		return tuiSGRDim // configured / simulated / unknown — visibly less solid
	}
}

func ablateStatusSGR(status string) string {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "active":
		return tuiSGRGreen
	case "unavailable":
		return tuiSGRYellowBold
	default:
		return tuiSGRDim
	}
}

func ablateTokEqSGR(v float64) string {
	if v > 0 {
		return tuiSGRGreenBold
	}
	return tuiSGRDim
}

func ablateDeltaSGR(n int64) string {
	switch {
	case n < 0:
		return tuiSGRGreen // fewer tokens re-read is the win
	case n > 0:
		return tuiSGRYellowBold
	default:
		return tuiSGRDim
	}
}
