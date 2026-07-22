package main

// guard_negframe_summary.go — the `fak guard` per-turn negframe telemetry line and the
// FAK_ABLATE lever that switches the emit-time reframe pass (issue #3568, epic #3538).
//
// Two halves of one A/B seam:
//
//  1. The LEVER. The emit-time positive-voice reframe (#3566, guard_sessionstart.go) is
//     DEFAULT-ON: every fak-authored string injected at SessionStart is routed through
//     negframe's token-superset-safe rewrite. `FAK_ABLATE=negframe_reframe` (or the
//     registry's canonical FAK_ABLATE_NEGFRAME_REFRAME=0, which an `fak ablate --sweep`
//     child carries) restores the RAW negative-framed injection — the control arm the
//     steerability A/B (#3546) measures its treatment against. One env toggle replaces
//     the hand-swapped strings that A/B used to need.
//
//  2. The SIGNAL. Each turn's reframe appends one best-effort row to the workspace
//     negframe journal, and guardNegframeSummaryLine folds those rows into ONE exit-summary
//     line naming the ACTIVE ARM plus the POST-reframe residual negatives and the
//     fail-safe-to-verbatim fallback count. The fallback count is the load-bearing one: a
//     degraded gate that refuses every mechanical candidate shows up here as a visible
//     spike instead of silently shipping unreframed prose that still reads as "reframed".
//
// Best-effort by contract, exactly like guardToolprocSummaryLine and
// guardHookLatencySummaryLine: a missing, unreadable, or malformed journal returns "" and
// the turn proceeds. An observability nicety never fails a turn, and never fails an inject.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ablate"
	"github.com/anthony-chaudhary/fak/internal/negframe"
)

// guardNegframeJournalRel is the workspace-relative per-turn negframe row stream, a sibling
// of guardToolprocJournalRel.
const guardNegframeJournalRel = ".fak/negframe/journal.jsonl"

// guardNegframeAblationToken is the `fak ablate --sweep` token for the reframe lever. It
// doubles as the value a human writes in the coarse FAK_ABLATE=<token,...> list.
var guardNegframeAblationToken = ablate.FeatureNegframeReframe

// guardNegframeEnvVar is the canonical per-feature env the rung-2 subprocess arm carries
// (internal/ablate/registry.go). "1" = reframe on, "0" = the unreframed control arm.
const guardNegframeEnvVar = "FAK_ABLATE_NEGFRAME_REFRAME"

// guardNegframeRow is one turn's reframe telemetry. Arm labels which side of the A/B the
// turn ran on, so the summary line can name it without re-reading the environment.
type guardNegframeRow struct {
	Arm              string `json:"arm"`               // "reframe_on" | "reframe_off"
	Applied          int    `json:"applied"`           // mechanical idioms flipped to their positive inverse
	Residual         int    `json:"residual"`          // POST-reframe negatives left in place
	VerbatimFallback int    `json:"verbatim_fallback"` // candidates refused to protect a must-keep token
}

const (
	guardNegframeArmOn  = "reframe_on"
	guardNegframeArmOff = "reframe_off"
)

// negframeReframeEnabled reports whether the emit-time reframe pass runs this process.
//
// DEFAULT-ON by design: an unset environment REFRAMES (the treatment arm). The pass is
// disabled only when a run EXPLICITLY ablates it, either through the coarse
// FAK_ABLATE=<token,...> list a human types or the canonical per-feature env an ablate
// sweep child carries. Inverting this default would silently flip #3546's arms.
func negframeReframeEnabled() bool {
	// The canonical per-feature env wins when present: the sweep sets it explicitly per arm,
	// so an arm's "1" must survive even if a stale FAK_ABLATE list is also inherited.
	if raw, ok := os.LookupEnv(guardNegframeEnvVar); ok {
		if v := strings.TrimSpace(raw); v != "" {
			if on, err := strconv.ParseBool(v); err == nil {
				return on
			}
		}
	}
	for _, tok := range strings.Split(os.Getenv("FAK_ABLATE"), ",") {
		if strings.EqualFold(strings.TrimSpace(tok), guardNegframeAblationToken) {
			return false
		}
	}
	return true
}

// guardNegframeArm names the arm this process runs, for the row and the summary label.
func guardNegframeArm() string {
	if negframeReframeEnabled() {
		return guardNegframeArmOn
	}
	return guardNegframeArmOff
}

// guardNegframeReframe is the gated emit-time reframe: the single seam guard_sessionstart.go
// calls instead of negframe.ReframeFakOnly directly. On the treatment arm it reframes every
// fak-authored fragment and reports that pass's telemetry; on the control arm it concatenates
// the fragments UNREFRAMED and reports the negatives that consequently ship in place.
//
// Opaque (non-fak-authored) fragments survive byte-for-byte on BOTH arms — the lever switches
// fak's own voice, never an operator's or a user's text.
func guardNegframeReframe(fragments ...negframe.Fragment) (string, guardNegframeRow) {
	row := guardNegframeRow{Arm: guardNegframeArm()}
	reframe := negframeReframeEnabled()
	var b strings.Builder
	for _, f := range fragments {
		if !f.FakAuthored {
			b.WriteString(f.Text)
			continue
		}
		if !reframe {
			// Control arm: ship the raw negative-framed prose, and count what it leaves behind
			// so the two arms report the SAME quantity over the same text.
			b.WriteString(f.Text)
			row.Residual += len(negframe.Classify("", f.Text))
			continue
		}
		res := negframe.ReframePass(f.Text)
		b.WriteString(res.Text)
		row.Applied += res.Applied
		row.Residual += res.ResidualNegatives
		row.VerbatimFallback += res.VerbatimFallback
	}
	return b.String(), row
}

// guardNegframeRecord best-effort APPENDS one turn row to the negframe journal. Every failure
// (unmakeable dir, unopenable file, unmarshalable row) is a silent no-op: telemetry never
// blocks the inject it is measuring.
func guardNegframeRecord(journalPath string, row guardNegframeRow) {
	guardNegframeWrite(journalPath, row, os.O_APPEND)
}

// guardNegframeBegin records the SessionStart row and TRUNCATES the journal first, because
// SessionStart IS the session boundary. Without the reset the journal accumulates across every
// session that ever ran in this workspace and the exit summary would report a lifetime total
// under a per-turn label — the counts #3546 reads would silently drift upward forever.
func guardNegframeBegin(journalPath string, row guardNegframeRow) {
	guardNegframeWrite(journalPath, row, os.O_TRUNC)
}

// guardNegframeWrite is the shared best-effort writer; mode is os.O_APPEND or os.O_TRUNC.
func guardNegframeWrite(journalPath string, row guardNegframeRow, mode int) {
	if err := os.MkdirAll(filepath.Dir(journalPath), 0o755); err != nil {
		return
	}
	data, err := json.Marshal(row)
	if err != nil {
		return
	}
	f, err := os.OpenFile(journalPath, mode|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintln(f, string(data))
}

// guardNegframeSummaryLine reads the workspace negframe journal and folds it into the
// exit-summary line: the ACTIVE ARM, the reframes applied, the POST-reframe residual
// negatives, and the fail-safe-to-verbatim fallback count.
//
// Returns "" when the journal is missing, unreadable, or carries no parseable row — the
// best-effort contract guardToolprocSummaryLine keeps. A malformed line among good ones is
// skipped rather than discarding the turn's real telemetry.
func guardNegframeSummaryLine(journalPath string) string {
	f, err := os.Open(journalPath)
	if err != nil {
		return ""
	}
	defer f.Close()

	var (
		total guardNegframeRow
		rows  int
	)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row guardNegframeRow
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue // skip a malformed row, keep the rest of the turn's telemetry
		}
		rows++
		total.Arm = row.Arm // last row wins: the arm this process actually ran
		total.Applied += row.Applied
		total.Residual += row.Residual
		total.VerbatimFallback += row.VerbatimFallback
	}
	if err := sc.Err(); err != nil {
		return ""
	}
	if rows == 0 {
		return "" // an empty or wholly-malformed journal stays quiet rather than print a vacuous row
	}

	arm := "reframe on (treatment)"
	if total.Arm == guardNegframeArmOff {
		arm = "reframe OFF (control — FAK_ABLATE=" + guardNegframeAblationToken + ")"
	}
	var b strings.Builder
	b.WriteString(guardSection("injected-directive negframe"))
	b.WriteString(guardRow("arm", arm))
	b.WriteString(guardRow("reframes applied", fmt.Sprintf("%d", total.Applied)))
	b.WriteString(guardRow("residual negatives", fmt.Sprintf("%d", total.Residual)))
	b.WriteString(guardRow("verbatim fallbacks", fmt.Sprintf("%d", total.VerbatimFallback)))
	if total.VerbatimFallback > total.Applied {
		// A gate refusing more candidates than it flips is the degradation this line exists
		// to surface: the prose ships unreframed while the arm still reads as "on".
		b.WriteString(guardNote("⚠ fallbacks exceed reframes — the reframe gate is refusing most candidates; injected prose is shipping largely unreframed"))
	}
	b.WriteString(guardNote("the A/B lever for #3546: `FAK_ABLATE=" + guardNegframeAblationToken + "` runs the unreframed control arm; unset reframes"))
	return b.String()
}

// guardNegframeSummary is the exit-summary caller: it folds the workspace journal the
// SessionStart reframe feeds.
func guardNegframeSummary() string { return guardNegframeSummaryLine(guardNegframeJournalRel) }
