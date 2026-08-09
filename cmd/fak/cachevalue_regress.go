package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cvregress"
)

// runCachevalueRegress handles `fak cachevalue regress` (issue #3695, epic #3569): the FIRST
// production consumer of internal/cvregress, the PINNED-baseline per-session cache-efficiency
// (hit% + write-amp) regression fold that landed under #3640.
//
// The fold shipped fully built and unit-tested but with ZERO callers outside its own package --
// exactly the `unwired-debt` #3695 carries. A regression fence nobody can run is not a fence, so
// this subcommand installs it on a real path: --gate turns a REGRESSED verdict into a non-zero
// exit, which is the checkable surface the epic asked for (the CI wiring that CALLS it is F1/F2,
// deliberately out of scope here).
//
// It reads the SAME Track-1 WITNESSED kernel ledger the sibling cachevalue verbs fold, through
// the same --since floor, so the corpus this grades is the corpus `fak cachevalue report` and
// `fak cachevalue shapes` already describe -- one evidence source read a second way, never a
// second source. The baseline is PINNED (not window-relative): that is the whole point of
// cvregress versus the report's median-relative outlier fold, since a self-relative median
// drifts WITH a uniform fleet-wide slide and is blind to it. The three --hit-rate-floor /
// --write-amp-ceiling / --min-prompt-tokens knobs let an operator ratchet the pin tighter than
// cvregress.DefaultBaseline without editing code; nothing here redefines the ledger schema.
func runCachevalueRegress(stdout, stderr io.Writer, argv []string) int {
	const prefix = "fak cachevalue regress"
	fs := flag.NewFlagSet(prefix, flag.ContinueOnError)
	fs.SetOutput(stderr)
	def := cvregress.DefaultBaseline()
	ledger := fs.String("ledger", cachevalueledger.DefaultLedgerRel, "Track-1 WITNESSED kernel ledger to fold (docs/nightrun/cache-value.jsonl)")
	since := fs.String("since", "", "fold only rows on or after this date (YYYY-MM-DD)")
	hitFloor := fs.Float64("hit-rate-floor", def.HitRatePctFloor, "pinned hit-rate floor in percent; a scored session below it is flagged")
	writeAmpCeiling := fs.Float64("write-amp-ceiling", def.WriteAmpCeiling, "pinned write-amplification ceiling; a scored session above it is flagged")
	minPromptTokens := fs.Uint64("min-prompt-tokens", def.MinPromptTokens, "drop sessions below this prompt-token size as noise before scoring")
	asJSON := fs.Bool("json", false, "emit the cvregress.Report as JSON instead of the table")
	gate := fs.Bool("gate", false, "exit 1 when the verdict is REGRESSED (opt-in; default exit stays 0). INSUFFICIENT stays green: a thin corpus is missing evidence, not a regression")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *since != "" {
		if _, err := time.Parse("2006-01-02", *since); err != nil {
			fmt.Fprintf(stderr, "%s: --since must be YYYY-MM-DD: %v\n", prefix, err)
			return 2
		}
	}

	rows := filterTrack1Since(cachevalueledger.ReadLedgerFile(*ledger), *since)
	rep := cvregress.Fold(rows, cvregress.Baseline{
		HitRatePctFloor: *hitFloor,
		WriteAmpCeiling: *writeAmpCeiling,
		MinPromptTokens: *minPromptTokens,
	})

	if *asJSON {
		b, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "%s: marshal: %v\n", prefix, err)
			return 1
		}
		fmt.Fprintln(stdout, string(b))
	} else {
		renderCachevalueRegress(stdout, rep, *ledger, *since)
	}

	// The gate is opt-in so the default render stays a read-only report. Only REGRESSED is a
	// non-zero exit -- rep.OK is already false ONLY for REGRESSED, so INSUFFICIENT (an empty or
	// all-dropped corpus) falls open green, matching the sibling ledger gates' posture.
	if *gate && !rep.OK {
		return 1
	}
	return 0
}

// renderCachevalueRegress prints the operator-facing view: the verdict line, the pin that was
// applied, the corpus that was actually scored, and -- when something tripped -- the flagged
// sessions worst-first with the per-session reason cvregress attached.
func renderCachevalueRegress(w io.Writer, rep cvregress.Report, ledger, since string) {
	fmt.Fprintf(w, "cachevalue regress: %s\n", rep.Finding)
	fmt.Fprintf(w, "baseline (pinned): hit_rate_floor=%.1f%% write_amp_ceiling=%.2f min_prompt_tokens=%d\n",
		rep.Baseline.HitRatePctFloor, rep.Baseline.WriteAmpCeiling, rep.Baseline.MinPromptTokens)
	fmt.Fprintf(w, "corpus: ledger=%s since=%s scored=%d skipped=%d\n",
		ledger, cachevalueEmptyDash(since), rep.Scored, rep.Skipped)
	if len(rep.Regressions) == 0 {
		return
	}
	fmt.Fprintln(w, "\nflagged sessions (worst first):")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  DATE\tSESSION_TYPE\tPROVIDER\tTURNS\tPROMPT\tREUSED\tHIT%\tWRITE_AMP\tREASON")
	for _, s := range rep.Regressions {
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%d\t%d\t%d\t%.1f\t%.2f\t%s\n",
			s.Date, cachevalueEmptyDash(s.SessionType), cachevalueEmptyDash(s.Provider),
			s.Turns, s.PromptTokens, s.ReusedTokens, s.HitRatePct, s.WriteAmp, s.Reason)
	}
	tw.Flush()
}
