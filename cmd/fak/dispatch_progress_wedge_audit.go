package main

// dispatch_progress_wedge_audit.go — `fak dispatch progress --weekly --file-issues`,
// the persistent-lane-wedge lens over the weekly retro (#3338).
//
// The weekly retro (`--weekly`) already COMPUTES lane wedges and top blockers, but it
// files nothing: a lane that stayed blocked all week is rendered once into a markdown
// section nobody diffs, and the same wedge is re-derived (and re-forgotten) the next
// week. This lens turns a PERSISTENT wedge into a fingerprinted, deduped
// improvement-ticket candidate.
//
//	fak dispatch progress --weekly --file-issues            # dry-run: print candidates only
//	fak dispatch progress --weekly --file-issues --confirm  # open a gh issue per NEW wedge
//	fak dispatch progress --weekly --file-issues --confirm --max-issues 3
//
// The detector (foldDispatchWeeklyWedgeFindings) is PURE — report in, deterministic
// findings out, no I/O — so a test drives it hermetically over a retro fixture. The
// filing half reuses the exact detect->dedup->file->mark substrate `fak dispatch audit
// --file-issues` ships (dispatchaudit.SelectFindingsToFile + AlreadyFiled + the shared
// fileAuditFindings gh shell), the same way the sibling session lens (#3336) does, so
// the three feeders never drift on how a candidate becomes an issue. Two load-bearing
// safety choices: --file-issues DEFAULTS to a dry run (it prints and touches nothing;
// only --confirm opens issues, so a routine retro can never storm the tracker), and the
// FILING bar is strictly stronger than the DISPLAY bar — see the thresholds below.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
)

// dispatchWedgeFilingThresholds is the PERSISTENCE bar a rendered wedge must clear
// before it earns a ticket. It is deliberately stricter than dispatchLaneIsWedged (the
// display bar: >=2 attempts and either a sub-50% close rate OR a twice-seen blocker).
// The display bar is tuned to SHOW an operator everything that looks slow this week; a
// filed issue is a durable artifact a human has to close, so it must mean "this lane
// was repeatedly blocked by the SAME thing and did not ship", not "this lane looked
// slow once". Every knob is a floor, and all three must hold at once.
type dispatchWedgeFilingThresholds struct {
	// MinAttempts is how many dispatch attempts the lane needs before a wedge is a
	// pattern rather than an anecdote. Two attempts is noise on a shared trunk.
	MinAttempts int
	// MinDominantFailures is how many times the SAME dominant failure class must recur.
	// This is the "repeatedly blocked" half: it also excludes the synthesized
	// LOW_WITNESSED_CLOSE_RATE fallback wedge (dominant count 0), which names no
	// actionable cause and would file a ticket with nothing in it.
	MinDominantFailures int
	// MaxCloseRate is the witnessed-close rate the lane must stay UNDER. A lane that
	// hits a recurring blocker but still ships is noisy, not wedged.
	MaxCloseRate float64
}

func defaultDispatchWedgeFilingThresholds() dispatchWedgeFilingThresholds {
	return dispatchWedgeFilingThresholds{
		MinAttempts:         3,
		MinDominantFailures: 2,
		MaxCloseRate:        0.5,
	}
}

// dispatchWeeklyWedgeFingerprint is the stable 16-hex hash over a code-site, namespaced
// away from dispatchaudit's outcome fingerprints AND from the session lens's
// "dispatch-sessions-waste/" namespace so the three feeders can never collide on a
// marker in the shared .audit-filed dir.
func dispatchWeeklyWedgeFingerprint(codeSite string) string {
	sum := sha256.Sum256([]byte("dispatch-weekly-wedge/" + codeSite))
	return hex.EncodeToString(sum[:])[:16]
}

// dispatchWeeklyWedgeCodeSite is the stable anchor a fingerprint hashes: the (lane,
// dominant failure class) pair and NOTHING that moves week to week. Attempt counts,
// close rates, and window stamps are deliberately excluded — folding them in would mint
// a fresh fingerprint every run and defeat the dedup the issue asks for, re-filing the
// same wedge every cadence. Same lane blocked by the same class → same fingerprint.
func dispatchWeeklyWedgeCodeSite(lane, failureClass string) string {
	return "lane-wedge/" + lane + "/" + failureClass
}

// dispatchWeeklyWedgeTitle is likewise count-free, because the shared filer dedups
// against OPEN ISSUE TITLES as well as on-disk markers: a title carrying "7 attempts"
// would miss last week's open "5 attempts" issue and file a near-duplicate.
func dispatchWeeklyWedgeTitle(lane, failureClass string) string {
	return "dispatch wedge: " + lane + " lane repeatedly blocked by " + failureClass
}

// foldDispatchWeeklyWedgeFindings is the PURE persistent-wedge detector: it folds an
// already-built weekly retro into deterministic, fingerprinted findings. It walks
// r.LaneWedges in the order the retro already sorted them (dominant-failure count, then
// attempts, then lane), so the returned slice is stable for the same input. A healthy
// week — no wedge clearing the persistence bar — returns nil, which is what makes
// "a healthy week files nothing" true by construction rather than by a caller's guard.
func foldDispatchWeeklyWedgeFindings(r dispatchWeeklyReport, th dispatchWedgeFilingThresholds) []dispatchaudit.Finding {
	var findings []dispatchaudit.Finding
	for _, w := range r.LaneWedges {
		lane := strings.TrimSpace(w.Lane)
		if lane == "" {
			continue
		}
		class := strings.TrimSpace(w.DominantFailureClass)
		if class == "" {
			continue
		}
		if w.Attempts < th.MinAttempts ||
			w.DominantFailureCount < th.MinDominantFailures ||
			w.CloseRate >= th.MaxCloseRate {
			continue
		}
		site := dispatchWeeklyWedgeCodeSite(lane, class)
		findings = append(findings, dispatchaudit.Finding{
			Fingerprint: dispatchWeeklyWedgeFingerprint(site),
			CodeSite:    site,
			Title:       dispatchWeeklyWedgeTitle(lane, class),
			Detail: fmt.Sprintf("Across the retro window %s..%s the %s lane took %d dispatch attempts and witnessed %d close(s) (%.0f%% close rate), with %s as the dominant failure class %d time(s). That is a persistent wedge, not a slow week: the lane keeps hitting the same blocker and is not shipping. Next action: %s.",
				r.WindowStartUTC, r.WindowEndUTC, lane, w.Attempts, w.WitnessedCloses,
				w.CloseRate*100, class, w.DominantFailureCount, w.NextAction),
		})
	}
	return findings
}

// runDispatchWeeklyWedgeAudit runs the wedge lens over an already-built weekly retro. It
// DEFAULTS to a dry run (live=false): it prints the deduped candidates and writes
// nothing — no gh call, no marker — so a routine retro sweep is hermetic and
// side-effect free. Only live=true (--confirm) reaches the shared fileAuditFindings gh
// shell, which opens an issue per NEW fingerprint and drops the marker so it is never
// re-filed.
func runDispatchWeeklyWedgeAudit(stdout, stderr io.Writer, runsDir string, r dispatchWeeklyReport, live bool, maxIssues int) int {
	findings := foldDispatchWeeklyWedgeFindings(r, defaultDispatchWedgeFilingThresholds())
	fmt.Fprintf(stdout, "dispatch weekly wedge audit — %d lane wedge(s) rendered, %d persistent wedge finding(s)\n",
		len(r.LaneWedges), len(findings))
	if len(findings) == 0 {
		fmt.Fprintln(stdout, "no persistent lane wedge detected.")
		return 0
	}

	if live {
		return fileAuditFindings(stdout, stderr, runsDir, dispatchaudit.Report{Findings: findings}, maxIssues)
	}

	// Dry run: dedup against on-disk markers ONLY (no gh call), then print what WOULD
	// be filed. The open-title half of the dedup runs only on the live path, where the
	// gh call is already being paid for.
	filed := map[string]bool{}
	for _, f := range findings {
		if dispatchaudit.AlreadyFiled(runsDir, f.Fingerprint) {
			filed[f.Fingerprint] = true
		}
	}
	fresh, withheld := dispatchaudit.SelectFindingsToFile(findings, filed, map[string]bool{}, maxIssues)
	if len(fresh) == 0 {
		fmt.Fprintln(stdout, "all wedges already filed (deduped by marker); nothing new. Pass --confirm to (re)file.")
		return 0
	}
	fmt.Fprintln(stdout, "\ncandidate improvement tickets (dry-run — pass --confirm to file):")
	for _, f := range fresh {
		fmt.Fprintf(stdout, "  %s  %s\n      %s\n", f.Fingerprint, f.Title, strings.TrimSpace(f.Detail))
	}
	if withheld > 0 {
		fmt.Fprintf(stdout, "(%d more withheld by --max-issues=%d)\n", withheld, maxIssues)
	}
	return 0
}
