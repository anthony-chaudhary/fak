package main

// superloop_drive.go — the impure shell for `fak superloop drive <intent>` (issue
// #2224). It is the DRIVE half over the pure WALK: WALK the members (same reads as
// `walk`), SELECT the single worst-first member (superloop.Drive), pass it through the
// SAME admission gate any spawn passes — region admission over the live lease fabric
// (COLLISION_RISK on lease overlap) — record the admission witness on the loop ledger,
// and RE-FOLD as the exit check.
//
// The super loop keeps its interior-node property: it mutates nothing at its own
// altitude. The single ACTION this invocation takes is surfacing the member's OWN front
// door (a loop member via `fak loop drive` / the dispatch tick; a scorecard member via
// its enter hint), reached through the shared region hold — the super loop gets NO
// private spawn path. Live execution of the member's child behind that front door is the
// named follow-on; this rung admits one member under a lease, surfaces its single action,
// and re-folds. A driven-but-unwitnessed member keeps the re-fold unsatisfied, so it can
// never satisfy the intent.

import (
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopdrive"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// superloopDriveReport is the folded drive verdict: the single member selected
// worst-first, how the shared admission gate ruled, and the re-fold that is the exit
// check. Outcome is one of "satisfied" (nothing to enter), "entered" (one member
// admitted under a lease + re-folded), or "refused" (the admission gate refused; the
// token is surfaced, never bypassed).
type superloopDriveReport struct {
	Schema    string                  `json:"schema"`
	Intent    string                  `json:"intent"`
	Outcome   string                  `json:"outcome"`
	Decision  superloop.DriveDecision `json:"decision"`
	Admission *superloopDriveAdmit    `json:"admission,omitempty"`
	Refold    *superloop.WalkReport   `json:"refold,omitempty"`
}

// superloopDriveAdmit records how the driven member fared at the shared admission gate.
// Status is "ADMITTED", "UNCOORDINATED" (no region declared), a closed refusal token
// (e.g. "COLLISION_RISK"), or "ADMIT_UNAVAILABLE" (infra fail-open).
type superloopDriveAdmit struct {
	Lane     string   `json:"lane,omitempty"`
	Tree     []string `json:"tree,omitempty"`
	Lease    string   `json:"lease,omitempty"`
	Status   string   `json:"status"`
	Admitted bool     `json:"admitted"`
	Detail   string   `json:"detail,omitempty"`
}

// superloopDriveAdmitGate is the admission seam: it passes the drive's one member
// through the SAME region admission `fak loop drive` and the dispatch tick pass, and
// returns the verdict plus a release closure held while the member is entered. It is a
// package var so a test forces ADMITTED / a refusal token without the live lease fabric.
var superloopDriveAdmitGate = superloopDriveRegionAdmit

// superloopDriveRegionAdmit arms region admission over the live lease fabric via the
// SAME loop-drive region hold — no private path. With no --lane/--tree it is the
// historical uncoordinated enter (admitted without a coordinating lease); with a region
// it refuses over a live overlapping lease (COLLISION_RISK) and otherwise holds a fenced
// lease while the member is entered. An infra error fails OPEN with a warning (the
// dispatch tick's posture), never a silent clean admission.
func superloopDriveRegionAdmit(root, lane string, tree []string, intent string) (superloopDriveAdmit, func()) {
	admit := superloopDriveAdmit{Lane: lane, Tree: tree}
	hold := newLoopDriveRegionHold(loopDriveOptions{Lane: lane, Region: tree}, loopdrive.Spec{Loop: "superloop-" + intent})
	if hold == nil {
		admit.Status = "UNCOORDINATED"
		admit.Admitted = true
		admit.Detail = "no --lane/--tree declared: entered without a region lease (declare a region to arm COLLISION_RISK on lease overlap)"
		return admit, func() {}
	}
	admit.Lease = hold.id
	refuse, err := hold.ensure(time.Now())
	switch {
	case err != nil:
		admit.Status = "ADMIT_UNAVAILABLE"
		admit.Admitted = true // fail open, but surface the infra gap
		admit.Detail = "region admission unavailable (fail-open): " + err.Error()
		return admit, func() {}
	case refuse != nil:
		admit.Status = refuse.Reason
		admit.Admitted = false
		admit.Detail = refuse.Detail
		return admit, func() {}
	default:
		admit.Status = "ADMITTED"
		admit.Admitted = true
		admit.Detail = "region lease " + hold.id + " held while the member is entered"
		return admit, hold.release
	}
}

func runSuperloopDrive(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop drive", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the drive report as JSON")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	lane := fs.String("lane", "", "dos.toml lane the driven member's writes stay inside; arms region admission (COLLISION_RISK on lease overlap) against the live lease fabric")
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger the drive records its admission witness to")
	var tree repeatedString
	fs.Var(&tree, "tree", "region glob the driven member's writes stay inside (repeatable); arms region admission against the live lease fabric")
	if err := fs.Parse(superloopInterspersedFlagArgs(argv, map[string]bool{"workspace": true, "lane": true, "ledger": true, "tree": true})); err != nil {
		return 2
	}
	name := fs.Arg(0)
	s, ok := superloop.Lookup(name)
	if !ok {
		fmt.Fprintf(stderr, "fak superloop drive: unknown super loop %q (try `fak superloop list`)\n", name)
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	// WALK — read every member's status from the cheap committed surfaces, then SELECT
	// the single worst-first member (one member per walk). The walk mutates nothing.
	// A declared issue-target folds its live progress in here (surface-only until a
	// dispatch ledger exists), so the headline gates the drive, not just decorates it.
	rep := superloop.Walk(s, collectSuperloopStatuses(root, s), issueProgressWalkOpts(root, s)...)
	decision := superloop.Drive(rep)
	report := superloopDriveReport{Schema: superloop.DriveSchema, Intent: s.Name, Decision: decision}

	if !decision.Enter {
		// A satisfied walk: nothing worst-first to enter, so the drive enters nothing
		// and reports the intent already at floor.
		report.Outcome = "satisfied"
		return finishSuperloopDrive(stdout, stderr, *asJSON, report, 0)
	}

	// ADMISSION — the one member passes the SAME gate any spawn passes. The super loop
	// reuses the loop-drive region hold, so it gets NO private spawn path.
	admit, release := superloopDriveAdmitGate(root, *lane, tree, s.Name)
	defer release()
	report.Admission = &admit

	if !admit.Admitted {
		// REFUSED — surface the token, record the refusal with the standing witness
		// vocabulary, and DO NOT enter (no bypass of the gate).
		report.Outcome = "refused"
		recordSuperloopDriveAdmit(*ledger, s.Name, decision, loopmgr.StatusRefused, admit.Status, admit.Detail, admit.Lease)
		fmt.Fprintf(stderr, "fak superloop drive: refused by admission gate: %s %s\n", admit.Status, admit.Detail)
		return finishSuperloopDrive(stdout, stderr, *asJSON, report, 3)
	}

	// ENTER — the single ACTION taken this invocation is the member's own front door
	// (surfaced). The super loop mutates nothing at its own altitude: it records the
	// admission witness; the member's own machinery (and its own witnessed_done) runs
	// behind that front door.
	recordSuperloopDriveAdmit(*ledger, s.Name, decision, loopmgr.StatusAdmitted, "ENTERED",
		"entered "+string(decision.Member.Kind)+" "+decision.Member.Ref+": "+decision.Action, admit.Lease)
	report.Outcome = "entered"

	// RE-FOLD — re-walk and fold; the aggregate re-fold after the member run is the exit
	// check. A driven-but-unwitnessed member (unmeasured/dark) keeps the re-fold
	// unsatisfied, so it can never satisfy the intent — the exit reflects that honestly.
	// The issue-target gate re-folds too: a member run that progressed issues moves the
	// live count, and an unmet headline keeps the re-fold unsatisfied.
	refold := superloop.Walk(s, collectSuperloopStatuses(root, s), issueProgressWalkOpts(root, s)...)
	report.Refold = &refold
	code := 1
	if refold.Satisfied {
		code = 0
	}
	return finishSuperloopDrive(stdout, stderr, *asJSON, report, code)
}

// recordSuperloopDriveAdmit appends the drive's admission decision to the loop ledger
// with the standing witness vocabulary (StatusAdmitted / StatusRefused), keyed on the
// super loop's own id and carrying the driven member and — when a region lease was
// consulted — the held lease as evidence. Best-effort: the drive's success is the enter
// + re-fold, not this observability row.
func recordSuperloopDriveAdmit(ledger, intent string, d superloop.DriveDecision, status loopmgr.RunStatus, reason, detail, leaseID string) {
	if strings.TrimSpace(ledger) == "" {
		return
	}
	ev := loopmgr.Event{
		LoopID:  "superloop-" + intent,
		Kind:    loopmgr.EventAdmit,
		Source:  "superloop-drive",
		Status:  status,
		Reason:  reason,
		Summary: detail,
		EvidenceRefs: []loopmgr.EvidenceRef{
			{Kind: "superloop", Ref: intent},
			{Kind: "member", Ref: string(d.Member.Kind) + ":" + d.Member.Ref},
		},
	}
	if leaseID != "" {
		ev.EvidenceRefs = append(ev.EvidenceRefs, loopmgr.EvidenceRef{Kind: "region_lease", Ref: leaseID})
	}
	_ = appendLoopRunEvent(ledger, ev)
}

func finishSuperloopDrive(stdout, stderr io.Writer, asJSON bool, report superloopDriveReport, code int) int {
	if asJSON {
		if rc := encodeJSONOrFail(stdout, stderr, report, "fak superloop drive"); rc != 0 {
			return rc
		}
		return code
	}
	renderSuperloopDrive(stdout, report)
	return code
}

func renderSuperloopDrive(w io.Writer, r superloopDriveReport) {
	fmt.Fprintf(w, "superloop drive: %s\n", r.Intent)
	if r.Outcome == "satisfied" {
		fmt.Fprintf(w, "  nothing to enter — %s\n", r.Decision.Reason)
		return
	}
	d := r.Decision
	dark := ""
	if d.Dark {
		dark = ", DARK"
	}
	fmt.Fprintf(w, "  worst-first: enter %s %s (rank %d, debt %d%s)\n", d.Member.Kind, d.Member.Ref, d.Rank, d.Debt, dark)
	fmt.Fprintf(w, "  action: %s\n", d.Action)
	if a := r.Admission; a != nil {
		fmt.Fprintf(w, "  admission: %s", a.Status)
		if a.Lease != "" {
			fmt.Fprintf(w, " (lease %s)", a.Lease)
		}
		if a.Detail != "" {
			fmt.Fprintf(w, " — %s", a.Detail)
		}
		fmt.Fprintln(w)
	}
	if r.Outcome == "refused" {
		fmt.Fprintf(w, "  → refused: the member's admission gate refused; the token is surfaced, not bypassed\n")
		return
	}
	if r.Refold != nil {
		sat := "not yet"
		if r.Refold.Satisfied {
			sat = "SATISFIED"
		}
		fmt.Fprintf(w, "  re-fold: %s — %s (debt %d, floor %d, unmeasured %d, dark %d)\n",
			r.Refold.Verdict, sat, r.Refold.TotalDebt, r.Refold.Floor, r.Refold.Unmeasured, r.Refold.Dark)
		fmt.Fprintf(w, "  → %s\n", r.Refold.NextAction)
	}
}
