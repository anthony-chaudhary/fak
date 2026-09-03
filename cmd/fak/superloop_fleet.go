package main

// superloop_fleet.go — the manager surface for the GENERIC fleet meta-walker
// (issue #4958), modelled on the bench-loop manager's verb set (benchloop.go): one
// small domain manager whose verbs are status / next / walk / run.
//
//	fak superloop fleet status  one-screen fold of the tend-fleet walk: verdict +
//	                            the three dimension counts (dark, spinning, orphaned)
//	                            + the worst-first head
//	fak superloop fleet next    the single worst-first SELECT (what would be entered),
//	                            with its front-door classification — no enter
//	fak superloop fleet walk    the full worst-first walk (== `fak superloop walk
//	                            tend-fleet`)
//	fak superloop fleet run     drive the worst member (== `fak superloop drive
//	                            tend-fleet ...`): the member is entered through its
//	                            OWN declared front door under the held region lease,
//	                            through the SAME admission gate any spawn passes —
//	                            the meta-walker gets NO private spawn path
//
// The manager adds no rival walker and no private fold: status/next are read-only
// folds over the same superloop.Walk every intent uses, and walk/run DELEGATE to the
// standing walk/drive shells verbatim, so every honesty gate those paths carry
// (unmeasured blocks satisfied, refusal tokens surfaced never bypassed, re-fold as
// the exit check) holds here by construction.

import (
	"flag"
	"fmt"
	"io"

	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// superloopFleetIntent is the registered intent this manager fronts — the generic
// meta-walker over the whole loop fleet (#4958). One const so the manager and its
// tests can never drift from the registry name.
const superloopFleetIntent = "tend-fleet"

// superloopFleetStatusSchema tags the `fleet status --json` payload.
const superloopFleetStatusSchema = "fak.superloop-fleet-status.v1"

func runSuperloopFleet(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		return superloopFleetStatus(stdout, stderr, nil)
	}
	sub, rest := argv[0], argv[1:]
	switch sub {
	case "status":
		return superloopFleetStatus(stdout, stderr, rest)
	case "next":
		return superloopFleetNext(stdout, stderr, rest)
	case "walk":
		// Delegate to the standing walk shell with the fleet intent appended as the
		// positional name (flags in rest are interspersed-safe there).
		return runSuperloopWalk(stdout, stderr, append(rest, superloopFleetIntent))
	case "run":
		// Delegate to the standing drive shell: worst-first SELECT, the SHARED region
		// admission gate, the member's own front door behind the held lease, re-fold
		// as the exit check. No private spawn path.
		return runSuperloopDrive(stdout, stderr, append(rest, superloopFleetIntent))
	case "-h", "--help", "help":
		superloopFleetUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak superloop fleet: unknown subcommand %q\n", sub)
		superloopFleetUsage(stderr)
		return 2
	}
}

// superloopFleetReport is the folded status readout: the tend-fleet walk reduced to
// the meta-walker's three dimension counts plus the worst-first head.
type superloopFleetReport struct {
	Schema     string                   `json:"schema"`
	Intent     string                   `json:"intent"`
	Verdict    string                   `json:"verdict"`
	Finding    string                   `json:"finding"`
	Satisfied  bool                     `json:"satisfied"`
	Members    int                      `json:"members"`
	Walked     int                      `json:"walked"`
	Unmeasured int                      `json:"unmeasured"`
	Dark       int                      `json:"dark"`
	Spinning   int                      `json:"spinning,omitempty"`
	Orphaned   int                      `json:"orphaned,omitempty"`
	TotalDebt  int                      `json:"total_debt"`
	Floor      int                      `json:"floor"`
	Head       *superloop.WorkItem      `json:"head,omitempty"`
	NextAction string                   `json:"next_action"`
	Rollup     *superloop.RollupSummary `json:"rollup,omitempty"`
}

// superloopFleetWalk is the one shared read both read-only verbs fold: look the
// intent up (registry drift is a hard error, never a silent empty walk) and walk it
// over the same collected surfaces + issue-progress gate every walk path uses.
func superloopFleetWalk(stderr io.Writer, workspace string) (superloop.Super, superloop.WalkReport, bool) {
	s, ok := superloop.Lookup(superloopFleetIntent)
	if !ok {
		fmt.Fprintf(stderr, "fak superloop fleet: intent %q is not registered (registry drift)\n", superloopFleetIntent)
		return superloop.Super{}, superloop.WalkReport{}, false
	}
	root := workspaceOrRepoRoot(workspace)
	rep := superloop.Walk(s, collectSuperloopStatuses(root, s), issueProgressWalkOpts(root, s)...)
	return s, rep, true
}

func superloopFleetStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop fleet status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the fleet status as JSON")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if !parseFlags(fs, argv) {
		return 2
	}
	s, rep, ok := superloopFleetWalk(stderr, *workspace)
	if !ok {
		return 2
	}
	report := superloopFleetReport{
		Schema:     superloopFleetStatusSchema,
		Intent:     s.Name,
		Verdict:    rep.Verdict,
		Finding:    rep.Finding,
		Satisfied:  rep.Satisfied,
		Members:    rep.Members,
		Walked:     rep.Walked,
		Unmeasured: rep.Unmeasured,
		Dark:       rep.Dark,
		Spinning:   rep.Spinning,
		Orphaned:   rep.Orphaned,
		TotalDebt:  rep.TotalDebt,
		Floor:      rep.Floor,
		NextAction: rep.NextAction,
		Rollup:     &rep.Rollup,
	}
	if len(rep.Worklist) > 0 {
		head := rep.Worklist[0]
		report.Head = &head
	}
	code := 1
	if rep.Satisfied {
		code = 0
	}
	if *asJSON {
		if rc := encodeJSONOrFail(stdout, stderr, report, "fak superloop fleet status"); rc != 0 {
			return rc
		}
		return code
	}
	fmt.Fprintf(stdout, "superloop fleet: %s — %s (%s)\n", report.Intent, report.Verdict, report.Finding)
	fmt.Fprintf(stdout, "  members %d  walked %d  unmeasured %d  |  dark %d  spinning %d  orphaned %d  |  debt %d (floor %d)\n",
		report.Members, report.Walked, report.Unmeasured, report.Dark, report.Spinning, report.Orphaned, report.TotalDebt, report.Floor)
	if report.Rollup != nil && report.Rollup.Members > 0 {
		fmt.Fprintf(stdout, "  rollup: leaves %d  walked %d  unmeasured %d  dark %d\n",
			report.Rollup.Members, report.Rollup.Walked, report.Rollup.Unmeasured, report.Rollup.Dark)
	}
	fmt.Fprintf(stdout, "  ranked worst-first on the product liveness × progress × follow-on (fleetwalk.go, #4958)\n")
	if report.Head != nil {
		fmt.Fprintf(stdout, "  worst: %s %s (debt %d) — %s\n", report.Head.Member.Kind, report.Head.Member.Ref, report.Head.Debt, report.Head.Detail)
	}
	fmt.Fprintf(stdout, "  → %s\n", report.NextAction)
	return code
}

func superloopFleetNext(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop fleet next", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the worst-first selection as JSON")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if !parseFlags(fs, argv) {
		return 2
	}
	_, rep, ok := superloopFleetWalk(stderr, *workspace)
	if !ok {
		return 2
	}
	// The SAME pure SELECT the drive uses — the head of the worst-first worklist —
	// plus the front-door classification, so the readout can never disagree with
	// what `fleet run` would actually enter or surface.
	decision := superloop.Drive(rep)
	if *asJSON {
		payload := struct {
			Schema    string                  `json:"schema"`
			Decision  superloop.DriveDecision `json:"decision"`
			FrontDoor superloop.FrontDoor     `json:"front_door"`
		}{superloop.DriveSchema, decision, superloop.FrontDoorFor(decision)}
		if rc := encodeJSONOrFail(stdout, stderr, payload, "fak superloop fleet next"); rc != 0 {
			return rc
		}
		if decision.Enter || decision.Satisfied {
			return 0
		}
		return 1
	}
	fmt.Fprintf(stdout, "superloop fleet next: %s\n", superloopFleetIntent)
	if !decision.Enter {
		// Nothing to enter reads clean ONLY when the walk is satisfied (#3147).
		fmt.Fprintf(stdout, "  nothing to enter — %s\n", decision.Reason)
		if decision.Satisfied {
			return 0
		}
		return 1
	}
	fd := superloop.FrontDoorFor(decision)
	dark := ""
	if decision.Dark {
		dark = ", DARK"
	}
	fmt.Fprintf(stdout, "  worst-first: %s %s (rank %d, debt %d%s)\n", decision.Member.Kind, decision.Member.Ref, decision.Rank, decision.Debt, dark)
	fmt.Fprintf(stdout, "  action: %s\n", decision.Action)
	fmt.Fprintf(stdout, "  front door: %s — %s\n", fd.Kind, fd.Note)
	fmt.Fprintf(stdout, "  → enter it under the shared gate: fak superloop fleet run [--lane L --tree G]\n")
	return 0
}

func superloopFleetUsage(w io.Writer) {
	fmt.Fprint(w, `fak superloop fleet - the generic fleet meta-walker (#4958): every ledgered loop,
worst-first on the PRODUCT liveness × progress × follow-on

  fak superloop fleet status [--json]  one-screen fold: verdict + dark/spinning/orphaned
                                       counts + the worst-first head
  fak superloop fleet next [--json]    the single worst-first SELECT + its front-door
                                       classification (read-only, no enter)
  fak superloop fleet walk [--json]    the full worst-first walk (== superloop walk tend-fleet)
  fak superloop fleet run [flags]      drive the worst member through the SAME admission
                                       gate any spawn passes, entering its OWN front door
                                       under the held region lease (== superloop drive
                                       tend-fleet; supports --lane/--tree/--batch/--execute)

The rank key is the product of the three per-loop dimensions (fleetwalk.go): a stale
loop that is also SPINNING (#4956) compounds to debt 3; one whose emitted work is
ORPHANED (#4957) doubles again. DARK stays the worst band via the Dark flag. The
reporting-family feeders (#4862) ride beneath this intent as one descended child.
`)
}
