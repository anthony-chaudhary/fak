package main

// superloop.go — the impure shell over internal/superloop: the operator-intent
// META-LOOP. `fak superloop walk improve-quality` is the operator command the concept
// is built for: it WALKS the curated member loops/scorecards/gardens FIRST to read
// their status, then folds a worst-first worklist of what to enter — the layer ABOVE
// a normal loop (see the package doc for the full normal-vs-super differentiation).
//
//	fak superloop list                  the named super loops + their members
//	fak superloop explain <name>        why <name> is a super loop, not a normal loop
//	fak superloop walk <name> [--json]  walk its members, fold the worst-first plan
//
// The walk reads CHEAP, committed status surfaces so it stays fast and honest: a
// scorecard member's debt from the pinned control-pane baseline (the last measured,
// committed value); a loop member's live/dark state from the cross-ledger loop-health
// fold (internal/loopfleet). A sub-super-loop member is DESCENDED inline — a
// recursive, cycle-guarded, in-process walk over the same surfaces, folded into one
// measured status (the recursion case is live; `fak superloop walk tend` exercises
// it). Garden / surface members stay DESCEND pointers — their status needs live
// member runs or an external command, which is the named drive follow-on. The walk
// mutates nothing.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/relay"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
	"github.com/anthony-chaudhary/fak/internal/superloop"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func cmdSuperloop(argv []string) { os.Exit(runSuperloop(os.Stdout, os.Stderr, argv)) }

func runSuperloop(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		superloopUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "list":
		return runSuperloopList(stdout, stderr, argv[1:])
	case "explain":
		return runSuperloopExplain(stdout, stderr, argv[1:])
	case "walk":
		return runSuperloopWalk(stdout, stderr, argv[1:])
	case "trigger":
		return runSuperloopTrigger(stdout, stderr, argv[1:])
	case "roster":
		return runRoster(stdout, stderr, argv[1:])
	case "fleet":
		return runSuperloopFleet(stdout, stderr, argv[1:])
	case "drive":
		return runSuperloopDrive(stdout, stderr, argv[1:])
	case "modelfit":
		return runSuperloopModelfit(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		superloopUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak superloop: unknown subcommand %q\n", argv[0])
		superloopUsage(stderr)
		return 2
	}
}

func runSuperloopList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the registry as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	reg := superloop.Registry()
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, reg, "fak superloop list")
	}
	fmt.Fprintf(stdout, "fak super loops — operator intents that walk a set of member loops\n\n")
	for _, s := range reg {
		fmt.Fprintf(stdout, "%-18s %s\n", s.Name, s.Title)
		fmt.Fprintf(stdout, "    %s\n", s.About)
		for _, m := range s.Members {
			fmt.Fprintf(stdout, "      - %-10s %-16s %s\n", m.Kind, m.Ref, m.Why)
		}
		fmt.Fprintln(stdout)
	}
	fmt.Fprintf(stdout, "walk one: fak superloop walk %s\n", firstName(reg))
	return 0
}

func runSuperloopExplain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop explain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the classification as JSON")
	if err := fs.Parse(superloopInterspersedFlagArgs(argv, nil)); err != nil {
		return 2
	}
	name := fs.Arg(0)
	s, ok := superloop.Lookup(name)
	if !ok {
		fmt.Fprintf(stderr, "fak superloop explain: unknown super loop %q (try `fak superloop list`)\n", name)
		return 2
	}
	super := superloop.Classify(superloop.FactsFor(s))
	normal := superloop.Classify(superloop.LeafFacts("a normal task loop (e.g. the dispatch tick)"))
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"super":  super,
			"normal": normal,
			"loop":   s,
		}, "fak superloop explain")
	}

	fmt.Fprintf(stdout, "%s — %s\n\n", s.Name, s.Title)
	fmt.Fprintf(stdout, "Is it a super loop? %s\n", superYesNo(super.IsSuper))
	fmt.Fprintf(stdout, "  %s\n\n", super.Reason)

	fmt.Fprintln(stdout, "The five properties that separate a super loop from a normal loop:")
	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "  PROPERTY\tSUPER LOOP\tNORMAL LOOP\tWHAT IT MEANS")
	for i, p := range super.Properties {
		np := normal.Properties[i]
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\n", p.Name, holdMark(p.Holds), holdMark(np.Holds), p.Detail)
	}
	_ = tw.Flush()

	fmt.Fprintf(stdout, "\nA super loop is an INTERIOR node: its unit of work is another LOOP, not a task.\n")
	fmt.Fprintf(stdout, "A normal loop is a LEAF: it ticks and does one concrete thing.\n\n")
	fmt.Fprintf(stdout, "walk it: fak superloop walk %s\n", s.Name)
	return 0
}

func runSuperloopWalk(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop walk", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the walk report as JSON")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if err := fs.Parse(superloopInterspersedFlagArgs(argv, map[string]bool{"workspace": true})); err != nil {
		return 2
	}
	name := fs.Arg(0)
	s, ok := superloop.Lookup(name)
	if !ok {
		fmt.Fprintf(stderr, "fak superloop walk: unknown super loop %q (try `fak superloop list`)\n", name)
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	// The declared issue-target headline gates the WALK exactly as it gates the
	// drive (#4958): a walk of an intent with a declared target folds its LIVE
	// progress count in (surface-only until a dispatch ledger exists), so `walk`
	// can never read satisfied over an unmet headline the drive would refuse.
	statuses := collectSuperloopStatuses(root, s)
	rep := superloop.Walk(s, statuses, issueProgressWalkOpts(root, s)...)

	if *asJSON {
		// Emit the machine-readable report, but the exit code must still reflect the
		// walk's satisfaction — same contract as the human path below, so a CI gate
		// reading $? is never told "satisfied" when it isn't.
		if rc := encodeJSONOrFail(stdout, stderr, rep, "fak superloop walk"); rc != 0 {
			return rc
		}
	} else {
		renderSuperloopWalk(stdout, rep)
	}
	if rep.Satisfied {
		return 0
	}
	return 1
}

// runSuperloopModelfit renders the C6 model-fit eval (#3043): the OFFLINE,
// fixture-backed grade of which cheaper models can do read-only super-loop
// watchdog/meta work reliably. It grades the built-in SIMULATED sample rows (no
// live model is queried, so it runs anywhere) against the built-in fixtures and
// emits per-model suitability + the risk class each passing model is cleared for.
// It mutates nothing and always exits 0 when it emits: this is an operator readout,
// not a gate — suitability is per-model, and a suitable verdict clears read-only
// meta RECOMMENDATION only, never mutation.
func runSuperloopModelfit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop modelfit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the eval report as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	rep := superloop.SimulatedReport()
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak superloop modelfit")
	}
	superloop.Render(stdout, rep)
	return 0
}

// superloopCollector holds the cheap committed surfaces read ONCE per walk (the
// pinned scorecard baseline, the cross-ledger loop-health fold), so a recursive
// descent through sub-super-loops re-reads nothing.
type superloopCollector struct {
	root          string
	baseline      scorecardpane.Baseline
	baseErr       error
	loopByKind    map[string]loopfleet.LoopHealth
	skippedLedger map[string]string
	// trajLoaded/trajCurve cache the folded open-objective curves so a recursive
	// descent (and multiple KindTrajectory members) reads the trajctl ledger once.
	// trajLedgerPresent records whether the ledger FILE existed at all, so an ABSENT
	// ledger reads as UNMEASURED (never silently clean) while a real ledger with zero
	// open objectives reads as measured-clean.
	trajLoaded        bool
	trajLedgerPresent bool
	trajCurve         trajctl.CurveReport
}

func newSuperloopCollector(root string) *superloopCollector {
	c := &superloopCollector{
		root:          root,
		loopByKind:    map[string]loopfleet.LoopHealth{},
		skippedLedger: map[string]string{},
	}
	c.baseline, c.baseErr = loadScorecardBaseline(root)
	fleet := loopfleet.Fold(root, time.Now(), loopmgr.HealthThresholds{})
	for _, l := range fleet.Loops {
		c.loopByKind[l.Kind] = l
	}
	for _, sk := range fleet.Skipped {
		c.skippedLedger[sk.Ledger] = sk.Reason
	}
	return c
}

// collectSuperloopStatuses reads each member's status from the cheap committed
// surfaces, off the hot path. Scorecard debt comes from the pinned control-pane
// baseline; loop liveness from the cross-ledger loop-health fold; a sub-super-loop
// member is DESCENDED inline (a recursive in-process walk over the same surfaces);
// garden / surface members stay descend pointers, not measured here.
func collectSuperloopStatuses(root string, s superloop.Super) []superloop.MemberStatus {
	c := newSuperloopCollector(root)
	return c.collect(s, map[string]bool{s.Name: true})
}

// collect walks one super loop's members. onPath is the set of super-loop names on
// the current descent path (ancestry), guarding a registry cycle.
func (c *superloopCollector) collect(s superloop.Super, onPath map[string]bool) []superloop.MemberStatus {
	out := make([]superloop.MemberStatus, 0, len(s.Members))
	for _, m := range s.Members {
		st := superloop.MemberStatus{Member: m}
		switch m.Kind {
		case superloop.KindScorecard:
			if c.baseErr != nil {
				st.Measured = false
				st.Detail = "no pinned baseline (run `fak scorecard --pin`): " + c.baseErr.Error()
				break
			}
			debt, present := c.baseline.Metrics[m.Ref]
			if !present {
				st.Measured = false
				st.Detail = fmt.Sprintf("key %q not in pinned baseline @%s", m.Ref, c.baseline.Commit)
				break
			}
			st.Measured = true
			st.Debt = debt
			st.Detail = fmt.Sprintf("baseline debt %d (pinned @%s)", debt, c.baseline.Commit)
		case superloop.KindLoop:
			lh, found := c.loopByKind[m.Ref]
			if found {
				st.Measured = true
				st.Dark = lh.Dark
				st.Progress, st.ProgressReason = c.memberProgress(m, lh)
				st.FollowOn, st.FollowOnReason = c.memberFollowon(m, lh)
				st.Debt = loopDebt(lh, st.Progress, st.FollowOn)
				st.Detail = fmt.Sprintf("state %s, %d run(s), keep %s", lh.State, lh.Runs, keepRateStr(lh.KeepRate))
				break
			}
			st.Measured = false
			if reason, sk := c.skippedLedger[m.Ref]; sk {
				st.Detail = "ledger " + reason
			} else {
				st.Detail = "no ledger on this host (loop has not run here)"
			}
		case superloop.KindSuperloop:
			st = c.descend(m, onPath)
		case superloop.KindTrajectory:
			// A trajectory member ENUMERATES into one status per OPEN objective, so it
			// appends its own (possibly several) statuses and skips the single-status
			// append below.
			out = append(out, c.trajectory(m)...)
			continue
		case superloop.KindLoopFleet:
			// The whole-fleet member (#4955) ENUMERATES into one status per ledgered
			// loop from the already-read cross-ledger fold, plus one UNMEASURED status
			// per skipped ledger (a known gap blocks Satisfied, never a healthy zero).
			out = append(out, c.loopFleet(m)...)
			continue
		case superloop.KindUtilization:
			st = c.utilization(m)
		case superloop.KindGarden, superloop.KindSurface:
			st.Container = true
			st.Measured = false
			st.Detail = m.Why
		default:
			st.Measured = false
			st.Detail = "unknown member kind"
		}
		out = append(out, st)
	}
	return out
}

// descend is the DESCEND move made real for a registered sub-super-loop: walk it
// in-process over the same already-read surfaces, and fold the sub-report into one
// measured member status (superloop.SubwalkStatus — an unsatisfied sub can never
// read clean). An unknown ref or a cycle is refused as UNMEASURED, which blocks
// Satisfied — a registry bug must red the walk, not vanish.
func (c *superloopCollector) descend(m superloop.Member, onPath map[string]bool) superloop.MemberStatus {
	st := superloop.MemberStatus{Member: m, Measured: false}
	sub, ok := superloop.Lookup(m.Ref)
	if !ok {
		st.Detail = fmt.Sprintf("unknown super loop %q (registry drift; try `fak superloop list`)", m.Ref)
		return st
	}
	if onPath[sub.Name] {
		st.Detail = fmt.Sprintf("cycle: %q is already on this descent path", sub.Name)
		return st
	}
	// The sub-intent's declared headline gates its sub-walk here too (#4958), so a
	// progress shortfall folds into SubwalkStatus debt at the PARENT's altitude —
	// the root walk ranks a big headline miss ahead of trivial member debt, not
	// only the drive path.
	onPath[sub.Name] = true
	rep := superloop.Walk(sub, c.collect(sub, onPath), issueProgressWalkOpts(c.root, sub)...)
	delete(onPath, sub.Name)
	return superloop.SubwalkStatus(m, rep)
}

// trajCurves folds the trajectory-control ledger into its worst-first open-objective
// curves, reading the ledger ONCE per collector (a recursive descent and multiple
// trajectory members reuse the cache). A missing or unreadable ledger folds to an
// empty report — the same conservative posture trajctl.ReadLedgerFile keeps — so an
// absent ledger reads as "no open objectives", not a crash.
func (c *superloopCollector) trajCurves() trajctl.CurveReport {
	if !c.trajLoaded {
		path := filepath.Join(c.root, filepath.FromSlash(trajctl.DefaultLedgerRel))
		if _, err := os.Stat(path); err == nil {
			c.trajLedgerPresent = true
		}
		c.trajCurve = trajctl.Fold(trajctl.ReadLedgerFile(path)).OpenCurves()
		c.trajLoaded = true
	}
	return c.trajCurve
}

// trajectory is the trajectory-member adapter (issue #2563): it ENUMERATES a single
// KindTrajectory registry member into one MemberStatus per OPEN objective, each weighed
// worst-first by its curve signal (trajctl.SignalDebt). A member Ref of "open" (or
// empty) takes every open objective; any other Ref selects one objective by id. When no
// open objective matches — an empty ledger, every objective closed, or a named id with
// no open curve — it folds to a SINGLE measured, zero-debt status so the intent reads
// SATISFIED (an on-course fleet is nothing to enter) rather than unmeasured.
func (c *superloopCollector) trajectory(m superloop.Member) []superloop.MemberStatus {
	rep := c.trajCurves()
	selectOne := strings.TrimSpace(m.Ref) != "" && m.Ref != "open"
	out := make([]superloop.MemberStatus, 0, len(rep.Objectives))
	for _, oc := range rep.Objectives {
		if selectOne && oc.ObjectiveID != m.Ref {
			continue
		}
		out = append(out, trajectoryStatus(m, oc))
	}
	if len(out) == 0 {
		if !c.trajLedgerPresent {
			// No ledger at all: trajectory health cannot be read, so it is UNMEASURED —
			// surfaced, never silently treated as clean (the same honesty the scorecard
			// and loop members keep for an unreadable surface).
			return []superloop.MemberStatus{{Member: m, Measured: false,
				Detail: "no trajctl ledger at " + trajctl.DefaultLedgerRel + " — trajectory health unmeasured (declare/score objectives via `fak trajctl`)"}}
		}
		detail := "no open trajectory objective — trajectory health reads clean"
		if selectOne {
			detail = fmt.Sprintf("objective %q has no open curve — nothing to steer", m.Ref)
		}
		return []superloop.MemberStatus{{Member: m, Measured: true, Debt: 0, Detail: detail}}
	}
	return out
}

// trajectoryStatus folds one open objective curve into the member status the walk
// weighs: debt is the signal severity (HEALTHY 0 < STALL 1 < DRIFT 2 < DETOUR_OVERRUN
// 3), so worst-first the walk enters the most off-course objective. Each enumerated
// status carries the concrete objective id as its Ref and a directly-runnable curve
// command as its Enter hint, inheriting the source member's Why for provenance.
func trajectoryStatus(src superloop.Member, oc trajctl.ObjectiveCurve) superloop.MemberStatus {
	return superloop.MemberStatus{
		Member: superloop.Member{
			Kind:  superloop.KindTrajectory,
			Ref:   oc.ObjectiveID,
			Why:   src.Why,
			Enter: fmt.Sprintf("fak trajctl curve --objective %s", oc.ObjectiveID),
		},
		Measured: true,
		Debt:     trajctl.SignalDebt(oc.Signal),
		Detail:   fmt.Sprintf("%s — %s (latest %.2f, delta %+.2f)", oc.Signal, oc.Detail, oc.Latest, oc.Delta),
	}
}

// loopFleet is the whole-fleet adapter (issue #4955): it ENUMERATES a single
// KindLoopFleet registry member into one MemberStatus per ledgered loop, from the
// SAME cross-ledger fold the collector already read (no extra I/O) — the
// trajectory() enumeration precedent, one level up. The pure expansion
// (superloop.LoopFleetStatuses) owns the honesty rules: a skipped ledger surfaces
// as an UNMEASURED known gap, a Ref of "all" takes the whole fleet, any other Ref
// selects one loop by its stable identity.
func (c *superloopCollector) loopFleet(m superloop.Member) []superloop.MemberStatus {
	folded := make([]superloop.RosterLoop, 0, len(c.loopByKind))
	for _, lh := range c.loopByKind {
		folded = append(folded, superloop.RosterLoop{Kind: lh.Kind, State: string(lh.State), Dark: lh.Dark})
	}
	gaps := make([]superloop.RosterGap, 0, len(c.skippedLedger))
	for ledger, reason := range c.skippedLedger {
		gaps = append(gaps, superloop.RosterGap{Ledger: ledger, Reason: reason})
	}
	sts := superloop.LoopFleetStatuses(m, folded, gaps)
	// Fold the per-loop PROGRESS verdict (#4956) onto each enumerated fleet loop
	// through the same ledger-verified read the hand-named KindLoop members use,
	// then RE-apply the meta-walk's worst-first key (#4958): debt becomes the
	// PRODUCT of the three loop dimensions (liveness × progress × follow-on, minus
	// one — superloop.FleetDebt), so a stale loop that is ALSO spinning compounds
	// (debt 3) instead of merely summing, and a clean live leaf stays 0. The
	// follow-on verdict (#4957) rides the same fold as its witness lands; until
	// then the axis reads "" and multiplies by 1 — surfaced, never fabricated.
	for i := range sts {
		lh, ok := c.loopByKind[sts[i].Member.Ref]
		if !ok || !sts[i].Measured {
			continue
		}
		sts[i].Progress, sts[i].ProgressReason = c.memberProgress(sts[i].Member, lh)
		sts[i].FollowOn, sts[i].FollowOnReason = c.memberFollowon(sts[i].Member, lh)
		sts[i].Debt = superloop.FleetDebt(string(lh.State), lh.Dark, sts[i].Progress, sts[i].FollowOn)
	}
	return sts
}

// memberProgress reads one loop member's ledger-verified progress (#4956) and folds
// it into the closed progress verdict. The read is relay.ReadVerifiedProgress over
// the file-backed ledger reader — the intent ledger's own re-verifiable rows, NEVER
// the ledger row's self-reported keep counters — and the fold is
// superloop.ClassifyProgress, which lets only a throughput (issue-drain) member trip
// SPINNING and parks a proven-idle watch member benign. Fail-closed at every edge: a
// non-ticking loop reads no progress axis at all (its urgency is the liveness
// verdict), and a loop with no bound intent-ledger anchor (progressLedgerRef "")
// reads ProgressUnknown, which the fold surfaces as unmeasured — shown, never debt,
// never a fabricated zero (the nightIssueProgress posture).
//
// The Baseline stays the zero value: the relay high-water starts at 0 (a first
// verified step counts as progress), so the live SPINNING bar is "ticking with NO
// verified step recorded at all"; a persisted per-window high-water cursor is the
// #4958 tend-wiring follow-on.
func (c *superloopCollector) memberProgress(m superloop.Member, lh loopfleet.LoopHealth) (superloop.MemberProgress, string) {
	ticking := lh.State == loopmgr.HealthLive || lh.State == loopmgr.HealthStale
	if !ticking {
		return "", ""
	}
	reader := relay.NewFileLedgerReader(relay.OSFileLedger(c.root))
	now := relay.ReadVerifiedProgress(relay.ProgressCursor{LedgerRef: progressLedgerRef(c.root, lh.Kind)}, reader)
	return superloop.ClassifyProgress(m, true, superloop.ProgressRead{Now: now})
}

// progressLedgerRef maps a fleet loop's stable identity to the repo-relative
// intent-ledger ref its verified progress is read from. NO live binding exists yet:
// every loop returns "" (no anchor), which relay.ReadVerifiedProgress fails closed
// to ProgressUnknown and the walk surfaces as an unmeasured, surface-only progress
// verdict — never a fabricated zero, never a slanderous SPINNING. A var so tests
// bind a hermetic ledger and drive the SPINNING path end to end; the production
// binding (a per-loop persisted cursor) rides the #4958 tend wiring.
var progressLedgerRef = func(root, kind string) string { return "" }

// loopDebt maps a loop-health row PLUS its progress verdict (#4956) and its
// follow-on verdict (#4957) to a debt integer for the worst-first fold: a dark loop
// carries its urgency in the Dark flag (debt 0), a stale loop is one unit of debt
// (slipping past its cadence), a SPINNING loop adds the progress term — one more unit
// for ticking with zero advanced verified progress — and an ORPHANED loop adds the
// follow-on term, one more unit for emitting work nobody advances. So neither a
// live-but-producing-nothing loop nor a producing-into-the-void one can read clean,
// and a loop that is both compounds. This is the hand-named KindLoop counterpart of
// the enumerated fleet's superloop.FleetDebt; the two weigh the same three axes.
func loopDebt(lh loopfleet.LoopHealth, prog superloop.MemberProgress, followOn superloop.MemberFollowon) int {
	debt := 0
	if !lh.Dark && lh.State == loopmgr.HealthStale {
		debt = 1
	}
	if prog == superloop.ProgressSpinning {
		debt++
	}
	if followOn == superloop.FollowonOrphaned {
		debt++
	}
	return debt
}

func keepRateStr(r float64) string {
	if r < 0 {
		return "-"
	}
	return fmt.Sprintf("%.2f", r)
}

func loadScorecardBaseline(root string) (scorecardpane.Baseline, error) {
	var b scorecardpane.Baseline
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(scorecardpane.BaselineRel)))
	if err != nil {
		return b, err
	}
	if err := json.Unmarshal(data, &b); err != nil {
		return b, err
	}
	return b, nil
}

func superloopInterspersedFlagArgs(argv []string, takesValue map[string]bool) []string {
	flags := make([]string, 0, len(argv))
	positionals := make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			positionals = append(positionals, arg)
			continue
		}
		flags = append(flags, arg)
		name := superloopFlagName(arg)
		if takesValue[name] && !strings.Contains(arg, "=") && i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "-") {
			i++
			flags = append(flags, argv[i])
		}
	}
	return append(flags, positionals...)
}

func superloopFlagName(arg string) string {
	arg = strings.TrimLeft(arg, "-")
	if before, _, ok := strings.Cut(arg, "="); ok {
		return before
	}
	return arg
}

func renderSuperloopWalk(w io.Writer, rep superloop.WalkReport) {
	fmt.Fprintf(w, "superloop walk: %s — %s (%s)\n", rep.Name, rep.Verdict, rep.Finding)
	fmt.Fprintf(w, "  aggregate debt %d (floor %d)  members %d  walked %d  unmeasured %d  dark %d\n",
		rep.TotalDebt, rep.Floor, rep.Members, rep.Walked, rep.Unmeasured, rep.Dark)
	if rep.Spinning > 0 {
		// #4956: a member can be live on cadence and still produce nothing verified.
		fmt.Fprintf(w, "  spinning %d — ticking on cadence with zero advanced verified progress (%s)\n",
			rep.Spinning, relay.ReasonNoProgress)
	}
	if rep.Orphaned > 0 {
		// #4957: a member can produce fine while its emitted work sits unowned.
		fmt.Fprintf(w, "  orphaned %d — emitting follow-on work nobody advances (%s)\n",
			rep.Orphaned, relay.ReasonOrphanedFollowon)
	}
	if rep.IssueTarget > 0 {
		// The intent declares a headline issue target (run-the-night's ~200 overnight).
		// When the dispatch ledger made live progress measurable, the walk folds it in
		// and shows the gate's live numbers (#4958); otherwise the declared number is
		// surfaced without the walk claiming a progress it did not witness.
		if rep.IssueProgressMeasured {
			fmt.Fprintf(w, "  issue target: %d — progressed %d, shortfall %d (a live gate, not a decoration)\n",
				rep.IssueTarget, rep.IssueProgressed, rep.IssueShortfall)
		} else {
			fmt.Fprintf(w, "  issue target: %d (declared operator headline; no dispatch ledger measurable here — surfaced, not folded)\n", rep.IssueTarget)
		}
	}
	fmt.Fprintf(w, "  %s\n\n", rep.Reason)

	renderSuperloopBudget(w, rep)

	if len(rep.Worklist) == 0 {
		fmt.Fprintf(w, "  nothing to enter — every measured member is clean and live\n\n")
	} else {
		fmt.Fprintln(w, "  worst-first — enter these in order:")
		tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
		fmt.Fprintln(tw, "  #\tMEMBER\tDEBT\tACTION\tWHY")
		for _, it := range rep.Worklist {
			debt := fmt.Sprintf("%d", it.Debt)
			if it.Dark {
				debt = "DARK"
			} else if it.Progress == superloop.ProgressSpinning {
				// live on cadence, zero advanced verified progress (#4956)
				debt = fmt.Sprintf("SPIN %d", it.Debt)
			} else if it.FollowOn == superloop.FollowonOrphaned {
				// emitting follow-on work nobody advances (#4957)
				debt = fmt.Sprintf("ORPH %d", it.Debt)
			} else if it.Container {
				// only an UNREAD pointer renders "→"; a descended sub-super-loop
				// carries its real folded debt.
				debt = "→"
			}
			fmt.Fprintf(tw, "  %d\t%s %s\t%s\t%s\t%s\n",
				it.Rank, it.Member.Kind, it.Member.Ref, debt, it.Action, it.Detail)
		}
		_ = tw.Flush()
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "  → %s\n", rep.NextAction)
}

// renderSuperloopBudget prints the intent's declared generation budget and the share
// the divide-down reserves for each worklist member. It shows the DECLARED cap and
// the PER-MEMBER share side by side so an operator can see reservation vs. dilution
// at a glance; an unbudgeted dimension prints HELD (the contract's hold state), never
// a blank that could read as "unlimited". These are planned reservations, not
// measured consumption — the measured comparison is the future scorecard hook.
func renderSuperloopBudget(w io.Writer, rep superloop.WalkReport) {
	if len(rep.Budget) == 0 {
		return
	}
	stream := rep.Budget[0].Stream
	members := rep.Budget[0].Members
	header := "  budget — declared reservation"
	if stream != "" {
		header = "  budget " + stream + " — declared reservation"
	}
	if members > 0 {
		fmt.Fprintf(w, "%s, divided across %d worklist member(s):\n", header, members)
	} else {
		fmt.Fprintf(w, "%s (no members to enter — nothing reserved this walk):\n", header)
	}
	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "  DIMENSION\tDECLARED\tPER-MEMBER")
	for _, row := range rep.Budget {
		if !row.Budgeted {
			fmt.Fprintf(tw, "  %s\t—\tHELD\n", row.Dimension)
			continue
		}
		fmt.Fprintf(tw, "  %s\t%d %s\t%d\n", row.Dimension, row.Total, row.Unit, row.PerMember)
	}
	_ = tw.Flush()
	fmt.Fprintln(w)
}

func firstName(reg []superloop.Super) string {
	if len(reg) == 0 {
		return "<name>"
	}
	return reg[0].Name
}

func superYesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func holdMark(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func superloopUsage(w io.Writer) {
	fmt.Fprint(w, `fak superloop - the operator-intent meta-loop: walk a set of member loops, worst-first

  fak superloop list                  the named super loops + their members
  fak superloop explain <name>        why <name> is a super loop, not a normal loop
  fak superloop walk <name> [--json]  walk its members' status, fold a worst-first plan
  fak superloop trigger [flags]        emit a read-only trigger decision receipt
  fak superloop roster [--json]       the ONE canonical loop-fleet roster (#4955): the
                                      deduped union of every folded loop (loopfleet),
                                      the loopmgr job registry, and the registered
                                      super loops — what the operator supervises,
                                      each counted exactly once; an unfoldable
                                      ledger surfaces as a KNOWN gap, never dropped
  fak superloop fleet <status|next|walk|run>  the generic fleet meta-walker (#4958):
                                      every ledgered loop worst-first on the PRODUCT
                                      liveness × progress × follow-on (dark/stale ×
                                      SPINNING × ORPHANED); run drives the worst
                                      member through the same admission gate as drive
  fak superloop drive <name> [--lane L]  walk, then ENTER the one worst-first member
                                      through the same admission gate any spawn passes
                                      (COLLISION_RISK on lease overlap), and re-fold
       [--batch N]                    admit up to N worst-first members whose regions
                                      are mutually disjoint (and disjoint from live
                                      leases) in one drive, through the SAME gate; N<=0
                                      offers every member. Default 1 (single member);
                                      FAK_SUPERLOOP_BATCH sets this default fleet-wide
                                      when --batch is not given (explicit flag wins)
  fak superloop modelfit [--json]     offline model-fit eval: which cheaper models can
                                      do read-only watchdog/meta work reliably (#3043)

A SUPER LOOP is keyed on an operator intent ("improve quality"), not a task. Its tick
WALKS its member loops/scorecards/gardens to read their status FIRST, SELECTS the
worst-first member to enter, and exits on the AGGREGATE clearing — the layer above a
normal (task + cadence) loop, which just ticks and does one concrete thing. walk reads
cheap committed surfaces (the pinned scorecard baseline, the loop-health ledger fold),
DESCENDS sub-super-loops inline ("fak superloop walk tend" walks every intent), and
mutates nothing; driving a garden/surface member is the named follow-on.
`)
}
