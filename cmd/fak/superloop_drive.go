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
// private spawn path. With --execute (opt-in, off by default) the drive now also RUNS
// that front door behind the held lease and lands the member's own witness — its exit
// code IS the witnessed_done (see superloop_drive_exec.go); a front door it cannot run
// headless (a skill, a container to descend) is surfaced, never faked. Without --execute
// it stays surface-only, byte-for-byte. A driven-but-unwitnessed member keeps the re-fold
// unsatisfied, so it can never satisfy the intent.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopdrive"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// superloopDriveReport is the folded drive verdict: the single member selected
// worst-first, how the shared admission gate ruled, and the re-fold that is the exit
// check. Outcome is one of "satisfied" (nothing to enter and the intent reads clean),
// "shortfall" (nothing to enter member-first but a declared headline gate is UNMET — an
// issue shortfall the drive must not read as clean, #3147), "entered" (one member
// admitted under a lease + re-folded), or "refused" (the admission gate refused; the
// token is surfaced, never bypassed).
type superloopDriveReport struct {
	Schema    string                  `json:"schema"`
	Intent    string                  `json:"intent"`
	Outcome   string                  `json:"outcome"`
	Decision  superloop.DriveDecision `json:"decision"`
	Admission *superloopDriveAdmit    `json:"admission,omitempty"`
	// Exec, when set, records the `--execute` run of the admitted member's own front
	// door behind the held lease (nil when --execute was not requested). It is the
	// difference between admit-and-surface and admit-and-run.
	Exec   *superloopDriveExec   `json:"exec,omitempty"`
	Refold *superloop.WalkReport `json:"refold,omitempty"`
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

// superloopDriveNoEnterOutcome classifies a non-entering drive decision into its
// operator outcome and process exit code. A satisfied walk reads clean — nothing to
// enter, exit 0. An UNSATISFIED empty-worklist walk is an unmet headline gate (an issue
// shortfall the drive surfaced instead of claiming clean, #3147): there is no member to
// enter, but the night is not done, so it reports "shortfall" and exits non-zero — an
// automated night loop can never mistake an unmet ~200-issue headline for a finished
// night. Pure over the decision so it is witnessed without the live surface.
func superloopDriveNoEnterOutcome(d superloop.DriveDecision) (string, int) {
	if d.Satisfied {
		return "satisfied", 0
	}
	return "shortfall", 1
}

func runSuperloopDrive(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop drive", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the drive report as JSON")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	repo := fs.String("repo", "", "expected GitHub repository identity (owner/name); required with --execute and verified against the selected workspace origin")
	lane := fs.String("lane", "", "dos.toml lane the driven member's writes stay inside; arms region admission (COLLISION_RISK on lease overlap) against the live lease fabric")
	ledger := fs.String("ledger", defaultLoopLedger(), "loop JSONL ledger the drive records its admission witness to")
	batch := fs.Int("batch", 1, "admit up to N worst-first members whose regions are mutually disjoint (and disjoint from live leases) in one invocation, through the SAME admission gate; 1 (default) is the historical one-member drive, <=0 offers every worklist member")
	execute := fs.Bool("execute", false, "actually RUN each admitted member's own front door behind the held lease (its exit code is the member's witness); default off = surface the front door only. A skill (/x) or container front door is always surfaced, never run headless")
	execTimeoutMin := fs.Int("exec-timeout", int(defaultSuperloopExecTimeout/time.Minute), "with --execute, per-member front-door timeout in minutes (0 = no timeout)")
	var tree repeatedString
	fs.Var(&tree, "tree", "region glob the driven member's writes stay inside (repeatable); arms region admission against the live lease fabric")
	if err := fs.Parse(superloopInterspersedFlagArgs(argv, map[string]bool{"workspace": true, "repo": true, "lane": true, "ledger": true, "tree": true, "batch": true, "exec-timeout": true})); err != nil {
		return 2
	}
	execTimeout := time.Duration(*execTimeoutMin) * time.Minute
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

	// A live drive is repository-scoped authority, not path-scoped intent. Carry the
	// operator's immutable owner/repo into the selected workspace and fail before WALK,
	// admission, leases, ledger writes, or execution when origin does not corroborate it.
	if *execute {
		if err := verifyDriveRepositoryTarget(root, *repo); err != nil {
			fmt.Fprintf(stderr, "fak superloop drive: %v\n", err)
			return 2
		}
	}

	// BATCH — a fan-out drive admits up to N worst-first members whose regions are
	// mutually disjoint (and disjoint from the live lease fabric) in one invocation,
	// through the SAME admission gate. `--batch 1` (the default) keeps the historical
	// one-member drive below byte-for-byte; only an explicit N>1 (or N<=0 = all)
	// widens the SELECT.
	//
	// FAK_SUPERLOOP_BATCH (opt-in, fail-closed) is the DEPLOYMENT lever that turns the
	// fan-out on fleet-wide without editing every caller: when --batch is not given on
	// the command line, a scheduled meta-loop (or an operator) can raise the default
	// drive throughput past one member per invocation with this single env knob. Unset
	// or unparseable ⇒ 1, byte-for-byte the historical single-member drive; an explicit
	// --batch on the command line ALWAYS wins over the env (the flag is the stronger
	// signal). This is the same fail-closed opt-in shape as FLEET_TIER_LAUNCH.
	effBatch := resolveSuperloopBatch(fs, *batch, stderr)
	if effBatch != 1 {
		return runSuperloopDriveBatch(stdout, stderr, *asJSON, root, s, *lane, tree, *ledger, effBatch, *execute, execTimeout)
	}

	// WALK — read every member's status from the cheap committed surfaces, then SELECT
	// the single worst-first member (one member per walk). The walk mutates nothing.
	// A declared issue-target folds its live progress in here (surface-only until a
	// dispatch ledger exists), so the headline gates the drive, not just decorates it.
	rep := superloopWalkNow(root, s)
	decision := superloop.Drive(rep)
	report := superloopDriveReport{Schema: superloop.DriveSchema, Intent: s.Name, Decision: decision}

	if !decision.Enter {
		// Nothing worst-first to enter — but "enters nothing" is only a CLEAN exit when
		// the walk is satisfied. An unsatisfied empty-worklist drive is an unmet headline
		// gate (an issue shortfall): there is no member to enter, yet the night is NOT
		// done. Gate the outcome + exit on the decision's Satisfied, never a clean exit
		// over an unmet headline (#3147).
		outcome, code := superloopDriveNoEnterOutcome(decision)
		report.Outcome = outcome
		return finishSuperloopDrive(stdout, stderr, *asJSON, report, code)
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

	// EXECUTE (opt-in) — with --execute, actually RUN the member's own front door behind
	// the held lease and land its witness (the follow-on to admit-and-surface, #2224). A
	// non-runnable front door (skill/container/none) is surfaced, never faked. Without
	// --execute this is skipped and the drive stays surface-only, byte-for-byte.
	if *execute {
		report.Exec = superloopExecuteMember(stderr, *ledger, s.Name, decision, admit.Lease, execTimeout)
	}

	// RE-FOLD — re-walk and fold; the aggregate re-fold after the member run is the exit
	// check. A driven-but-unwitnessed member (unmeasured/dark) keeps the re-fold
	// unsatisfied, so it can never satisfy the intent — the exit reflects that honestly.
	// The issue-target gate re-folds too: a member run that progressed issues moves the
	// live count, and an unmet headline keeps the re-fold unsatisfied.
	refold, code := superloopRefoldExit(root, s)
	report.Refold = refold
	return finishSuperloopDrive(stdout, stderr, *asJSON, report, code)
}

// superloopBatchEnv is the deployment-wide opt-in knob for the default drive batch
// size (see the BATCH comment in runSuperloopDrive). Named as a const so the test and
// the reader share one source of truth for the env name.
const superloopBatchEnv = "FAK_SUPERLOOP_BATCH"

// resolveSuperloopBatch folds the --batch flag with the FAK_SUPERLOOP_BATCH env into
// the effective batch size. An explicit --batch on the command line wins (it is the
// stronger, more local signal); otherwise a parseable env value raises the default.
// Unset or unparseable env ⇒ the flag's own default (1), so a box without the knob is
// byte-for-byte the historical single-member drive. An unparseable value is surfaced
// on stderr and ignored (fail-closed to the safe default), never silently swallowed.
func resolveSuperloopBatch(fs *flag.FlagSet, flagBatch int, stderr io.Writer) int {
	explicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "batch" {
			explicit = true
		}
	})
	if explicit {
		return flagBatch
	}
	env := strings.TrimSpace(os.Getenv(superloopBatchEnv))
	if env == "" {
		return flagBatch
	}
	n, err := strconv.Atoi(env)
	if err != nil {
		fmt.Fprintf(stderr, "fak superloop drive: ignoring unparseable %s=%q (want an integer); using batch %d\n",
			superloopBatchEnv, env, flagBatch)
		return flagBatch
	}
	return n
}

// superloopDriveBatchReport is the folded verdict of a batch drive: the top-K
// worst-first members offered, how each fared at the SHARED admission gate, and the
// single re-fold that is the exit check. Outcome mirrors the single drive's
// vocabulary at the aggregate: "satisfied"/"shortfall" when nothing was offered,
// "entered" when at least one member was admitted (others may have been refused —
// each refusal token is surfaced, never bypassed), or "refused" when every offered
// member was refused by the gate (no member entered, no re-fold).
type superloopDriveBatchReport struct {
	Schema   string                       `json:"schema"`
	Intent   string                       `json:"intent"`
	Outcome  string                       `json:"outcome"`
	Batch    int                          `json:"batch"`
	Offered  int                          `json:"offered"`
	Admitted int                          `json:"admitted"`
	Refused  int                          `json:"refused"`
	Decision superloop.BatchDriveDecision `json:"decision"`
	Entries  []superloopDriveBatchEntry   `json:"entries,omitempty"`
	Refold   *superloop.WalkReport        `json:"refold,omitempty"`
}

// superloopDriveBatchEntry is one member's outcome within a batch drive: the pure
// per-member decision, the gate's ruling, and whether it was entered. A non-entered
// entry carries the gate's refusal token (Admission.Status) — the batch surfaces it
// rather than bypassing it.
type superloopDriveBatchEntry struct {
	Decision  superloop.DriveDecision `json:"decision"`
	Admission superloopDriveAdmit     `json:"admission"`
	Entered   bool                    `json:"entered"`
	// Exec, when set, records the `--execute` run of this admitted member's front door
	// behind its member-scoped lease (nil when not requested or the member was refused).
	Exec *superloopDriveExec `json:"exec,omitempty"`
}

// verifyDriveRepositoryTarget binds a live drive to ONE declared GitHub identity.
// Repository scope is authority, not intent: the operator's immutable owner/repo must be
// corroborated by the selected workspace's own origin before the drive walks candidates,
// takes a lease, writes a ledger line, or runs a front door. Missing, malformed, unknown,
// and mismatched targets all refuse with a typed token so an operator can tell "you did
// not declare a target" from "this checkout is not the repository you declared"
// (docs/notes/FLEET-REPOSITORY-TARGETING-INCIDENT-2026-08-13.md).
func verifyDriveRepositoryTarget(root, expected string) error {
	expected = strings.TrimSpace(expected)
	if expected == "" {
		return fmt.Errorf("REPO_TARGET_REQUIRED: --execute requires --repo owner/name")
	}
	if !validOwnerRepo(expected) {
		return fmt.Errorf("REPO_TARGET_INVALID: --repo must be owner/name, got %q", expected)
	}
	cmd := exec.Command("git", "-C", root, "config", "--get", "remote.origin.url")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("REPO_TARGET_UNKNOWN: cannot resolve origin for workspace %q", root)
	}
	actual := repoFromRemoteURL(strings.TrimSpace(string(out)))
	if !validOwnerRepo(actual) {
		return fmt.Errorf("REPO_TARGET_UNKNOWN: workspace %q origin does not resolve to GitHub owner/name", root)
	}
	if !strings.EqualFold(expected, actual) {
		return fmt.Errorf("REPO_TARGET_MISMATCH: requested %s, workspace resolves %s", expected, actual)
	}
	return nil
}

// runSuperloopDriveBatch is the impure shell for a fan-out drive. It WALKs the
// members (same reads as the single drive), SELECTs the top-K worst-first via the
// pure superloop.DriveBatch, then passes EACH offered member through the SAME
// admission gate any spawn passes — under a MEMBER-SCOPED lease identity so K holds
// coexist. Because each admitted member's lease stays held while the next is gated,
// the gate itself enforces the batch's mutual disjointness (a later member whose
// region overlaps an already-admitted one refuses COLLISION_RISK) on top of
// disjointness from live peer leases. Admitted members are entered (their front
// door surfaced, an admitted witness recorded); refused members surface their token
// and are skipped (no private spawn path). One re-fold over the whole intent is the
// exit check.
func runSuperloopDriveBatch(stdout, stderr io.Writer, asJSON bool, root string, s superloop.Super, lane string, tree []string, ledger string, k int, execute bool, execTimeout time.Duration) int {
	rep := superloopWalkNow(root, s)
	bdec := superloop.DriveBatch(rep, k)
	report := superloopDriveBatchReport{Schema: superloop.BatchDriveSchema, Intent: s.Name, Batch: k, Decision: bdec, Offered: len(bdec.Members)}

	if !bdec.Enter {
		// Nothing worst-first to offer — clean ONLY when satisfied; an unsatisfied
		// empty worklist is an unmet headline gate (#3147), not a finished night.
		outcome, code := superloopDriveNoEnterOutcome(superloop.DriveDecision{
			Enter: false, Satisfied: bdec.Satisfied, IssueShortfall: bdec.IssueShortfall, Reason: bdec.Reason,
		})
		report.Outcome = outcome
		return finishSuperloopDriveBatch(stdout, stderr, asJSON, report, code)
	}

	// ADMISSION — offer each member to the SAME gate under a member-scoped lease
	// identity. Every admitted lease stays held (LIFO release at return) while the
	// remaining members are gated, so the gate refuses a later member whose region
	// overlaps one already admitted this batch (mutual disjointness) exactly as it
	// refuses overlap with a live peer lease.
	var releases []func()
	defer func() {
		for i := len(releases) - 1; i >= 0; i-- {
			releases[i]()
		}
	}()
	for _, m := range bdec.Members {
		scope := superloopBatchMemberScope(s.Name, m.Member)
		memberLane, memberTree := superloopBatchMemberRegion(lane, tree, s.Name, m.Member)
		admit, release := superloopDriveAdmitGate(root, memberLane, memberTree, scope)
		releases = append(releases, release)
		entry := superloopDriveBatchEntry{Decision: m, Admission: admit}
		if admit.Admitted {
			entry.Entered = true
			report.Admitted++
			recordSuperloopDriveAdmit(ledger, s.Name, m, loopmgr.StatusAdmitted, "ENTERED",
				"entered "+string(m.Member.Kind)+" "+m.Member.Ref+": "+m.Action, admit.Lease)
			// EXECUTE (opt-in) — run this admitted member's own front door behind its
			// member-scoped lease. Non-runnable front doors are surfaced, never faked.
			if execute {
				entry.Exec = superloopExecuteMember(stderr, ledger, s.Name, m, admit.Lease, execTimeout)
			}
		} else {
			report.Refused++
			recordSuperloopDriveAdmit(ledger, s.Name, m, loopmgr.StatusRefused, admit.Status, admit.Detail, admit.Lease)
			fmt.Fprintf(stderr, "fak superloop drive: batch member %s %s refused: %s %s\n",
				m.Member.Kind, m.Member.Ref, admit.Status, admit.Detail)
		}
		report.Entries = append(report.Entries, entry)
	}

	if report.Admitted == 0 {
		// Every offered member was refused by the gate — surface it as a refusal
		// (exit 3), enter nothing, and DO NOT re-fold (a re-fold would imply work
		// happened). The tokens are already on the ledger and stderr.
		report.Outcome = "refused"
		return finishSuperloopDriveBatch(stdout, stderr, asJSON, report, 3)
	}

	// RE-FOLD — one re-walk over the whole intent after the batch is the exit check.
	// A driven-but-unwitnessed member keeps the re-fold unsatisfied, so a batch can
	// never satisfy the intent on surfacing alone.
	report.Outcome = "entered"
	refold, code := superloopRefoldExit(root, s)
	report.Refold = refold
	return finishSuperloopDriveBatch(stdout, stderr, asJSON, report, code)
}

// superloopWalkNow reads every member's status off the cheap committed surfaces and folds
// ONE walk report, with the issue-target progress opts applied. Both drives walk twice --
// once to select the worst-first member(s), once to re-fold as the exit check -- and every
// one of those walks has to read the intent the same way, or a re-fold could disagree with
// the selection that preceded it for no reason but a drifted argument list. The walk mutates
// nothing.
func superloopWalkNow(root string, s superloop.Super) superloop.WalkReport {
	return superloop.Walk(s, collectSuperloopStatuses(root, s), issueProgressWalkOpts(root, s)...)
}

// superloopRefoldExit performs the post-run re-fold and returns it together with the drive's
// exit code. The rule is the same for the single and the batch drive: 0 only when the
// re-folded intent is SATISFIED, otherwise 1. A driven-but-unwitnessed (unmeasured/dark)
// member keeps the re-fold unsatisfied, so neither drive can report done on surfacing alone.
func superloopRefoldExit(root string, s superloop.Super) (*superloop.WalkReport, int) {
	refold := superloopWalkNow(root, s)
	if refold.Satisfied {
		return &refold, 0
	}
	return &refold, 1
}

// superloopBatchMemberScope builds a per-member intent token so the shared gate
// mints a DISTINCT lease id per member ("loop-superloop-<scope>"), letting K holds
// coexist within one batch. The intent stays the prefix so the leases remain
// recognizably this super loop's.
func superloopBatchMemberScope(intent string, m superloop.Member) string {
	return intent + "-m-" + superloopMemberSlug(m)
}

// superloopMemberSlug reduces a member's kind+ref to a lease-safe slug (alnum, dash,
// underscore; every other rune folds to a dash), so distinct members yield distinct
// lease identities and region tokens.
func superloopMemberSlug(m superloop.Member) string {
	return strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		default:
			return '-'
		}
	}, string(m.Kind)+"-"+m.Ref)
}

// superloopBatchMemberRegion resolves the region a batch member is admitted under.
// When the operator declared an explicit shared region (--lane/--tree), it is
// honored verbatim: the operator has fenced the batch to one tree and accepts that
// members serialize on it (the gate refuses overlap). With no operator region, the
// region is scoped PER MEMBER so distinct members are mutually disjoint (admitted
// concurrently) while a peer entering the SAME member refuses COLLISION_RISK — the
// interior-node coordination unit is the member, not raw files (the member's child
// re-acquires its own file lease behind its front door).
func superloopBatchMemberRegion(opLane string, opTree []string, intent string, m superloop.Member) (string, []string) {
	if strings.TrimSpace(opLane) != "" || len(opTree) > 0 {
		return opLane, opTree
	}
	return "", []string{"superloop/" + cleanDispatchLeaseToken(intent) + "/" + superloopMemberSlug(m) + "/**"}
}

func finishSuperloopDriveBatch(stdout, stderr io.Writer, asJSON bool, report superloopDriveBatchReport, code int) int {
	if asJSON {
		if rc := encodeJSONOrFail(stdout, stderr, report, "fak superloop drive"); rc != 0 {
			return rc
		}
		return code
	}
	renderSuperloopDriveBatch(stdout, report)
	return code
}

func renderSuperloopDriveBatch(w io.Writer, r superloopDriveBatchReport) {
	fmt.Fprintf(w, "superloop drive (batch %d): %s\n", r.Batch, r.Intent)
	if r.Outcome == "satisfied" || r.Outcome == "shortfall" {
		fmt.Fprintf(w, "  nothing to enter — %s\n", r.Decision.Reason)
		return
	}
	fmt.Fprintf(w, "  offered %d worst-first, admitted %d, refused %d\n", r.Offered, r.Admitted, r.Refused)
	for _, e := range r.Entries {
		d := e.Decision
		verb := "REFUSED"
		if e.Entered {
			verb = "entered"
		}
		dark := ""
		if d.Dark {
			dark = ", DARK"
		}
		fmt.Fprintf(w, "  [%s] %s %s (rank %d, debt %d%s) — admission %s", verb, d.Member.Kind, d.Member.Ref, d.Rank, d.Debt, dark, e.Admission.Status)
		if e.Admission.Lease != "" {
			fmt.Fprintf(w, " (lease %s)", e.Admission.Lease)
		}
		fmt.Fprintln(w)
		if e.Entered {
			fmt.Fprintf(w, "        action: %s\n", d.Action)
			renderSuperloopDriveExec(w, "        ", e.Exec)
		}
	}
	if r.Outcome == "refused" {
		fmt.Fprintf(w, "  → refused: every offered member's admission gate refused; the tokens are surfaced, not bypassed\n")
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

// recordSuperloopDriveAdmit appends the drive's admission decision to the loop ledger
// with the standing witness vocabulary (StatusAdmitted / StatusRefused), keyed on the
// super loop's own id and carrying the driven member and — when a region lease was
// consulted — the held lease as evidence. Best-effort: the drive's success is the enter
// + re-fold, not this observability row.
func recordSuperloopDriveAdmit(ledger, intent string, d superloop.DriveDecision, status loopmgr.RunStatus, reason, detail, leaseID string) {
	recordSuperloopDriveEvent(ledger, intent, d, loopmgr.EventAdmit, status, reason, detail, leaseID)
}

// recordSuperloopDriveEvent is the general ledger-append the admit and execute rungs
// share: same LoopID/Source/evidence shape, but the caller chooses the event KIND —
// EventAdmit for an admission ruling, EventStart/EventEnd for the `--execute` run's
// running/witnessed_done/failed witnesses — so the member's own lifecycle reads on the
// loop ledger with the standing vocabulary. Best-effort, same as the admit row.
func recordSuperloopDriveEvent(ledger, intent string, d superloop.DriveDecision, kind loopmgr.EventKind, status loopmgr.RunStatus, reason, detail, leaseID string) {
	if strings.TrimSpace(ledger) == "" {
		return
	}
	ev := loopmgr.Event{
		LoopID:  "superloop-" + intent,
		Kind:    kind,
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
	// Both "satisfied" and "shortfall" enter nothing; the decision Reason already carries
	// the honest clean-vs-unmet-headline text, and the exit code distinguishes them.
	if r.Outcome == "satisfied" || r.Outcome == "shortfall" {
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
	renderSuperloopDriveExec(w, "  ", r.Exec)
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

// renderSuperloopDriveExec prints the `--execute` outcome for one member (nothing when
// exec was not requested). A runnable front door that ran shows its exit and whether it
// witnessed_done; a non-runnable one (skill/container/none) shows it was surfaced, not
// run — so the operator can always tell a real member run from a surfaced pointer.
func renderSuperloopDriveExec(w io.Writer, indent string, ex *superloopDriveExec) {
	if ex == nil {
		return
	}
	if !ex.Ran {
		fmt.Fprintf(w, "%sexecute: %s front door surfaced, not run — %s\n", indent, ex.Kind, ex.Note)
		return
	}
	verdict := "WITNESSED_DONE"
	if !ex.Witnessed {
		verdict = "FAILED"
	}
	deadline := ""
	if ex.TimeoutMinutes > 0 {
		deadline = fmt.Sprintf(", deadline %dm (%s)", ex.TimeoutMinutes, ex.TimeoutSource)
	}
	fmt.Fprintf(w, "%sexecute: ran `%s` → exit %d (%s%s)\n", indent, ex.Command, ex.ExitCode, verdict, deadline)
}
