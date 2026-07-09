// scenario_stamp.go — the commit-stamp + trunk-fidelity scenario + grader
// (#2733, epic #2721 concept #1). Does the model produce a commit the fak way:
// a Conventional-Commits subject ending in the correct `(fak <leaf>)` trailer,
// committed BY EXPLICIT PATH (never `git add -A`), landing on `main` (never a
// branch/worktree), with a diff that actually backs the subject's claim? This is
// the rule set (AGENTS.md, dos.toml [stamp], OFF_TRUNK) that most often bites an
// agent, so the row records the SPECIFIC failure class rather than a bare
// pass/fail — the localizable signal the benchmark exists to produce.
//
// The scenario grades through two real referees, never the model's own prose:
//
//   - the ship-stamp grammar + leaf-match rung is the REAL pre-commit lint,
//     hooks.LintCommitMessage — the same path-aware `(fak <leaf>)` oracle the
//     commit hook binds to (the in-binary twin of `fak commit --preview`), so
//     the stamped leaf is checked against the lane the committed paths actually
//     live in, not against a recording;
//   - the diff-witness rung is a real dos_verify + dos_commit_audit reading
//     (the #2732 Referee surface): shipped from git evidence AND the subject
//     matches its own diff (verdict OK, witness diff-witnessed), so a
//     CLAIM_UNWITNESSED subject-only commit fails even when the model says
//     "done".
//
// A pass requires ALL of the #2733 DoD rungs: the stamp grammar parses, the
// `(fak <leaf>)` leaf matches the touched paths, the commit is on `main`, it was
// committed by explicit path (no `git add -A` over-staging), and the audit reads
// diff-witnessed. Each failing episode buckets into exactly one recorded class:
//
//   - off_trunk         — committed on a branch/worktree, not main (OFF_TRUNK).
//   - absent_trailer    — no `(fak <leaf>)` ship-stamp parses at all.
//   - wrong_leaf        — a stamp parses but its leaf is not a lane the touched
//     paths live in (a `(fak gateway)` on an internal/conceptbench edit).
//   - over_staging      — a `git add -A` swept sibling paths beyond the task's
//     touched set into the commit (the by-explicit-path rule violated).
//   - claim_unwitnessed — dos_commit_audit reports CLAIM_UNWITNESSED / the diff
//     does not back the subject (subject-only), or dos_verify says it never
//     shipped.
package conceptbench

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hooks"
)

// StampOutcome is the bucket for one commit-stamp episode: a pass, or exactly
// one of the #2733 named failure classes.
type StampOutcome string

const (
	StampPass          StampOutcome = "pass"
	StampOffTrunk      StampOutcome = "off_trunk"
	StampAbsentTrailer StampOutcome = "absent_trailer"
	StampWrongLeaf     StampOutcome = "wrong_leaf"
	StampOverStaging   StampOutcome = "over_staging"
	StampUnwitnessed   StampOutcome = "claim_unwitnessed"
)

// Pass reports whether the episode is a clean fak-way commit. It is all-or-
// nothing: the #2733 DoD requires ALL rungs (stamp parses, leaf matches, on
// main, by explicit path, diff-witnessed), so any failure class is a fail.
func (o StampOutcome) Pass() bool { return o == StampPass }

// Score folds the bucket to a per-row score: a clean commit is 1, any failure
// class is 0 (no partial credit — a commit that lands off-trunk or unwitnessed
// is not partially "the fak way").
func (o StampOutcome) Score() float64 {
	if o == StampPass {
		return 1
	}
	return 0
}

// StampTask is one corpus task: the ground-truth files the change requires
// editing and the leaf those paths imply. TouchedPaths is the by-explicit-path
// reference the over-staging check measures against (a `git add -A` stages more
// than this); ExpectLeaf is the primary leaf a correct `(fak <leaf>)` stamp
// names (for evidence — the leaf-match rung is decided by the real lint, which
// accepts any touched lane so a two-leaf commit that picks either leaf passes).
type StampTask struct {
	Name         string
	Prompt       string   // the instruction naming the change to make and the fak-way rules
	TouchedPaths []string // the ground-truth files the task requires editing
	ExpectLeaf   string   // the primary leaf those paths imply
}

// StampTasks returns the committed task set (>=2 per the #2733 scope): a clean
// single-file change (one leaf), and a change that spans two leaves where the
// model must pick the right leaf or split the commit.
func StampTasks() []StampTask {
	return []StampTask{
		{
			Name:         "single_file",
			Prompt:       "Edit internal/conceptbench/report.go to add a column to the leaderboard, then commit it the fak way: a verb-led Conventional-Commits subject ending in the correct `(fak <leaf>)` trailer, staged by explicit path, on main.",
			TouchedPaths: []string{"internal/conceptbench/report.go"},
			ExpectLeaf:   "conceptbench",
		},
		{
			Name:         "spans_two_leaves",
			Prompt:       "Your change edits both internal/conceptbench/grade.go and internal/hooks/commitstamp.go. Pick the primary leaf for the ship-stamp (or split into two commits) and commit the fak way, staged by explicit path, on main.",
			TouchedPaths: []string{"internal/conceptbench/grade.go", "internal/hooks/commitstamp.go"},
			ExpectLeaf:   "conceptbench",
		},
	}
}

// StampCommit is the model's proposed commit for one episode — the act the
// grader reads, never the model's "done" prose. Subject is the first line it
// wrote; StagedPaths is the set it actually committed (a `git add -A` sweeps in
// paths beyond the task's touched set); Branch is where it landed ("main" or a
// feature branch); Ref is the commit ref the audit reads.
type StampCommit struct {
	Subject     string
	StagedPaths []string
	Branch      string
	Ref         string
}

// StampLint is the ship-stamp lint's reading of a proposed commit: whether a
// `(fak <leaf>)` stamp parses, the leaf it names, and whether that leaf matches
// the lane the committed paths live in.
type StampLint struct {
	StampParses bool
	Kind        string   // "trailer" | "direct" | "release" | "exempt" | "none"
	Leaf        string   // the stamped leaf, "" if unstamped
	LeafMatches bool     // the leaf is one of the lanes the paths fall in
	PathLanes   []string // the lanes the committed paths fall in
	Raw         string
}

// StampLinter is the narrow ship-stamp surface this scenario consumes — one
// path-aware stamp lint over the model's subject and committed paths.
// RuleStampLint binds it to the real hooks.LintCommitMessage twin.
type StampLinter interface {
	LintStamp(subject string, paths []string) StampLint
}

// RuleStampLint is the real ship-stamp reading: it routes the model's subject
// and committed paths through hooks.LintCommitMessage — the same pre-commit,
// path-aware `(fak <leaf>)` oracle the commit hook binds to — reading the lane
// taxonomy from the repo's own dos.toml at Root. So the leaf-match rung is the
// live grammar (a `(fak gateway)` stamp on an internal/conceptbench edit reads
// as a mismatch), not a canned answer.
type RuleStampLint struct {
	Root string // repo root, for reading dos.toml's lane taxonomy
}

var _ StampLinter = RuleStampLint{}

func (r RuleStampLint) LintStamp(subject string, paths []string) StampLint {
	rep := hooks.LintCommitMessage(subject, paths, r.Root)
	parses := (rep.StampKind == "trailer" || rep.StampKind == "direct") && rep.Leaf != ""
	// LeafMatches is vacuously true in the lint when no lane could be inferred;
	// require a resolved lane so an unbindable-path commit is not credited.
	leafMatches := rep.LeafMatches && len(rep.PathLanes) > 0
	return StampLint{
		StampParses: parses,
		Kind:        rep.StampKind,
		Leaf:        rep.Leaf,
		LeafMatches: leafMatches,
		PathLanes:   rep.PathLanes,
		Raw:         fmt.Sprintf("stamp_kind=%s leaf=%q leaf_matches=%v path_lanes=%v", rep.StampKind, rep.Leaf, rep.LeafMatches, rep.PathLanes),
	}
}

// CommitReferee is the narrow dos referee surface this scenario consumes — one
// dos_verify (shipped from git evidence) and one dos_commit_audit (the subject
// matches its own diff) call. The #2732 Referee satisfies it; a test binds
// RecordedReferee (a response recorded from a live referee and bound to the
// fixture), the honest offline path the #2732 acceptance permits.
type CommitReferee interface {
	Verify(ref, claim string) VerifyResult
	CommitAudit(ref, subject string) CommitAuditResult
}

// the #2732 adapter's referee is this scenario's referee — pinned at compile time.
var _ CommitReferee = Referee(nil)

// StampRow is the scenario's graded row for one (task, commit) episode. It names
// every rung's reading and the SPECIFIC failure class (FailureClass), so a
// localizable signal survives — never collapsed to a bare pass/fail, the #2733
// DoD's recorded distinction.
type StampRow struct {
	Task          string
	Subject       string
	Leaf          string
	PathLanes     []string
	StampParses   bool
	LeafMatches   bool
	OnMain        bool
	OverStaged    bool
	ExtraStaged   []string // the sibling paths a `git add -A` swept in, for audit
	Shipped       bool
	DiffWitnessed bool
	Outcome       StampOutcome
	FailureClass  string // the recorded failure class ("" on pass)
	Score         float64
	Pass          bool
	WitnessSource string // dos_verify + dos_commit_audit (stamp lint in Evidence)
	Evidence      string
}

// GradeStamp grades one episode: run the model's subject + committed paths
// through the real ship-stamp lint, the branch through the on-main check, the
// committed set through the by-explicit-path (over-staging) check, and the
// commit through a real dos_verify + dos_commit_audit reading — then bucket the
// FIRST failing rung, in the precedence a fak-way commit is refused: off-trunk
// (the OFF_TRUNK guard refuses before the commit lands) → the ship-stamp grammar
// (absent trailer, then wrong leaf) → by-explicit-path (over-staging) → the
// post-commit diff-witness. A test fixture that isolates one failure lands in
// exactly its class.
func GradeStamp(task StampTask, commit StampCommit, lint StampLinter, ref CommitReferee) StampRow {
	ln := lint.LintStamp(commit.Subject, commit.StagedPaths)
	v := ref.Verify(commit.Ref, commit.Subject)
	a := ref.CommitAudit(commit.Ref, commit.Subject)

	onMain := strings.EqualFold(strings.TrimSpace(commit.Branch), "main")
	extra := overStagedPaths(task.TouchedPaths, commit.StagedPaths)
	overStaged := len(extra) > 0
	diffWitnessed := strings.EqualFold(a.Verdict, "OK") && a.Witness == "diff-witnessed" && !a.ClaimUnwitnessed
	witnessed := v.Shipped && diffWitnessed

	var outcome StampOutcome
	switch {
	case !onMain:
		outcome = StampOffTrunk
	case !ln.StampParses:
		outcome = StampAbsentTrailer
	case !ln.LeafMatches:
		outcome = StampWrongLeaf
	case overStaged:
		outcome = StampOverStaging
	case !witnessed:
		outcome = StampUnwitnessed
	default:
		outcome = StampPass
	}

	failureClass := ""
	if outcome != StampPass {
		failureClass = string(outcome)
	}

	ev := fmt.Sprintf("task=%s branch=%q on_main=%v stamp_parses=%v leaf=%q leaf_matches=%v over_staged=%v extra=%v shipped=%v diff_witnessed=%v outcome=%s",
		task.Name, commit.Branch, onMain, ln.StampParses, ln.Leaf, ln.LeafMatches, overStaged, extra, v.Shipped, diffWitnessed, outcome)

	return StampRow{
		Task:          task.Name,
		Subject:       commit.Subject,
		Leaf:          ln.Leaf,
		PathLanes:     ln.PathLanes,
		StampParses:   ln.StampParses,
		LeafMatches:   ln.LeafMatches,
		OnMain:        onMain,
		OverStaged:    overStaged,
		ExtraStaged:   extra,
		Shipped:       v.Shipped,
		DiffWitnessed: diffWitnessed,
		Outcome:       outcome,
		FailureClass:  failureClass,
		Score:         outcome.Score(),
		Pass:          outcome.Pass(),
		WitnessSource: WitnessDosVerify + "+" + WitnessDosCommitAudit,
		Evidence:      joinRaw(ev, ln.Raw, v.Raw, a.Raw),
	}
}

// overStagedPaths returns the committed paths that are NOT in the task's
// ground-truth touched set — the sibling files a `git add -A` swept in. An empty
// result means the commit was staged by explicit path (the fak-way rule). Paths
// are normalized (backslashes → slashes, "./" stripped) so a Windows-committed
// path matches its POSIX touched-set entry.
func overStagedPaths(touched, staged []string) []string {
	want := map[string]bool{}
	for _, p := range touched {
		want[normStampPath(p)] = true
	}
	var extra []string
	for _, p := range staged {
		if !want[normStampPath(p)] {
			extra = append(extra, p)
		}
	}
	sort.Strings(extra)
	return extra
}

func normStampPath(p string) string {
	return strings.TrimPrefix(strings.ReplaceAll(p, "\\", "/"), "./")
}
