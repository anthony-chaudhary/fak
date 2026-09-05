package hooks

// gatescope.go — the committed SCOPE CLASSIFICATION for every gate in this package (#5931).
//
// landstree.go builds the `HEAD ⊕ staged` view. This file decides WHICH gates read through it,
// and — the harder half — writes down why each of the others does not. A seam fixed for six
// gates and left silently open for twenty is not a fixed seam: the next reader has no way to tell
// a gate that was reasoned about from one that was missed. So the table is exhaustive over
// internal/hooks/gate_*.go and gatescope_test.go re-derives that file list from the directory,
// which means adding a gate file without classifying it FAILS THE BUILD rather than quietly
// inheriting the working-tree read.
//
// THE THREE CLASSES:
//
//	LANDS_TREE          judged over HEAD ⊕ staged — the tree this commit produces. Wired through
//	                    LandsTreeView by scopeGates below; the gate body is untouched.
//	WORKTREE_BY_DESIGN  NOT moved, with a written reason. The name is about the DEFAULT it keeps,
//	                    not a claim that every such gate reads the disk: the class covers a gate
//	                    whose input is deliberately something other than the landing tree — the
//	                    shared checkout, an untracked private file, or the commit message.
//	TREE_TWIN           the whole-tree `--audit-tree` / `fak hygiene` mode, whose denominator is
//	                    the tree ON PURPOSE (a non-goal of #5931 is removing it). These run over
//	                    *TrackedTree, never over a staged diff, and cannot refuse a commit.
//
// ⛔ WHY NOT "MOVE EVERYTHING". Two gates would LOSE information they need, and moving them would
// trade a false-refusal bug for a silent-miss bug — the strictly worse direction for a security
// gate. PUBLIC_LEAK's needle list (tools/_registry/scrub_needles.private.json) is untracked by
// design, so the change-set view cannot see it and the gate would quietly scan with fewer needles.
// DUPLICATION reads sibling sources off disk on purpose, with its own written argument
// (gate_duplication.go neighborBytes): a block that was MOVED is correctly not flagged there,
// where the staged-blob read would still see the old copy and cry duplicate.

// The classification vocabulary, named once so the table, the wiring and the test cannot drift.
const (
	ClassLandsTree = "LANDS_TREE"
	ClassWorktree  = "WORKTREE_BY_DESIGN"
	ClassTreeTwin  = "TREE_TWIN"
)

// The seam a gate is invoked from. It is recorded because the class alone is ambiguous: a gate can
// only be LANDS_TREE if something actually hands it a staged diff, and gatescope_test.go asserts
// exactly that rather than trusting the label.
const (
	SeamPreCommit = "pre-commit" // PreCommitGates(), over *StagedDiff
	SeamHygiene   = "hygiene"    // HygieneGates(), over *TrackedTree
	SeamCommitMsg = "commit-msg" // the commit-message seam, over the message text
)

// gateScopeRow is one gate's row: where it lives, which seam runs it, what it is judged over, and —
// for anything not moved — the one line saying why.
type gateScopeRow struct {
	Gate  string // the Finding.Gate value / registry name
	File  string // the internal/hooks/gate_*.go it is defined in
	Seam  string // SeamPreCommit | SeamHygiene | SeamCommitMsg
	Class string // ClassLandsTree | ClassWorktree | ClassTreeTwin
	Why   string // required for every class except LANDS_TREE
}

// gateScopes returns the classification table. Exported so `fak hooks` (and any auditor) can print
// the same rows the wiring reads, rather than a second copy of the same claim in a doc.
func gateScopes() []gateScopeRow {
	return []gateScopeRow{
		// ---- pre-commit, moved to the landing tree -------------------------------------------
		{"DOC_PLACEMENT", "gate_docplacement.go", SeamPreCommit, ClassLandsTree, ""},
		{"BROKEN_LINK", "gate_brokenlink.go", SeamPreCommit, ClassLandsTree, ""},
		{"INDEX_SYNC", "gate_indexsync.go", SeamPreCommit, ClassLandsTree, ""},
		{"FILE_ADMISSION", "gate_fileadmission.go", SeamPreCommit, ClassLandsTree, ""},
		{"UNTIERED_LEAF", "gate_untieredleaf.go", SeamPreCommit, ClassLandsTree, ""},
		{"CART_BEFORE_HORSE", "gate_cartbeforehorse.go", SeamPreCommit, ClassLandsTree, ""},
		{"CONCEPT_ADMISSION", "gate_conceptadmission.go", SeamPreCommit, ClassLandsTree, ""},
		{"TRUST_WIDENING", "gate_trustwidening.go", SeamPreCommit, ClassLandsTree, ""},
		{"GOFMT", "gate_gofmt.go", SeamPreCommit, ClassLandsTree, ""},
		{"SECRET_SHAPE", "gate_secretshape.go", SeamPreCommit, ClassLandsTree, ""},
		{"PROVENANCE_LABEL", "gate_provenance.go", SeamPreCommit, ClassLandsTree, ""},
		{"HARDWARE_TELL", "gate_hardware.go", SeamPreCommit, ClassLandsTree, ""},
		{"NATIVE_FIRST", "gate_nativefirst.go", SeamPreCommit, ClassLandsTree, ""},
		{"E2E_OVER_MOCKS", "gate_e2eovermocks.go", SeamPreCommit, ClassLandsTree, ""},
		{"PRIOR_ART", "gate_priorart.go", SeamPreCommit, ClassLandsTree, ""},
		{"MICROHARNESS_WITNESS", "gate_microharnesswitness.go", SeamPreCommit, ClassLandsTree, ""},
		{"BARE_COMMIT_SWEEP", "gate_barecommitsweep.go", SeamPreCommit, ClassLandsTree, ""},
		{"PARALLEL_FABRIC_NUDGE", "gate_microcontext_nudge.go", SeamPreCommit, ClassLandsTree, ""},
		// GIT_HYGIENE_BYPASS judges ADDED LINES only — it makes no fileProbe read at all — so the
		// wrapper is a no-op for it in the same way it is for CONCEPT_FRESHNESS. It is listed
		// LANDS_TREE because that is the change set its verdict is about: the view shares the very
		// AddedByFile/StagedPaths slices the bare diff carries, so "judged over HEAD ⊕ staged" is
		// true of it by construction rather than by wiring.
		{"GIT_HYGIENE_BYPASS", "gate_githygiene.go", SeamPreCommit, ClassLandsTree, ""},
		// CONCEPT_FRESHNESS is the prior art for this whole ticket inside fak: it already scores
		// HEAD-plus-your-pathspec via conceptcatalog.CheckGitTree (#5534/#5829) and says so in the
		// refusal text it prints. Listed LANDS_TREE because that is what it judges; the wrapper is
		// a no-op for it (it takes no reads through fileProbe), and that is the point — one gate
		// solved this for itself, and #5931 is the same answer applied to the shared seam.
		{"CONCEPT_FRESHNESS", "gate_conceptfreshness.go", SeamPreCommit, ClassLandsTree, ""},

		// ---- pre-commit, deliberately NOT moved ----------------------------------------------
		{"PUBLIC_LEAK", "gate_publicleak.go", SeamPreCommit, ClassWorktree,
			"its private needle list (tools/_registry/scrub_needles.private.json) is UNTRACKED by design, so the change-set view cannot see it and the gate would silently scan with fewer needles — a scope fix must never turn a security gate quiet"},
		{"DESKTOP_POPUP_REGRESSION", "gate_desktoppopup.go", SeamPreCommit, ClassWorktree,
			"reads each complete candidate-index source file independently; no sibling-state dependency"},
		{"DUPLICATION", "gate_duplication.go", SeamPreCommit, ClassWorktree,
			"neighborBytes reads sibling sources off disk deliberately (its own doc): a MOVED block is correctly not flagged there, where the staged-blob read would still see the old copy and cry duplicate — and it is advisory by default, so a peer-dirty read cannot refuse a commit"},
		{"COMMENT_QUALITY", "gate_commentquality.go", SeamPreCommit, ClassWorktree,
			"reviews changed implementation comments from the staged diff; full-file reads provide comment context and the gate remains advisory by default"},
		{"COMMIT_MSG", "gate_commitmsg.go", SeamCommitMsg, ClassWorktree,
			"judges the commit MESSAGE, not a tree — its input is already scoped to this commit and no peer WIP can reach it"},
		{"FRESH_DELETION", "gate_freshdeletion.go", SeamCommitMsg, ClassWorktree,
			"judges the commit message against paths this commit deletes — message-seam, not a tree probe, so the view has nothing to substitute"},

		// ---- whole-tree twins: the reporting mode, unaffected ---------------------------------
		// Every row below runs over *TrackedTree from `fak hygiene` / `--audit-tree`, where the
		// denominator IS the tree on purpose (#5931 non-goal). None can refuse a commit, so a
		// peer's unstaged edit shows up as a hygiene report line, never as a fleet-wide refusal.
		{"DEAD_CODE", "gate_deadcode.go", SeamHygiene, ClassTreeTwin,
			"registered ONLY as a hygiene tree gate (HygieneGates, DefaultOff) — it has no staged twin to scope, and its whole-tree denominator is the audit sweep that proves the retirement; give it a staged twin and that twin lands LANDS_TREE"},
		{"TIER_DECLARED", "gate_tierdeclared.go", SeamHygiene, ClassTreeTwin,
			"whole-tree hygiene twin of UNTIERED_LEAF, which IS scoped — the commit-boundary refusal is the staged one"},
		{"BRAND_CONSISTENCY", "gate_brandconsistency.go", SeamHygiene, ClassTreeTwin, "whole-tree hygiene sweep"},
		{"SWALLOWED_ERROR", "gate_swallowederror.go", SeamHygiene, ClassTreeTwin, "whole-tree hygiene sweep (DefaultOff migration ratchet)"},
		{"GOD_FILE_GROWTH", "gate_godfile.go", SeamHygiene, ClassTreeTwin, "whole-tree hygiene sweep against a frozen baseline"},
		{"NEW_PYTHON_TOOL", "gate_pythongate.go", SeamHygiene, ClassTreeTwin, "whole-tree hygiene sweep against a frozen baseline"},
		{"DEMO_COMMAND", "gate_democommand.go", SeamHygiene, ClassTreeTwin, "whole-tree hygiene sweep (reads the Makefile and demo sources off disk by design)"},
		{"BROWSER_CONTRACT", "gate_browsercontract.go", SeamHygiene, ClassTreeTwin, "whole-tree hygiene sweep (reads browser-demo metadata off disk by design)"},
		{"DEMO_LIVE_LINKS", "gate_demolivelinks.go", SeamHygiene, ClassTreeTwin, "whole-tree hygiene sweep (reads docs/demos.html and social preview off disk by design)"},
		{"GUARD_MCP_STATUS", "gate_guardmcpstatus.go", SeamHygiene, ClassTreeTwin, "whole-tree hygiene sweep (reads agent-live proof packet and evidence artifacts off disk by design)"},
	}
}

// gateScopeFilesWithoutGates names the gate_*.go files that define NO gate, so the exhaustiveness
// test can distinguish "deliberately has no row" from "somebody forgot one". Each entry carries
// its reason for exactly the same purpose the Why column serves in the table.
var gateScopeFilesWithoutGates = map[string]string{
	"gate_tuning.go":     "shared operator knobs (gateEnvInt) for the SIZE gates — declares no gate of its own",
	"gate_affordance.go": "affordance validator (GateAffordanceCompleteness) — validates refusals carry next actions",
}

// gateClass returns the classification for a gate name, or "" when the name is not in the table.
// An unknown name is treated as unclassified and left on the working tree, which is the SAFE
// direction: a gate this file has never heard of keeps the behaviour it shipped with.
func gateClass(name string) string {
	for _, s := range gateScopes() {
		if s.Gate == name {
			return s.Class
		}
	}
	return ""
}

// scopeGates wires the table onto a gate registry: every LANDS_TREE gate is run over the
// HEAD ⊕ staged view, and EVERY gate — moved or not — has its findings stamped with the view that
// produced them.
//
// ⭐ The wrapper is why "no gate body changes to adopt it" is literally true. A gate's Check keeps
// taking *StagedDiff and keeps making the same d.Exists / d.Size / d.FileBytes calls; only which
// change set answers them moves. Classification therefore stays a one-line table edit rather than
// a diff through twenty gate files, which is the property that makes the seam auditable at all.
func scopeGates(gs []Gate) []Gate {
	out := make([]Gate, 0, len(gs))
	for _, g := range gs {
		g.Check = scopedCheck(g.Name, g.Check)
		out = append(out, g)
	}
	return out
}

func scopedCheck(name string, check func(*StagedDiff) ([]Finding, error)) func(*StagedDiff) ([]Finding, error) {
	if check == nil {
		return nil
	}
	if gateClass(name) != ClassLandsTree {
		return func(d *StagedDiff) ([]Finding, error) {
			findings, err := check(d)
			return stampView(findings, ViewWorktree), err
		}
	}
	return func(d *StagedDiff) ([]Finding, error) {
		v, ok := d.LandsTreeView()
		if !ok {
			// Fail-CLOSED to the OLD behaviour, never to silence: no view means no verdict from
			// the view, and the gate must still run. The distinct stamp is what keeps this from
			// being an invisible degradation — a reader chasing a false refusal can see, in the
			// JSON, that the scoping did not happen this run.
			findings, err := check(d)
			return stampView(findings, ViewWorktreeFallback), err
		}
		findings, err := check(v)
		// The gate recorded its #5602 denominator on the view it was handed; move it back before
		// the runner reads it off the original, or every scoped gate regresses to UNREPORTED.
		mergeCandidateNotes(d, v)
		return stampView(findings, ViewLandsTree), err
	}
}

// stampView records, per finding, which change set produced it — the #5931 deliverable that makes
// a false refusal diagnosable from `fak hooks pre-commit -json` alone, WITHOUT taking the commit
// lock to reproduce it. An already-stamped finding keeps its own attribution (a gate that
// deliberately mixes views can say so per finding, and a wrapper must not overwrite that).
func stampView(findings []Finding, view string) []Finding {
	for i := range findings {
		if findings[i].View == "" {
			findings[i].View = view
		}
	}
	return findings
}
