// Package corelockgate is the single owner of the hard-self core-lock question
// that EVERY path to the trunk must ask before a change lands.
//
// WHY IT IS ITS OWN FOUNDATION LEAF (#5392).
//
// The check was born inside internal/safecommit — tier 2, the `fak commit` path —
// because that was the only path that could ask it. Then the sanctioned detached
// per-worker worktree land (internal/workerworktree, tier 1: the path CLAUDE.md
// blesses for build isolation, #1334 / epic #3165) had to ask the SAME question,
// because `fak commit` REFUSES a detached HEAD and so the lander had no core-lock
// question of its own — a kernel edit reached the trunk through Land with no
// witness ever demanded.
//
// Two cures were unavailable. Re-implementing the classifier in the lander would
// give the fleet two policies wearing one name, which is exactly what #5392 set
// out to prevent. Importing safecommit(2) from workerworktree(1) is an upward
// foundation -> mechanism edge the layered-DAG gate refuses outright
// (internal/architest, TestNoUpwardImports).
//
// So the shared question was pushed DOWN a layer instead: it lives here at the
// foundation, below both callers. safecommit(2) and workerworktree(1) both import
// corelockgate(1), and neither owns the policy. The locked path classification and
// the confirm/refute/abstain witness semantics therefore cannot drift apart; only
// the REMEDY sentence differs between callers, because only the way to SUPPLY the
// witness differs between the two commands.
//
// THE ONE THING THE MOVE COULD NOT CARRY DOWN is the concrete witness resolver:
// internal/witness is tier 2, and a foundation leaf may not import it. That edge is
// inverted into a registration seam (RegisterResolverFactory) which internal/witness
// calls from the same init that already performs abi.RegisterWitnessResolver —
// witness(2) -> corelockgate(1) is an ordinary downward edge. A binary with no
// factory registered and no injected Resolver cannot corroborate a claim at all, so
// the gate FAILS CLOSED there; see CheckCoreLockHardSelf.
package corelockgate

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/corelockaudit"
	"github.com/anthony-chaudhary/fak/internal/corelocks"
)

// Runner is the git seam the witness resolver reads its evidence through. It is
// the same contract internal/safecommit and internal/witness already use: a
// non-zero git exit is returned in code, never in err; err means git could not be
// EXECUTED at all.
type Runner func(ctx context.Context, dir string, args ...string) (stdout string, code int, err error)

// ResolverFactory builds the concrete witness resolver this gate resolves claims
// through. It is the inverted edge that lets a foundation leaf reach the tier-2
// resolver without importing it: internal/witness registers the real,
// git-evidence-backed factory from its init. A nil run means "use the factory's own
// real git seam".
type ResolverFactory func(run Runner, dir string) abi.WitnessResolver

var (
	factoryMu sync.RWMutex
	factory   ResolverFactory
)

// RegisterResolverFactory installs the process-wide witness resolver factory. It is
// called from internal/witness's init; a later registration replaces an earlier one,
// and registering nil clears it (which returns the gate to its fail-closed default).
func RegisterResolverFactory(f ResolverFactory) {
	factoryMu.Lock()
	factory = f
	factoryMu.Unlock()
}

// registeredFactory reads the installed factory, or nil when none was registered.
func registeredFactory() ResolverFactory {
	factoryMu.RLock()
	f := factory
	factoryMu.RUnlock()
	return f
}

// CoreLockCheck is one hard-self core-lock decision request: the changed pathset
// to classify, the maintenance witness claim offered to clear it, the evidence
// seams the resolver reads, and the caller-specific remedy named in the refusal.
//
// It exists so EVERY commit path that must honour the lock asks the SAME question
// of the SAME classifier with the SAME witness semantics. `fak commit` asks
// through safecommit.Options (checkCoreLockHardSelf); the sanctioned detached
// worker-worktree land asks directly (internal/workerworktree, #5392) — that path
// is the one CLAUDE.md blesses for build isolation and it used to reach the trunk
// with no core-lock question asked at all. Only Remedy differs between the two
// callers, because only the way to SUPPLY the witness differs between the two
// commands; the locked path set and the confirm/refute/abstain semantics are shared.
type CoreLockCheck struct {
	// Dir is the repo the witness resolver reads its evidence from.
	Dir string
	// Run is the git seam the default resolver runs through. Nil hands the
	// registered factory a nil runner, which means "your own real git seam", so a
	// caller with no injected runner still gets a real, independently-checked
	// witness.
	Run Runner
	// Resolver overrides the registered witness resolver (tests, or an alternate
	// evidence source). Nil builds one from the registered factory.
	Resolver abi.WitnessResolver
	// Changed is the changed repo-relative pathset to classify.
	Changed []string
	// Witness is the maintenance witness claim offered to clear a hard-self
	// pathset. Empty means none was offered, which is a refusal.
	Witness string
	// Remedy is the caller's own "how to supply that witness" sentence, quoted in
	// the refusal detail. Empty falls back to the `fak commit` remedy.
	Remedy string
	// Observe, when non-nil, receives the witness/change correlation computed for a
	// claim that was actually RESOLVED (see correlate.go). It is ADVISORY: the
	// allow/refuse decision below is not affected by it, so a caller can record —
	// and an operator can read — a witness that points away from the change it
	// cleared, before that mismatch is ever made blocking. A caller that passes no
	// hook simply does not observe; nothing about the gate changes.
	Observe func(WitnessCorrelation)
}

// CheckCoreLockHardSelf reports the hard-self core-lock refusal for a changed
// pathset, or ("", false) when the set raises no hard-self lock or when the
// offered witness resolves CONFIRMED. It performs no staging and no writes, so a
// caller can run it before anything touches an index, a worktree, or HEAD.
//
// FAIL-CLOSED on the lock, fail-open on the taxonomy: a missing, refuted, or
// merely abstaining witness keeps the refusal (an abstain is not clearance), while
// an unreadable/malformed taxonomy classifies nothing and therefore refuses
// nothing — the same posture the `fak commit` path has always had.
func CheckCoreLockHardSelf(ctx context.Context, c CoreLockCheck) (detail string, fired bool) {
	f, ok := HardSelfFinding(c.Changed)
	if !ok {
		return "", false
	}
	claim := strings.TrimSpace(c.Witness)
	if claim == "" {
		return hardSelfDetail(f, "missing maintenance witness", c.Remedy), true
	}
	// THE ONE CHANGE-RELATIVE VERB (changedwitness.go). `changed:<path>` is resolved
	// HERE, from c.Changed, and is deliberately absent from the shared resolver's
	// grammar: the other producers of witness claims (the file-admission hook, the
	// dispatch-tick witness, the agent turn and workflow journals) hold no changed
	// pathset, so a change-relative question is meaningless there — and for them an
	// unrecognized kind still ABSTAINS, which is not clearance. It exists because this
	// gate runs BEFORE any `git add`, so `committed:<path>` is REFUTED for a file the
	// change ADDS and an additive maintainer could not name their own work.
	//
	// It needs no resolver, because it needs no new evidence: the changed pathset was
	// obtained by the caller from git before the claim was read. That is why it is
	// decided ahead of the fail-closed resolver branch below rather than through it.
	var outcome abi.WitnessOutcome
	var cause string
	if arg, ok := isChangedWitnessClaim(claim); ok {
		outcome, cause = resolveChangedWitness(arg, c.Changed)
	} else {
		resolver := c.Resolver
		if resolver == nil {
			// FAIL CLOSED. No injected resolver and no registered factory means this
			// binary has NO way to corroborate the claim against evidence the author did
			// not write — so the claim is an unverified self-report, and a self-report is
			// precisely what the hard-self lock exists to refuse. Treating "I could not
			// check" as clearance would make the whole gate disappear in exactly the
			// build where the checker was left out, which is the failure mode #5392 was
			// opened to close. The refusal stands and names the missing resolver so the
			// cause is diagnosable rather than looking like a bad claim.
			mk := registeredFactory()
			if mk == nil {
				return hardSelfDetail(f, "no witness resolver is registered in this binary, so the maintenance claim cannot be corroborated", c.Remedy), true
			}
			resolver = mk(c.Run, c.Dir)
			if resolver == nil {
				// Same posture for a factory that declines to build one: still no
				// corroboration, still not clearance.
				return hardSelfDetail(f, "the registered witness resolver factory produced no resolver, so the maintenance claim cannot be corroborated", c.Remedy), true
			}
		}
		outcome = resolver.Resolve(ctx, nil, claim)
	}

	// OBSERVE THE CORRELATION THE RESOLVER CANNOT SEE. The resolve above is handed a
	// nil *abi.ToolCall on purpose — the resolver's rungs are claim-local — so the
	// only place that can ask "does this claim name the change it is clearing?" is
	// right here, where c.Changed is in hand. correlate.go answers it from those
	// inputs alone (no git, no filesystem). The reading is reported, NOT enforced:
	// see correlate.go for why a mismatch is recorded before it is made blocking on
	// a surface whose lock has no environment escape.
	if c.Observe != nil {
		c.Observe(CorrelateWitness(claim, c.Changed))
	}

	if outcome == abi.WitnessConfirmed {
		return "", false
	}
	why := fmt.Sprintf("maintenance witness %q resolved %s", claim, witnessOutcome(outcome))
	if cause != "" {
		why += " — " + cause
	}
	return hardSelfDetail(f, why, c.Remedy), true
}

// CoreLockRemedyCommit is the `fak commit` way to supply the maintenance witness.
// It is the default remedy, so a caller that names none still prints a real cure.
//
// It names `changed:<path>` explicitly because that verb is otherwise undiscoverable
// at exactly the moment it is needed: this gate runs before any `git add`, so a
// maintainer whose change ADDS a core-locked file gets REFUTED by `committed:<path>`
// on their own new file and has no way to learn, from the refusal alone, that a
// change-relative claim exists. See changedwitness.go.
const CoreLockRemedyCommit = "Use a privileged maintenance path, or rerun fak commit with --core-lock-maintenance-witness <claim> after independent read-back confirms the edit. " +
	"A file this change ADDS is not tracked yet, so committed:<path> is refuted for it — name it with changed:<path>, which confirms only for a path this very commit carries."

// HardSelfFinding reports the hard-self CORE_SELF_MODIFY finding a changed pathset
// raises, or (zero, false) when it raises none. An unreadable/malformed taxonomy
// classifies nothing, so it refuses nothing — the deliberate fail-open half.
func HardSelfFinding(paths []string) (corelockaudit.Finding, bool) {
	tax, err := corelocks.LoadFixture()
	if err != nil {
		return corelockaudit.Finding{}, false
	}
	rep := corelockaudit.Audit(tax, paths)
	for _, f := range rep.Findings {
		if f.Class == corelocks.ClassHardSelf && f.ReasonToken == corelocks.ReasonCoreSelfModify {
			return f, true
		}
	}
	return corelockaudit.Finding{}, false
}

func hardSelfDetail(f corelockaudit.Finding, cause, remedy string) string {
	paths := append([]string(nil), f.Paths...)
	sort.Strings(paths)
	if strings.TrimSpace(remedy) == "" {
		remedy = CoreLockRemedyCommit
	}
	return fmt.Sprintf(
		"hard-self core-lock path(s) require an external maintenance witness before this change may land; %s. %s Paths: %s",
		cause, remedy, strings.Join(paths, ", "),
	)
}

func witnessOutcome(outcome abi.WitnessOutcome) string {
	switch outcome {
	case abi.WitnessConfirmed:
		return "confirmed"
	case abi.WitnessRefuted:
		return "refuted"
	default:
		return "abstain"
	}
}
