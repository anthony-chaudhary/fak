package corelockgate

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// refusingResolver stands in for the SHARED witness resolver, and it refuses
// everything. Every `changed:` case below drives it, so a test that passes proves the
// verb was decided from the gate's own changed pathset and NOT by the resolver — the
// same reason it must keep working with no resolver registered at all.
type refusingResolver struct{ calls int }

func (r *refusingResolver) Resolve(context.Context, *abi.ToolCall, string) abi.WitnessOutcome {
	r.calls++
	return abi.WitnessRefuted
}

// TestAdditiveMaintainerCanNameTheirOwnNewFile is the gap this verb closes, taken
// verbatim from the record: commit 58a8924131 was the single file
// `A internal/adjudicator/codereview_agent_test.go`, and e24c78a028 was the single
// file `A internal/adjudicator/reversibility_confirm_supervision_test.go`. Because the
// gate runs before any `git add`, `committed:<that file>` was REFUTED for both, and
// each maintainer reached for a neighbouring tracked path instead. `changed:` is the
// spelling they lacked.
func TestAdditiveMaintainerCanNameTheirOwnNewFile(t *testing.T) {
	for _, added := range []string{
		"internal/adjudicator/codereview_agent_test.go",
		"internal/adjudicator/reversibility_confirm_supervision_test.go",
	} {
		res := &refusingResolver{}
		var seen WitnessCorrelation
		detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
			Resolver: res,
			Changed:  []string{added},
			Witness:  ChangedWitnessKind + ":" + added,
			Observe:  func(c WitnessCorrelation) { seen = c },
		})
		if fired {
			t.Fatalf("naming the file this change ADDS must clear the lock, got refusal:\n%s", detail)
		}
		if res.calls != 0 {
			t.Fatalf("the shared resolver must not be consulted for a changed: claim (it has no changed set); calls=%d", res.calls)
		}
		if seen.Outcome != CorrelationCorrelated {
			t.Fatalf("a confirmed changed: claim must read correlated, got %s (%s)", seen.Outcome, seen.Reason)
		}
	}
}

// TestChangedWitnessNeedsNoResolver pins that the verb is decided from inputs the gate
// already holds. It is the reason `changed:` is resolved ahead of the fail-closed
// resolver branch: there is no new evidence to fetch, so there is nothing to fail
// closed ON. A binary with no resolver registered still lets an additive maintainer
// name their own work.
func TestChangedWitnessNeedsNoResolver(t *testing.T) {
	withFactory(t, nil)

	if detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Changed: []string{lockedPath},
		Witness: ChangedWitnessKind + ":" + lockedPath,
	}); fired {
		t.Fatalf("changed: needs no resolver, got refusal:\n%s", detail)
	}
	// ... and every OTHER claim kind still fails closed in that same binary.
	if _, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Changed: []string{lockedPath},
		Witness: "committed:" + lockedPath,
	}); !fired {
		t.Fatal("the fail-closed branch must still refuse a resolver-backed claim with no resolver registered")
	}
}

// TestChangedWitnessRefusesAFileTheChangeDoesNotCarry is the forgery the verb exists
// to resist. Merely creating a file — anywhere, including inside the repository —
// does not put it in the changed pathset, because that set is git's report scoped to
// the paths this commit will actually carry. The only way to satisfy the claim is to
// make the cited path part of the change, which lands it on the trunk in the same
// commit where anyone can see it.
func TestChangedWitnessRefusesAFileTheChangeDoesNotCarry(t *testing.T) {
	changed := []string{"internal/adjudicator/oot_mention.go", "internal/adjudicator/outoftree.go"}
	for _, tc := range []struct {
		claim string
		want  string
	}{
		// The exact shape found four times in the record: a `path:` witness on an
		// absolute temp file the clearing agent had written itself. Spelled as
		// `changed:` it is refused by name.
		{ChangedWitnessKind + `:C:\Users\USER\AppData\Local\Temp\fak-5146-adjudicator-witness.txt`, "outside the repository"},
		{ChangedWitnessKind + ":/tmp/fak-core-witness.txt", "outside the repository"},
		// A tracked file that is not part of THIS change — the whole weakness of
		// `committed:`, which any of the repository's tracked paths satisfies.
		{ChangedWitnessKind + ":README.md", "not one of the 2 path(s)"},
		{ChangedWitnessKind + ":internal/adjudicator/decide.go", "not one of the 2 path(s)"},
		// The containing directory is a restatement of the refusal, not a witness.
		{ChangedWitnessKind + ":internal/adjudicator", "name the file"},
	} {
		detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
			Resolver: &refusingResolver{},
			Changed:  changed,
			Witness:  tc.claim,
		})
		if !fired {
			t.Fatalf("claim %q must not clear the lock", tc.claim)
		}
		if !strings.Contains(detail, tc.want) {
			t.Fatalf("refusal for %q should say %q:\n%s", tc.claim, tc.want, detail)
		}
		if !strings.Contains(detail, "refuted") {
			t.Fatalf("a claim positively outside the change is REFUTED, not an abstain:\n%s", detail)
		}
	}
}

// TestChangedWitnessAbstainsWhenItCannotJudge preserves abstain-over-refute exactly
// where the surrounding code does. An empty changed set and an empty path are "no
// evidence either way" — both keep the lock closed, but neither is recorded as an
// accusation.
func TestChangedWitnessAbstainsWhenItCannotJudge(t *testing.T) {
	// No changed set: HardSelfFinding needs a locked path to fire at all, so the
	// abstain branch is exercised through resolveChangedWitness directly.
	if out, cause := resolveChangedWitness("internal/adjudicator/x.go", nil); out != abi.WitnessAbstain {
		t.Fatalf("an empty changed set must abstain, got %v (%s)", out, cause)
	}
	if out, cause := resolveChangedWitness("   ", []string{lockedPath}); out != abi.WitnessAbstain {
		t.Fatalf("an empty path must abstain, got %v (%s)", out, cause)
	}
	if out, _ := resolveChangedWitness("README.md", []string{lockedPath}); out != abi.WitnessRefuted {
		t.Fatal("a path positively outside the change is a measurement, so it must REFUTE")
	}
}

// TestChangedWitnessAcceptsHonestSpellings mirrors the correlation's own generosity:
// separators, "./", git's ":/" magic prefix, surrounding whitespace and case must
// never turn an honest maintainer's own changed file into a refusal, because this
// lock has no environment escape.
func TestChangedWitnessAcceptsHonestSpellings(t *testing.T) {
	changed := []string{"internal/adjudicator/oot_mention.go"}
	for _, arg := range []string{
		"internal/adjudicator/oot_mention.go",
		`internal\adjudicator\oot_mention.go`,
		"./internal/adjudicator/oot_mention.go",
		":/internal/adjudicator/oot_mention.go",
		"internal/adjudicator/OOT_MENTION.go",
		" internal/adjudicator/oot_mention.go ",
	} {
		if out, cause := resolveChangedWitness(arg, changed); out != abi.WitnessConfirmed {
			t.Fatalf("spelling %q of a changed path must confirm, got %v (%s)", arg, out, cause)
		}
	}
}

// TestChangedVerbDoesNotDisturbTheOtherKinds is the non-weakening half. Adding a verb
// must leave every existing claim exactly as it was: still routed to the shared
// resolver, still cleared only on CONFIRMED.
func TestChangedVerbDoesNotDisturbTheOtherKinds(t *testing.T) {
	for _, claim := range []string{
		"committed:README.md",
		"ancestor:HEAD",
		"commit:0f1e2d3c4b5a",
		"path:/tmp/whatever",
		"changedish:internal/adjudicator/x.go", // a near-miss kind is NOT the verb
	} {
		res := &refusingResolver{}
		if _, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
			Resolver: res,
			Changed:  []string{lockedPath},
			Witness:  claim,
		}); !fired {
			t.Fatalf("a REFUTED %q must keep the lock closed", claim)
		}
		if res.calls != 1 {
			t.Fatalf("claim %q must still be routed to the shared resolver, calls=%d", claim, res.calls)
		}
	}
}

// TestRefusalTeachesTheChangedVerb pins the discoverability that makes the grammar
// usable: a maintainer blocked by the lock learns from the refusal itself that a
// change-relative claim exists. Without this the verb is real but unreachable.
func TestRefusalTeachesTheChangedVerb(t *testing.T) {
	detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{Changed: []string{lockedPath}})
	if !fired {
		t.Fatal("expected a refusal")
	}
	if !strings.Contains(detail, ChangedWitnessKind+":<path>") {
		t.Fatalf("the default remedy must name the changed: verb:\n%s", detail)
	}
}
