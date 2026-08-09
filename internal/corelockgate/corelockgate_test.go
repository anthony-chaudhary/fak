package corelockgate

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// A hard-self core-locked path (internal/adjudicator/** is declared CORE_SELF_MODIFY
// in internal/corelocks' embedded taxonomy) and an ordinary leaf path, so every case
// below drives the REAL classifier rather than a stub.
const (
	lockedPath   = "internal/adjudicator/decide.go"
	ordinaryLeaf = "internal/tools/a.go"
)

type fixedResolver struct{ outcome abi.WitnessOutcome }

func (f fixedResolver) Resolve(context.Context, *abi.ToolCall, string) abi.WitnessOutcome {
	return f.outcome
}

// withFactory installs a factory for one test and restores the previous one, so a
// test that needs a resolver cannot leak registration into the fail-closed tests
// (which must run in a binary state where NOTHING is registered).
func withFactory(t *testing.T, f ResolverFactory) {
	t.Helper()
	prev := registeredFactory()
	RegisterResolverFactory(f)
	t.Cleanup(func() { RegisterResolverFactory(prev) })
}

func TestOrdinaryLeafRaisesNoHardSelfLock(t *testing.T) {
	detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{Changed: []string{ordinaryLeaf}})
	if fired {
		t.Fatalf("an open-leaf path must raise no hard-self lock, got %q", detail)
	}
}

func TestMissingWitnessRefuses(t *testing.T) {
	detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{Changed: []string{lockedPath}})
	if !fired {
		t.Fatal("a hard-self path with no witness must be refused")
	}
	for _, want := range []string{"hard-self", "missing maintenance witness", lockedPath, "--core-lock-maintenance-witness"} {
		if !strings.Contains(detail, want) {
			t.Fatalf("refusal detail missing %q:\n%s", want, detail)
		}
	}
}

// TestUnregisteredResolverFactoryFailsClosed is the gate's fail-closed branch: this
// test binary links no resolver at all (package corelockgate imports nothing that
// registers one), so a PRESENT claim still cannot be corroborated — and an
// uncorroborated claim is not clearance. Without this branch the whole core lock
// would silently evaporate in any build that left the resolver out.
func TestUnregisteredResolverFactoryFailsClosed(t *testing.T) {
	withFactory(t, nil)

	detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Changed: []string{lockedPath},
		Witness: "commit:0f1e2d3c4b5a",
	})
	if !fired {
		t.Fatalf("an unresolvable claim must keep the lock closed, got detail=%q", detail)
	}
	if !strings.Contains(detail, "no witness resolver is registered") {
		t.Fatalf("refusal should name the missing resolver:\n%s", detail)
	}
	if !strings.Contains(detail, lockedPath) {
		t.Fatalf("refusal should name the locked path:\n%s", detail)
	}
}

// TestNilBuildingFactoryFailsClosed covers the other half of the same branch: a
// factory that is registered but declines to build a resolver leaves the claim just
// as uncorroborated, so it must refuse identically.
func TestNilBuildingFactoryFailsClosed(t *testing.T) {
	withFactory(t, func(Runner, string) abi.WitnessResolver { return nil })

	detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Changed: []string{lockedPath},
		Witness: "commit:0f1e2d3c4b5a",
	})
	if !fired {
		t.Fatalf("a factory that produces no resolver must keep the lock closed, got detail=%q", detail)
	}
	if !strings.Contains(detail, "produced no resolver") {
		t.Fatalf("refusal should name the empty factory:\n%s", detail)
	}
}

// TestRegisteredFactoryReceivesTheCallersGitSeam pins the inverted edge: the runner
// and dir the caller injected are the ones handed to the factory, so the witness
// resolves through the caller's git seam rather than opening a second, unmockable one.
func TestRegisteredFactoryReceivesTheCallersGitSeam(t *testing.T) {
	var gotDir string
	var gotRun bool
	withFactory(t, func(run Runner, dir string) abi.WitnessResolver {
		gotDir, gotRun = dir, run != nil
		return fixedResolver{outcome: abi.WitnessConfirmed}
	})

	detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Dir:     "/repo",
		Run:     func(context.Context, string, ...string) (string, int, error) { return "", 0, nil },
		Changed: []string{lockedPath},
		Witness: "commit:0f1e2d3c4b5a",
	})
	if fired {
		t.Fatalf("a CONFIRMED witness must clear the lock, got %q", detail)
	}
	if gotDir != "/repo" || !gotRun {
		t.Fatalf("factory got dir=%q run!=nil=%v, want the caller's own seam", gotDir, gotRun)
	}
}

// TestOnlyConfirmedClears is the non-weakening half: REFUTED and ABSTAIN are both
// refusals. An abstain in particular is "no evidence either way", which is not
// clearance — a gate that accepted it would pass on any claim the resolver cannot
// parse.
func TestOnlyConfirmedClears(t *testing.T) {
	for _, tc := range []struct {
		name    string
		outcome abi.WitnessOutcome
		fired   bool
		cause   string
	}{
		{"confirmed", abi.WitnessConfirmed, false, ""},
		{"refuted", abi.WitnessRefuted, true, "refuted"},
		{"abstain", abi.WitnessAbstain, true, "abstain"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
				Resolver: fixedResolver{outcome: tc.outcome},
				Changed:  []string{lockedPath},
				Witness:  "commit:0f1e2d3c4b5a",
			})
			if fired != tc.fired {
				t.Fatalf("fired = %v, want %v (detail=%q)", fired, tc.fired, detail)
			}
			if tc.cause != "" && !strings.Contains(detail, tc.cause) {
				t.Fatalf("refusal should name the %s outcome:\n%s", tc.cause, detail)
			}
		})
	}
}

// TestInjectedResolverNeedsNoFactory pins that an explicitly injected resolver is
// honoured even when nothing is registered — the fail-closed branch guards the
// ABSENCE of any resolver, not the absence of the registration.
func TestInjectedResolverNeedsNoFactory(t *testing.T) {
	withFactory(t, nil)

	if detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{
		Resolver: fixedResolver{outcome: abi.WitnessConfirmed},
		Changed:  []string{lockedPath},
		Witness:  "commit:0f1e2d3c4b5a",
	}); fired {
		t.Fatalf("an injected CONFIRMED resolver must clear the lock, got %q", detail)
	}
}

func TestRemedyDefaultsToTheCommitPath(t *testing.T) {
	detail, fired := CheckCoreLockHardSelf(context.Background(), CoreLockCheck{Changed: []string{lockedPath}})
	if !fired {
		t.Fatal("expected a refusal")
	}
	if !strings.Contains(detail, CoreLockRemedyCommit) {
		t.Fatalf("an unnamed remedy should fall back to the fak commit remedy:\n%s", detail)
	}
}

func TestHardSelfFindingNamesTheLockedPaths(t *testing.T) {
	f, ok := HardSelfFinding([]string{ordinaryLeaf, lockedPath})
	if !ok {
		t.Fatal("a mixed pathset containing a locked path must still raise the finding")
	}
	found := false
	for _, p := range f.Paths {
		if p == lockedPath {
			found = true
		}
		if p == ordinaryLeaf {
			t.Fatalf("the finding must carry only the locked paths, got %v", f.Paths)
		}
	}
	if !found {
		t.Fatalf("finding did not name %s: %v", lockedPath, f.Paths)
	}
}
