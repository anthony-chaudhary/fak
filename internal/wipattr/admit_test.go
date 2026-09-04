package wipattr

import (
	"strings"
	"testing"
)

// The start-of-task admission is the PROSPECTIVE mirror of SweepGuard. SweepGuard
// answers "is it safe to STAGE what is already written"; AdmitStart answers "is it
// safe to START writing here at all", from the same evidence. These tests pin the
// three properties the retrospective side also holds — totality, determinism, and
// fail-safe defaults — plus the one property only the prospective side can have:
// a HOLD that names an action taken BEFORE the first edit exists.

func liveSet(ids ...string) map[string]bool {
	m := map[string]bool{}
	for _, id := range ids {
		m[id] = true
	}
	return m
}

func findingFor(r AdmitReport, path string) (AdmitFinding, bool) {
	for _, f := range r.Findings {
		if f.Path == path {
			return f, true
		}
	}
	return AdmitFinding{}, false
}

func hasReason(r AdmitReport, want AdmitReason) bool {
	for _, f := range r.Findings {
		if f.Reason == want {
			return true
		}
	}
	return false
}

// A path a LIVE peer is already editing is the duplicative-WIP case this gate
// exists to stop: two sessions writing one file, neither knowing until both have
// a delta. It must HOLD, and the hold must name the peer.
func TestAdmitStartHoldsPathClaimedByLivePeer(t *testing.T) {
	got := AdmitStart(AdmitInput{
		Self:    "self",
		Intends: []string{"cmd/fak/guard.go"},
		Attrs: []Attribution{
			{File: "cmd/fak/guard.go", State: AttrOwned, Owner: "peer-live"},
		},
		Live: liveSet("peer-live"),
	})
	if got.Verdict != AdmitHold {
		t.Fatalf("verdict = %q, want %q", got.Verdict, AdmitHold)
	}
	f, ok := findingFor(got, "cmd/fak/guard.go")
	if !ok {
		t.Fatalf("no finding for the intended path; report = %+v", got)
	}
	if f.Reason != ReasonPathClaimedByPeer {
		t.Errorf("reason = %q, want %q", f.Reason, ReasonPathClaimedByPeer)
	}
	if !f.Hard {
		t.Errorf("a live peer's claim must be a HARD hold, got soft")
	}
	if !strings.Contains(f.Detail, "peer-live") {
		t.Errorf("detail %q does not name the peer holding the path", f.Detail)
	}
}

// A dirty path no checkpoint claims is unattributed WIP. Starting on top of it
// both risks destroying it and hides whose it was, so it HOLDs and routes to the
// recovery verb rather than to "just start".
func TestAdmitStartHoldsOrphanPath(t *testing.T) {
	got := AdmitStart(AdmitInput{
		Self:    "self",
		Intends: []string{"internal/x/x.go"},
		Attrs: []Attribution{
			{File: "internal/x/x.go", State: AttrOrphan},
		},
	})
	if got.Verdict != AdmitHold {
		t.Fatalf("verdict = %q, want %q", got.Verdict, AdmitHold)
	}
	f, _ := findingFor(got, "internal/x/x.go")
	if f.Reason != ReasonPathOrphaned || !f.Hard {
		t.Fatalf("finding = %+v, want a hard %s", f, ReasonPathOrphaned)
	}
	if !strings.Contains(got.Next, "reconcile") {
		t.Errorf("Next = %q, want it to route to the reconcile recovery verb", got.Next)
	}
}

// A checkpointed peer who is NOT live is still not yours to overwrite: the delta
// is recoverable work, and its owner being dead is what makes it recoverable, not
// what makes it free. Distinct reason from the live case, because the remedy differs.
func TestAdmitStartHoldsDeadPeerPathAsOrphaned(t *testing.T) {
	got := AdmitStart(AdmitInput{
		Self:    "self",
		Intends: []string{"internal/x/x.go"},
		Attrs: []Attribution{
			{File: "internal/x/x.go", State: AttrOwned, Owner: "peer-dead"},
		},
		Live: liveSet("someone-else"),
	})
	f, _ := findingFor(got, "internal/x/x.go")
	if got.Verdict != AdmitHold || f.Reason != ReasonPathOrphaned {
		t.Fatalf("report = %+v, want HOLD/%s", got, ReasonPathOrphaned)
	}
	if !strings.Contains(f.Detail, "peer-dead") {
		t.Errorf("detail %q does not name the dead owner", f.Detail)
	}
}

// SHARED is ambiguous ownership; it can never be resolved to "mine" by a machine,
// so it holds like a live claim rather than being collapsed to one owner.
func TestAdmitStartHoldsSharedPath(t *testing.T) {
	got := AdmitStart(AdmitInput{
		Self:    "self",
		Intends: []string{"a.go"},
		Attrs:   []Attribution{{File: "a.go", State: AttrShared, Owners: []string{"p1", "p2"}}},
	})
	f, _ := findingFor(got, "a.go")
	if got.Verdict != AdmitHold || f.Reason != ReasonPathClaimedByPeer {
		t.Fatalf("report = %+v, want HOLD/%s", got, ReasonPathClaimedByPeer)
	}
}

// An untracked file already sitting at an intended path is WIP that NO checkpoint
// can attribute (a checkpoint captures the tracked delta), so attribution reads it
// as absent. Fail-safe: the gate must not read that silence as a clean path.
func TestAdmitStartHoldsUntrackedPath(t *testing.T) {
	got := AdmitStart(AdmitInput{
		Self:      "self",
		Intends:   []string{"internal/x/new.go"},
		Untracked: []string{"internal/x/new.go"},
	})
	f, _ := findingFor(got, "internal/x/new.go")
	if got.Verdict != AdmitHold || f.Reason != ReasonPathUntrackedWIP || !f.Hard {
		t.Fatalf("report = %+v, want a hard HOLD/%s", got, ReasonPathUntrackedWIP)
	}
}

// Resuming your OWN checkpointed work is the case that must never be held —
// otherwise the gate punishes exactly the finish-before-start behaviour it exists
// to encourage.
func TestAdmitStartAdmitsSelfOwnedPath(t *testing.T) {
	got := AdmitStart(AdmitInput{
		Self:    "self",
		Intends: []string{"a.go"},
		Attrs:   []Attribution{{File: "a.go", State: AttrOwned, Owner: "self"}},
		Live:    liveSet("self"),
	})
	if got.Verdict != AdmitOK {
		t.Fatalf("verdict = %q, want %q; findings = %+v", got.Verdict, AdmitOK, got.Findings)
	}
}

// A clean intended path on a tree with plenty of unrelated dirt still admits: the
// gate is scoped to what THIS task will touch, not to the shared tree's total mess,
// which no single session can fix and which would make the gate fire forever.
func TestAdmitStartAdmitsCleanPathOnDirtyTree(t *testing.T) {
	attrs := []Attribution{}
	for _, f := range []string{"b.go", "c.go", "d.go"} {
		attrs = append(attrs, Attribution{File: f, State: AttrOrphan})
	}
	got := AdmitStart(AdmitInput{Self: "self", Intends: []string{"a.go"}, Attrs: attrs})
	if got.Verdict != AdmitOK {
		t.Fatalf("verdict = %q, want %q; findings = %+v", got.Verdict, AdmitOK, got.Findings)
	}
}

// Finish-before-start: a session already holding more unlanded paths than the
// ceiling is warned but NOT refused by default, the same warn-first posture the
// fleet's focus WIP cap uses. Strict promotes it, so the policy is a flag, not a fork.
func TestAdmitStartSelfWIPIsSoftUntilStrict(t *testing.T) {
	attrs := []Attribution{}
	for _, f := range []string{"a.go", "b.go", "c.go"} {
		attrs = append(attrs, Attribution{File: f, State: AttrOwned, Owner: "self"})
	}
	in := AdmitInput{Self: "self", Intends: []string{"z.go"}, Attrs: attrs, SelfWIPCeiling: 2}

	warn := AdmitStart(in)
	if warn.Verdict != AdmitOK {
		t.Fatalf("default verdict = %q, want %q (warn-first)", warn.Verdict, AdmitOK)
	}
	if !hasReason(warn, ReasonSelfWIPUnlanded) {
		t.Fatalf("no %s finding; report = %+v", ReasonSelfWIPUnlanded, warn)
	}
	for _, f := range warn.Findings {
		if f.Reason == ReasonSelfWIPUnlanded && f.Hard {
			t.Errorf("%s must be soft by default", ReasonSelfWIPUnlanded)
		}
	}
	if warn.SelfDirty != 3 {
		t.Errorf("SelfDirty = %d, want 3", warn.SelfDirty)
	}

	in.Strict = true
	if strict := AdmitStart(in); strict.Verdict != AdmitHold {
		t.Fatalf("strict verdict = %q, want %q", strict.Verdict, AdmitHold)
	}
}

// Declaring no intent must not read as a clean answer — the per-path rules did not
// run, and "nothing checked" is not "nothing wrong".
func TestAdmitStartUndeclaredIntentIsSurfaced(t *testing.T) {
	got := AdmitStart(AdmitInput{Self: "self"})
	if !hasReason(got, ReasonIntentUndeclared) {
		t.Fatalf("no %s finding; report = %+v", ReasonIntentUndeclared, got)
	}
	if got.Verdict != AdmitOK {
		t.Errorf("verdict = %q, want %q (advisory, not a refusal)", got.Verdict, AdmitOK)
	}
}

// Determinism + totality: the same input yields the same ordered findings, and every
// engaged intended path yields exactly one finding.
func TestAdmitStartIsDeterministicAndTotal(t *testing.T) {
	in := AdmitInput{
		Self:    "self",
		Intends: []string{"z.go", "a.go", "m.go", "clean.go"},
		Attrs: []Attribution{
			{File: "z.go", State: AttrOrphan},
			{File: "a.go", State: AttrOwned, Owner: "peer"},
			{File: "m.go", State: AttrShared, Owners: []string{"p1", "p2"}},
		},
		Live: liveSet("peer"),
	}
	first := AdmitStart(in)
	second := AdmitStart(in)
	if len(first.Findings) != len(second.Findings) {
		t.Fatalf("nondeterministic length: %d vs %d", len(first.Findings), len(second.Findings))
	}
	for i := range first.Findings {
		if first.Findings[i] != second.Findings[i] {
			t.Fatalf("finding %d differs: %+v vs %+v", i, first.Findings[i], second.Findings[i])
		}
	}
	if len(first.Findings) != 3 {
		t.Fatalf("got %d findings, want exactly one per engaged path (3); %+v", len(first.Findings), first.Findings)
	}
	if _, ok := findingFor(first, "clean.go"); ok {
		t.Errorf("a clean intended path must yield no finding")
	}
	// Hard findings sort ahead of soft ones so the first row printed is the binding one.
	for i := 1; i < len(first.Findings); i++ {
		if !first.Findings[i-1].Hard && first.Findings[i].Hard {
			t.Fatalf("soft finding sorted ahead of a hard one: %+v", first.Findings)
		}
	}
}

// Path normalisation: a declared intent must match the attribution's slash-separated,
// repo-relative spelling even when the operator types a Windows path or a "./" prefix.
func TestAdmitStartNormalisesDeclaredPaths(t *testing.T) {
	for _, spelling := range []string{`cmd\fak\guard.go`, "./cmd/fak/guard.go", "cmd/fak/guard.go"} {
		got := AdmitStart(AdmitInput{
			Self:    "self",
			Intends: []string{spelling},
			Attrs:   []Attribution{{File: "cmd/fak/guard.go", State: AttrOrphan}},
		})
		if got.Verdict != AdmitHold {
			t.Errorf("intent %q did not match the dirty path; report = %+v", spelling, got)
		}
	}
}

// A declared DIRECTORY covers the dirty files beneath it: a task that says it will
// work in internal/x must be held by a collision inside internal/x.
func TestAdmitStartDeclaredDirectoryCoversChildren(t *testing.T) {
	got := AdmitStart(AdmitInput{
		Self:    "self",
		Intends: []string{"internal/x"},
		Attrs:   []Attribution{{File: "internal/x/deep/y.go", State: AttrOrphan}},
	})
	if got.Verdict != AdmitHold {
		t.Fatalf("a declared directory must cover its dirty children; report = %+v", got)
	}
}

// If Scope is empty but session/lane is active and dos.toml (or lane manifest) matches
// the dirty files, ownership is inferred (AttrOwned by Self) so it does not hold as an orphan.
func TestAdmitStartInfersOwnershipWhenScopeEmptyAndLaneActive(t *testing.T) {
	got := AdmitStart(AdmitInput{
		Self:         "self",
		Intends:      []string{"internal/gateway/server.go"},
		Attrs:        []Attribution{{File: "internal/gateway/server.go", State: AttrOrphan}},
		Live:         liveSet("self"),
		Lane:         "gateway",
		LaneManifest: []string{"internal/gateway/**"},
	})
	if got.Verdict != AdmitOK {
		t.Fatalf("verdict = %q, want %q; findings = %+v", got.Verdict, AdmitOK, got.Findings)
	}
	if got.SelfDirty != 1 {
		t.Errorf("SelfDirty = %d, want 1", got.SelfDirty)
	}
}
