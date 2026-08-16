package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/stallscan"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

const (
	skewOldRev = "70defaa112aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	skewTipRev = "b2eef82268bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)

// TestStallscanSkewGuard_closedSet pins WHICH verdicts make the detector warn about itself. The
// four refusable tokens all mean "the axes compiled into this file may not be the axes in the
// source", so all four must fire. FRESH/AHEAD are current enough, and UNKNOWN is the honest
// residual — warning on it would refuse on a fact the guard could not establish, so it stays
// silent. If a future edit lets SKEWED or UNSTAMPED go quiet, this reds.
func TestStallscanSkewGuard_closedSet(t *testing.T) {
	warns := map[versionskew.Verdict]bool{
		versionskew.Skewed:    true,
		versionskew.Diverged:  true,
		versionskew.Unstamped: true,
		versionskew.Dirty:     true,
	}
	for _, v := range []versionskew.Verdict{
		versionskew.Unknown, versionskew.Fresh, versionskew.Skewed,
		versionskew.Ahead, versionskew.Diverged, versionskew.Unstamped, versionskew.Dirty,
	} {
		got := stallscanSkewGuard(versionskew.Assessment{Verdict: v, Running: skewOldRev, TrunkTip: skewTipRev})
		if want := warns[v]; (got != nil) != want {
			t.Errorf("stallscanSkewGuard(%v) fired = %v, want %v", v, got != nil, want)
		}
		if got != nil && got.Verdict != v.String() {
			t.Errorf("verdict token = %q, want %q", got.Verdict, v.String())
		}
	}
}

// TestStallscanSkewGuard_namesTheEmptyAxisTrap locks the load-bearing WORDING. The #3668 defect is
// not that a stale binary errors — it is that a missing axis renders EMPTY, which reads exactly
// like a healthy host. A warning that only said "your fak is old" would leave the reader with no
// reason to distrust the calm lines below it, so the note must say the quiet axis itself is
// suspect, and must carry both revisions so the operator can see the gap.
func TestStallscanSkewGuard_namesTheEmptyAxisTrap(t *testing.T) {
	sk := stallscanSkewGuard(versionskew.Assessment{
		Verdict: versionskew.Skewed, Running: skewOldRev, TrunkTip: skewTipRev, Relation: versionskew.RelBehind,
	})
	if sk == nil {
		t.Fatal("a provably-behind detector must warn about itself")
	}
	if !strings.Contains(sk.Note, "EMPTY") {
		t.Errorf("note must say a missing axis reports EMPTY, not missing:\n%s", sk.Note)
	}
	if !strings.Contains(sk.Note, "BEHIND") {
		t.Errorf("note must name the behind-trunk condition:\n%s", sk.Note)
	}
	if !strings.Contains(sk.Note, skewOldRev[:12]) || !strings.Contains(sk.Note, skewTipRev[:12]) {
		t.Errorf("note must carry both short revs:\n%s", sk.Note)
	}
	if !strings.Contains(sk.Note, "go build -o tools/.bin/fak.exe ./cmd/fak") {
		t.Errorf("note must name the redeploy command that clears it:\n%s", sk.Note)
	}
	if sk.Running != skewOldRev || sk.TrunkTip != skewTipRev {
		t.Errorf("machine fields must carry the full revs, got %+v", sk)
	}
}

// TestStallscanSkewGuard_unstampedNeedsNoRevs covers the token a hand-copied or
// `go install …@latest` deploy actually produces: no stamp at all, hence no revisions to compare.
// The note must still stand on its own rather than printing "(unknown)" ancestry as if it meant
// something.
func TestStallscanSkewGuard_unstampedNeedsNoRevs(t *testing.T) {
	sk := stallscanSkewGuard(versionskew.Assessment{Verdict: versionskew.Unstamped})
	if sk == nil {
		t.Fatal("an unstamped detector cannot attest its axes and must warn")
	}
	if !strings.Contains(sk.Note, "NO VCS stamp") {
		t.Errorf("note must name the unstamped condition:\n%s", sk.Note)
	}
	if strings.Contains(sk.Note, "(unknown)") {
		t.Errorf("note must not print a placeholder rev it has no basis for:\n%s", sk.Note)
	}
}

// TestStallFingerprintSkewed_stampsTheRecord is the wiring witness: the guard must ride IN-BAND on
// the fingerprint, because the --watch JSONL trail is read long after the console banner scrolled
// away. Without this, a trail full of empty thread axes has nothing in it to explain why.
func TestStallFingerprintSkewed_stampsTheRecord(t *testing.T) {
	s := stallscan.Sample{TotalFaultsPerSec: 1000, AvailableMB: 229000}
	v := stallscan.Classify(s, stallscan.DefaultThresholds())
	sk := stallscanSkewGuard(versionskew.Assessment{Verdict: versionskew.Skewed, Running: skewOldRev, TrunkTip: skewTipRev})

	b, err := json.Marshal(stallFingerprintSkewed(s, v, sk))
	if err != nil {
		t.Fatal(err)
	}
	var back struct {
		Schema    string          `json:"schema"`
		BuildSkew *stallBuildSkew `json:"build_skew"`
	}
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if back.Schema != "fak.stallscan.v1" {
		t.Errorf("the skew stamp must not change the record schema, got %q", back.Schema)
	}
	if back.BuildSkew == nil || back.BuildSkew.Verdict != "SKEWED" || back.BuildSkew.TrunkTip != skewTipRev {
		t.Fatalf("build_skew missing or wrong in the record: %+v", back.BuildSkew)
	}
}

// TestStallFingerprintSkewed_currentBinaryRecordIsUnchanged is the other half of the contract: a
// detector that IS current must emit exactly the record it emitted before this guard existed. The
// key must be absent, not present-and-empty — a `"build_skew": null` row would train every reader
// to ignore the field.
func TestStallFingerprintSkewed_currentBinaryRecordIsUnchanged(t *testing.T) {
	s := stallscan.Sample{TotalFaultsPerSec: 1000, AvailableMB: 229000}
	v := stallscan.Classify(s, stallscan.DefaultThresholds())

	rec := stallFingerprintSkewed(s, v, stallscanSkewGuard(versionskew.Assessment{Verdict: versionskew.Fresh, Running: skewTipRev, TrunkTip: skewTipRev}))
	if _, present := rec["build_skew"]; present {
		t.Fatalf("a fresh build must add no key at all, got %v", rec["build_skew"])
	}
	b, err := json.Marshal(rec)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "build_skew") {
		t.Fatalf("build_skew must not appear in a current binary's JSON:\n%s", b)
	}
}
