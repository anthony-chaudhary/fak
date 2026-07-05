package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// TestKnownBadRecordThenMatch is the captured, durable form of the epic's
// done-condition witness: record --tree internal/foo/** --reason build, then a
// second (in-process) shell matches internal/foo/bar.go (matched:true, rc 3) and
// reports matched:false (rc 0) for a disjoint internal/other/** — over a
// temp-file ledger with an injected clock.
func TestKnownBadRecordThenMatch(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "known-bad.jsonl")
	const now = int64(1_700_000_000)

	// record
	var recOut, recErr bytes.Buffer
	rc := runKnownBad(&recOut, &recErr, []string{
		"record", "--tree", "internal/foo/**", "--reason", "build",
		"--note", "shared foo break", "--by", "agent-1", "--ledger", ledger,
	}, now)
	if rc != 0 {
		t.Fatalf("record rc=%d stderr=%q", rc, recErr.String())
	}
	if !strings.Contains(recOut.String(), "recorded known-bad sha256:") {
		t.Fatalf("record stdout unexpected: %q", recOut.String())
	}

	// match intersecting -> matched:true, rc 3 (short-circuit signal)
	var mOut, mErr bytes.Buffer
	rc = runKnownBad(&mOut, &mErr, []string{
		"match", "--tree", "internal/foo/bar.go", "--ledger", ledger, "--json",
	}, now)
	if rc != 3 {
		t.Fatalf("intersecting match rc=%d (want 3) stderr=%q", rc, mErr.String())
	}
	var res struct {
		Matched bool              `json:"matched"`
		Count   int               `json:"count"`
		Records []knownbad.Record `json:"records"`
	}
	if err := json.Unmarshal(mOut.Bytes(), &res); err != nil {
		t.Fatalf("match --json not valid JSON: %v (%q)", err, mOut.String())
	}
	if !res.Matched || res.Count != 1 {
		t.Fatalf("intersecting match = %+v, want matched:true count:1", res)
	}
	if res.Records[0].ReasonClass != "build" || res.Records[0].DiscoveredBy != "agent-1" {
		t.Errorf("matched record fields wrong: %+v", res.Records[0])
	}

	// match disjoint -> matched:false, rc 0
	var dOut, dErr bytes.Buffer
	rc = runKnownBad(&dOut, &dErr, []string{
		"match", "--tree", "internal/other/**", "--ledger", ledger, "--json",
	}, now)
	if rc != 0 {
		t.Fatalf("disjoint match rc=%d (want 0) stderr=%q", rc, dErr.String())
	}
	if err := json.Unmarshal(dOut.Bytes(), &res); err != nil {
		t.Fatalf("disjoint match --json not valid JSON: %v (%q)", err, dOut.String())
	}
	if res.Matched || res.Count != 0 {
		t.Errorf("disjoint match = %+v, want matched:false count:0", res)
	}
}

// An expired record must not match once now passes its TTL — the liveness gate
// wired through the shell + clock injection.
func TestKnownBadMatchLivenessExpiry(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "known-bad.jsonl")
	const recAt = int64(1_700_000_000)

	var b bytes.Buffer
	if rc := runKnownBad(&b, &b, []string{
		"record", "--tree", "internal/foo/**", "--reason", "flaky",
		"--ttl", "100", "--ledger", ledger,
	}, recAt); rc != 0 {
		t.Fatalf("record rc=%d out=%q", rc, b.String())
	}

	// within TTL -> match (rc 3)
	b.Reset()
	if rc := runKnownBad(&b, &b, []string{"match", "--tree", "internal/foo/x.go", "--ledger", ledger}, recAt+10); rc != 3 {
		t.Fatalf("within-ttl match rc=%d (want 3) out=%q", rc, b.String())
	}
	// past TTL -> no match (rc 0)
	b.Reset()
	if rc := runKnownBad(&b, &b, []string{"match", "--tree", "internal/foo/x.go", "--ledger", ledger}, recAt+1000); rc != 0 {
		t.Fatalf("expired match rc=%d (want 0) out=%q", rc, b.String())
	}
}

// Missing required flags and an unknown subcommand are rc-2 usage errors; a
// missing ledger is simply "no match", not an error.
func TestKnownBadUsageAndMissingLedger(t *testing.T) {
	var b bytes.Buffer
	if rc := runKnownBad(&b, &b, []string{"record", "--reason", "build"}, 0); rc != 2 {
		t.Errorf("record without --tree rc=%d, want 2", rc)
	}
	b.Reset()
	if rc := runKnownBad(&b, &b, []string{"record", "--tree", "internal/foo/**"}, 0); rc != 2 {
		t.Errorf("record without --reason rc=%d, want 2", rc)
	}
	b.Reset()
	if rc := runKnownBad(&b, &b, []string{"bogus"}, 0); rc != 2 {
		t.Errorf("unknown subcommand rc=%d, want 2", rc)
	}
	b.Reset()
	missing := filepath.Join(t.TempDir(), "nope.jsonl")
	if rc := runKnownBad(&b, &b, []string{"match", "--tree", "internal/foo/**", "--ledger", missing}, 0); rc != 0 {
		t.Errorf("match on missing ledger rc=%d, want 0 (no match)", rc)
	}
}
