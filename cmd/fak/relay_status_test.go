package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/relay"
)

// emitLeg writes one faithful shadow-baton sidecar for the test via the real relay codec
// (relay.EmitShadowBaton), so the fixtures decode through exactly the C2 path the verb reads.
func emitLeg(t *testing.T, dir, relayID string, leg int, reason, atSHA, note string) {
	t.Helper()
	in := relay.ShadowBatonInput{
		RelayID:    relayID,
		Leg:        leg,
		StartSHA:   atSHA,
		AtSHA:      atSHA,
		ExitReason: reason,
		NextAction: note,
		HeldRegion: []string{"cmd/fak/**"},
	}
	if _, err := relay.EmitShadowBaton(dir, in); err != nil {
		t.Fatalf("emit leg %d: %v", leg, err)
	}
}

// TestRelayStatusFoldsLegs writes three shadow-baton legs with distinct reasons, then asserts
// the human render surfaces every leg + reason in leg order and reports cost as unknown.
func TestRelayStatusFoldsLegs(t *testing.T) {
	dir := t.TempDir()
	const relayID = "RID-2026-07-17-a"
	// Emit out of leg order to prove the fold sorts by leg number, not glob order.
	emitLeg(t, dir, relayID, 2, "RELAY_ROTATED", "bbbb222", "")
	emitLeg(t, dir, relayID, 0, "RELAY_ARMED", "aaaa000", "first leg")
	emitLeg(t, dir, relayID, 1, "RELAY_PARKED_UNSAFE", "cccc111", "")

	var out, errb bytes.Buffer
	if code := runRelayStatus(&out, &errb, []string{relayID, "--dir", dir}); code != 0 {
		t.Fatalf("runRelayStatus exit=%d stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"relay " + relayID,
		"legs=3",
		"cost=unknown",
		"leg 0: RELAY_ARMED @ aaaa000",
		"leg 1: RELAY_PARKED_UNSAFE @ cccc111",
		"leg 2: RELAY_ROTATED @ bbbb222",
		"first leg",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("status render missing %q\n--- got ---\n%s", want, got)
		}
	}
	// Leg order: leg 0 must render before leg 1 before leg 2.
	i0, i1, i2 := strings.Index(got, "leg 0:"), strings.Index(got, "leg 1:"), strings.Index(got, "leg 2:")
	if !(i0 >= 0 && i0 < i1 && i1 < i2) {
		t.Errorf("legs not rendered in ascending order: i0=%d i1=%d i2=%d\n%s", i0, i1, i2, got)
	}
}

// TestRelayStatusJSON asserts --json emits the folded per-leg view as a canonical array,
// sorted by leg, carrying the reason and observed SHA for each leg.
func TestRelayStatusJSON(t *testing.T) {
	dir := t.TempDir()
	const relayID = "RID-json"
	emitLeg(t, dir, relayID, 1, "RELAY_ROTATED", "sha1", "")
	emitLeg(t, dir, relayID, 0, "RELAY_ARMED", "sha0", "")

	var out, errb bytes.Buffer
	if code := runRelayStatus(&out, &errb, []string{"--dir", dir, "--json", relayID}); code != 0 {
		t.Fatalf("runRelayStatus --json exit=%d stderr=%s", code, errb.String())
	}
	var legs []relayLegStatus
	if err := json.Unmarshal(out.Bytes(), &legs); err != nil {
		t.Fatalf("unmarshal --json output: %v\n%s", err, out.String())
	}
	if len(legs) != 2 {
		t.Fatalf("want 2 legs, got %d: %s", len(legs), out.String())
	}
	if legs[0].Leg != 0 || legs[0].Reason != "RELAY_ARMED" || legs[0].AtSHA != "sha0" {
		t.Errorf("leg[0] wrong: %+v", legs[0])
	}
	if legs[1].Leg != 1 || legs[1].Reason != "RELAY_ROTATED" || legs[1].AtSHA != "sha1" {
		t.Errorf("leg[1] wrong: %+v", legs[1])
	}
}

// TestRelayStatusEmptyFold asserts a relay id with no sidecars is a clean, non-error "(no
// legs)" report — distinguishing "not yet emitted" from a failure.
func TestRelayStatusEmptyFold(t *testing.T) {
	dir := t.TempDir()
	var out, errb bytes.Buffer
	if code := runRelayStatus(&out, &errb, []string{"RID-absent", "--dir", dir}); code != 0 {
		t.Fatalf("empty fold should exit 0, got %d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "(no legs)") {
		t.Errorf("empty fold missing (no legs) line: %s", out.String())
	}
}

// TestRelayStatusUsage asserts a missing relay id is a usage error (exit 2), not a silent
// empty fold.
func TestRelayStatusUsage(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runRelayStatus(&out, &errb, []string{"--dir", "."}); code != 2 {
		t.Fatalf("missing relay id should exit 2, got %d", code)
	}
}
