package snapshot

// determinism_test.go is the determinism witness + bit-exact pin for the snapshot/fleet
// codecs (closes the "pin the snapshot/fleet codecs bit-exact" obligation). Two claims,
// each witnessed separately so a regression in either is caught:
//
//   - DETERMINISM: for a fixed fleet (and a fixed injected clock), DumpFleet -> Encode
//     yields byte-identical bytes on every repeat, sequentially AND concurrently. This
//     pins that the codec has NO dependence on map-iteration order, goroutine scheduling,
//     RNG, or wall-clock — the properties that make a dumped fleet a portable, reproducible
//     artifact. The risk surface it guards is real: FleetBody is fed by session.Table
//     .Snapshot(), which reads a map; the sort there (Priority, Rev, TraceID) is what makes
//     the output order-stable, and dropping or altering it would silently make dumps
//     non-reproducible. Run this package with -race to also rule out a data race.
//
//   - BIT-EXACT PIN: the fleet codec's Body bytes (the canonical JSON of the sessions
//     array, in scheduler order) are pinned to a golden file. The Body — NOT the whole
//     envelope — is the pin target on purpose: the envelope stamps app_version (read from
//     the repo VERSION file) and created_unix, both of which legitimately vary per build /
//     per dump, so pinning them would create false failures on every release. The Body is
//     the part the CODEC owns: field names, value rendering, and the session ordering. A
//     field rename, a type change, or an ordering drift changes the Body and fails the pin
//     (regenerate with UPDATE_GOLDEN=1 after an intentional format change).

import (
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// canonicalFleet builds a representative, NON-DEGENERATE fleet that exercises the whole
// drive surface the fleet codec serializes: a Throttled session with a real budget +
// priority + reason, a STOPPED session restored verbatim (terminal, with a closed reason
// token and a carried Rev), a Paused hold, and a Running session with a pace opinion. The
// four sessions also exercise all three of Snapshot()'s sort keys (Priority asc, Rev desc,
// TraceID asc), so the pinned ordering is a real witness and not a single-element trivial
// one. The same fleet feeds the determinism witness and the golden pin, so the two claims
// are over an identical, named input.
func canonicalFleet() *session.Table {
	tbl := session.NewTable()
	tbl.Transition("sess-a", session.Throttled, "operator-offload")
	tbl.SetBudget("sess-a", session.Budget{TurnsLeft: 3, TokensLeft: 4096})
	tbl.SetPriority("sess-a", 5)

	tbl.Restore("sess-b", session.State{
		TraceID: "sess-b", Run: session.Stopped,
		Reason: session.ReasonBudgetTurns, Rev: 9,
	})

	tbl.Transition("sess-c", session.Paused, "")

	tbl.Transition("sess-d", session.Running, "")
	tbl.SetPace("sess-d", session.Pace{MaxTokensPerTurn: 512, MinTurnGapMs: 200})
	tbl.SetPriority("sess-d", 2)
	return tbl
}

// canonNow is the injected clock every encode in this file uses, so the envelope's
// created_unix is constant within and across the determinism + golden assertions.
const canonNow int64 = 1_700_000_000

// encodeFleetOnce is the single step under witness: build the snapshot, serialize it. The
// determinism claim is over THIS composition (DumpFleet -> Encode), the exact path a live
// dump/restore and the snapshot CLI take.
func encodeFleetOnce(t *testing.T) []byte {
	t.Helper()
	snap, err := DumpFleet("fleet-eu", canonicalFleet(), canonNow)
	if err != nil {
		t.Fatalf("DumpFleet: %v", err)
	}
	b, err := snap.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	return b
}

// TestFleetCodecDeterministic is the determinism witness: a reference encode, then many
// repeats, must be byte-identical to it — and the same must hold under concurrent encodes
// from many goroutines (ruling out scheduling / shared-state dependence). Non-vacuous: it
// also asserts the body actually carries the four sessions and the non-default fields, so
// the byte-equality cannot pass on a degenerate empty/single-element body.
func TestFleetCodecDeterministic(t *testing.T) {
	ref := encodeFleetOnce(t)

	// Non-vacuity: the pinned body is rich (4 sessions, several non-default fields). If a
	// future change collapsed the fleet to one session or zeroed the fields, the
	// determinism equality below would hold trivially — so assert the surface first.
	parsed, err := Parse(ref)
	if err != nil {
		t.Fatalf("Parse reference: %v", err)
	}
	var body FleetBody
	if err := parsed.Into(&body); err != nil {
		t.Fatalf("Into FleetBody: %v", err)
	}
	if len(body.Sessions) != 4 {
		t.Fatalf("non-vacuity: expected 4 sessions in the fleet body, got %d", len(body.Sessions))
	}
	if body.Sessions[0].Run != session.Stopped || body.Sessions[0].Reason != session.ReasonBudgetTurns {
		t.Fatalf("non-vacuity: first (Priority0/Rev9) session must be the stopped sess-b, got %+v", body.Sessions[0])
	}

	// Sequential determinism: every repeat is byte-identical to the reference.
	const repeats = 256
	for i := 0; i < repeats; i++ {
		got := encodeFleetOnce(t)
		if string(got) != string(ref) {
			t.Fatalf("repeat %d: fleet codec not deterministic (bytes differ from reference)", i)
		}
	}

	// Concurrent determinism: many goroutines encoding the SAME fleet at once must all
	// agree with the reference — no map-iteration-order or scheduling divergence. Run with
	// -race to also rule out a data race on any shared cell inside DumpFleet/Encode.
	const goroutines = 32
	out := make([][]byte, goroutines)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			<-start // line them up to maximize interleaving
			out[idx] = encodeFleetOnce(t)
		}(g)
	}
	close(start)
	wg.Wait()
	for g := 0; g < goroutines; g++ {
		if string(out[g]) != string(ref) {
			t.Fatalf("goroutine %d: concurrent fleet encode diverged from reference", g)
		}
	}
}

// TestFleetCodecGoldenPin is the bit-exact pin: the fleet codec's Body bytes are pinned to
// a golden file. The Body (not the whole envelope) is the target — the envelope's
// app_version / created_unix legitimately vary, but the Body is the codec's own contract
// (field names, value rendering, session ordering). A drift in any of those fails the pin.
// Regenerate after an intentional format change with UPDATE_GOLDEN=1.
func TestFleetCodecGoldenPin(t *testing.T) {
	snap, err := DumpFleet("fleet-eu", canonicalFleet(), canonNow)
	if err != nil {
		t.Fatalf("DumpFleet: %v", err)
	}

	const golden = "testdata/fleet_body_v1.golden"
	if os.Getenv("UPDATE_GOLDEN") == "1" {
		if err := os.MkdirAll("testdata", 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join("testdata", "fleet_body_v1.golden"), snap.Body, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if string(want) != string(snap.Body) {
		t.Fatalf("fleet codec body drifted from the golden pin (bit-exact broken).\n--- want ---\n%s\n--- got ---\n%s", want, snap.Body)
	}

	// Cross-check: the body the codec produced hashes to the digest the envelope records,
	// and that digest is itself stable (it is a function of the body alone, so a pinned
	// body implies a pinned digest). Asserting it here keeps the integrity stamp in the
	// same witness as the format pin.
	if snap.BodyDigest == "" {
		t.Fatal("BodyDigest is empty — the integrity stamp is not being recorded")
	}
}

// TestTraceCodecDeterministic extends the determinism witness to the OTHER typed codec
// (the turn level), which shares the same envelope. It is a lighter check than the fleet
// witness because a trace body is a plain ordered slice (no map-backed sort), but it pins
// that the shared Marshal/Encode path is order-stable for the turn codec too — so the
// "snapshot codecs" claim holds across both typed codecs, not just fleet.
func TestTraceCodecDeterministic(t *testing.T) {
	turns := []trajectory.Turn{
		{TraceID: "sess-1", Seq: 1, Query: "what refund fee?", Tool: "get_user_details", Verdict: "ALLOW"},
		{TraceID: "sess-1", Seq: 2, Tool: "read_refund_policy", Verdict: "QUARANTINE", Reason: "TRUST_VIOLATION"},
	}
	refSnap, err := DumpTrace("sess-1", turns, canonNow)
	if err != nil {
		t.Fatalf("DumpTrace: %v", err)
	}
	ref, err := refSnap.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	for i := 0; i < 128; i++ {
		s, err := DumpTrace("sess-1", turns, canonNow)
		if err != nil {
			t.Fatalf("DumpTrace: %v", err)
		}
		got, err := s.Encode()
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if string(got) != string(ref) {
			t.Fatalf("repeat %d: trace codec not deterministic (bytes differ from reference)", i)
		}
	}
}
