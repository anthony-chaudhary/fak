package fleetaccounts

import (
	"encoding/json"
	"reflect"
	"testing"
)

// sampleSeatLedger is a fixture rolling ledger in the JSONL shape the fleet's dispatch/probe
// ledgers use. It exercises every outcome bucket, the alias normalization (success/ok ->
// witnessed, rate-limited -> throttled, error/timeout), an ignored unknown token
// ("cancelled"), and an unscored seat that only ever started.
const sampleSeatLedger = `{"seat":"uuid:aaa","outcome":"started"}
{"seat":"uuid:aaa","outcome":"started"}
{"seat":"uuid:aaa","outcome":"started"}
{"seat":"uuid:aaa","outcome":"witnessed"}
{"seat":"uuid:aaa","outcome":"witnessed"}
{"seat":"uuid:aaa","outcome":"witnessed"}
{"seat":"dir:beta","outcome":"started"}
{"seat":"dir:beta","outcome":"started"}
{"seat":"dir:beta","outcome":"started"}
{"seat":"dir:beta","outcome":"started"}
{"seat":"dir:beta","outcome":"success"}
{"seat":"dir:beta","outcome":"ok"}
{"seat":"dir:beta","outcome":"failed"}
{"seat":"dir:beta","outcome":"rate-limited"}
{"seat":"dir:beta","outcome":"cancelled"}
{"seat":"dir:gamma","outcome":"started"}
{"seat":"dir:gamma","outcome":"started"}
{"seat":"dir:gamma","outcome":"error"}
{"seat":"dir:gamma","outcome":"timeout"}
{"seat":"dir:delta","outcome":"started"}
{"seat":"dir:delta","outcome":"started"}`

// TestFoldSeatLedgerDeterministicScores is the #1804 witness: a sample ledger folds into
// deterministic per-seat health scores. It pins the exact folded slice (counts, derived
// rates, best-first order) and re-folds to prove the fold is reproducible.
func TestFoldSeatLedgerDeterministicScores(t *testing.T) {
	got, err := FoldSeatLedger([]byte(sampleSeatLedger))
	if err != nil {
		t.Fatalf("FoldSeatLedger: %v", err)
	}
	want := []SeatScore{
		{Seat: "uuid:aaa", Started: 3, Witnessed: 3, Resolved: 3, Health: 1.0, Scored: true},
		{Seat: "dir:beta", Started: 4, Witnessed: 2, Failed: 1, Throttled: 1, Resolved: 4, Health: 0.5, Scored: true},
		{Seat: "dir:gamma", Started: 2, Failed: 1, TimedOut: 1, Resolved: 2, Health: 0.0, Scored: true},
		{Seat: "dir:delta", Started: 2, Resolved: 0, InFlight: 2, Health: 0, Scored: false},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("folded scores mismatch:\n got=%+v\nwant=%+v", got, want)
	}
	// Determinism: the same ledger folds byte-for-byte identically a second time.
	again, err := FoldSeatLedger([]byte(sampleSeatLedger))
	if err != nil {
		t.Fatalf("second FoldSeatLedger: %v", err)
	}
	if !reflect.DeepEqual(got, again) {
		t.Fatalf("fold not deterministic:\n a=%+v\n b=%+v", got, again)
	}
}

// TestFoldSeatScoresOrderBestFirst pins the routing order: highest health first, then more
// resolved evidence, then seat name; every unscored seat sorts after every scored one.
func TestFoldSeatScoresOrderBestFirst(t *testing.T) {
	scores := FoldSeatLedgerOrPanic(t, sampleSeatLedger)
	if len(scores) != 4 {
		t.Fatalf("want 4 seats, got %d", len(scores))
	}
	order := []string{scores[0].Seat, scores[1].Seat, scores[2].Seat, scores[3].Seat}
	wantOrder := []string{"uuid:aaa", "dir:beta", "dir:gamma", "dir:delta"}
	if !reflect.DeepEqual(order, wantOrder) {
		t.Fatalf("routing order = %v, want %v", order, wantOrder)
	}
	// The last seat is the only unscored one; no scored seat may follow an unscored seat.
	seenUnscored := false
	for _, s := range scores {
		if seenUnscored && s.Scored {
			t.Fatalf("scored seat %q sorted after an unscored seat", s.Seat)
		}
		if !s.Scored {
			seenUnscored = true
		}
	}
}

// TestFoldSeatScoresHealthTieBreakByEvidence checks that among equal-health seats, the one
// with more resolved evidence ranks first (a 5/5 seat outranks a 1/1 seat).
func TestFoldSeatScoresHealthTieBreakByEvidence(t *testing.T) {
	var evs []SeatEvent
	for i := 0; i < 5; i++ {
		evs = append(evs, SeatEvent{Seat: "dir:deep", Outcome: SeatWitnessed})
	}
	evs = append(evs, SeatEvent{Seat: "dir:thin", Outcome: SeatWitnessed})
	scores := FoldSeatScores(evs)
	if scores[0].Seat != "dir:deep" {
		t.Fatalf("more-resolved seat should rank first; got order %v", []string{scores[0].Seat, scores[1].Seat})
	}
}

// TestParseSeatOutcomeAliasesAndUnknown checks the token normalization and the ignore path.
func TestParseSeatOutcomeAliasesAndUnknown(t *testing.T) {
	cases := map[string]SeatOutcome{
		"WITNESSED":    SeatWitnessed,
		"success":      SeatWitnessed,
		" OK ":         SeatWitnessed,
		"rate-limited": SeatThrottled,
		"Rate_Limited": SeatThrottled,
		"timeout":      SeatTimedOut,
		"timed out":    SeatTimedOut,
		"error":        SeatFailed,
		"dispatched":   SeatStarted,
	}
	for raw, want := range cases {
		got, ok := ParseSeatOutcome(raw)
		if !ok || got != want {
			t.Fatalf("ParseSeatOutcome(%q) = (%q,%v), want (%q,true)", raw, got, ok, want)
		}
	}
	for _, raw := range []string{"", "   ", "cancelled", "weird"} {
		if got, ok := ParseSeatOutcome(raw); ok {
			t.Fatalf("ParseSeatOutcome(%q) = (%q,true), want ok=false", raw, got)
		}
	}
}

// TestSeatScoreCarriesNoSecret enforces the "no secrets exposed" done-condition
// structurally: the JSON of a folded score (and of a ledger event) contains only the closed
// set of count/score keys — no room for a token, reason, prompt, or credential field.
func TestSeatScoreCarriesNoSecret(t *testing.T) {
	scoreKeys := jsonKeys(t, SeatScore{Seat: "dir:beta", Started: 1, Witnessed: 1})
	wantScoreKeys := map[string]bool{
		"seat": true, "started": true, "witnessed": true, "failed": true,
		"throttled": true, "timed_out": true, "resolved": true, "in_flight": true,
		"health": true, "scored": true,
	}
	if !reflect.DeepEqual(scoreKeys, wantScoreKeys) {
		t.Fatalf("SeatScore keys = %v, want %v", scoreKeys, wantScoreKeys)
	}
	eventKeys := jsonKeys(t, SeatEvent{Seat: "dir:beta", Outcome: SeatWitnessed})
	wantEventKeys := map[string]bool{"seat": true, "outcome": true}
	if !reflect.DeepEqual(eventKeys, wantEventKeys) {
		t.Fatalf("SeatEvent keys = %v, want %v", eventKeys, wantEventKeys)
	}
	for k := range scoreKeys {
		for _, bad := range []string{"token", "secret", "reason", "prompt", "cred", "key", "auth"} {
			if k == bad {
				t.Fatalf("SeatScore exposes a %q field — a seat report must carry no secret", bad)
			}
		}
	}
}

// TestDecodeSeatLedgerArrayAndJSONL checks both accepted ledger encodings decode equally,
// and that a malformed row is a hard error (not a silent under-count).
func TestDecodeSeatLedgerArrayAndJSONL(t *testing.T) {
	jsonl := "{\"seat\":\"dir:a\",\"outcome\":\"witnessed\"}\n{\"seat\":\"dir:a\",\"outcome\":\"failed\"}"
	arr := `[{"seat":"dir:a","outcome":"witnessed"},{"seat":"dir:a","outcome":"failed"}]`
	fromJSONL, err := DecodeSeatLedger([]byte(jsonl))
	if err != nil {
		t.Fatalf("decode jsonl: %v", err)
	}
	fromArr, err := DecodeSeatLedger([]byte(arr))
	if err != nil {
		t.Fatalf("decode array: %v", err)
	}
	if !reflect.DeepEqual(fromJSONL, fromArr) {
		t.Fatalf("jsonl and array decode differ:\n jsonl=%+v\n arr=%+v", fromJSONL, fromArr)
	}
	if _, err := DecodeSeatLedger([]byte("{not json}")); err == nil {
		t.Fatal("malformed ledger row should be a hard decode error")
	}
	if evs, err := DecodeSeatLedger(nil); err != nil || evs != nil {
		t.Fatalf("empty ledger should decode to (nil,nil), got (%v,%v)", evs, err)
	}
}

// FoldSeatLedgerOrPanic is a tiny test helper: fold a JSONL fixture or fail the test.
func FoldSeatLedgerOrPanic(t *testing.T, ledger string) []SeatScore {
	t.Helper()
	scores, err := FoldSeatLedger([]byte(ledger))
	if err != nil {
		t.Fatalf("FoldSeatLedger: %v", err)
	}
	return scores
}

// jsonKeys marshals v and returns its top-level object keys as a set.
func jsonKeys(t *testing.T, v any) map[string]bool {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(data, &obj); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	keys := map[string]bool{}
	for k := range obj {
		keys[k] = true
	}
	return keys
}
