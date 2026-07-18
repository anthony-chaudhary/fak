package session

// compact_schema_drift_test.go — the shared-schema drift gate (#5138): the compact-audit
// consumers (#3152 durable witness, #3187 dogfood decoder) must stay bound to
// internal/session.CompactSessionReport as the SINGLE schema source. Three ways a
// parallel shape could creep in, each pinned here:
//
//  1. the witness row re-declaring (and thereby shadowing) a report field —
//     caught structurally by reflection over CompactWitnessRow;
//  2. the dogfood decoder diverging from the miner's JSON — caught by round-tripping
//     the miner's own WriteCompactAuditJSON output byte-for-byte through
//     DecodeCompactAudit;
//  3. the durable ledger dropping or renaming report fields — caught by re-marshalling
//     the embedded report out of a re-read row and comparing it to the miner's bytes.
//
// The corpus fixtures under testdata/compactaudit drive the round trips, so the fire
// counts a consumer reads are the same ones `fak session compact-audit` reports.

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

// TestCompactWitnessRowEmbedsMinerSchema is the structural half of the drift gate: the
// durable witness row must carry the miner's report as ONE anonymous embedding, and its
// own envelope fields must not shadow any report JSON key — a shadowed key is exactly
// the "parallel shape" #5138 forbids.
func TestCompactWitnessRowEmbedsMinerSchema(t *testing.T) {
	rowT := reflect.TypeOf(CompactWitnessRow{})
	repT := reflect.TypeOf(CompactSessionReport{})

	embedded := 0
	envelopeTags := map[string]bool{}
	for i := 0; i < rowT.NumField(); i++ {
		f := rowT.Field(i)
		if f.Anonymous {
			if f.Type != repT {
				t.Fatalf("CompactWitnessRow embeds %v; the one embedding must be session.CompactSessionReport", f.Type)
			}
			embedded++
			continue
		}
		tag := strings.Split(f.Tag.Get("json"), ",")[0]
		if tag == "" || tag == "-" {
			t.Fatalf("CompactWitnessRow envelope field %s needs an explicit json tag", f.Name)
		}
		envelopeTags[tag] = true
	}
	if embedded != 1 {
		t.Fatalf("CompactWitnessRow embeds CompactSessionReport %d times, want exactly 1", embedded)
	}
	for i := 0; i < repT.NumField(); i++ {
		tag := strings.Split(repT.Field(i).Tag.Get("json"), ",")[0]
		if tag != "" && envelopeTags[tag] {
			t.Errorf("CompactWitnessRow envelope re-declares report JSON key %q — schema drift", tag)
		}
	}
}

// TestDecodeCompactAuditRoundTripsMinerJSON is the #3187 half: DecodeCompactAudit must
// consume the exact document `fak session compact-audit --json` emits, losslessly —
// re-encoding a decoded sweep reproduces the miner's bytes, and the fire counts a
// dogfood reads are the miner's fire counts.
func TestDecodeCompactAuditRoundTripsMinerJSON(t *testing.T) {
	res, err := AuditCompactCorpus(CompactAuditOptions{Root: filepath.Join("testdata", "compactaudit")})
	if err != nil {
		t.Fatalf("AuditCompactCorpus: %v", err)
	}
	if len(res.Sessions) == 0 || res.Aggregate.Fires == 0 {
		t.Fatalf("fixture corpus yielded no fires (%d sessions) — round trip would be vacuous", len(res.Sessions))
	}

	var minted bytes.Buffer
	if err := WriteCompactAuditJSON(&minted, res); err != nil {
		t.Fatalf("WriteCompactAuditJSON: %v", err)
	}
	decoded, err := DecodeCompactAudit(bytes.NewReader(minted.Bytes()))
	if err != nil {
		t.Fatalf("DecodeCompactAudit: %v", err)
	}
	var reencoded bytes.Buffer
	if err := WriteCompactAuditJSON(&reencoded, decoded); err != nil {
		t.Fatalf("re-encode: %v", err)
	}
	if !bytes.Equal(minted.Bytes(), reencoded.Bytes()) {
		t.Fatalf("decode/re-encode drifted from the miner's JSON:\nminer:  %.400s\nreread: %.400s", minted.String(), reencoded.String())
	}

	if decoded.Aggregate.Fires != res.Aggregate.Fires {
		t.Fatalf("decoded aggregate fires = %d, miner reported %d", decoded.Aggregate.Fires, res.Aggregate.Fires)
	}
	for i, s := range res.Sessions {
		if decoded.Sessions[i].FireCount != s.FireCount {
			t.Errorf("session %s: decoded fire_count %d, miner %d", s.SessionID, decoded.Sessions[i].FireCount, s.FireCount)
		}
	}
}

// TestCompactWitnessLedgerRoundTrip is the #3152 half: every report appended to the
// durable ledger must come back with the miner's schema intact — same session keys,
// same fire counts, and the embedded report re-marshals to the miner's own bytes.
func TestCompactWitnessLedgerRoundTrip(t *testing.T) {
	res, err := AuditCompactCorpus(CompactAuditOptions{Root: filepath.Join("testdata", "compactaudit")})
	if err != nil {
		t.Fatalf("AuditCompactCorpus: %v", err)
	}
	path := filepath.Join(t.TempDir(), "compact-witness.jsonl")
	at := time.Date(2026, 7, 17, 12, 0, 0, 0, time.UTC)
	if err := AppendCompactWitnesses(path, res.Sessions, at); err != nil {
		t.Fatalf("AppendCompactWitnesses: %v", err)
	}
	rows, err := ReadCompactWitnesses(path)
	if err != nil {
		t.Fatalf("ReadCompactWitnesses: %v", err)
	}
	if len(rows) != len(res.Sessions) {
		t.Fatalf("ledger returned %d rows, want %d", len(rows), len(res.Sessions))
	}
	for i, row := range rows {
		if row.SchemaVersion != CompactWitnessSchemaVersion {
			t.Errorf("row %d schema_version = %d, want %d", i, row.SchemaVersion, CompactWitnessSchemaVersion)
		}
		if row.RecordedAt != at.Format(time.RFC3339) {
			t.Errorf("row %d recorded_at = %q, want %q", i, row.RecordedAt, at.Format(time.RFC3339))
		}
		want, err := json.Marshal(res.Sessions[i])
		if err != nil {
			t.Fatalf("marshal miner report: %v", err)
		}
		got, err := json.Marshal(row.CompactSessionReport)
		if err != nil {
			t.Fatalf("marshal re-read report: %v", err)
		}
		if !bytes.Equal(want, got) {
			t.Errorf("row %d (%s): durable report drifted from the miner's:\nminer:  %.300s\nledger: %.300s",
				i, res.Sessions[i].SessionID, want, got)
		}
	}
}

// TestAuditResidentCeiling pins the #3187 dogfood view: sessions over the ceiling come
// from PeakResidentTokens on the shared report, a bounded session stays out, and a
// non-positive ceiling disables the term (mirroring compactcohere's contract).
func TestAuditResidentCeiling(t *testing.T) {
	res := CompactAuditResult{Sessions: []CompactSessionReport{
		{SessionID: "over", PeakResidentTokens: 180_000, ContextWindow: 200_000, FireCount: 3, Verdict: VerdictFiredAndHeld},
		{SessionID: "under", PeakResidentTokens: 40_000, ContextWindow: 200_000, Verdict: VerdictNoFireBounded},
	}}
	w := AuditResidentCeiling(res, 160_000)
	if w.Sessions != 2 || len(w.OverCeiling) != 1 {
		t.Fatalf("witness = %+v, want 2 sessions with 1 over ceiling", w)
	}
	got := w.OverCeiling[0]
	if got.SessionID != "over" || got.PeakResidentTokens != 180_000 || got.FireCount != 3 || got.Verdict != VerdictFiredAndHeld {
		t.Errorf("over-ceiling row = %+v, fields must copy the shared report", got)
	}
	if off := AuditResidentCeiling(res, 0); len(off.OverCeiling) != 0 {
		t.Errorf("non-positive ceiling must disable the term, got %+v", off.OverCeiling)
	}
}
