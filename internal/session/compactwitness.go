package session

// compactwitness.go — the consumer seams for the compact-audit schema (#5138).
//
// #4763 shipped the miner (compactaudit.go) and its schema, CompactSessionReport. Two
// separately-owned consumers were asked to READ that schema rather than grow parallel
// shapes, and this file is where their wiring lives so neither ever re-declares a field:
//
//   - #3152 (durable per-session compaction-health witness): CompactWitnessRow EMBEDS
//     CompactSessionReport — the miner's report IS the witness schema, wrapped only in
//     the durable-row envelope (schema version + recorded-at). AppendCompactWitnesses /
//     ReadCompactWitnesses give the row a session-keyed JSONL ledger readable without
//     any live gateway process, which is exactly the gap #3152 names.
//
//   - #3187 (live-session dogfood): DecodeCompactAudit is THE parser for the machine
//     form `fak session compact-audit --json` emits (WriteCompactAuditJSON), and
//     AuditResidentCeiling folds a decoded sweep into the resident-token-ceiling view
//     the dogfood scores against — so the dogfood consumes the audit, never a bespoke
//     rollout parser.
//
// The shared-schema drift test (compact_schema_drift_test.go) holds both consumers to
// this: the witness row must stay a pure embedding (no shadowed JSON keys) and the
// decoder must round-trip the miner's own output byte-for-byte.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"time"
)

// CompactWitnessSchemaVersion stamps every durable witness row so a later reader can
// tell which shape wrote it. It moves only when CompactSessionReport itself moves in a
// way an old reader cannot absorb.
const CompactWitnessSchemaVersion = 1

// CompactWitnessRow is the durable per-session compaction-health row #3152 asks for.
// It EMBEDS CompactSessionReport rather than re-declaring any field: the miner's schema
// is the single source, and the row adds only the durable-ledger envelope. A row is
// self-contained — session id, fires, verdict, and every resident-context witness are
// readable from the JSONL alone, with no live gateway process.
type CompactWitnessRow struct {
	SchemaVersion int    `json:"schema_version"`
	RecordedAt    string `json:"recorded_at"` // RFC3339 UTC — when the row was witnessed, not when the session ran
	CompactSessionReport
}

// NewCompactWitnessRow wraps one mined report in the durable-row envelope.
func NewCompactWitnessRow(rep CompactSessionReport, at time.Time) CompactWitnessRow {
	return CompactWitnessRow{
		SchemaVersion:        CompactWitnessSchemaVersion,
		RecordedAt:           at.UTC().Format(time.RFC3339),
		CompactSessionReport: rep,
	}
}

// AppendCompactWitnesses appends one durable witness row per report to the JSONL ledger
// at path, creating it if absent. Append-only by construction (O_APPEND): a later sweep
// adds rows, it never rewrites history — the same discipline as every other durable
// ledger in the tree.
func AppendCompactWitnesses(path string, reports []CompactSessionReport, at time.Time) error {
	if len(reports) == 0 {
		return nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, rep := range reports {
		if err := enc.Encode(NewCompactWitnessRow(rep, at)); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
	}
	return nil
}

// ReadCompactWitnesses reads every durable witness row back from the JSONL ledger at
// path, in append order. Blank lines are skipped; a malformed row is an error, not a
// silent drop — a durable witness that cannot be re-read is the failure mode the ledger
// exists to rule out.
func ReadCompactWitnesses(path string) ([]CompactWitnessRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []CompactWitnessRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64<<10), 16<<20)
	line := 0
	for sc.Scan() {
		line++
		b := sc.Bytes()
		if len(b) == 0 {
			continue
		}
		var row CompactWitnessRow
		if err := json.Unmarshal(b, &row); err != nil {
			return rows, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		return rows, fmt.Errorf("%s: %w", path, err)
	}
	return rows, nil
}

// DecodeCompactAudit parses the machine form `fak session compact-audit --json` emits
// (the exact document WriteCompactAuditJSON writes). It is the one parser a consumer —
// #3187's live-session dogfood in particular — should use, so the audit's schema stays
// the single source instead of each consumer growing its own rollout reader.
func DecodeCompactAudit(r io.Reader) (CompactAuditResult, error) {
	var res CompactAuditResult
	if err := json.NewDecoder(r).Decode(&res); err != nil {
		return CompactAuditResult{}, fmt.Errorf("compact-audit json: %w", err)
	}
	return res, nil
}

// CompactCeilingSession is one session scored against a resident-token ceiling — the
// per-session slice of the #3187 dogfood view, carrying just the fields the ceiling
// question needs, each copied from the shared report (never re-derived from rollouts).
type CompactCeilingSession struct {
	SessionID          string `json:"session_id"`
	PeakResidentTokens int    `json:"peak_resident_tokens"`
	ContextWindow      int    `json:"context_window"`
	FireCount          int    `json:"fire_count"`
	Verdict            string `json:"verdict"`
}

// ResidentCeilingWitness is the resident-token-ceiling answer #3187's dogfood scores
// against: of the audited sessions, which ones carried a peak RESIDENT window over the
// ceiling, and did compaction fire for them. Derived entirely from a decoded
// CompactAuditResult, so the dogfood's witness and the miner's report can never drift.
type ResidentCeilingWitness struct {
	Ceiling     int                     `json:"ceiling"`
	Sessions    int                     `json:"sessions"`
	OverCeiling []CompactCeilingSession `json:"over_ceiling,omitempty"`
}

// AuditResidentCeiling folds a decoded compact-audit sweep into the resident-ceiling
// witness. ceiling is the resident-token bar (compactcohere's FAK_CTX_YIELD_CEILING in
// the #3187 dogfood); a non-positive ceiling yields an empty over-ceiling set, matching
// compactcohere's "non-positive ceiling disables every resident-token term".
func AuditResidentCeiling(res CompactAuditResult, ceiling int) ResidentCeilingWitness {
	w := ResidentCeilingWitness{Ceiling: ceiling, Sessions: len(res.Sessions)}
	if ceiling <= 0 {
		return w
	}
	for _, s := range res.Sessions {
		if s.PeakResidentTokens <= ceiling {
			continue
		}
		w.OverCeiling = append(w.OverCeiling, CompactCeilingSession{
			SessionID:          s.SessionID,
			PeakResidentTokens: s.PeakResidentTokens,
			ContextWindow:      s.ContextWindow,
			FireCount:          s.FireCount,
			Verdict:            s.Verdict,
		})
	}
	return w
}
