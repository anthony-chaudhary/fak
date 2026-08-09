package session

// compactaudit_provenance.go — restrict a compaction-health sweep to sessions whose
// traffic actually crossed fak's wire (#5254).
//
// Why this exists. The #5254 dedup port (`internal/agent/message_elide.go`) is a
// GATEWAY-side transform: it can only fold bytes for a session that went through fak.
// The Codex rollout corpus under ~/.codex/sessions is mixed-provenance — it records
// every Codex session on the box, guarded or not. Measuring the port by re-running
// `compact-audit` over that whole corpus therefore reads a population fak mostly never
// saw, so the number moves for reasons unrelated to the change. The 2026-07-19 decision
// note measured the mix directly: 120 of 2,448 corpus sessions (4.9%) appeared in the
// guard ledger. A "materially reduced" reading off that corpus is unfalsifiable, not
// passing.
//
// The discriminator is the ledger `fak guard` writes per guarded Codex session:
// <codex-home>/fak-guarded-sessions/<session-id>.json, schema fak.codex_guard_witness.v1
// (cmd/fak/sessions_codex_loop.go). Config files are NOT a discriminator — `fak guard`
// injects its provider with per-invocation `-c model_providers.fak.*` argv overrides and
// leaves no durable trace in ~/.codex/config.toml, so config absence is uninformative in
// both directions.
//
// The refusal posture is the load-bearing part. An empty or unreadable ledger under
// GuardedOnly yields an empty sweep whose aggregate reads as a perfect zero — the exact
// shape of "the anomaly class is gone". That would be a green-looking lie, so a
// guarded-only sweep with no ledger to stand on is an ERROR, never an empty pass.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// GuardWitnessSchema is the schema tag `fak guard` stamps on each per-session witness.
const GuardWitnessSchema = "fak.codex_guard_witness.v1"

// GuardWitnessDirName is the ledger directory, relative to the Codex home.
const GuardWitnessDirName = "fak-guarded-sessions"

// CompactProvenance records WHICH subset of the corpus a sweep measured. It rides in the
// result so a checked-in witness JSON says on its face whether it is a fak-routed cohort
// or the whole mixed-provenance box — two documents that are otherwise identical in shape
// and wildly different in what they license you to claim.
type CompactProvenance struct {
	// GuardedOnly is true when the sweep kept only ledger-present sessions.
	GuardedOnly bool `json:"guarded_only"`
	// LedgerSessions is how many guarded session ids the ledger held. The remaining
	// fields describe the filter's effect and are omitted when it is off.
	LedgerSessions int `json:"ledger_sessions,omitempty"`
	// Guarded/Unguarded split the rollouts that survived the other filters, so the
	// cohort's share of its own corpus slice is visible rather than inferred.
	Guarded   int `json:"guarded_sessions,omitempty"`
	Unguarded int `json:"unguarded_sessions,omitempty"`
}

// LoadGuardWitnessIDs reads the guard ledger directory and returns the set of session ids
// fak is witnessed to have routed. Junk files are skipped rather than fataled — the
// directory is live and a half-written witness must not sink the sweep — but a directory
// that yields NO ids is an error, because silently returning an empty set turns a
// guarded-only sweep into a zero-row pass.
func LoadGuardWitnessIDs(dir string) (map[string]struct{}, error) {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return nil, fmt.Errorf("guard witness ledger: no directory (pass --guard-witness-dir)")
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("guard witness ledger %s: %w", dir, err)
	}
	ids := make(map[string]struct{}, len(entries))
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		var w struct {
			Schema    string `json:"schema"`
			SessionID string `json:"session_id"`
		}
		if json.Unmarshal(raw, &w) != nil || w.Schema != GuardWitnessSchema || w.SessionID == "" {
			continue
		}
		ids[w.SessionID] = struct{}{}
	}
	if len(ids) == 0 {
		return nil, fmt.Errorf("guard witness ledger %s: no %s records — a guarded-only sweep would measure nothing and report it as zero", dir, GuardWitnessSchema)
	}
	return ids, nil
}
