package sessionaudit

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

// fixedPostureTime is a stable timestamp so the dated audit row is deterministic under test
// (the reconciler takes the clock as an argument precisely so the fold stays pure).
var fixedPostureTime = time.Date(2026, 7, 11, 12, 0, 0, 0, time.UTC)

// TestReconcileManagedCachePosture is the core done-condition witness (#3624): a session
// whose banner said ACTIVE but whose wire never carried a 1h upgrade reconciles to
// POSTURE_MISMATCH; a consistent session reconciles to POSTURE_OK. It sweeps every
// claim×wire quadrant over synthetic ManagedCacheVars-vs-ledger inputs.
func TestReconcileManagedCachePosture(t *testing.T) {
	cases := []struct {
		name        string
		claim       ManagedCacheClaim
		ledger      ManagedCacheLedger
		wantVerdict ManagedCachePostureVerdict
		wantWire    bool
		wantReasons []string
	}{
		{
			// The done-condition headline: banner ACTIVE, wire never upgraded (every head
			// refused) -> claimed ACTIVE, behaved PASSIVE -> POSTURE_MISMATCH.
			name:        "active claim, no wire upgrade -> mismatch",
			claim:       ManagedCacheClaim{Active: true, Inert: true, Upgraded: 0, Reasons: map[string]uint64{"no_stable_breakpoint": 4}},
			ledger:      ManagedCacheLedger{Upgraded: 0, Reasons: map[string]uint64{"no_stable_breakpoint": 4}},
			wantVerdict: PostureMismatch,
			wantWire:    false,
			wantReasons: []string{"no_stable_breakpoint"},
		},
		{
			// Consistent ACTIVE: the durable ledger witnessed a real 1h upgrade.
			name:        "active claim, ledger upgrade -> ok",
			claim:       ManagedCacheClaim{Active: true, Upgraded: 3},
			ledger:      ManagedCacheLedger{Upgraded: 3},
			wantVerdict: PostureOK,
			wantWire:    true,
		},
		{
			// Either witness suffices: the /debug/vars self-report saw an upgrade even though
			// the (absent) ledger read zero — the wire still carried one, so OK.
			name:        "active claim, self-report upgrade only -> ok",
			claim:       ManagedCacheClaim{Active: true, Upgraded: 1},
			ledger:      ManagedCacheLedger{Upgraded: 0},
			wantVerdict: PostureOK,
			wantWire:    true,
		},
		{
			// Consistent PASSIVE: lever off, nothing shipped — the two agree, so OK.
			name:        "passive claim, no wire upgrade -> ok",
			claim:       ManagedCacheClaim{Active: false, Upgraded: 0},
			ledger:      ManagedCacheLedger{Upgraded: 0},
			wantVerdict: PostureOK,
			wantWire:    false,
		},
		{
			// Inverse gap: banner PASSIVE but the wire DID upgrade -> still a claim-vs-wire
			// disagreement -> POSTURE_MISMATCH.
			name:        "passive claim, wire upgrade -> mismatch",
			claim:       ManagedCacheClaim{Active: false, Upgraded: 0},
			ledger:      ManagedCacheLedger{Upgraded: 2},
			wantVerdict: PostureMismatch,
			wantWire:    true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := ReconcileManagedCachePosture(tc.claim, tc.ledger, fixedPostureTime)
			if got.Verdict != tc.wantVerdict {
				t.Fatalf("verdict = %q, want %q (finding: %s)", got.Verdict, tc.wantVerdict, got.Finding)
			}
			if got.WireUpgraded != tc.wantWire {
				t.Fatalf("wire_upgraded = %v, want %v", got.WireUpgraded, tc.wantWire)
			}
			if got.Schema != managedCachePostureSchema {
				t.Fatalf("schema = %q, want %q", got.Schema, managedCachePostureSchema)
			}
			if got.Generated != fixedPostureTime.Format(time.RFC3339) {
				t.Fatalf("generated = %q, want dated row %q", got.Generated, fixedPostureTime.Format(time.RFC3339))
			}
			if got.ClaimedUpgrades != tc.claim.Upgraded || got.LedgerUpgrades != tc.ledger.Upgraded {
				t.Fatalf("witnesses not surfaced separately: claimed=%d ledger=%d", got.ClaimedUpgrades, got.LedgerUpgrades)
			}
			if tc.wantReasons != nil && !reflect.DeepEqual(got.RefusalReasons, tc.wantReasons) {
				t.Fatalf("refusal_reasons = %v, want %v", got.RefusalReasons, tc.wantReasons)
			}
			if got.Finding == "" {
				t.Fatalf("finding must not be empty for verdict %q", got.Verdict)
			}
		})
	}
}

// TestReconcileManagedCachePostureReasonUnion proves the refusal breakdown surfaced with a
// mismatch is the SORTED UNION across both witnesses' reason maps — the deterministic WHY an
// operator reads to see why an ACTIVE session behaved passive.
func TestReconcileManagedCachePostureReasonUnion(t *testing.T) {
	claim := ManagedCacheClaim{Active: true, Inert: true, Reasons: map[string]uint64{"volatile_head": 2}}
	ledger := ManagedCacheLedger{Reasons: map[string]uint64{"no_stable_breakpoint": 5, "volatile_head": 1}}
	got := ReconcileManagedCachePosture(claim, ledger, fixedPostureTime)
	if got.Verdict != PostureMismatch {
		t.Fatalf("verdict = %q, want POSTURE_MISMATCH", got.Verdict)
	}
	want := []string{"no_stable_breakpoint", "volatile_head"}
	if !reflect.DeepEqual(got.RefusalReasons, want) {
		t.Fatalf("refusal_reasons = %v, want %v", got.RefusalReasons, want)
	}
}

// TestManagedCacheClaimFromDebugVars decodes the managed_cache block out of a full
// /debug/vars body, and proves an absent block (a passive/cold session) is reported as
// not-present rather than an error — a captured-session round-trip witness.
func TestManagedCacheClaimFromDebugVars(t *testing.T) {
	// A captured /debug/vars body whose managed_cache block claims ACTIVE with zero upgrades
	// (Inert) — the exact "claimed ACTIVE, behaved PASSIVE" shape the done-condition targets.
	body := []byte(`{
		"gateway": {"up": true},
		"managed_cache": {"active": true, "inert": true, "upgraded": 0, "reasons": {"no_stable_breakpoint": 3}}
	}`)
	claim, ok := ManagedCacheClaimFromDebugVars(body)
	if !ok {
		t.Fatal("expected managed_cache block to be present")
	}
	if !claim.Active || claim.Upgraded != 0 || claim.Reasons["no_stable_breakpoint"] != 3 {
		t.Fatalf("decoded claim mismatch: %+v", claim)
	}

	// A ledger counters object witnessing zero real upgrades on the wire.
	counters := []byte(`{"cache_ttl_upgrades_upgraded": 0, "cache_ttl_upgrade_reasons": {"no_stable_breakpoint": 3}}`)
	ledger, ok := ManagedCacheLedgerFromCounters(counters)
	if !ok {
		t.Fatal("expected ledger counters to decode")
	}
	audit := ReconcileManagedCachePosture(claim, ledger, fixedPostureTime)
	if audit.Verdict != PostureMismatch {
		t.Fatalf("captured-session reconciliation verdict = %q, want POSTURE_MISMATCH (finding: %s)", audit.Verdict, audit.Finding)
	}

	// Absent block: a passive/cold session omits managed_cache -> not present, not an error.
	if _, ok := ManagedCacheClaimFromDebugVars([]byte(`{"gateway": {"up": true}}`)); ok {
		t.Fatal("expected absent managed_cache block to report not-present")
	}
	// Malformed JSON fails closed on both decoders.
	if _, ok := ManagedCacheClaimFromDebugVars([]byte(`{not json`)); ok {
		t.Fatal("expected malformed /debug/vars body to fail closed")
	}
	if _, ok := ManagedCacheLedgerFromCounters([]byte(`{not json`)); ok {
		t.Fatal("expected malformed counters body to fail closed")
	}
}

// TestManagedCachePostureAuditJSON proves the audit row marshals to the versioned schema
// with the dated generated stamp — a machine-readable dated audit row.
func TestManagedCachePostureAuditJSON(t *testing.T) {
	audit := ReconcileManagedCachePosture(
		ManagedCacheClaim{Active: true, Inert: true},
		ManagedCacheLedger{Upgraded: 0},
		fixedPostureTime,
	)
	b, err := json.Marshal(audit)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var round map[string]any
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if round["schema"] != managedCachePostureSchema {
		t.Fatalf("schema = %v, want %q", round["schema"], managedCachePostureSchema)
	}
	if round["verdict"] != string(PostureMismatch) {
		t.Fatalf("verdict = %v, want %q", round["verdict"], PostureMismatch)
	}
	if round["generated"] != fixedPostureTime.Format(time.RFC3339) {
		t.Fatalf("generated = %v, want %q", round["generated"], fixedPostureTime.Format(time.RFC3339))
	}
}
