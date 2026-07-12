package sessionaudit

// Managed-cache posture reconciliation (#3624, epic #3569 cache-verify): the post-run
// trust-but-verify LOOP that reconciles a session's CLAIMED managed-cache posture against
// what the wire actually did. The guard startup banner prints a posture (ACTIVE/PASSIVE)
// from the resolved lever state (--managed-cache / Config.CacheTTL1H), but nothing after
// the run checked that claim against the wire: a session can print ACTIVE and behave
// PASSIVE — the lever is on but no 1h `ttl` body ever shipped. This lens closes that gap.
//
// THE TWO INPUTS (kept in distinct trust classes, never conflated).
//   - The CLAIM is the /debug/vars managed_cache block (guardvars.ManagedCacheVars): Active
//     is the posture the banner printed; Upgraded/Reasons are that block's own live
//     self-report of the outbound TTL-upgrade outcome.
//   - The WIRE witness is the durable gateway usage ledger's 1h TTL-upgrade counters
//     (gatewayusageledger.Counters.CacheTTLUpgrades*), folded from a persisted row rather
//     than the live process — the independent record of what actually shipped.
// Both mirror the same AdjudicationSummary.CacheTTLUpgrade* counter, so a 1h upgrade the
// wire really carried shows up in AT LEAST one of them; "the wire never carried a 1h
// upgrade" means BOTH witnesses read zero. The reconciliation therefore treats a positive
// count from EITHER witness as proof the wire upgraded, and the ACTIVE-vs-zero gap (or its
// inverse) as the POSTURE_MISMATCH.
//
// Like the Behavior/Confusion lenses this stays pure and deterministic: stdlib-only, no
// clock of its own (the caller supplies `generated`), stable ordering — same inputs in,
// same audit row out. Pricing and live in-session alarming are explicitly out of scope
// (#3624): this is a post-run reconciliation, not a pricing model or an A-class alarm.

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// managedCachePostureSchema versions the audit-row wire shape so a reader can detect a
// future field addition without guessing.
const managedCachePostureSchema = "fak.session_audit.managed_cache_posture.v1"

// ManagedCachePostureVerdict is the closed-vocabulary outcome of reconciling the claimed
// managed-cache posture against the observed wire behavior.
type ManagedCachePostureVerdict string

const (
	// PostureOK: the claim and the wire agree — an ACTIVE session that carried at least one
	// 1h upgrade, or a PASSIVE session that carried none.
	PostureOK ManagedCachePostureVerdict = "POSTURE_OK"
	// PostureMismatch: the claimed posture and the observed wire behavior disagree — the
	// banner said ACTIVE but the wire never carried a 1h upgrade (claimed ACTIVE, behaved
	// PASSIVE), or the inverse (claimed PASSIVE but the wire DID upgrade).
	PostureMismatch ManagedCachePostureVerdict = "POSTURE_MISMATCH"
)

// ManagedCacheClaim mirrors the /debug/vars managed_cache block (guardvars.ManagedCacheVars):
// the session's CLAIMED managed-cache posture as the banner printed it. Active is the
// resolved lever state (the ACTIVE/PASSIVE the banner shows); Inert is the block's own
// ACTIVE-but-zero-upgrade signal; Upgraded/Reasons are its live self-report of the TTL
// upgrade outcome (Reasons is refusal-only). The JSON tags match the producer so a captured
// /debug/vars block decodes straight in (see ManagedCacheClaimFromDebugVars).
type ManagedCacheClaim struct {
	Active   bool              `json:"active"`
	Inert    bool              `json:"inert"`
	Upgraded uint64            `json:"upgraded"`
	Reasons  map[string]uint64 `json:"reasons,omitempty"`
}

// ManagedCacheLedger mirrors the gateway usage-ledger 1h TTL-upgrade counters
// (gatewayusageledger.Counters.CacheTTLUpgrades*): the INDEPENDENT, durable wire witness of
// what actually shipped, folded from a persisted ledger row rather than the live
// /debug/vars block. Upgraded counts the actual 1h-tier upgrades; Reasons is the refusal
// breakdown. The JSON tags match the ledger's Counters so a row's counters object decodes
// straight in (see ManagedCacheLedgerFromCounters).
type ManagedCacheLedger struct {
	Upgraded uint64            `json:"cache_ttl_upgrades_upgraded"`
	Reasons  map[string]uint64 `json:"cache_ttl_upgrade_reasons,omitempty"`
}

// ManagedCachePostureAudit is one dated reconciliation verdict row: the CLAIMED posture, the
// two upgrade witnesses kept separate (the /debug/vars self-report and the durable ledger),
// whether the wire carried any 1h upgrade, and the refusal breakdown that explains a passive
// wire. Generated stamps WHEN the reconciliation ran so the row is a dated audit artifact.
type ManagedCachePostureAudit struct {
	Schema    string                     `json:"schema"`
	Generated string                     `json:"generated"`
	Verdict   ManagedCachePostureVerdict `json:"verdict"`
	// Claimed is the banner posture in words ("ACTIVE" | "PASSIVE"); ClaimedActive is the
	// raw bool it derives from.
	Claimed       string `json:"claimed"`
	ClaimedActive bool   `json:"claimed_active"`
	// ClaimedUpgrades is the /debug/vars block's own self-reported 1h-upgrade count;
	// LedgerUpgrades is the durable ledger's independent count. Kept apart so a reader sees
	// each trust class, never a conflated sum.
	ClaimedUpgrades uint64 `json:"claimed_upgrades"`
	LedgerUpgrades  uint64 `json:"ledger_upgrades"`
	// WireUpgraded is true when EITHER witness recorded at least one 1h upgrade — the
	// "did the wire actually carry a 1h upgrade" answer the verdict turns on.
	WireUpgraded bool `json:"wire_upgraded"`
	// RefusalReasons is the sorted union of the two witnesses' refusal breakdowns — the WHY
	// behind a passive wire (no_stable_breakpoint, already_1h, volatile_head, ...). Empty
	// when neither witness recorded a refusal.
	RefusalReasons []string `json:"refusal_reasons,omitempty"`
	Finding        string   `json:"finding"`
}

// ReconcileManagedCachePosture reconciles a session's CLAIMED managed-cache posture (the
// /debug/vars managed_cache block) against the OBSERVED wire behavior (the durable usage
// ledger's 1h TTL-upgrade counters), returning a dated POSTURE_OK / POSTURE_MISMATCH audit
// row. generated stamps the row (the caller supplies it, keeping this pure).
//
// The wire is deemed to have carried a 1h upgrade iff EITHER witness recorded a positive
// count (both mirror the same underlying counter, so one positive is proof). The verdict:
//   - ACTIVE claim + no wire upgrade  -> POSTURE_MISMATCH (claimed ACTIVE, behaved PASSIVE).
//   - PASSIVE claim + a wire upgrade  -> POSTURE_MISMATCH (claimed PASSIVE, behaved ACTIVE).
//   - otherwise                       -> POSTURE_OK (claim and wire agree).
func ReconcileManagedCachePosture(claim ManagedCacheClaim, ledger ManagedCacheLedger, generated time.Time) ManagedCachePostureAudit {
	wireUpgraded := claim.Upgraded > 0 || ledger.Upgraded > 0
	claimedLabel := "PASSIVE"
	if claim.Active {
		claimedLabel = "ACTIVE"
	}
	audit := ManagedCachePostureAudit{
		Schema:          managedCachePostureSchema,
		Generated:       generated.UTC().Format(time.RFC3339),
		Claimed:         claimedLabel,
		ClaimedActive:   claim.Active,
		ClaimedUpgrades: claim.Upgraded,
		LedgerUpgrades:  ledger.Upgraded,
		WireUpgraded:    wireUpgraded,
		RefusalReasons:  unionReasonKeys(claim.Reasons, ledger.Reasons),
	}
	switch {
	case claim.Active && !wireUpgraded:
		audit.Verdict = PostureMismatch
		why := "no 1h TTL upgrade attempt was recorded"
		if len(audit.RefusalReasons) > 0 {
			why = "every eligible head refused (" + strings.Join(audit.RefusalReasons, ", ") + ")"
		}
		audit.Finding = fmt.Sprintf(
			"POSTURE_MISMATCH — the banner claimed ACTIVE but the wire never carried a 1h TTL upgrade (behaved PASSIVE); %s",
			why)
	case !claim.Active && wireUpgraded:
		audit.Verdict = PostureMismatch
		audit.Finding = fmt.Sprintf(
			"POSTURE_MISMATCH — the banner claimed PASSIVE but the wire carried a 1h TTL upgrade (self_report=%d ledger=%d)",
			claim.Upgraded, ledger.Upgraded)
	default:
		audit.Verdict = PostureOK
		if claim.Active {
			audit.Finding = fmt.Sprintf(
				"POSTURE_OK — the banner claimed ACTIVE and the wire carried a 1h TTL upgrade (self_report=%d ledger=%d)",
				claim.Upgraded, ledger.Upgraded)
		} else {
			audit.Finding = "POSTURE_OK — the banner claimed PASSIVE and the wire carried no 1h TTL upgrade (consistent)"
		}
	}
	return audit
}

// unionReasonKeys returns the sorted union of the keys across the two refusal maps — the
// stable, deterministic refusal breakdown the audit row surfaces as the WHY behind a passive
// wire. Nil when both maps are empty.
func unionReasonKeys(a, b map[string]uint64) []string {
	if len(a) == 0 && len(b) == 0 {
		return nil
	}
	set := make(map[string]struct{}, len(a)+len(b))
	for k := range a {
		set[k] = struct{}{}
	}
	for k := range b {
		set[k] = struct{}{}
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// ManagedCacheClaimFromDebugVars extracts the managed_cache block from a full /debug/vars
// JSON body (the session's captured live vars). ok is false when the body does not parse or
// carries no managed_cache block — a passive, cold session omits it, which the caller should
// treat as a PASSIVE claim (the zero value), not an error.
func ManagedCacheClaimFromDebugVars(raw []byte) (ManagedCacheClaim, bool) {
	var envelope struct {
		ManagedCache *ManagedCacheClaim `json:"managed_cache"`
	}
	if err := json.Unmarshal(raw, &envelope); err != nil || envelope.ManagedCache == nil {
		return ManagedCacheClaim{}, false
	}
	return *envelope.ManagedCache, true
}

// ManagedCacheLedgerFromCounters extracts the 1h TTL-upgrade witness from a gateway
// usage-ledger row's counters object (the JSON under a Row's "counters" key). ok is false
// when the body does not parse; an absent counter simply decodes to zero (a session that
// never upgraded), which is a valid PASSIVE-wire witness, not an error.
func ManagedCacheLedgerFromCounters(raw []byte) (ManagedCacheLedger, bool) {
	var l ManagedCacheLedger
	if err := json.Unmarshal(raw, &l); err != nil {
		return ManagedCacheLedger{}, false
	}
	return l, true
}
