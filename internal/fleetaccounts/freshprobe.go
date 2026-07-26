package fleetaccounts

import (
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountprobe"
)

// probeStatusKind maps the account-probe ledger's closed status vocabulary to the
// roster's block-kind, mirroring fleet_accounts._PROBE_STATUS_KIND. OK is handled
// separately (available); statuses not in this map (APIERR/TRANSPORT/unknown) are not
// a clean availability signal and yield no verdict.
var probeStatusKind = map[string]string{
	"AUTH":   "auth",
	"ACCESS": "access",
	"CREDIT": "credit",
	"LIMIT":  "usage",
}

// ProbeLedgerFreshMin resolves the freshness window (minutes) within which an active
// probe verdict overrides a carried registry block, honoring FLEET_PROBE_FRESH_MIN
// (default 20). Mirrors fleet_accounts.PROBE_LEDGER_FRESH_MIN.
func ProbeLedgerFreshMin() float64 {
	if v := strings.TrimSpace(os.Getenv("FLEET_PROBE_FRESH_MIN")); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return 20
}

// entitlementKinds are the block kinds produced by an authorization or billing
// decision rather than by a quota wall. The distinction decides how fast the verdict
// may be aged out: a "usage" block expires on its own — the window resets and the seat
// comes back — so forgetting it after ProbeLedgerFreshMin is correct. These do not
// expire. No amount of waiting turns a 403 oauth_org_not_allowed into a 200.
var entitlementKinds = map[string]bool{"auth": true, "access": true, "credit": true}

// ProbeLedgerEntitlementFreshMin resolves the freshness window (minutes) for an
// entitlement-class probe verdict, honoring FLEET_PROBE_ENTITLEMENT_FRESH_MIN
// (default 1440 = 24h).
//
// It is deliberately much longer than ProbeLedgerFreshMin, and deliberately still
// finite. Longer, because aging out one of these verdicts does not merely lose
// information, it inverts it: a seat whose org disabled its subscription answers 403
// to every request, and a 403 burns no quota, so the moment the verdict expires the
// headroom-weighted allocator sees a seat with full headroom and ranks the dead seat
// among the EMPTIEST in the fleet — handing it the most workers, each of which dies in
// well under a second. The stale-verdict failure mode is not "we under-use a seat", it
// is "we preferentially route to the one seat that cannot work".
//
// Still finite, because a prober that stops running must not strand a seat forever on
// its last bad verdict. Past this window the fold falls back to the registry's own
// status exactly as before, so the worst case of a dead prober is the behavior that
// shipped before this window existed, not a permanently evicted account.
func ProbeLedgerEntitlementFreshMin() float64 {
	if v := strings.TrimSpace(os.Getenv("FLEET_PROBE_ENTITLEMENT_FRESH_MIN")); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return 1440
}

// FreshProbe is the roster fresh-probe verdict shape, mirroring the dict
// fleet_accounts._fresh_probe_from_ledger returns for the fresh-probe branch of
// runtime_status. A blocked verdict carries the reset/weekly windows so a downstream
// fold can surface them.
type FreshProbe struct {
	Available   bool
	BlockKind   string
	BlockReason string
	Reset       string
	Weekly      string
	AgeMin      float64
}

// FreshProbeFromLedger returns the freshest active-probe verdict for account from the
// account-probe ledger under regDir, IF it is within freshMin minutes of now — else
// nil. It is the Go port of fleet_accounts._fresh_probe_from_ledger: the missing link
// between the prober (account_probe writes OK/LIMIT/AUTH to probe_ledger.jsonl) and
// the roster (runtime_status reads sessions.json). Consulting it lets a recent probe
// override a carried block with one freshness gate so a stale OK cannot mask a real
// current limit.
//
// The freshness window depends on what the verdict SAYS, not only on how old it is.
// An OK or a usage wall ages out after ProbeLedgerFreshMin, because both are claims
// about a moment: quota resets, and a stale OK must never mask a real current limit.
// An entitlement verdict (auth/access/credit) ages out after the much longer
// ProbeLedgerEntitlementFreshMin instead, because waiting cannot clear it — see that
// function for why an expired one is worse than no verdict at all.
//
// Honoring an old entitlement verdict cannot pin a seat that has recovered:
// LastProbeByAccount keeps only the NEWEST row per account, so the instant a probe
// reads OK that OK *is* the entry and the block is gone on the next read. The window
// governs how long a verdict survives with no newer probe, never whether a newer probe
// wins. That keeps the self-healing property intact and is the specific reason this is
// a widened window rather than a sticky block or an eviction.
//
// freshMin <= 0 uses the pair of defaults above. A caller passing an explicit freshMin
// gets exactly that window for every kind, so an explicit window still means what it
// says. now is injected for determinism (pass time.Now().UTC() in production).
// computeRuntimeStatus consults this fold when FLEET_REG_DIR names the prober's
// registry dir (see shouldConsultProbeLedger), so the whole roster surface —
// available/rotation/resolve — self-heals off a fresh probe.
func FreshProbeFromLedger(account, regDir string, now time.Time, freshMin float64) *FreshProbe {
	explicitWindow := freshMin > 0
	if !explicitWindow {
		freshMin = ProbeLedgerFreshMin()
	}
	entry, ok := accountprobe.LastProbeByAccount(regDir)[account]
	if !ok {
		return nil
	}
	age := accountprobe.RecentProbeAgeMin(account, regDir, now)
	if age == nil {
		return nil
	}
	status := strings.ToUpper(strings.TrimSpace(entry.Status))
	kind, known := probeStatusKind[status]
	window := freshMin
	if !explicitWindow && known && entitlementKinds[kind] {
		if w := ProbeLedgerEntitlementFreshMin(); w > window {
			window = w
		}
	}
	if *age > window {
		return nil
	}
	if status == "OK" {
		return &FreshProbe{Available: true, AgeMin: *age}
	}
	if !known {
		// Any other status (APIERR/TRANSPORT/unknown) is not a clean availability
		// signal — fall through to the registry's own status.
		return nil
	}
	reset := entry.Reset
	reason := entry.BlockReason
	if reason == "" {
		reason = entry.Reason
	}
	if reason == "" {
		if kind == "usage" {
			if reset != "" {
				reason = "usage limit; resets " + reset
			} else {
				reason = "usage limit"
			}
		} else {
			reason = kind + " block"
		}
	}
	return &FreshProbe{
		Available:   false,
		BlockKind:   kind,
		BlockReason: reason,
		Reset:       reset,
		Weekly:      entry.Weekly,
		AgeMin:      *age,
	}
}
