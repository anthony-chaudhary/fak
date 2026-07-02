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
// freshMin <= 0 uses ProbeLedgerFreshMin(). now is injected for determinism (pass
// time.Now().UTC() in production). computeRuntimeStatus consults this fold when
// FLEET_REG_DIR names the prober's registry dir (see shouldConsultProbeLedger), so the
// whole roster surface — available/rotation/resolve — self-heals off a fresh probe.
func FreshProbeFromLedger(account, regDir string, now time.Time, freshMin float64) *FreshProbe {
	if freshMin <= 0 {
		freshMin = ProbeLedgerFreshMin()
	}
	entry, ok := accountprobe.LastProbeByAccount(regDir)[account]
	if !ok {
		return nil
	}
	age := accountprobe.RecentProbeAgeMin(account, regDir, now)
	if age == nil || *age > freshMin {
		return nil
	}
	status := strings.ToUpper(strings.TrimSpace(entry.Status))
	if status == "OK" {
		return &FreshProbe{Available: true, AgeMin: *age}
	}
	kind, ok := probeStatusKind[status]
	if !ok {
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
