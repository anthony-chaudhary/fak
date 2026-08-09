package main

import (
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

// The billing seat this process is running on (#3664), stamped onto every Track-2 savings
// row it writes so the fleet report can split its OBSERVED reduction into a REAL-dollar
// column and a NOTIONAL one.
//
// Track-2 rows are priced at published list $/MTok with no billing input whatsoever
// (cachevalueSavingsPricing). That projection lands on a real invoice line only when the
// session bills per token; on a flat-rate Pro/Max subscription the same arithmetic is
// notional. Until now the ledger recorded no way to tell the two apart, so the headline
// reduction was an API-key-EQUIVALENT projection over a mixed corpus.
//
// The class is derived from exactly the credential fields resolveGuardManagedCache reads,
// so the ledger and the managed-cache banner can never disagree about which seat is which.
// It is deliberately NOT the resolved active/passive bit: `--managed-cache on` forces
// ACTIVE on a passthrough credential whose billing fak cannot see, and stamping that as
// real-dollar would re-introduce the very overstatement this splits apart.
var (
	billingModeMu    sync.RWMutex
	billingModeStamp string
)

// billingModeFrom classifies the resolved upstream credential into the seat posture the
// savings ledger records. A pinned subscription OAuth token is flat-rate (notional); an
// api key with no subscription token pinned is per-token billing (real dollars); anything
// else — plain passthrough, an empty credential, the pure local in-kernel branch — is
// UNKNOWN and folds notional, because a seat fak cannot prove must never be counted as
// real money.
func billingModeFrom(in guardManagedCacheInputs) string {
	switch {
	case strings.TrimSpace(in.oauthSource) != "":
		return cachevaluereport.BillingModeOAuth
	case strings.TrimSpace(in.apiKey) != "":
		return cachevaluereport.BillingModeAPIKey
	default:
		return cachevaluereport.BillingModeUnknown
	}
}

// recordBillingMode publishes the resolved seat for the rest of this process, so the exit
// path that writes the Track-2 rows can stamp it without threading a credential class
// through every supervision frame between startup and teardown.
func recordBillingMode(mode string) {
	billingModeMu.Lock()
	defer billingModeMu.Unlock()
	billingModeStamp = strings.TrimSpace(mode)
}

// resolvedBillingMode reports the seat recorded for this process, or "" when nothing
// resolved one — a front door that never classifies its credential (`fak serve`, a
// fixture, a test) writes rows with NO billing_mode at all, exactly as every row written
// before #3664 did, and the fleet fold reads that absence as notional.
func resolvedBillingMode() string {
	billingModeMu.RLock()
	defer billingModeMu.RUnlock()
	return billingModeStamp
}
