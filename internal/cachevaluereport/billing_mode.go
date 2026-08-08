package cachevaluereport

import "strings"

// Billing mode (#3664): WHICH seat paid for the tokens a Track-2 savings row prices.
//
// Every Track-2 dollar figure is priced at published list $/MTok with no billing input at
// all, so the blended fleet reduction is an API-key-EQUIVALENT projection over the whole
// corpus. On a per-token-billed API-key seat that projection lands on a real invoice line;
// on a flat-rate Pro/Max subscription seat the identical arithmetic is NOTIONAL — the
// tokens were metered, but the marginal price that seat actually paid for them is flat.
// Blending both into one headline turns "real dollars even on API-key billing" into a
// corpus projection rather than a per-seat figure.
//
// The producer stamps the resolved posture on the row at write time and the fleet fold
// partitions the same dollars into the two columns. Only api_key is real-dollar; oauth AND
// unknown are notional, because a row that cannot PROVE its seat must never be counted as
// real money — which is also what every row written before this field existed reads as.
const (
	BillingModeAPIKey  = "api_key"
	BillingModeOAuth   = "oauth"
	BillingModeUnknown = "unknown"
)

// NormalizeBillingMode maps a raw stamped posture onto the closed set above. Anything the
// producer left blank, or stamped with a token this reader does not know, resolves to
// unknown — never to api_key. That fail-safe direction is the whole point: the entire
// ledger written before #3664 carries no billing mode at all and must fold notional.
func NormalizeBillingMode(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case BillingModeAPIKey:
		return BillingModeAPIKey
	case BillingModeOAuth:
		return BillingModeOAuth
	default:
		return BillingModeUnknown
	}
}

// RealDollarBillingMode reports whether a row's stamped posture proves the tokens were
// billed per token, i.e. whether its OBSERVED projection is a real-dollar one.
//
// It does NOT upgrade the fence: BOTH columns stay OBSERVED, list-priced from published
// rates and never reconciled against an invoice. This only decides which of the two a
// row lands in, so the API-key-vs-OAuth attribution is auditable from the ledger instead
// of asserted over the blend.
func RealDollarBillingMode(raw string) bool {
	return NormalizeBillingMode(raw) == BillingModeAPIKey
}
