package main

// audit_usage_selfhosted.go — the one line of `fak audit usage` that answers "what
// fraction of the tokens we served did we generate on our own hardware?"
//
// Everything hard about rendering it is the refusal case. Right now every ledger on
// every box is full of rows written before the split was recorded, so the honest
// output is "not instrumented" — and this file exists largely to make sure that
// sentence never quietly becomes "0.0% self-hosted", which is the same characters a
// reader would see if we had measured everything and bought every token.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/auditusage"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// selfHostedLine renders the self-hosted share for the text report, or "" when
// there is nothing to say (no gateway rows folded at all).
//
// The share is NEVER printed without the coverage it was computed over. A share
// over a sliver of the volume is a sample, and an unqualified percentage is the
// form in which a sample gets quoted as a fleet number.
func selfHostedLine(sh *auditusage.SelfHostedRollup) string {
	if sh == nil {
		return ""
	}
	switch {
	case sh.OutputShare != nil:
		return fmt.Sprintf("%.1f%% of classified output generated in-kernel (%d/%d tokens) — %.1f%% of served output classified, %d local / %d vendor turns",
			*sh.OutputShare*100,
			sh.SelfHostedOutputTokens, sh.SelfHostedOutputTokens+sh.VendorOutputTokens,
			sh.ClassifiedOutputFraction*100,
			sh.SelfHostedTurns, sh.VendorTurns)
	case sh.Reason == string(gatewayusageledger.ShareNotInstrumented):
		// The distinction the whole split exists to preserve, spelled out where an
		// operator reads it, because "0%" here would be a claim nobody measured.
		return fmt.Sprintf("NOT INSTRUMENTED — no row among %d recorded which side served it (%d output tokens unattributed; this is not a measured zero)",
			sh.Rows, sh.OutputTokens)
	case sh.Reason == string(gatewayusageledger.ShareNoClassifiedOutput):
		return fmt.Sprintf("undefined — %d turns classified across %d rows, none generated output tokens",
			sh.SelfHostedTurns+sh.VendorTurns, sh.Rows)
	default:
		// A reason the fold grew and this renderer has not been taught. Print it
		// rather than swallowing it into a blank or a fabricated percentage.
		return fmt.Sprintf("unavailable (%s) — %d rows", sh.Reason, sh.Rows)
	}
}
