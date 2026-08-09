package main

// reason.go — bind this leaf's artifact refusals to the closed refusal vocabulary
// (dos.toml [reasons.MICROCONTEXT_LEDGER_REFUSED], declared by #5841).
//
// The floor already refused correctly: every -verify-* path exits non-zero when a witness,
// quality ledger, cache A/B, or health scorecard fails schema or accounting reconciliation.
// What it never did was NAME the refusal. dos.toml declared the reason and pointed see_also at
// cmd/microcontextdemo, so the table read as total while no code path could produce the code —
// a consumer routing on reason codes waited for a token that never arrived (the same drift
// #5608 caught on ISSUEFANOUT_CONTRACT_REFUSED, and the drift
// internal/architest.TestEveryDeclaredReasonHasAnEmitter now fails on).
//
// This is not a new enforcement path — #5841 put those out of scope. It is the existing
// refusal, spoken in the vocabulary that was declared for it.

import (
	"fmt"
	"os"
)

// LedgerRefusedReason is the closed-vocabulary code every -verify-* refusal in this leaf
// carries. It is the token dos.toml declares and `dos check-reason` resolves.
const LedgerRefusedReason = "MICROCONTEXT_LEDGER_REFUSED"

// ledgerRefusal renders one machine-routable refusal line. The reason code leads so a consumer
// can route on the first colon-delimited field without parsing prose; the failing flag and path
// say which artifact to regenerate; the verifier's own message rides at the tail unchanged, so
// existing prose witnesses keep matching.
func ledgerRefusal(flagName, path string, err error) string {
	return fmt.Sprintf("%s: -%s %s: %v", LedgerRefusedReason, flagName, path, err)
}

// runVerify is the single door every -verify-* flag leaves through. Routing all of them here is
// the point: a verify path added later cannot refuse anonymously, because refusing at all means
// calling this and naming the reason.
func runVerify(flagName, path string, verify func(string) error) {
	if err := verify(path); err != nil {
		fmt.Fprintln(os.Stderr, ledgerRefusal(flagName, path, err))
		os.Exit(1)
	}
	fmt.Println("PASS: verified", path)
}
