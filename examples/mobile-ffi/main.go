package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/pkg/abi"
)

// main is the pure-Go witness of the FFI core: it drives the SAME deny/allow
// round-trip the C shim carries to Android/iOS, printing each decision and
// exiting NON-ZERO if any leg is wrong. `go run .` in this module is therefore a
// witness that the floor is reachable and correct, not merely that it compiles —
// the mirror of ../extdriver. It needs no cgo, so it runs on any host (this
// repo's CI is CGO_ENABLED=0).
func main() {
	failed := false
	fail := func(format string, a ...any) {
		failed = true
		fmt.Fprintf(os.Stderr, "FAIL: "+format+"\n", a...)
	}

	fmt.Println("fak mobile FFI — on-device tool calls routed through the adjudicator floor")
	fmt.Printf("  ABI version: v%d.%d\n\n", abi.ABIMajor, abi.ABIMinor)

	// [1] A dangerous proposed call is DENIED at the floor — it never reaches the
	// Android Intent / Apple App Intent dispatcher.
	danger := `{"tool":"send_sms","args":{"to":"+1900PREMIUM","body":"drain"}}`
	d := Decide(danger)
	printDecision("[1] dangerous", d)
	if d.Allow || d.Verdict != "DENY" || d.Reason != "POLICY_BLOCK" {
		fail("send_sms should DENY(POLICY_BLOCK), got %+v", d)
	}

	// [2] A benign read-shaped call is CONTINUED — dispatch proceeds.
	benign := `{"tool":"get_battery_level","args":{}}`
	d = Decide(benign)
	printDecision("[2] benign", d)
	if !d.Allow || d.Verdict != "ALLOW" {
		fail("get_battery_level should ALLOW, got %+v", d)
	}

	// [3] An unrecognized call FAILS CLOSED — default-deny, not silent dispatch.
	unknown := `{"tool":"transfer_funds","args":{"amount":9999}}`
	d = Decide(unknown)
	printDecision("[3] unknown ", d)
	if d.Allow || d.Verdict != "DEFAULT_DENY" {
		fail("transfer_funds should DEFAULT_DENY, got %+v", d)
	}

	if failed {
		fmt.Fprintln(os.Stderr, "\nmobile-ffi: ROUND-TRIP FAILED")
		os.Exit(1)
	}
	fmt.Println("\nmobile-ffi: OK — dangerous denied, benign continued, unknown failed closed")
}

func printDecision(label string, d Decision) {
	b, _ := json.Marshal(d)
	fmt.Printf("%s %s\n", label, string(b))
}
