package main

import (
	"os"
	"strings"
	"testing"
)

// TestUpstreamErrorKindsIsBridgedToTheDurableRow is the #5487 bridge witness, and it exists
// for the same reason TestEverySelfHostedSplitFieldIsBridgedToTheDurableRow does: the
// gateway computes the failure KIND, the durable row now has a field for it, and the two are
// joined by one hand-written struct literal in gatewayUsageCounters. The field is omitempty,
// so forgetting the line here fails SILENTLY — every row would omit the key and every reader
// would go on seeing NOT INSTRUMENTED, which is indistinguishable from the bug the ticket
// reports. A compiler cannot catch an unwritten map, so the bridge gets a witness.
//
// It also pins the SOURCE of the value. The gateway exposes three readers of the same
// counter map and only one of them carries every kind: RotationEvidenceSnapshot keeps just
// auth/rate_limited and TransientWireErrorSnapshot just the transport scalar, so bridging
// from either would drop "stalled" again while looking correct.
func TestUpstreamErrorKindsIsBridgedToTheDurableRow(t *testing.T) {
	src, err := os.ReadFile("serve.go")
	if err != nil {
		t.Fatalf("read serve.go: %v", err)
	}
	body, ok := funcBodyText(string(src), "func gatewayUsageCounters(")
	if !ok {
		t.Fatal("gatewayUsageCounters not found in serve.go — if it moved, move this witness with it")
	}
	if !strings.Contains(body, "UpstreamErrorKinds:") {
		t.Fatal("Counters.UpstreamErrorKinds is never assigned in gatewayUsageCounters — the gateway " +
			"classifies the upstream failure kind and the durable row drops it, so a stall stays " +
			"unattributable after the process exits (#5487)")
	}
	if !strings.Contains(body, "srv.UpstreamErrorKindsSnapshot()") {
		t.Fatal("UpstreamErrorKinds is not sourced from srv.UpstreamErrorKindsSnapshot() — the narrower " +
			"RotationEvidenceSnapshot/TransientWireErrorSnapshot accessors omit the stalled kind")
	}
}
