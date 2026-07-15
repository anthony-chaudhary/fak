package adjudicator

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestPolicySnapshotIsIsolated(t *testing.T) {
	a := New(Policy{
		Allow:           map[string]bool{"read": true},
		Deny:            map[string]abi.ReasonCode{"write": abi.ReasonPolicyBlock},
		AllowPrefix:     []string{"get_"},
		SelfModifyGlobs: []string{"config/**"},
		AdvisoryReasons: map[abi.ReasonCode]bool{abi.ReasonDefaultDeny: true},
	})
	got := a.PolicySnapshot()
	got.Allow["read"] = false
	got.Deny["write"] = abi.ReasonDefaultDeny
	got.AllowPrefix[0] = "mutated_"
	got.SelfModifyGlobs[0] = "mutated/**"
	got.AdvisoryReasons[abi.ReasonDefaultDeny] = false

	again := a.PolicySnapshot()
	if !again.Allow["read"] || again.Deny["write"] != abi.ReasonPolicyBlock || again.AllowPrefix[0] != "get_" || again.SelfModifyGlobs[0] != "config/**" || !again.AdvisoryReasons[abi.ReasonDefaultDeny] {
		t.Fatalf("snapshot mutation escaped into live policy: %+v", again)
	}
}
