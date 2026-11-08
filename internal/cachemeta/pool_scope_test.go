package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=1ms lane=default
func TestPoolReuseClosedScope(t *testing.T) {
	// ShareScope is a closed uint8 enum. Trusted bytes and a matching key
	// cannot authorize sharing under an unknown scope value.
	for raw := 0; raw <= 255; raw++ {
		scope := abi.ShareScope(raw)
		stored := poolKVEntry("qwen3", abi.TaintTrusted, scope)
		verdict := PoolReuseVerdict(stored, wantKey("qwen3"))
		knownShared := scope == abi.ScopeFleet || scope == abi.ScopeTenant
		if knownShared {
			if !verdict.CanServe() {
				t.Fatalf("known scope %d refused: %+v", raw, verdict)
			}
		} else if verdict.CanServe() || verdict.Reason != ReasonScopeDenied {
			t.Fatalf("private/unknown scope %d not refused: %+v", raw, verdict)
		}
	}
	stored := poolKVEntry("qwen3", abi.TaintQuarantined, abi.ShareScope(255))
	if verdict := PoolReuseVerdict(stored, wantKey("qwen3")); verdict.CanServe() || verdict.Reason != ReasonTaintDenied {
		t.Fatalf("unknown scope changed quarantine precedence: %+v", verdict)
	}
}
