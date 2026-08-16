package dispatchtick

import (
	"reflect"
	"strings"
	"testing"
)

func wipBaseResult() PreflightResult {
	return PreflightResult{OK: true, Verdict: PreflightOKVerdict, Cap: 8, Live: 2, Headroom: 6, CapTerms: CapTerms{EffectiveCap: 8}}
}

func TestApplyWIPLimitBelowCapLowersWave(t *testing.T) {
	got := ApplyWIPLimit(wipBaseResult(), WIPCensus{Measured: true, Started: 3, Inventory: 99, Limit: 5})
	if !got.OK || got.Cap != 4 || got.Headroom != 2 || got.CapTerms.Limiting != WIPLimiting {
		t.Fatalf("result = %+v", got)
	}
}

func TestApplyWIPLimitAtCapRefusesWithoutChargingInventory(t *testing.T) {
	got := ApplyWIPLimit(wipBaseResult(), WIPCensus{Measured: true, Started: 5, Inventory: 99, Limit: 5})
	if got.OK || got.Verdict != PreflightRefuseWIPLimit || got.Cap != got.Live || got.Headroom != 0 {
		t.Fatalf("result = %+v", got)
	}
	if !strings.Contains(got.Reason, "99 unstarted") || !strings.Contains(got.Reason, "not WIP") {
		t.Fatalf("reason does not preserve inventory distinction: %q", got.Reason)
	}
}

func TestApplyWIPLimitDisabledIsIdentity(t *testing.T) {
	base := wipBaseResult()
	for _, c := range []WIPCensus{{}, {Started: 8, Limit: 2}, {Measured: true, Started: 8}} {
		got := ApplyWIPLimit(base, c)
		if !reflect.DeepEqual(got, base) {
			t.Fatalf("census %+v changed result: %+v", c, got)
		}
	}
}

func TestApplyWIPLimitPreservesEarlierRefusal(t *testing.T) {
	base := wipBaseResult()
	base.OK = false
	base.Verdict = PreflightRefuseNoAccount
	base.Reason = "account unavailable"
	got := ApplyWIPLimit(base, WIPCensus{Measured: true, Started: 8, Limit: 2})
	if !reflect.DeepEqual(got, base) {
		t.Fatalf("earlier refusal changed: %+v", got)
	}
}
