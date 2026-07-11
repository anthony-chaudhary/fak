package fleetaccounts

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestFreshProbeSurfacesActiveDailyCapAsUsageSoon is the day26 near-cap fix: a fresh OK probe
// correctly reopens a seat whose carried DAILY usage cap is still counting down, but the reopen
// must not SILENTLY drop the cap -- an operator seeing a plain "serving" row with no usage could
// not tell the seat was one request from the wall. The seat stays offered (available, not
// throttled) and carries the still-future reset as an advisory usage_soon_reset.
func TestFreshProbeSurfacesActiveDailyCapAsUsageSoon(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	reset := futureResetStr(30 * time.Minute)
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{"reset": reset},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)

	// Availability is unchanged by the advisory: the seat is still offered.
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("near-cap seat must stay offered, got %+v", st)
	}
	if !st.hasUsageSoon || st.UsageSoonReset != reset {
		t.Fatalf("usage_soon_reset = %q (has=%v), want %q", st.UsageSoonReset, st.hasUsageSoon, reset)
	}

	// applyStatus + MarshalJSON: the key rides last in the runtime-status block, present only
	// because this row carries a live daily cap.
	var acc Account
	applyStatus(&acc, st)
	if acc.UsageSoonReset == nil || *acc.UsageSoonReset != reset {
		t.Fatalf("Account.UsageSoonReset = %v, want %q", acc.UsageSoonReset, reset)
	}
	data, err := json.Marshal(acc)
	if err != nil {
		t.Fatal(err)
	}
	keys := topLevelKeysInOrder(t, data)
	if last := keys[len(keys)-1]; last != "usage_soon_reset" {
		t.Fatalf("last JSON key = %q, want usage_soon_reset (keys=%v)", last, keys)
	}
	if !strings.Contains(string(data), `"usage_soon_reset":"`+reset+`"`) {
		t.Fatalf("marshaled row missing usage_soon_reset: %s", data)
	}
}

// TestNormalServingRowOmitsUsageSoon proves the additive contract: a serving seat with no
// carried cap gains no key, so every existing row's byte-parity JSON is unchanged.
func TestNormalServingRowOmitsUsageSoon(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	st := computeRuntimeStatus(".claude-a", "", Registry{})
	if st.hasUsageSoon || st.UsageSoonReset != "" {
		t.Fatalf("no carried cap -> no advisory, got %+v", st)
	}
	var acc Account
	applyStatus(&acc, st)
	if acc.UsageSoonReset != nil {
		t.Fatalf("Account.UsageSoonReset = %v, want nil", acc.UsageSoonReset)
	}
	data, err := json.Marshal(acc)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "usage_soon_reset") {
		t.Fatalf("serving row must not emit usage_soon_reset: %s", data)
	}
}

// TestActiveWeeklyCapIsWalledNotUsageSoon guards the boundary: a still-active WEEKLY cap holds
// the seat CLOSED (blocked/throttled) rather than reopening it as a near-cap advisory. The
// advisory is strictly for daily caps the seat is being served through.
func TestActiveWeeklyCapIsWalledNotUsageSoon(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{
			"reset":  futureResetStr(4 * time.Hour),
			"weekly": futureResetStr(72 * time.Hour),
		},
	}}
	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("active weekly cap must wall the seat, got %+v", st)
	}
	if st.hasUsageSoon || st.UsageSoonReset != "" {
		t.Fatalf("walled seat must not carry a usage_soon advisory, got %+v", st)
	}
}
