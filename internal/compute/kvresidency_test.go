package compute

import "testing"

// kvCfg768 is a small KV geometry whose per-token cost is exactly 768 bytes:
// 2 layers * 4 kv heads * 8 dims * 3 rows (Kraw,K,V) * 4-byte f32 = 768.
var kvCfg768 = KVConfig{NumLayers: 2, NumKVHeads: 4, HeadDim: 8}

func TestPlanKVResidencyKeepsLargerContextThanDeviceOnly(t *testing.T) {
	const perToken = 768
	if got := EstimateKVStoreBytes(kvCfg768, 1); got != perToken {
		t.Fatalf("per-token KV bytes = %d, want %d", got, perToken)
	}
	// Device pool is tight: room for only 100 tokens of KV. Host pool is roomy.
	deviceBudget := int64(perToken * 100)
	hostBudget := int64(perToken * 5000)
	const want = 1000

	// The device-only baseline is what auto-sizing-that-only-shrinks would settle on: the
	// most tokens the tight device pool alone can hold.
	deviceOnly := tokensWithinBudget(perToken, deviceBudget, want)
	if deviceOnly != 100 {
		t.Fatalf("device-only sizing = %d tokens, want 100", deviceOnly)
	}

	split := PlanKVResidency(kvCfg768, want, deviceBudget, hostBudget)
	if split.HotTokens != 100 {
		t.Fatalf("hot tokens = %d, want 100 (fills the tight device pool first)", split.HotTokens)
	}
	if split.ColdTokens != 900 {
		t.Fatalf("cold tokens = %d, want 900 (the overflow spills to the roomy host pool)", split.ColdTokens)
	}
	if split.Tokens() != want {
		t.Fatalf("effective context = %d, want %d", split.Tokens(), want)
	}
	if !split.Spilled() {
		t.Fatal("split should report a spill — that is the context the device alone could not hold")
	}
	// The acceptance: a device-tight / host-roomy config keeps a LARGER effective context
	// than the device-only sizing would allow.
	if split.Tokens() <= deviceOnly {
		t.Fatalf("tiered residency (%d tokens) must exceed device-only sizing (%d)", split.Tokens(), deviceOnly)
	}

	// ...and no OOM: the rendered plan must fit BOTH pools (hot on device, cold on host).
	plan := KVResidencyMemoryPlan(split)
	if got, want := plan.DeviceTotal(), int64(perToken*100); got != want {
		t.Fatalf("plan device (hot) bytes = %d, want %d", got, want)
	}
	if got, want := plan.HostTotal(), int64(perToken*900); got != want {
		t.Fatalf("plan host (cold) bytes = %d, want %d", got, want)
	}
	dev := capDevice{
		total: deviceBudget, free: deviceBudget, known: true,
		hostTotal: hostBudget, hostFree: hostBudget, hostKnown: true, hostProbe: true,
	}
	if err := RefuseMemoryPlanIfTooBig(dev, plan, 0); err != nil {
		t.Fatalf("the spilled plan must fit both pools without OOM, got refusal: %v", err)
	}
}

func TestPlanKVResidencyNeverShrinksBelowDeviceOnly(t *testing.T) {
	const perToken = 768
	deviceBudget := int64(perToken * 100)
	const want = 1000

	// No roomy pool at all (hostBudget = 0): the split must degenerate to device-only sizing,
	// never worse — tiered residency only ever ADDS context.
	noHost := PlanKVResidency(kvCfg768, want, deviceBudget, 0)
	if noHost.ColdTokens != 0 || noHost.Spilled() {
		t.Fatalf("a zero roomy budget must spill nothing, got %+v", noHost)
	}
	if noHost.Tokens() != 100 {
		t.Fatalf("with no host pool the effective context = %d, want the device-only 100", noHost.Tokens())
	}

	// A small-but-nonzero roomy pool adds only what fits, still >= device-only.
	smallHost := PlanKVResidency(kvCfg768, want, deviceBudget, int64(perToken*50))
	if smallHost.ColdTokens != 50 {
		t.Fatalf("small host pool cold tokens = %d, want 50", smallHost.ColdTokens)
	}
	if smallHost.Tokens() != 150 {
		t.Fatalf("effective context = %d, want 150 (100 hot + 50 cold)", smallHost.Tokens())
	}
	if smallHost.Tokens() < noHost.Tokens() {
		t.Fatalf("adding a roomy pool must never shrink context: %d < %d", smallHost.Tokens(), noHost.Tokens())
	}
}

func TestPlanKVResidencyCapsAtWantAndFitsDeviceWhenRoomy(t *testing.T) {
	const perToken = 768
	// Both pools roomy and the whole window fits the device: nothing spills, capped at want.
	split := PlanKVResidency(kvCfg768, 200, int64(perToken*1000), int64(perToken*1000))
	if split.HotTokens != 200 || split.ColdTokens != 0 {
		t.Fatalf("a window that fits the device must stay fully hot, got %+v", split)
	}
	if split.Tokens() != 200 {
		t.Fatalf("effective context = %d, want the requested 200", split.Tokens())
	}
}

func TestPlanKVResidencyFailsOpenOnIncompleteGeometryOrWant(t *testing.T) {
	// Incomplete KV geometry (per-token cost 0) yields an empty split — fail open, parity
	// with EstimateKVStoreMemoryPlan.
	if got := PlanKVResidency(KVConfig{NumLayers: 2, HeadDim: 8}, 1000, 1<<40, 1<<40); got != (KVResidencySplit{}) {
		t.Fatalf("incomplete geometry should yield an empty split, got %+v", got)
	}
	if got := PlanKVResidency(kvCfg768, 0, 1<<40, 1<<40); got != (KVResidencySplit{}) {
		t.Fatalf("non-positive want should yield an empty split, got %+v", got)
	}
	if got := KVResidencyMemoryPlan(KVResidencySplit{}); got != nil {
		t.Fatalf("an empty split should render a nil plan, got %+v", got)
	}
}
