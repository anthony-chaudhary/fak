package ggufload

import (
	"bytes"
	"io"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const ramTierGiB = int64(1) << 30

func ramTierHost(total, avail int64) func() (int64, int64, bool) {
	return func() (int64, int64, bool) { return total, avail, true }
}

func ramTierEnv(kv map[string]string) func(string) (string, bool) {
	return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
}

// ramTierSmall plans Inkling-2-Small (48 layers, ~182 GiB of experts in a
// ~190.9 GiB UD-Q2_K_XL) on a 16 GiB card, returning the device plan.
func ramTierSmall(t *testing.T) (order []int, resident []bool, layerBytes []int64, p Inkling2DevicePlacement) {
	t.Helper()
	const layers = 48
	layerBytes = make([]int64, layers)
	for i := range layerBytes {
		layerBytes[i] = 182 * ramTierGiB / layers
	}
	order = inkling2ExpertResidencyOrder(model.Inkling2LoopHParams{Layers: layers, Passes: 1}, layers)
	resident, p, err := inkling2PlanExpertResidency(order, layerBytes, 6*ramTierGiB, 14*ramTierGiB)
	if err != nil {
		t.Fatal(err)
	}
	if p.FullyResident() || p.ResidentLayers == 0 {
		t.Fatalf("Small on 16 GiB should spill some but not all experts: %s", p)
	}
	return order, resident, layerBytes, p
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2RAMTierLargeHostHoldsAllColdExperts(t *testing.T) {
	order, resident, layerBytes, p := ramTierSmall(t)
	info, err := inkling2RAMBudget(ramTierHost(256*ramTierGiB, 250*ramTierGiB), ramTierEnv(nil))
	if err != nil {
		t.Fatal(err)
	}
	if want := 250*ramTierGiB - 256*ramTierGiB/10; info.Budget != want {
		t.Fatalf("budget = %d, want available minus 10%% of total = %d", info.Budget, want)
	}
	inRAM := inkling2PlanRAMTier(order, resident, layerBytes, info, &p)
	if p.NVMeBytes != 0 || p.NVMeLayers != 0 || p.RAMBytes != p.ColdBytes || p.RAMLayers != p.ColdLayers || p.RAMNeededForAllExpertsBytes != 0 {
		t.Fatalf("256 GiB host: %+v", p)
	}
	for i := range inRAM {
		if inRAM[i] == resident[i] {
			t.Fatalf("layer %d: resident %v, ram %v; every layer must be in exactly one tier", i, resident[i], inRAM[i])
		}
	}
	if p.RAMAvailableBytes != 250*ramTierGiB || !strings.Contains(p.String(), "no NVMe streaming") {
		t.Fatalf("summary: %+v %q", p, p.String())
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2RAMTierSmallHostSplitsAndReportsNeed(t *testing.T) {
	order, resident, layerBytes, p := ramTierSmall(t)
	total, avail := 30*ramTierGiB, 28*ramTierGiB
	info, err := inkling2RAMBudget(ramTierHost(total, avail), ramTierEnv(nil))
	if err != nil {
		t.Fatal(err)
	}
	if info.Budget != avail-inkling2RAMHeadroomMin {
		t.Fatalf("budget = %d, want available minus the 4 GiB floor", info.Budget)
	}
	inRAM := inkling2PlanRAMTier(order, resident, layerBytes, info, &p)
	if p.RAMLayers == 0 || p.NVMeLayers == 0 || p.RAMBytes > info.Budget || p.RAMBytes+p.NVMeBytes != p.ColdBytes {
		t.Fatalf("30 GiB host should split RAM/NVMe: %+v", p)
	}
	// RAM fills in residency order: the first cold layer in order is in RAM.
	for _, i := range order {
		if !resident[i] {
			if !inRAM[i] {
				t.Fatalf("first cold layer %d in residency order not in RAM", i)
			}
			break
		}
	}
	// Adding exactly the reported RAM makes the default budget cover every
	// cold expert; materially less does not.
	need := p.RAMNeededForAllExpertsBytes
	cold := p.RAMBytes + p.NVMeBytes
	grown, _ := inkling2RAMBudget(ramTierHost(total+need, avail+need), ramTierEnv(nil))
	short, _ := inkling2RAMBudget(ramTierHost(total+need-ramTierGiB, avail+need-ramTierGiB), ramTierEnv(nil))
	if grown.Budget < cold || short.Budget >= cold {
		t.Fatalf("need %d: grown budget %d, short budget %d, cold %d", need, grown.Budget, short.Budget, cold)
	}
	if s := p.String(); !strings.Contains(s, "GiB RAM would remove NVMe streaming") || !strings.HasPrefix(s, "inkling2 placement: vram ") {
		t.Fatalf("summary %q", s)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2RAMTierEnvOverride(t *testing.T) {
	info, err := inkling2RAMBudget(ramTierHost(256*ramTierGiB, 250*ramTierGiB), ramTierEnv(map[string]string{inkling2RAMBudgetEnv: " 5368709120 "}))
	if err != nil || info.Budget != 5*ramTierGiB || !info.Override || info.Available != 250*ramTierGiB {
		t.Fatalf("override: %+v %v", info, err)
	}
	// Unprobed host: the override still applies; without it the tier is empty.
	info, err = inkling2RAMBudget(nil, ramTierEnv(map[string]string{inkling2RAMBudgetEnv: "7"}))
	if err != nil || info.Budget != 7 || info.Known {
		t.Fatalf("override without probe: %+v %v", info, err)
	}
	if info, _ = inkling2RAMBudget(nil, ramTierEnv(nil)); info.Budget != 0 {
		t.Fatalf("unprobed default budget = %d, want 0", info.Budget)
	}
	for _, bad := range []string{"-1", "lots"} {
		if _, err := inkling2RAMBudget(nil, ramTierEnv(map[string]string{inkling2RAMBudgetEnv: bad})); err == nil {
			t.Fatalf("%q accepted", bad)
		}
	}
	// Override shortfall is reported as plain extra budget.
	order, resident, layerBytes, p := ramTierSmall(t)
	info, _ = inkling2RAMBudget(nil, ramTierEnv(map[string]string{inkling2RAMBudgetEnv: strconv.FormatInt(10*ramTierGiB, 10)}))
	inkling2PlanRAMTier(order, resident, layerBytes, info, &p)
	if p.RAMBytes > 10*ramTierGiB || p.RAMNeededForAllExpertsBytes != p.ColdBytes-10*ramTierGiB || p.RAMAvailableBytes != -1 {
		t.Fatalf("override plan %+v", p)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2RAMTierHeadroomNeverNegative(t *testing.T) {
	for _, h := range [][2]int64{{2 * ramTierGiB, ramTierGiB}, {8 * ramTierGiB, 0}, {1024 * ramTierGiB, 50 * ramTierGiB}, {0, 3 * ramTierGiB}} {
		info, err := inkling2RAMBudget(ramTierHost(h[0], h[1]), ramTierEnv(nil))
		if err != nil || info.Budget < 0 {
			t.Fatalf("host %v: %+v %v", h, info, err)
		}
	}
	info, _ := inkling2RAMBudget(ramTierHost(2*ramTierGiB, ramTierGiB), ramTierEnv(nil))
	var p Inkling2DevicePlacement
	inRAM := inkling2PlanRAMTier([]int{0, 1}, []bool{false, false}, []int64{ramTierGiB, ramTierGiB}, info, &p)
	if inRAM[0] || inRAM[1] || p.RAMBytes != 0 || p.NVMeBytes != 2*ramTierGiB || p.RAMNeededForAllExpertsBytes <= 0 {
		t.Fatalf("starved host: %+v", p)
	}
	// Unknown free reading is not a probe.
	info, _ = inkling2RAMBudget(func() (int64, int64, bool) { return 64 * ramTierGiB, compute.FreeUnknown, true }, ramTierEnv(nil))
	if info.Known || info.Budget != 0 {
		t.Fatalf("FreeUnknown: %+v", info)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2RAMTierMiniStaysVRAMResident(t *testing.T) {
	// Inkling-2-Mini UD-Q3_K_XL on a 16 GiB card (15.3 GiB free).
	const fileBytes, expertBytes = int64(13_630_000_000), int64(12_200_000_000)
	hp := model.Inkling2LoopHParams{Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3}
	layerBytes := make([]int64, 24)
	for i := range layerBytes {
		layerBytes[i] = expertBytes / 24
	}
	be := &inkling2BudgetBackend{Backend: compute.Default(), free: 15*ramTierGiB + 3*ramTierGiB/10}
	budget, err := inkling2DeviceBudget(be, ramTierEnv(nil))
	if err != nil {
		t.Fatal(err)
	}
	order := inkling2ExpertResidencyOrder(hp, 24)
	resident, p, err := inkling2PlanExpertResidency(order, layerBytes, fileBytes-expertBytes, budget)
	if err != nil {
		t.Fatal(err)
	}
	info, _ := inkling2RAMBudget(ramTierHost(32*ramTierGiB, 20*ramTierGiB), ramTierEnv(nil))
	inkling2PlanRAMTier(order, resident, layerBytes, info, &p)
	if !p.FullyResident() || p.RAMBytes != 0 || p.NVMeBytes != 0 || p.RAMNeededForAllExpertsBytes != 0 || p.VRAMBytes() != fileBytes-expertBytes+24*(expertBytes/24) {
		t.Fatalf("Mini on 16 GiB: %+v", p)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2RAMTierBindPrefetchesColdAndStopsOnClose(t *testing.T) {
	m, dense, layer := inkling2BudgetModel(t)
	host := inkling2HostExpertsBytes(m.Weights.Layers[0].GateUp) + inkling2HostExpertsBytes(m.Weights.Layers[0].Down)
	t.Setenv(inkling2VRAMBudgetEnv, strconv.FormatInt(dense+2*layer, 10))
	t.Setenv(inkling2RAMBudgetEnv, strconv.FormatInt(host, 10))
	if err := bindInkling2Device(m, &inkling2BudgetBackend{Backend: compute.Default()}); err != nil {
		t.Fatal(err)
	}
	p, _ := Inkling2DevicePlacementOf(m)
	// Residency order 1, 2, 0, 3: layer 0 is the first cold layer, so it takes the RAM tier.
	if p.ColdLayers != 2 || p.RAMLayers != 1 || p.NVMeLayers != 1 || p.RAMBytes != host || p.NVMeBytes != host || p.RAMBudgetBytes != host {
		t.Fatalf("placement %+v", p)
	}
	s, ok := m.closer.(*inkling2DeviceSet)
	if !ok {
		t.Fatalf("closer %T", m.closer)
	}
	pf, ok := s.next.(*inkling2RAMPrefetch)
	if !ok {
		t.Fatalf("mapping closer %T, want the RAM prefetch chained in front", s.next)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-pf.done:
	default:
		t.Fatal("Close returned before the prefetch stopped")
	}

	// A stopped prefetch touches nothing; a running one touches every byte once.
	region := make([]byte, 3<<20+5)
	stopped := &inkling2RAMPrefetch{done: make(chan struct{})}
	stopped.stop.Store(true)
	stopped.run([][]byte{region})
	full := &inkling2RAMPrefetch{done: make(chan struct{})}
	full.run([][]byte{region, region[:7]})
	if stopped.touched.Load() != 0 || full.touched.Load() != int64(len(region)+7) {
		t.Fatalf("touched: stopped %d, full %d", stopped.touched.Load(), full.touched.Load())
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2RAMTierBindPrefetchStartsAfterUploads(t *testing.T) {
	m, dense, layer := inkling2BudgetModel(t)
	host := inkling2HostExpertsBytes(m.Weights.Layers[0].GateUp) + inkling2HostExpertsBytes(m.Weights.Layers[0].Down)
	t.Setenv(inkling2VRAMBudgetEnv, strconv.FormatInt(dense+2*layer, 10))
	t.Setenv(inkling2RAMBudgetEnv, strconv.FormatInt(host, 10))
	be := &inkling2BudgetBackend{Backend: compute.Default()}
	startedAt, starts := -1, 0
	orig := inkling2StartRAMPrefetch
	t.Cleanup(func() { inkling2StartRAMPrefetch = orig })
	inkling2StartRAMPrefetch = func(regions [][]byte, next io.Closer) *inkling2RAMPrefetch {
		startedAt, starts = be.uploads, starts+1
		return orig(regions, next)
	}
	if err := bindInkling2Device(m, be); err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if starts != 1 || be.uploads == 0 || startedAt != be.uploads {
		t.Fatalf("prefetch started %d times after %d of %d uploads; want once, after all", starts, startedAt, be.uploads)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2RAMTierCgroupClampReportsPlainShortfall(t *testing.T) {
	// A 2 TiB host under a 64 GiB cgroup: host-total headroom (~205 GiB) would
	// leave no tier at all, and the "add RAM" figure would be meaningless.
	info, err := inkling2RAMBudget(ramTierHost(2048*ramTierGiB, 64*ramTierGiB), ramTierEnv(nil))
	if err != nil || info.Budget != 0 {
		t.Fatalf("unclamped: %+v %v", info, err)
	}
	info = inkling2ApplyCgroupClamp(info, true)
	if !info.CgroupClamped || info.Total != 64*ramTierGiB || info.Budget != 64*ramTierGiB-64*ramTierGiB/10 {
		t.Fatalf("clamped: %+v", info)
	}
	order, resident, layerBytes, p := ramTierSmall(t)
	inkling2PlanRAMTier(order, resident, layerBytes, info, &p)
	if p.RAMBytes == 0 || p.RAMNeededForAllExpertsBytes != p.ColdBytes-info.Budget {
		t.Fatalf("clamped plan: %+v", p)
	}
	if got := inkling2ApplyCgroupClamp(inkling2RAMBudgetInfo{Budget: 9}, true); got.CgroupClamped || got.Budget != 9 {
		t.Fatalf("unknown host must not be clamped: %+v", got)
	}
	ov, _ := inkling2RAMBudget(ramTierHost(2048*ramTierGiB, 64*ramTierGiB), ramTierEnv(map[string]string{inkling2RAMBudgetEnv: "7"}))
	if got := inkling2ApplyCgroupClamp(ov, true); got.Budget != 7 {
		t.Fatalf("override must survive the clamp: %+v", got)
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkling2RAMTierMlockFallsBackToTouch(t *testing.T) {
	var log bytes.Buffer
	orig := inkling2RAMLog
	t.Cleanup(func() { inkling2RAMLog = orig })
	inkling2RAMLog = &log
	region := make([]byte, 1<<16)
	p := &inkling2RAMPrefetch{done: make(chan struct{})}
	if p.lock([][]byte{region}) {
		if p.touched.Load() != int64(len(region)) {
			t.Fatalf("locked %d of %d bytes", p.touched.Load(), len(region))
		}
		return
	}
	if p.touched.Load() != 0 || !strings.Contains(log.String(), inkling2RAMMlockEnv) {
		t.Fatalf("failed mlock: touched %d, log %q", p.touched.Load(), log.String())
	}
}
