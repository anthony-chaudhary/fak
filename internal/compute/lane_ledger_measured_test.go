package compute

import (
	"errors"
	"math"
	"testing"
)

// measuredWant is one expected MeasuredFinding row.
type measuredWant struct {
	name     string
	cycles   int64
	additive bool
}

func checkMeasuredFindings(t *testing.T, got []MeasuredFinding, want []measuredWant) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("findings = %+v, want %+v", got, want)
	}
	for i, w := range want {
		if got[i].Name != w.name || got[i].LaneSlotCycles != w.cycles || got[i].Additive != w.additive {
			t.Errorf("finding[%d] = %+v, want %+v", i, got[i], w)
		}
	}
}

func mustLaneLedger(t *testing.T, cap NVCompute, l DecodeLaunch, deviceSMs int) LaneLedger {
	t.Helper()
	arch, ok := LookupNVArch(cap)
	if !ok {
		t.Fatalf("no occupancy row for compute capability %d", cap)
	}
	ledger, err := arch.BuildLaneLedger(l, deviceSMs)
	if err != nil {
		t.Fatalf("BuildLaneLedger: %v", err)
	}
	return ledger
}

// singleWaveLedger is sm_80, 128 threads, 40 registers (blocksPerSM 12, register-bound), 768 B smem, 9 blocks
// on 108 SMs: one wave, 9 SMs busy with one block each.
func singleWaveLedger(t *testing.T) LaneLedger {
	return mustLaneLedger(t, SM80, DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, SmemBytesPerBlock: 768, GridBlocks: 9}, 108)
}

// raggedLedger is sm_80 with 128 threads and 40 registers on 2 SMs, 49 blocks: 24 slots per wave gives three
// waves, the last holding one block on one SM. Per full wave: active 3072, resource_capped 1024. The final wave:
// active 128, resident_unfilled 1408, resource_capped 512, unit_idle 2048. Totals are twice the full wave plus the
// final wave, and each wave is 2*2048 = 4096 lane-slots.
func raggedLedger(t *testing.T) LaneLedger {
	return mustLaneLedger(t, SM80, DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, GridBlocks: 49}, 2)
}

// TestLaneLedgerMeasuredSingleWaveHandValues: one wave covering the whole 1000-cycle window scales every
// slot-wave cell by 1000 and leaves nothing unattributed. Capacity is 108*2048*1000 = 221184000.
func TestLaneLedgerMeasuredSingleWaveHandValues(t *testing.T) {
	m, err := singleWaveLedger(t).JoinMeasured(MeasuredWindow{CyclesPerSM: 1000})
	if err != nil {
		t.Fatalf("JoinMeasured: %v", err)
	}
	if m.CapacityLaneCycles != 221184000 {
		t.Errorf("CapacityLaneCycles = %d, want 221184000", m.CapacityLaneCycles)
	}
	wantCells := [numLaneCells]int64{
		CellActive: 1152000, CellResidentUnfilled: 12672000, CellResourceCapped: 4608000, CellUnitIdle: 202752000,
	}
	if m.Cells != wantCells {
		t.Errorf("Cells = %v, want %v", m.Cells, wantCells)
	}
	if m.Unattributed != 0 || !m.AssumedEqualWaves || m.ActiveSupplied || m.Verdict != "" {
		t.Errorf("Unattributed=%d AssumedEqualWaves=%v ActiveSupplied=%v Verdict=%q, want 0 true false \"\"",
			m.Unattributed, m.AssumedEqualWaves, m.ActiveSupplied, m.Verdict)
	}
	checkMeasuredFindings(t, m.Findings(), []measuredWant{
		{"unit_idle", 202752000, true}, {"resident_unfilled", 12672000, true}, {"resource_capped", 4608000, true},
	})
}

// TestLaneLedgerMeasuredRaggedWeightsHandValues: explicit weights 100, 250, 200 sum to 550 of a 1000-cycle window.
// The full waves run 100+250 = 350 cycles and the final wave 200. Cells are full*350 + final*200. Unattributed is
// 2*2048*(1000-550) = 1843200.
func TestLaneLedgerMeasuredRaggedWeightsHandValues(t *testing.T) {
	m, err := raggedLedger(t).JoinMeasured(MeasuredWindow{CyclesPerSM: 1000, WaveWeights: []int64{100, 250, 200}})
	if err != nil {
		t.Fatalf("JoinMeasured: %v", err)
	}
	if m.CapacityLaneCycles != 4096000 {
		t.Errorf("CapacityLaneCycles = %d, want 4096000", m.CapacityLaneCycles)
	}
	wantCells := [numLaneCells]int64{
		CellActive: 1100800, CellResidentUnfilled: 281600, CellResourceCapped: 460800, CellUnitIdle: 409600,
	}
	if m.Cells != wantCells {
		t.Errorf("Cells = %v, want %v", m.Cells, wantCells)
	}
	if m.Unattributed != 1843200 {
		t.Errorf("Unattributed = %d, want 1843200", m.Unattributed)
	}
	if m.AssumedEqualWaves {
		t.Errorf("AssumedEqualWaves = true with explicit weights")
	}
	checkMeasuredFindings(t, m.Findings(), []measuredWant{
		{"unattributed", 1843200, true}, {"resource_capped", 460800, true},
		{"unit_idle", 409600, true}, {"resident_unfilled", 281600, true},
	})
}

// TestLaneLedgerMeasuredEqualWaveRemainder: 1000 cycles over three equal-length waves does not divide evenly.
// The final wave gets floor(1000/3) = 333 cycles and the two full waves share the remaining 667 (334 and 333),
// so the cells are full*667 + final*333 and nothing is unattributed. The split is deterministic, not measured.
func TestLaneLedgerMeasuredEqualWaveRemainder(t *testing.T) {
	m, err := raggedLedger(t).JoinMeasured(MeasuredWindow{CyclesPerSM: 1000})
	if err != nil {
		t.Fatalf("JoinMeasured: %v", err)
	}
	wantCells := [numLaneCells]int64{
		CellActive: 2091648, CellResidentUnfilled: 468864, CellResourceCapped: 853504, CellUnitIdle: 681984,
	}
	if m.Cells != wantCells {
		t.Errorf("Cells = %v, want %v", m.Cells, wantCells)
	}
	if m.Unattributed != 0 || !m.AssumedEqualWaves {
		t.Errorf("Unattributed=%d AssumedEqualWaves=%v, want 0 true", m.Unattributed, m.AssumedEqualWaves)
	}
}

// TestLaneLedgerMeasuredWindowRefusals: every refusal returns its own sentinel, and the exact and sampled
// weight overruns are distinguishable.
func TestLaneLedgerMeasuredWindowRefusals(t *testing.T) {
	ragged := raggedLedger(t)
	tests := []struct {
		name string
		w    MeasuredWindow
		want error
	}{
		{"zero cycles", MeasuredWindow{CyclesPerSM: 0}, ErrMeasuredCycles},
		{"negative cycles", MeasuredWindow{CyclesPerSM: -5}, ErrMeasuredCycles},
		{"weights too short", MeasuredWindow{CyclesPerSM: 1000, WaveWeights: []int64{100, 200}}, ErrMeasuredWeightsLength},
		{"weights empty", MeasuredWindow{CyclesPerSM: 1000, WaveWeights: []int64{}}, ErrMeasuredWeightsLength},
		{"window shorter than waves under equal waves", MeasuredWindow{CyclesPerSM: 2}, ErrMeasuredWindowShorterThanWaves},
		{"zero weight", MeasuredWindow{CyclesPerSM: 1000, WaveWeights: []int64{100, 0, 200}}, ErrMeasuredWeightNotPositive},
		{"negative weight", MeasuredWindow{CyclesPerSM: 1000, WaveWeights: []int64{100, -1, 200}}, ErrMeasuredWeightNotPositive},
		{"exact weights exceed window", MeasuredWindow{CyclesPerSM: 1000, Exact: true, WaveWeights: []int64{400, 400, 400}}, ErrMeasuredWeightsExceedExact},
		{"sampled weights exceed window", MeasuredWindow{CyclesPerSM: 1000, WaveWeights: []int64{400, 400, 400}}, ErrMeasuredWeightsExceedSampled},
		{"negative active lane-cycles", MeasuredWindow{CyclesPerSM: 1000, ActiveLaneCycles: -1}, ErrMeasuredActiveNegative},
		{"active above capacity", MeasuredWindow{CyclesPerSM: 1000, ActiveLaneCycles: 4096001}, ErrMeasuredActiveOverCapacity},
		{"tolerance NaN", MeasuredWindow{CyclesPerSM: 1000, ActiveTolerance: math.NaN()}, ErrMeasuredTolerance},
		{"tolerance negative", MeasuredWindow{CyclesPerSM: 1000, ActiveTolerance: -0.5}, ErrMeasuredTolerance},
		{"tolerance above one", MeasuredWindow{CyclesPerSM: 1000, ActiveTolerance: 1.5}, ErrMeasuredTolerance},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, err := ragged.JoinMeasured(tc.w)
			if !errors.Is(err, tc.want) {
				t.Fatalf("err = %v, want %v", err, tc.want)
			}
			if m != (MeasuredLedger{}) {
				t.Errorf("refused join returned a non-zero ledger: %+v", m)
			}
		})
	}
	_, sampled := ragged.JoinMeasured(MeasuredWindow{CyclesPerSM: 1000, WaveWeights: []int64{400, 400, 400}})
	if errors.Is(sampled, ErrMeasuredWeightsExceedExact) {
		t.Errorf("sampled overrun reported the exact sentinel: %v", sampled)
	}
}

// TestLaneLedgerMeasuredLedgerRefusals: a ledger that does not conserve, or whose totals do not split into
// waves, is refused before any window is applied.
func TestLaneLedgerMeasuredLedgerRefusals(t *testing.T) {
	tampered := func(mut func(*LaneLedger)) LaneLedger {
		x := raggedLedger(t)
		mut(&x)
		return x
	}
	single := func(mut func(*LaneLedger)) LaneLedger {
		x := singleWaveLedger(t)
		mut(&x)
		return x
	}
	tests := []struct {
		name   string
		ledger LaneLedger
	}{
		{"zero value", LaneLedger{}},
		{"cells not conserved", tampered(func(x *LaneLedger) { x.Cells[CellActive]++ })},
		{"final wave does not fill one wave", tampered(func(x *LaneLedger) { x.FinalWave[CellActive] += 1 })},
		{"full-wave shape not integral", tampered(func(x *LaneLedger) {
			x.Cells[CellActive]++
			x.Cells[CellUnitIdle]--
		})},
		{"single wave cells differ from final", single(func(x *LaneLedger) {
			x.Cells[CellActive]++
			x.Cells[CellUnitIdle]--
		})},
		{"no device SMs", tampered(func(x *LaneLedger) { x.DeviceSMs = 0 })},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.ledger.JoinMeasured(MeasuredWindow{CyclesPerSM: 1000}); !errors.Is(err, ErrMeasuredLedgerNotConserved) {
				t.Fatalf("err = %v, want ErrMeasuredLedgerNotConserved", err)
			}
		})
	}
}

// TestLaneLedgerMeasuredActiveVerdicts checks the model-versus-measured active comparison on the single-wave
// ledger, whose model-active lane-cycles are 1152*1000 = 1152000 of capacity 221184000.
func TestLaneLedgerMeasuredActiveVerdicts(t *testing.T) {
	tests := []struct {
		name      string
		w         MeasuredWindow
		verdict   string
		shortfall int64
	}{
		{"exact agreement", MeasuredWindow{CyclesPerSM: 1000, ActiveLaneCycles: 1152000}, MeasuredVerdictConsistent, 0},
		{"one lane-cycle short of exact", MeasuredWindow{CyclesPerSM: 1000, ActiveLaneCycles: 1151999}, MeasuredVerdictOverstates, 1},
		{"model overstates active", MeasuredWindow{CyclesPerSM: 1000, ActiveLaneCycles: 1000000}, MeasuredVerdictOverstates, 152000},
		{"model understates active", MeasuredWindow{CyclesPerSM: 1000, ActiveLaneCycles: 1200000}, MeasuredVerdictUnderstates, -48000},
		{"inside tolerance", MeasuredWindow{CyclesPerSM: 1000, ActiveLaneCycles: 1000000, ActiveTolerance: 0.001}, MeasuredVerdictConsistent, 152000},
		{"outside tolerance", MeasuredWindow{CyclesPerSM: 1000, ActiveLaneCycles: 1000000, ActiveTolerance: 0.0001}, MeasuredVerdictOverstates, 152000},
		{"measured equals capacity", MeasuredWindow{CyclesPerSM: 1000, ActiveLaneCycles: 221184000}, MeasuredVerdictUnderstates, -220032000},
		{"measured zero is supplied", MeasuredWindow{CyclesPerSM: 1000, ActiveSupplied: true}, MeasuredVerdictOverstates, 1152000},
		{"not supplied", MeasuredWindow{CyclesPerSM: 1000}, "", 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, err := singleWaveLedger(t).JoinMeasured(tc.w)
			if err != nil {
				t.Fatalf("JoinMeasured: %v", err)
			}
			if m.Verdict != tc.verdict || m.ActiveShortfall != tc.shortfall {
				t.Errorf("verdict=%q shortfall=%d, want %q %d", m.Verdict, m.ActiveShortfall, tc.verdict, tc.shortfall)
			}
			supplied := tc.w.ActiveSupplied || tc.w.ActiveLaneCycles != 0
			if !supplied {
				if m.ActiveSupplied || m.ModelActive != 0 || m.MeasuredActive != 0 {
					t.Errorf("unsupplied active populated: %+v", m)
				}
				checkActiveResiduals(t, m, "")
				return
			}
			if !m.ActiveSupplied || m.ModelActive != 1152000 || m.MeasuredActive != tc.w.ActiveLaneCycles {
				t.Errorf("ModelActive=%d MeasuredActive=%d, want 1152000 %d", m.ModelActive, m.MeasuredActive, tc.w.ActiveLaneCycles)
			}
			checkActiveResiduals(t, m, tc.verdict)
		})
	}
}

// checkActiveResiduals pins the unexplained_* rows to the verdict: an overstating verdict gives unexplained_inactive
// of the shortfall, an understating verdict gives unexplained_active of its magnitude, and a consistent or absent
// verdict gives neither. Each row is non-additive.
func checkActiveResiduals(t *testing.T, m MeasuredLedger, verdict string) {
	t.Helper()
	var want []measuredWant
	switch verdict {
	case MeasuredVerdictOverstates:
		want = []measuredWant{{"unexplained_inactive", m.ActiveShortfall, false}}
	case MeasuredVerdictUnderstates:
		want = []measuredWant{{"unexplained_active", -m.ActiveShortfall, false}}
	}
	var got []MeasuredFinding
	for _, f := range m.Findings() {
		if f.Name == "unexplained_inactive" || f.Name == "unexplained_active" {
			got = append(got, f)
		}
	}
	checkMeasuredFindings(t, got, want)
}

// TestLaneLedgerMeasuredActiveAboveFloatPrecision: the tolerance bound is compared in exact integers. One lane over
// 2^54 cycles has capacity 2^54; float64 rounds 2^53+1 down to 2^53, so a float comparison would call the gap
// within a 0.5 tolerance. The exact bound is 2^53, so 2^53+1 overstates and 2^53 is consistent.
func TestLaneLedgerMeasuredActiveAboveFloatPrecision(t *testing.T) {
	one := LaneLedger{DeviceSMs: 1, LanesPerSM: 1, Waves: 1, Capacity: 1}
	one.Cells[CellActive] = 1
	one.FinalWave[CellActive] = 1
	const window = int64(1) << 54
	const half = 0.5
	tests := []struct {
		name    string
		active  int64
		verdict string
	}{
		{"gap of 2^53+1 is outside the 2^53 bound", (int64(1) << 53) - 1, MeasuredVerdictOverstates},
		{"gap of exactly 2^53 is inside the bound", int64(1) << 53, MeasuredVerdictConsistent},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m, err := one.JoinMeasured(MeasuredWindow{CyclesPerSM: window, ActiveLaneCycles: tc.active, ActiveTolerance: half})
			if err != nil {
				t.Fatalf("JoinMeasured: %v", err)
			}
			if m.CapacityLaneCycles != window {
				t.Fatalf("capacity = %d, want %d", m.CapacityLaneCycles, window)
			}
			if m.Verdict != tc.verdict {
				t.Errorf("verdict = %q, want %q (shortfall %d)", m.Verdict, tc.verdict, m.ActiveShortfall)
			}
		})
	}
}

// TestLaneLedgerMeasuredOverflow: the capacity product is checked, so a window whose capacity exceeds int64 is
// refused. The largest exact capacity on one SM is accepted.
func TestLaneLedgerMeasuredOverflow(t *testing.T) {
	if _, err := singleWaveLedger(t).JoinMeasured(MeasuredWindow{CyclesPerSM: math.MaxInt64}); !errors.Is(err, ErrMeasuredOverflow) {
		t.Errorf("108-SM overflow: err = %v, want ErrMeasuredOverflow", err)
	}
	one := mustLaneLedger(t, SM80, DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, GridBlocks: 9}, 1)
	if _, err := one.JoinMeasured(MeasuredWindow{CyclesPerSM: math.MaxInt64}); !errors.Is(err, ErrMeasuredOverflow) {
		t.Errorf("one-SM overflow: err = %v, want ErrMeasuredOverflow", err)
	}
	fits := int64(math.MaxInt64 / 2048)
	m, err := one.JoinMeasured(MeasuredWindow{CyclesPerSM: fits})
	if err != nil {
		t.Fatalf("boundary window: %v", err)
	}
	if m.CapacityLaneCycles != fits*2048 || m.Unattributed != 0 {
		t.Errorf("boundary capacity=%d unattributed=%d, want %d 0", m.CapacityLaneCycles, m.Unattributed, fits*2048)
	}
}

// TestLaneLedgerMeasuredConservationSweep joins many launches against many windows and checks the exact
// identities: sum(cells)+Unattributed == capacity, every cell and Unattributed non-negative, the findings
// partition the non-active capacity, and when the window is exactly one cycle per wave the cells equal the
// model's cells.
func TestLaneLedgerMeasuredConservationSweep(t *testing.T) {
	caps := []NVCompute{SM80, SM89, SM100, SM120}
	threads := []int{1, 31, 32, 100, 128, 256, 1024}
	regs := []int{16, 40, 64, 255}
	smems := []int{0, 768, 48000}
	grids := []int{1, 9, 49, 108, 1000}
	sms := []int{1, 2, 84, 108}
	windows := []int64{1, 7, 333, 1000, 1 << 20}
	joins := 0
	for _, cap := range caps {
		arch, _ := LookupNVArch(cap)
		for _, tpb := range threads {
			for _, r := range regs {
				for _, smem := range smems {
					for _, grid := range grids {
						for _, d := range sms {
							l := DecodeLaunch{Kernel: "k", ThreadsPerBlock: tpb, RegsPerThread: r, SmemBytesPerBlock: smem, GridBlocks: grid}
							ledger, err := arch.BuildLaneLedger(l, d)
							if errors.Is(err, ErrLaneUnlaunchable) {
								continue
							}
							if err != nil {
								t.Fatalf("BuildLaneLedger(%+v, %d): %v", l, d, err)
							}
							joins += sweepJoins(t, ledger, append(windows, int64(ledger.Waves)))
						}
					}
				}
			}
		}
	}
	if joins == 0 {
		t.Fatal("sweep joined no ledgers")
	}
}

// sweepJoins runs equal-wave and explicit-weight joins for each window and asserts the identities.
func sweepJoins(t *testing.T, ledger LaneLedger, windows []int64) int {
	t.Helper()
	n := 0
	slots := int64(ledger.DeviceSMs) * int64(ledger.LanesPerSM)
	for _, c := range windows {
		equal := MeasuredWindow{CyclesPerSM: c}
		if c < int64(ledger.Waves) {
			if _, err := ledger.JoinMeasured(equal); !errors.Is(err, ErrMeasuredWindowShorterThanWaves) {
				t.Fatalf("equal waves over %d cycles and %d waves: err = %v, want ErrMeasuredWindowShorterThanWaves", c, ledger.Waves, err)
			}
		} else {
			checkSweepJoin(t, ledger, equal, slots)
			n++
		}
		waves := int64(ledger.Waves)
		if c >= 2*waves {
			partial := make([]int64, waves)
			for i := range partial {
				partial[i] = c / (2 * waves)
			}
			checkSweepJoin(t, ledger, MeasuredWindow{CyclesPerSM: c, WaveWeights: partial}, slots)
			n++
		}
		if c >= waves {
			exact := make([]int64, waves)
			var used int64
			for i := range exact[:waves-1] {
				exact[i] = c / waves
				used += exact[i]
			}
			exact[waves-1] = c - used
			checkSweepJoin(t, ledger, MeasuredWindow{CyclesPerSM: c, WaveWeights: exact, Exact: true}, slots)
			n++
		}
	}
	return n
}

func checkSweepJoin(t *testing.T, ledger LaneLedger, w MeasuredWindow, slots int64) {
	t.Helper()
	m, err := ledger.JoinMeasured(w)
	if err != nil {
		t.Fatalf("JoinMeasured(c=%d, weights=%v): %v", w.CyclesPerSM, w.WaveWeights, err)
	}
	capacity := w.CyclesPerSM * slots
	if m.CapacityLaneCycles != capacity {
		t.Fatalf("capacity = %d, want %d", m.CapacityLaneCycles, capacity)
	}
	var sum int64
	for c, n := range m.Cells {
		if n < 0 {
			t.Fatalf("cell %d = %d is negative", c, n)
		}
		sum += n
	}
	if m.Unattributed < 0 || sum+m.Unattributed != capacity {
		t.Fatalf("sum %d + unattributed %d != capacity %d", sum, m.Unattributed, capacity)
	}
	if w.WaveWeights == nil && m.Unattributed != 0 {
		t.Fatalf("equal waves left unattributed %d", m.Unattributed)
	}
	if w.Exact && w.WaveWeights != nil && m.Unattributed != 0 {
		t.Fatalf("exact fill left unattributed %d", m.Unattributed)
	}
	if ledger.Waves == 1 && w.WaveWeights == nil {
		for c := range m.Cells {
			if m.Cells[c] != ledger.Cells[c]*w.CyclesPerSM {
				t.Fatalf("single wave cell %d = %d, want %d*%d", c, m.Cells[c], ledger.Cells[c], w.CyclesPerSM)
			}
		}
	}
	if w.CyclesPerSM == int64(ledger.Waves) && m.Cells != ledger.Cells {
		t.Fatalf("one cycle per wave gave cells %v, want the model cells %v", m.Cells, ledger.Cells)
	}
	var additive int64
	for _, f := range m.Findings() {
		if f.Additive {
			additive += f.LaneSlotCycles
		}
	}
	if additive != capacity-m.Cells[CellActive] {
		t.Fatalf("additive findings %d, want capacity-active %d", additive, capacity-m.Cells[CellActive])
	}
}
