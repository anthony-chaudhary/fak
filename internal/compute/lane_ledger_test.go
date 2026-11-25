package compute

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"testing"
)

// laneFinding is one expected Findings row: the name and the exact lane-slot-wave count.
type laneFinding struct {
	name  string
	slots int64
}

// checkFindings asserts the findings match the expected names and counts in order.
func checkFindings(t *testing.T, got []GapFinding, want []laneFinding) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("findings = %+v, want %d rows %+v", got, len(want), want)
	}
	for i, w := range want {
		if got[i].Name != w.name || got[i].LaneSlotWaves != w.slots {
			t.Errorf("finding[%d] = %s:%d, want %s:%d", i, got[i].Name, got[i].LaneSlotWaves, w.name, w.slots)
		}
	}
}

// TestLaneLedgerHandValues pins the ledger against arithmetic worked out by hand from the occupancy
// formulas. Each row's cells, capacity and findings are independent of the implementation's loop order.
func TestLaneLedgerHandValues(t *testing.T) {
	sm80, _ := LookupNVArch(SM80)
	sm120, _ := LookupNVArch(SM120)
	tests := []struct {
		name          string
		arch          NVArch
		launch        DecodeLaunch
		deviceSMs     int
		waves         int
		blocksPerSM   int
		limiter       string
		capacity      int64
		cells         [numLaneCells]int64
		slotsUnfilled int64
		findings      []laneFinding
	}{
		{
			// grid 9 on 108 SMs: 9 SMs take one block, 99 are empty. The register ceiling (12) binds.
			name: "sm_80 register-bound grid underfill",
			arch: sm80, launch: DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, SmemBytesPerBlock: 768, GridBlocks: 9},
			deviceSMs: 108, waves: 1, blocksPerSM: 12, limiter: "register", capacity: 221184,
			cells:         [numLaneCells]int64{CellActive: 1152, CellResidentUnfilled: 12672, CellResourceCapped: 4608, CellUnitIdle: 202752},
			slotsUnfilled: 1287,
			findings: []laneFinding{
				{"underfill", 202752}, {"resident_unfilled", 12672}, {"resource_capped", 4608},
			},
		},
		{
			// 100 threads fill 3 full warps and leave 28 lanes of a fourth warp partial.
			name: "sm_80 partial warp on one SM",
			arch: sm80, launch: DecodeLaunch{Kernel: "k", ThreadsPerBlock: 100, RegsPerThread: 40, GridBlocks: 1},
			deviceSMs: 1, waves: 1, blocksPerSM: 12, limiter: "register", capacity: 2048,
			cells:         [numLaneCells]int64{CellActive: 100, CellPartialWarp: 28, CellResidentUnfilled: 1408, CellResourceCapped: 512},
			slotsUnfilled: 11,
			findings: []laneFinding{
				{"resident_unfilled", 1408}, {"resource_capped", 512}, {"partial_warp", 28},
			},
		},
		{
			// 25 blocks on 2 SMs with 12 slots each: the full wave is exact, the final wave holds one
			// block, so one SM is empty and eleven of the twelve slots on the other are unfilled.
			name: "sm_80 final-wave tail",
			arch: sm80, launch: DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, GridBlocks: 25},
			deviceSMs: 2, waves: 2, blocksPerSM: 12, limiter: "register", capacity: 8192,
			cells:         [numLaneCells]int64{CellActive: 3200, CellResidentUnfilled: 1408, CellResourceCapped: 1536, CellUnitIdle: 2048},
			slotsUnfilled: 23,
			findings: []laneFinding{
				{"wave_tail", 3456}, {"resource_capped", 1536},
			},
		},
		{
			// SM count is a caller input: 84 SMs is passed explicitly, not defaulted. 100 blocks put 16 SMs
			// at two blocks and 68 at one; sm_120 has no resource cap left once 12 blocks are resident.
			name: "sm_120 caller-supplied 84 SMs",
			arch: sm120, launch: DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, GridBlocks: 100},
			deviceSMs: 84, waves: 1, blocksPerSM: 12, limiter: "register", capacity: 129024,
			cells:         [numLaneCells]int64{CellActive: 12800, CellResidentUnfilled: 116224},
			slotsUnfilled: 908,
			findings: []laneFinding{
				{"resident_unfilled", 116224},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ledger, err := tc.arch.BuildLaneLedger(tc.launch, tc.deviceSMs)
			if err != nil {
				t.Fatalf("BuildLaneLedger: %v", err)
			}
			if ledger.Waves != tc.waves || ledger.BlocksPerSM != tc.blocksPerSM || ledger.Limiter != tc.limiter {
				t.Errorf("waves=%d blocksPerSM=%d limiter=%q, want %d %d %q",
					ledger.Waves, ledger.BlocksPerSM, ledger.Limiter, tc.waves, tc.blocksPerSM, tc.limiter)
			}
			if ledger.Capacity != tc.capacity {
				t.Errorf("Capacity = %d, want %d", ledger.Capacity, tc.capacity)
			}
			if ledger.Cells != tc.cells {
				t.Errorf("Cells = %v, want %v", ledger.Cells, tc.cells)
			}
			if ledger.BlockSlotsUnfilled != tc.slotsUnfilled {
				t.Errorf("BlockSlotsUnfilled = %d, want %d", ledger.BlockSlotsUnfilled, tc.slotsUnfilled)
			}
			if !ledger.Conserved() {
				t.Errorf("ledger does not conserve capacity: %+v", ledger.Cells)
			}
			checkFindings(t, ledger.Findings(), tc.findings)
		})
	}
}

// TestLaneLedgerPartialWarpCount pins the partial-lane count per block the repo's occupancy notes state:
// a 100-thread block occupies four warps, so 128−100 = 28 lanes are unusable in each block.
func TestLaneLedgerPartialWarpCount(t *testing.T) {
	a, _ := LookupNVArch(SM80)
	for _, tc := range []struct{ tpb, partialPerBlock int }{{1, 31}, {31, 1}, {32, 0}, {33, 31}, {100, 28}, {1000, 24}} {
		ledger, err := a.BuildLaneLedger(DecodeLaunch{Kernel: "k", ThreadsPerBlock: tc.tpb, RegsPerThread: 16, GridBlocks: 1}, 1)
		if err != nil {
			t.Fatalf("tpb=%d: %v", tc.tpb, err)
		}
		if got := ledger.Cells[CellPartialWarp]; got != int64(tc.partialPerBlock) {
			t.Errorf("tpb=%d: partial lanes = %d, want %d for the one block", tc.tpb, got, tc.partialPerBlock)
		}
	}
}

// TestLaneLedgerFindingsAttribution checks on a synthetic three-wave ledger that the final-wave split
// feeds wave_tail only, and that the findings still account for every lost lane-slot-wave.
func TestLaneLedgerFindingsAttribution(t *testing.T) {
	x := LaneLedger{
		Limiter:   "shared",
		Waves:     3,
		Capacity:  100,
		Cells:     [numLaneCells]int64{CellActive: 40, CellResidentUnfilled: 30, CellResourceCapped: 20, CellUnitIdle: 10},
		FinalWave: [numLaneCells]int64{CellResidentUnfilled: 7, CellUnitIdle: 10},
	}
	checkFindings(t, x.Findings(), []laneFinding{
		{"resident_unfilled", 23}, {"resource_capped", 20}, {"wave_tail", 17},
	})
	sum := int64(0)
	for _, f := range x.Findings() {
		sum += f.LaneSlotWaves
	}
	if sum != x.Capacity-x.Cells[CellActive] {
		t.Errorf("findings sum %d, want Capacity-Active = %d", sum, x.Capacity-x.Cells[CellActive])
	}
	for _, f := range x.Findings() {
		if f.Name == "resource_capped" && f.Trigger != "binding limiter=shared" {
			t.Errorf("resource_capped trigger = %q, want it to name the limiter", f.Trigger)
		}
	}
	for _, f := range x.Findings() {
		if f.Name == "wave_tail" && f.Cell != CellUnitIdle.String() {
			t.Errorf("wave_tail cell = %q, want %q", f.Cell, CellUnitIdle)
		}
	}
}

// TestLaneLedgerFindingsTieBreak pins the order of equal findings: equal lane-slot-wave counts sort by name.
func TestLaneLedgerFindingsTieBreak(t *testing.T) {
	x := LaneLedger{
		Limiter:  "register",
		Waves:    1,
		Capacity: 10,
		Cells:    [numLaneCells]int64{CellPartialWarp: 5, CellResidentUnfilled: 5},
	}
	checkFindings(t, x.Findings(), []laneFinding{{"partial_warp", 5}, {"resident_unfilled", 5}})
}

// TestLaneLedgerJSONRoundTrip decodes a multi-wave ledger back from its own JSON. The final wave travels in
// the document, so the wave_tail finding survives. 4 waves of 324 slots hold 1000 blocks; the final wave has
// 28 blocks, leaving 80 of 108 SMs empty (80*2048 = 163840 unit_idle) and 2 free slots on each of the 28
// busy SMs (28*2*128 = 7168 resident_unfilled), so wave_tail = 171008.
func TestLaneLedgerJSONRoundTrip(t *testing.T) {
	sm80, _ := LookupNVArch(SM80)
	launch := DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, SmemBytesPerBlock: 48000, GridBlocks: 1000}
	want, err := sm80.BuildLaneLedger(launch, 108)
	if err != nil {
		t.Fatalf("BuildLaneLedger: %v", err)
	}
	if want.Waves != 4 || want.FinalWave[CellUnitIdle] != 163840 || want.FinalWave[CellResidentUnfilled] != 7168 {
		t.Fatalf("hand values: waves=%d final=%v", want.Waves, want.FinalWave)
	}
	raw, err := json.Marshal(want)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var got LaneLedger
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("json.Unmarshal of the ledger's own output: %v", err)
	}
	if got != want {
		t.Errorf("round trip changed the ledger:\n got %+v\nwant %+v", got, want)
	}
	if !got.Conserved() {
		t.Errorf("decoded ledger does not conserve capacity")
	}
	// Each populated SM keeps 2048 - 3*128 = 1664 lanes out of its three blocks: 3 full waves of 108 SMs plus
	// the 28 populated SMs of the final wave give 585728. Those and wave_tail (171008) cover all 756736 lost lanes.
	checkFindings(t, got.Findings(), []laneFinding{{"resource_capped", 585728}, {"wave_tail", 171008}})
	var sum int64
	for _, f := range got.Findings() {
		sum += f.LaneSlotWaves
	}
	if sum != got.Capacity-got.Cells[CellActive] {
		t.Errorf("decoded findings sum %d, want Capacity-Active = %d", sum, got.Capacity-got.Cells[CellActive])
	}
}

// TestLaneLedgerJSONRefusesIncompleteCells checks that a cell map missing a name or carrying an unknown one
// is refused rather than decoded to zeros.
func TestLaneLedgerJSONRefusesIncompleteCells(t *testing.T) {
	for name, doc := range map[string]string{
		"missing names": `{"capacity":10,"cells":{"active":10},"final_wave":{}}`,
		"unknown name": `{"capacity":10,"cells":{"active":4,"partial_warp":0,"resident_unfilled":0,"resource_capped":0,"bogus":6},` +
			`"final_wave":{"active":0,"partial_warp":0,"resident_unfilled":0,"resource_capped":0,"unit_idle":0}}`,
	} {
		var x LaneLedger
		if err := json.Unmarshal([]byte(doc), &x); err == nil {
			t.Errorf("%s: decoded without error: %+v", name, x)
		}
	}
}

// TestLaneLedgerSharedMemoryBoundary pins the shared-memory ceiling: a block whose rounded footprint equals
// the SM's capacity is launchable with shared memory binding, and one allocation unit more is refused.
func TestLaneLedgerSharedMemoryBoundary(t *testing.T) {
	sm80, _ := LookupNVArch(SM80)
	limit := sm80.SmemPerSM
	for _, tc := range []struct {
		name    string
		smem    int
		wantErr error
		blocks  int
	}{
		{"one byte under the SM capacity", limit - 1, nil, 1},
		{"exactly the SM capacity", limit, nil, 1},
		{"one byte over the SM capacity", limit + 1, ErrLaneUnlaunchable, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			launch := DecodeLaunch{Kernel: "k", ThreadsPerBlock: 32, RegsPerThread: 1, SmemBytesPerBlock: tc.smem, GridBlocks: 1}
			ledger, err := sm80.BuildLaneLedger(launch, 1)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("BuildLaneLedger err = %v, want %v", err, tc.wantErr)
			}
			if tc.wantErr != nil {
				return
			}
			if ledger.BlocksPerSM != tc.blocks || ledger.Limiter != "shared" {
				t.Errorf("blocksPerSM=%d limiter=%q, want %d shared", ledger.BlocksPerSM, ledger.Limiter, tc.blocks)
			}
		})
	}
}

// TestLaneCellString pins the snake_case names the JSON and findings expose.
func TestLaneCellString(t *testing.T) {
	for _, tc := range []struct {
		cell LaneCell
		want string
	}{
		{CellActive, "active"},
		{CellPartialWarp, "partial_warp"},
		{CellResidentUnfilled, "resident_unfilled"},
		{CellResourceCapped, "resource_capped"},
		{CellUnitIdle, "unit_idle"},
		{LaneCell(numLaneCells), "unknown"},
	} {
		if got := tc.cell.String(); got != tc.want {
			t.Errorf("LaneCell(%d).String() = %q, want %q", tc.cell, got, tc.want)
		}
	}
}

// TestLaneLedgerJSON checks that cells marshal as a name->count object and the arch provenance travels.
func TestLaneLedgerJSON(t *testing.T) {
	a, _ := LookupNVArch(SM80)
	ledger, err := a.BuildLaneLedger(DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, SmemBytesPerBlock: 768, GridBlocks: 9}, 108)
	if err != nil {
		t.Fatalf("BuildLaneLedger: %v", err)
	}
	raw, err := json.Marshal(ledger)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}
	cells, ok := got["cells"].(map[string]any)
	if !ok {
		t.Fatalf("cells is not a name->count object: %s", raw)
	}
	want := map[string]float64{
		"active": 1152, "partial_warp": 0, "resident_unfilled": 12672, "resource_capped": 4608, "unit_idle": 202752,
	}
	if len(cells) != len(want) {
		t.Errorf("cells has %d keys, want %d: %v", len(cells), len(want), cells)
	}
	for name, v := range want {
		if cells[name] != v {
			t.Errorf("cells[%s] = %v, want %v", name, cells[name], v)
		}
	}
	if got["capacity"] != float64(221184) || got["arch"] != "sm_80" || got["provenance"] != nvSourceLimits {
		t.Errorf("capacity/arch/provenance wrong in %s", raw)
	}
}

// TestRegisterFootprintHandValues pins the register partition against hand arithmetic. The R=41 row is
// the one the rounding and granularity formulas were checked on: 41×32 = 1312 rounds to 1536 per warp,
// the 65536-register file fits 42 such warps, 40 of which are granted after rounding down to 4.
func TestRegisterFootprintHandValues(t *testing.T) {
	a, _ := LookupNVArch(SM80)
	tests := []struct {
		name        string
		launch      DecodeLaunch
		blocksPerSM int
		limiter     string
		want        RegisterFootprint
		findings    []string
	}{
		{
			name:        "R=41 rounding and granularity, register-bound",
			launch:      DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 41, GridBlocks: 1},
			blocksPerSM: 10, limiter: "register",
			want: RegisterFootprint{RegsPerSM: 65536, Useful: 52480, RoundingWaste: 8960,
				GranularityWaste: 3072, HeadroomUnused: 0, Remainder: 1024},
			findings: []string{"register_rounding", "register_granularity"},
		},
		{
			name:        "R=40 shared-bound leaves whole-warp headroom",
			launch:      DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, SmemBytesPerBlock: 20480, GridBlocks: 1},
			blocksPerSM: 8, limiter: "shared",
			want: RegisterFootprint{RegsPerSM: 65536, Useful: 40960, RoundingWaste: 0,
				GranularityWaste: 3840, HeadroomUnused: 20480, Remainder: 256},
			findings: []string{"register_headroom", "register_granularity"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			bpsm, limiter := a.blocksPerSM(tc.launch)
			if bpsm != tc.blocksPerSM || limiter != tc.limiter {
				t.Fatalf("blocksPerSM = %d/%q, want %d/%q", bpsm, limiter, tc.blocksPerSM, tc.limiter)
			}
			got, err := a.RegisterFootprintOf(tc.launch)
			if err != nil {
				t.Fatalf("RegisterFootprintOf: %v", err)
			}
			if got != tc.want {
				t.Errorf("footprint = %+v, want %+v", got, tc.want)
			}
			if !got.Conserved() {
				t.Errorf("footprint does not partition the register file: %+v", got)
			}
			findings := got.Findings(limiter)
			if len(findings) != len(tc.findings) {
				t.Fatalf("register findings = %+v, want names %v", findings, tc.findings)
			}
			for i, name := range tc.findings {
				if findings[i].Name != name {
					t.Errorf("register finding[%d] = %s, want %s", i, findings[i].Name, name)
				}
			}
		})
	}
}

// TestRegisterFootprintHeadroomNeedsNonRegisterLimiter pins the rule that headroom is reported only when a
// non-register limiter binds, even though the arithmetic value is the same under a register limit.
func TestRegisterFootprintHeadroomNeedsNonRegisterLimiter(t *testing.T) {
	x := RegisterFootprint{RegsPerSM: 65536, Useful: 40960, GranularityWaste: 3840, HeadroomUnused: 20480, Remainder: 256}
	for _, tc := range []struct {
		limiter string
		want    []string
	}{
		{"register", []string{"register_granularity"}},
		{"shared", []string{"register_headroom", "register_granularity"}},
		{"warp", []string{"register_headroom", "register_granularity"}},
	} {
		got := x.Findings(tc.limiter)
		if len(got) != len(tc.want) {
			t.Errorf("limiter=%s: findings %+v, want %v", tc.limiter, got, tc.want)
			continue
		}
		for i, name := range tc.want {
			if got[i].Name != name {
				t.Errorf("limiter=%s: finding[%d]=%s, want %s", tc.limiter, i, got[i].Name, name)
			}
		}
	}
}

// refusalLaunch is a launch every refusal case mutates one field of.
var refusalLaunch = DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, SmemBytesPerBlock: 768, GridBlocks: 9}

// TestLaneLedgerRefusals checks every refusal path returns its own sentinel and an empty result, and that
// RegisterFootprintOf refuses the same launch-shape inputs with the same sentinels.
func TestLaneLedgerRefusals(t *testing.T) {
	sm80, _ := LookupNVArch(SM80)
	mutate := func(f func(*DecodeLaunch)) DecodeLaunch {
		l := refusalLaunch
		f(&l)
		return l
	}
	mutatedArch := sm80
	mutatedArch.RegsPerSM++
	tests := []struct {
		name      string
		arch      NVArch
		launch    DecodeLaunch
		deviceSMs int
		want      error
	}{
		{"zero device SMs", sm80, refusalLaunch, 0, ErrLaneNoDeviceSMs},
		{"negative device SMs", sm80, refusalLaunch, -1, ErrLaneNoDeviceSMs},
		{"zero registers", sm80, mutate(func(l *DecodeLaunch) { l.RegsPerThread = 0 }), 108, ErrLaneRegsUnmeasured},
		{"negative registers", sm80, mutate(func(l *DecodeLaunch) { l.RegsPerThread = -1 }), 108, ErrLaneRegsUnmeasured},
		{"registers above 255", sm80, mutate(func(l *DecodeLaunch) { l.RegsPerThread = 256 }), 108, ErrLaneRegsOverLimit},
		{"zero threads", sm80, mutate(func(l *DecodeLaunch) { l.ThreadsPerBlock = 0 }), 108, ErrLaneThreads},
		{"negative threads", sm80, mutate(func(l *DecodeLaunch) { l.ThreadsPerBlock = -32 }), 108, ErrLaneThreads},
		{"threads above 1024", sm80, mutate(func(l *DecodeLaunch) { l.ThreadsPerBlock = 1025 }), 108, ErrLaneThreads},
		{"zero grid", sm80, mutate(func(l *DecodeLaunch) { l.GridBlocks = 0 }), 108, ErrLaneGrid},
		{"negative grid", sm80, mutate(func(l *DecodeLaunch) { l.GridBlocks = -1 }), 108, ErrLaneGrid},
		{"negative shared memory", sm80, mutate(func(l *DecodeLaunch) { l.SmemBytesPerBlock = -1 }), 108, ErrLaneSmem},
		{"zero-value arch", NVArch{}, refusalLaunch, 108, ErrLaneUnknownArch},
		{"arch row altered from the table", mutatedArch, refusalLaunch, 108, ErrLaneUnknownArch},
		{"registers exhaust the SM", sm80, mutate(func(l *DecodeLaunch) {
			l.RegsPerThread = 255
			l.ThreadsPerBlock = 1024
		}), 108, ErrLaneUnlaunchable},
		{"shared memory exceeds the SM", sm80, mutate(func(l *DecodeLaunch) { l.SmemBytesPerBlock = 200 * 1024 }), 108, ErrLaneUnlaunchable},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ledger, err := tc.arch.BuildLaneLedger(tc.launch, tc.deviceSMs)
			if !errors.Is(err, tc.want) {
				t.Fatalf("BuildLaneLedger err = %v, want %v", err, tc.want)
			}
			if ledger.Capacity != 0 || ledger.Cells != [numLaneCells]int64{} {
				t.Errorf("refused build returned a non-zero ledger: %+v", ledger)
			}
			if ledger.Conserved() {
				t.Errorf("refused build reports Conserved")
			}
			if tc.want == ErrLaneNoDeviceSMs {
				return
			}
			if _, err := tc.arch.RegisterFootprintOf(tc.launch); !errors.Is(err, tc.want) {
				t.Errorf("RegisterFootprintOf err = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestLaneLedgerOverflowRefused needs a 64-bit int to express a capacity above MaxInt64.
func TestLaneLedgerOverflowRefused(t *testing.T) {
	if strconv.IntSize != 64 {
		t.Skip("int is narrower than 64 bits; the overflow boundary is not representable")
	}
	sm80, _ := LookupNVArch(SM80)
	shift := 52
	huge := 1 << shift // 2^52 SMs × 2048 lanes = 2^63, one past MaxInt64
	if _, err := sm80.BuildLaneLedger(refusalLaunch, huge); !errors.Is(err, ErrLaneOverflow) {
		t.Errorf("BuildLaneLedger(deviceSMs=2^52) err = %v, want ErrLaneOverflow", err)
	}
	if _, err := sm80.BuildLaneLedger(refusalLaunch, huge-1); err != nil {
		t.Errorf("BuildLaneLedger(deviceSMs=2^52-1) err = %v, want success (largest representable)", err)
	}
}

// TestLaneLedgerConservedRejectsBrokenCells checks Conserved against a hand-broken ledger.
func TestLaneLedgerConservedRejectsBrokenCells(t *testing.T) {
	ok := LaneLedger{Capacity: 10, Cells: [numLaneCells]int64{CellActive: 4, CellUnitIdle: 6}}
	if !ok.Conserved() {
		t.Fatal("a partition summing to Capacity must be conserved")
	}
	if (RegisterFootprint{}).Conserved() {
		t.Error("zero register footprint reported conserved")
	}
	short := ok
	short.Cells[CellUnitIdle] = 5
	negative := ok
	negative.Cells[CellActive] = -1
	negative.Cells[CellUnitIdle] = 11
	for name, x := range map[string]LaneLedger{"short": short, "negative": negative, "zero value": {}} {
		if x.Conserved() {
			t.Errorf("%s ledger reported conserved: %+v", name, x.Cells)
		}
	}
}

// oracleSMWaveBudget bounds the per-SM oracle, which visits every SM of every wave. Launches with more
// SM-waves than this are checked by the closed-form invariants alone.
const oracleSMWaveBudget = 1 << 14

// perSMOracle classifies a launch by visiting every SM of every wave and summing each SM's cells, so it
// shares no arithmetic with the closed form in waveCells. It returns the total and the final wave.
func perSMOracle(a NVArch, l DecodeLaunch, deviceSMs, bpsm int) (total, final [numLaneCells]int64) {
	lanesPerSM := int64(a.MaxWarpsPerSM * a.WarpSize)
	tpb := int64(l.ThreadsPerBlock)
	warpLanes := int64(ceilDivInt(l.ThreadsPerBlock, a.WarpSize) * a.WarpSize)
	slots := deviceSMs * bpsm
	for first := 0; first < l.GridBlocks; first += slots {
		n := l.GridBlocks - first
		if n > slots {
			n = slots
		}
		var wave [numLaneCells]int64
		for sm := 0; sm < deviceSMs; sm++ {
			c := int64(n / deviceSMs)
			if sm < n%deviceSMs {
				c++
			}
			var sc [numLaneCells]int64
			if c == 0 {
				sc[CellUnitIdle] = lanesPerSM
			} else {
				sc[CellActive] = c * tpb
				sc[CellPartialWarp] = c * (warpLanes - tpb)
				sc[CellResidentUnfilled] = (int64(bpsm) - c) * warpLanes
				sc[CellResourceCapped] = lanesPerSM - int64(bpsm)*warpLanes
			}
			for k := range wave {
				wave[k] += sc[k]
			}
		}
		for k := range total {
			total[k] += wave[k]
		}
		final = wave
	}
	return total, final
}

// perSMParity checks the closed-form cells and final wave against the per-SM oracle. It reports false, and
// checks nothing, when the launch has more SM-waves than oracleSMWaveBudget.
func perSMParity(a NVArch, l DecodeLaunch, ledger LaneLedger) (bool, error) {
	if int64(ledger.Waves)*int64(ledger.DeviceSMs) > oracleSMWaveBudget {
		return false, nil
	}
	total, final := perSMOracle(a, l, ledger.DeviceSMs, ledger.BlocksPerSM)
	if total != ledger.Cells {
		return true, fmt.Errorf("cells %v != per-SM oracle %v", ledger.Cells, total)
	}
	if final != ledger.FinalWave {
		return true, fmt.Errorf("final wave %v != per-SM oracle %v", ledger.FinalWave, final)
	}
	return true, nil
}

// TestLaneLedgerPerSMOracle runs the brute-force oracle on launches inside its budget, and checks that it
// rejects a ledger whose lanes were moved between classes, so it is not agreeing with anything.
func TestLaneLedgerPerSMOracle(t *testing.T) {
	sm80, _ := LookupNVArch(SM80)
	sm120, _ := LookupNVArch(SM120)
	for _, tc := range []struct {
		name    string
		arch    NVArch
		launch  DecodeLaunch
		devices int
	}{
		{"sm_80 final-wave tail", sm80, DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, SmemBytesPerBlock: 48000, GridBlocks: 1000}, 108},
		{"sm_80 grid below the device", sm80, DecodeLaunch{Kernel: "k", ThreadsPerBlock: 128, RegsPerThread: 40, GridBlocks: 9}, 108},
		// SM count is a caller input: 84 is passed explicitly.
		{"sm_120 caller-supplied 84 SMs", sm120, DecodeLaunch{Kernel: "k", ThreadsPerBlock: 100, RegsPerThread: 40, GridBlocks: 1000}, 84},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ledger, err := tc.arch.BuildLaneLedger(tc.launch, tc.devices)
			if err != nil {
				t.Fatalf("BuildLaneLedger: %v", err)
			}
			if ran, err := perSMParity(tc.arch, tc.launch, ledger); !ran || err != nil {
				t.Fatalf("oracle ran=%v err=%v, want a passing check", ran, err)
			}
			moved := ledger
			moved.Cells[CellActive]--
			moved.Cells[CellUnitIdle]++
			if _, err := perSMParity(tc.arch, tc.launch, moved); err == nil {
				t.Error("oracle accepted a ledger with one lane moved from active to unit_idle")
			}
		})
	}
}

// The sweep dimensions below cover every input the ledger takes. 1, 31, 33 and 1000 are block sizes
// that are not multiples of the warp size; 255 and 1024 sit at the per-thread and per-block ceilings.
var (
	sweepTPB     = []int{1, 31, 32, 33, 100, 128, 255, 256, 1000, 1024}
	sweepRegs    = []int{1, 16, 32, 40, 41, 64, 128, 255}
	sweepSmem    = []int{0, 768, 48000}
	sweepDevices = []int{1, 84, 108, 132}
)

// sweepGrids returns the grid sizes that bracket the wave boundaries for a device of deviceSMs SMs with
// slots resident blocks; non-positive sizes are not launches and are dropped.
func sweepGrids(deviceSMs, slots int) []int {
	all := []int{1, 2, deviceSMs - 1, deviceSMs, deviceSMs + 1, slots - 1, slots, slots + 1, 3*slots + 7, 1 << 20}
	var out []int
	for _, g := range all {
		if g >= 1 {
			out = append(out, g)
		}
	}
	return out
}

// occupancyParity checks the ledger against the existing occupancy calculator: same blocks per SM,
// binding limiter, wave count and capacity, and the exact integer identity that ties the unfilled block
// slots to WaveQuantWaste.
func occupancyParity(ledger LaneLedger, occ DecodeOccupancy) error {
	slots := occ.TotalSlots
	switch {
	case ledger.BlocksPerSM != occ.BlocksPerSM:
		return fmt.Errorf("BlocksPerSM %d != occupancy %d", ledger.BlocksPerSM, occ.BlocksPerSM)
	case ledger.Limiter != occ.BindingLimiter:
		return fmt.Errorf("limiter %q != occupancy %q", ledger.Limiter, occ.BindingLimiter)
	case ledger.Waves != occ.Waves:
		return fmt.Errorf("Waves %d != occupancy %d", ledger.Waves, occ.Waves)
	case ledger.Capacity != int64(occ.Waves)*int64(occ.DeviceSMs)*int64(ledger.LanesPerSM):
		return fmt.Errorf("Capacity %d != Waves*DeviceSMs*LanesPerSM", ledger.Capacity)
	case ledger.BlockSlotsUnfilled != int64(math.Round(occ.WaveQuantWaste*float64(occ.Waves*slots))):
		return fmt.Errorf("BlockSlotsUnfilled %d != round(WaveQuantWaste*Waves*S)", ledger.BlockSlotsUnfilled)
	case occ.Waves == 1 && ledger.Cells[CellUnitIdle] != int64(occ.IdleSMs)*int64(ledger.LanesPerSM):
		return fmt.Errorf("UnitIdle %d != IdleSMs %d * lanes", ledger.Cells[CellUnitIdle], occ.IdleSMs)
	}
	return nil
}

// verifySweepPoint asserts every invariant one priceable launch must satisfy, and that a zero-block
// launch is refused with ErrLaneUnlaunchable by both the ledger and the register footprint.
func verifySweepPoint(a NVArch, l DecodeLaunch, deviceSMs, bpsm int) error {
	ledger, err := a.BuildLaneLedger(l, deviceSMs)
	footprint, fpErr := a.RegisterFootprintOf(l)
	if bpsm == 0 {
		if errors.Is(err, ErrLaneUnlaunchable) && errors.Is(fpErr, ErrLaneUnlaunchable) {
			return nil
		}
		return fmt.Errorf("zero-block launch: ledger err=%v, footprint err=%v", err, fpErr)
	}
	if err != nil || fpErr != nil {
		return fmt.Errorf("priceable launch refused: ledger err=%v, footprint err=%v", err, fpErr)
	}
	if !ledger.Conserved() {
		return fmt.Errorf("cells %v do not partition capacity %d", ledger.Cells, ledger.Capacity)
	}
	var lost int64
	for _, f := range ledger.Findings() {
		lost += f.LaneSlotWaves
	}
	if lost != ledger.Capacity-ledger.Cells[CellActive] {
		return fmt.Errorf("findings sum %d != Capacity-Active %d", lost, ledger.Capacity-ledger.Cells[CellActive])
	}
	if !footprint.Conserved() {
		return fmt.Errorf("register footprint %+v does not partition the register file", footprint)
	}
	warps := ceilDivInt(l.ThreadsPerBlock, a.WarpSize)
	if footprint.Useful != int64(bpsm*warps*l.RegsPerThread*a.WarpSize) {
		return fmt.Errorf("register Useful %d != B*w*R*WS", footprint.Useful)
	}
	var regLost int64
	for _, f := range footprint.Findings(ledger.Limiter) {
		regLost += f.RegistersPerSM
	}
	// Headroom is reported only under a non-register limiter, so under a register limiter it is the
	// one part of the partition the findings leave out.
	var unattributed int64
	if ledger.Limiter == "register" {
		unattributed = footprint.HeadroomUnused
	}
	if regLost != footprint.RegsPerSM-footprint.Useful-footprint.Remainder-unattributed {
		return fmt.Errorf("register findings sum %d != partition minus unattributed headroom", regLost)
	}
	if err := occupancyParity(ledger, a.Occupancy(l, deviceSMs)); err != nil {
		return err
	}
	_, err = perSMParity(a, l, ledger)
	return err
}

// TestLaneLedgerConservationSweep runs the invariants over every known arch crossed with block sizes,
// register counts, shared memory, device SM counts and grid sizes around the wave boundaries.
func TestLaneLedgerConservationSweep(t *testing.T) {
	points := 0
	for _, a := range KnownNVArches() {
		for _, tpb := range sweepTPB {
			for _, regs := range sweepRegs {
				for _, smem := range sweepSmem {
					l := DecodeLaunch{Kernel: "sweep", ThreadsPerBlock: tpb, RegsPerThread: regs, SmemBytesPerBlock: smem}
					bpsm, _ := a.blocksPerSM(l)
					for _, devices := range sweepDevices {
						for _, grid := range sweepGrids(devices, devices*bpsm) {
							l.GridBlocks = grid
							if err := verifySweepPoint(a, l, devices, bpsm); err != nil {
								t.Fatalf("%s tpb=%d regs=%d smem=%d D=%d G=%d: %v",
									a.Label, tpb, regs, smem, devices, grid, err)
							}
							points++
						}
					}
				}
			}
		}
	}
	if points == 0 {
		t.Fatal("sweep covered no points")
	}
}
