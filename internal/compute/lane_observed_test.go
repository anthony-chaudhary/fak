package compute

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

// observedTestInputs models kernel_a, kernel_b and kernel_c on sm_80 with the A100 SM count. Each kernel has 128
// threads at 32 registers and no dynamic shared memory, so each SM holds 16 resident blocks.
func observedTestInputs(t *testing.T) ObservedInputs {
	t.Helper()
	sm80, ok := LookupNVArch(SM80)
	if !ok {
		t.Fatal("sm_80 row missing")
	}
	return ObservedInputs{
		Arch:      sm80,
		DeviceSMs: A100SMs,
		Regs:      map[string]int{"kernel_a": 32, "kernel_b": 32, "kernel_c": 32},
		Threads:   map[string]int{"kernel_a": 128, "kernel_b": 128, "kernel_c": 128},
		SmemBytes: map[string]int{"kernel_a": 0, "kernel_b": 0, "kernel_c": 0},
	}
}

func observedFixture(t *testing.T, name string) []ObservedLaunch {
	t.Helper()
	f, err := os.Open("testdata/lane_observed/" + name)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rows, err := ParseObservedProfile(f)
	if err != nil {
		t.Fatalf("parse %s: %v", name, err)
	}
	return rows
}

func observedRowNamed(t *testing.T, r ObservedReport, kernel string) ObservedRow {
	t.Helper()
	for _, row := range r.Rows {
		if row.Kernel == kernel {
			return row
		}
	}
	t.Fatalf("no row named %q in %+v", kernel, r.Rows)
	return ObservedRow{}
}

func observedClose(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// TestObservedHandDerivedTwoRows checks the join against arithmetic worked out by hand.
//
// sm_80, 108 SMs, 128 threads at 32 registers: the register ceiling gives 16 blocks per SM and the warp ceiling
// also gives 16, so the register limiter binds. One SM holds 16*128 = 2048 lanes = 16 blocks' worth of lanes.
//
// kernel_a, grid 108: one block per SM, one wave, capacity 1*108*2048 = 221184 lane-slot-waves. Active cells
// 108*1*128 = 13824 (share 1/16). Resident_unfilled 108*15*128 = 207360. Time 1000 ns splits 13824*1000/221184 =
// 62.5 and 207360*1000/221184 = 937.5. The tie goes to the lower-numbered cell (active), so active = 63 and
// resident_unfilled = 937, and time_weighted_idle_ns = 1000 - 63 = 937.
//
// kernel_b, grid 1728: 16 full SMs' worth per SM, every lane active, so 3000 ns is all active and idle is 0.
func TestObservedHandDerivedTwoRows(t *testing.T) {
	rows := observedFixture(t, "two_launches.json")
	report, err := JoinObserved(rows, observedTestInputs(t))
	if err != nil {
		t.Fatal(err)
	}
	if report.Caveat != ObservedCaveat || !strings.Contains(report.Caveat, "not measured occupancy") {
		t.Fatalf("caveat missing or wrong: %q", report.Caveat)
	}
	if len(report.Rows) != 2 || report.Rows[0].Kernel != "kernel_a" || report.Rows[1].Kernel != "kernel_b" {
		t.Fatalf("rank order = %+v, want kernel_a (937 ns idle) then kernel_b (0 ns idle)", report.Rows)
	}

	a := report.Rows[0]
	if a.Model == nil || a.Reason != "" {
		t.Fatalf("kernel_a must be modelled, got reason %q", a.Reason)
	}
	if a.TimeShare != 0.25 {
		t.Errorf("kernel_a time share = %v, want 1000/4000", a.TimeShare)
	}
	if a.Model.BlocksPerSM != 16 || a.Model.Limiter != "register" || a.Model.Waves != 1 {
		t.Errorf("kernel_a blocks/limiter/waves = %d/%s/%d, want 16/register/1", a.Model.BlocksPerSM, a.Model.Limiter, a.Model.Waves)
	}
	if a.Model.LaneWaveCapacity != 221184 {
		t.Errorf("kernel_a capacity = %d, want 221184", a.Model.LaneWaveCapacity)
	}
	if a.Model.ActiveShare != 0.0625 || a.Model.CellShare["resident_unfilled"] != 0.9375 {
		t.Errorf("kernel_a shares active=%v resident=%v, want 1/16 and 15/16",
			a.Model.ActiveShare, a.Model.CellShare["resident_unfilled"])
	}
	if a.Model.CellTimeNS["active"] != 63 || a.Model.CellTimeNS["resident_unfilled"] != 937 {
		t.Errorf("kernel_a cell time active=%d resident=%d, want 63 and 937",
			a.Model.CellTimeNS["active"], a.Model.CellTimeNS["resident_unfilled"])
	}
	if a.Model.TimeWeightedIdleNS != 937 {
		t.Errorf("kernel_a time_weighted_idle_ns = %d, want 937", a.Model.TimeWeightedIdleNS)
	}

	b := report.Rows[1]
	if b.Model == nil || b.Model.ActiveShare != 1 || b.Model.TimeWeightedIdleNS != 0 || b.Model.CellTimeNS["active"] != 3000 {
		t.Errorf("kernel_b = %+v, want fully active, idle 0, active time 3000", b.Model)
	}
	if b.TimeShare != 0.75 {
		t.Errorf("kernel_b time share = %v, want 0.75", b.TimeShare)
	}

	tot := report.Totals
	if tot.TotalNS != 4000 || tot.ModelledNS != 4000 || tot.UnmodelledNS != 0 || tot.ModelledShare != 1 {
		t.Errorf("totals = %+v, want 4000 modelled and 0 unmodelled", tot)
	}
	if tot.TimeWeightedIdleNS != 937 || tot.ModelledRows != 2 || tot.UnmodelledRows != 0 {
		t.Errorf("totals idle/rows = %d/%d/%d, want 937 and 2 modelled rows", tot.TimeWeightedIdleNS, tot.ModelledRows, tot.UnmodelledRows)
	}
	if tot.CellTimeNS["active"] != 3063 || tot.CellTimeNS["resident_unfilled"] != 937 {
		t.Errorf("totals cell time active=%d resident=%d, want 3063 and 937",
			tot.CellTimeNS["active"], tot.CellTimeNS["resident_unfilled"])
	}
}

// TestObservedExactBigIntegerTime checks that a time whose product with a cell count overflows int64 still splits
// exactly. For kernel_a, 2^62*13824/221184 = 2^58 active and 2^62*207360/221184 = 15*2^58 resident_unfilled.
func TestObservedExactBigIntegerTime(t *testing.T) {
	rows := []ObservedLaunch{{Kernel: "kernel_a", GridX: 108, GridY: 1, Calls: 1, TotalNS: 1 << 62}}
	report, err := JoinObserved(rows, observedTestInputs(t))
	if err != nil {
		t.Fatal(err)
	}
	m := report.Rows[0].Model
	if m == nil {
		t.Fatalf("row unmodelled: %s", report.Rows[0].Reason)
	}
	if m.CellTimeNS["active"] != 1<<58 || m.CellTimeNS["resident_unfilled"] != 15<<58 {
		t.Errorf("cell time active=%d resident=%d, want 2^58 and 15*2^58", m.CellTimeNS["active"], m.CellTimeNS["resident_unfilled"])
	}
	if m.TimeWeightedIdleNS != 15<<58 {
		t.Errorf("idle = %d, want 15*2^58", m.TimeWeightedIdleNS)
	}
}

// TestObservedAccountingIdentitySweep checks that modelled plus unmodelled time equals the profile total over a
// sweep of grids, thread counts, register counts and every unmodelled reason, and that each modelled row's cells
// sum to its own time.
func TestObservedAccountingIdentitySweep(t *testing.T) {
	in := observedTestInputs(t)
	in.Threads["kernel_b"], in.Regs["kernel_b"], in.SmemBytes = 100, 64, map[string]int{"kernel_a": 0, "kernel_b": 4096, "kernel_c": 0, "kernel_d": 0, "kernel_e": 200 * 1024}
	in.Threads["kernel_d"], in.Regs["kernel_d"] = 1025, 32
	in.Threads["kernel_e"], in.Regs["kernel_e"] = 128, 32
	delete(in.Regs, "kernel_c")

	grids := []int{1, 2, 107, 108, 109, 1727, 1728, 1729, 5000}
	var rows []ObservedLaunch
	var total int64
	n := int64(0)
	for _, kernel := range []string{"kernel_a", "kernel_b", "kernel_c", "kernel_d", "kernel_e"} {
		for _, gx := range grids {
			for _, gy := range []int{1, 3} {
				n++
				ns := n*7919%100003 + 1
				rows = append(rows, ObservedLaunch{Kernel: kernel, GridX: gx, GridY: gy, Calls: n, TotalNS: ns})
				total += ns
			}
		}
	}
	report, err := JoinObserved(rows, in)
	if err != nil {
		t.Fatal(err)
	}
	tot := report.Totals
	if tot.TotalNS != total || tot.ModelledNS+tot.UnmodelledNS != total {
		t.Fatalf("identity broken: total=%d modelled=%d unmodelled=%d sum=%d",
			total, tot.ModelledNS, tot.UnmodelledNS, tot.ModelledNS+tot.UnmodelledNS)
	}
	var reasonSum, shareSum int64
	for _, ns := range tot.UnmodelledNSByReason {
		reasonSum += ns
	}
	if reasonSum != tot.UnmodelledNS {
		t.Errorf("reason split %d != unmodelled %d", reasonSum, tot.UnmodelledNS)
	}
	var shares float64
	for _, row := range report.Rows {
		shares += row.TimeShare
		shareSum += row.TotalNS
		if row.Model == nil {
			if row.Reason == "" {
				t.Errorf("unmodelled row %s %dx%d has no reason", row.Kernel, row.GridX, row.GridY)
			}
			continue
		}
		var cells int64
		for _, c := range AllLaneCells() {
			cells += row.Model.CellTimeNS[c.String()]
		}
		if cells != row.TotalNS {
			t.Errorf("%s %dx%d cells sum to %d, want %d", row.Kernel, row.GridX, row.GridY, cells, row.TotalNS)
		}
		if row.Model.TimeWeightedIdleNS != row.TotalNS-row.Model.CellTimeNS["active"] {
			t.Errorf("%s %dx%d idle %d inconsistent with cells", row.Kernel, row.GridX, row.GridY, row.Model.TimeWeightedIdleNS)
		}
	}
	if shareSum != total || !observedClose(shares, 1) {
		t.Errorf("row times sum to %d with shares %v, want %d and 1", shareSum, shares, total)
	}
	wantReason := map[string]string{"kernel_c": ObservedReasonRegsUnmeasured, "kernel_d": ObservedReasonUnlaunchable, "kernel_e": ObservedReasonUnlaunchable}
	for _, row := range report.Rows {
		if want, ok := wantReason[row.Kernel]; ok && row.Reason != want {
			t.Errorf("%s reason = %q, want %q", row.Kernel, row.Reason, want)
		}
	}
	if report.Totals.ModelledRows == 0 || report.Totals.UnmodelledRows == 0 {
		t.Fatalf("sweep must mix modelled and unmodelled rows: %+v", report.Totals)
	}
}

// TestObservedUnmodelledReasons checks each unmodelled reason on a single row. Every case keeps the row in the
// report with its full time share, so the identity still holds.
func TestObservedUnmodelledReasons(t *testing.T) {
	tests := []struct {
		name   string
		grid   [2]int
		edit   func(in *ObservedInputs)
		reason string
		errTxt string
	}{
		{name: "regs absent", grid: [2]int{108, 1}, edit: func(in *ObservedInputs) { delete(in.Regs, "kernel_a") }, reason: ObservedReasonRegsUnmeasured},
		{name: "regs zero", grid: [2]int{108, 1}, edit: func(in *ObservedInputs) { in.Regs["kernel_a"] = 0 }, reason: ObservedReasonRegsUnmeasured},
		{name: "threads absent", grid: [2]int{108, 1}, edit: func(in *ObservedInputs) { delete(in.Threads, "kernel_a") }, reason: ObservedReasonThreadsUnjudged},
		{name: "threads zero", grid: [2]int{108, 1}, edit: func(in *ObservedInputs) { in.Threads["kernel_a"] = 0 }, reason: ObservedReasonThreadsUnjudged},
		{name: "smem absent", grid: [2]int{108, 1}, edit: func(in *ObservedInputs) { delete(in.SmemBytes, "kernel_a") }, reason: ObservedReasonSmemUnjudged},
		{name: "threads above block ceiling", grid: [2]int{108, 1}, edit: func(in *ObservedInputs) { in.Threads["kernel_a"] = 1025 }, reason: ObservedReasonUnlaunchable, errTxt: ErrLaneThreads.Error()},
		{name: "regs above register ceiling", grid: [2]int{108, 1}, edit: func(in *ObservedInputs) { in.Regs["kernel_a"] = 256 }, reason: ObservedReasonUnlaunchable, errTxt: ErrLaneRegsOverLimit.Error()},
		{name: "no block fits shared memory", grid: [2]int{108, 1}, edit: func(in *ObservedInputs) { in.SmemBytes["kernel_a"] = 200 * 1024 }, reason: ObservedReasonUnlaunchable, errTxt: ErrLaneUnlaunchable.Error()},
		{name: "grid product overflows", grid: [2]int{1 << 40, 1 << 40}, edit: func(*ObservedInputs) {}, reason: ObservedReasonOverflow, errTxt: ErrLaneOverflow.Error()},
		{name: "capacity overflows", grid: [2]int{1 << 60, 1}, edit: func(*ObservedInputs) {}, reason: ObservedReasonOverflow, errTxt: ErrLaneOverflow.Error()},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := observedTestInputs(t)
			tc.edit(&in)
			rows := []ObservedLaunch{{Kernel: "kernel_a", GridX: tc.grid[0], GridY: tc.grid[1], Calls: 3, TotalNS: 500}}
			report, err := JoinObserved(rows, in)
			if err != nil {
				t.Fatal(err)
			}
			row := report.Rows[0]
			if row.Model != nil || row.Reason != tc.reason {
				t.Fatalf("row = %+v, want unmodelled reason %q", row, tc.reason)
			}
			if tc.errTxt != "" && !strings.Contains(row.Error, tc.errTxt) {
				t.Errorf("error %q does not carry %q", row.Error, tc.errTxt)
			}
			if tc.errTxt == "" && row.Error != "" {
				t.Errorf("input-missing row carries an error %q", row.Error)
			}
			if row.TimeShare != 1 || report.Totals.UnmodelledNS != 500 || report.Totals.ModelledNS != 0 {
				t.Errorf("accounting = share %v totals %+v, want the whole 500 ns unmodelled", row.TimeShare, report.Totals)
			}
			if report.Totals.UnmodelledNSByReason[tc.reason] != 500 {
				t.Errorf("reason %q carries %d ns, want 500", tc.reason, report.Totals.UnmodelledNSByReason[tc.reason])
			}
		})
	}
}

// TestObservedMixedFixtureAccounting reads a fixture with an unknown field and an unmodelled kernel, and checks
// that modelled rank first and that the two parts sum to the total.
func TestObservedMixedFixtureAccounting(t *testing.T) {
	in := observedTestInputs(t)
	delete(in.Regs, "kernel_c")
	report, err := JoinObserved(observedFixture(t, "mixed_unmodelled.json"), in)
	if err != nil {
		t.Fatal(err)
	}
	tot := report.Totals
	if tot.TotalNS != 1500 || tot.ModelledNS != 1000 || tot.UnmodelledNS != 500 {
		t.Fatalf("totals = %+v, want 1500 = 1000 modelled + 500 unmodelled", tot)
	}
	if tot.UnmodelledNSByReason[ObservedReasonRegsUnmeasured] != 500 || tot.UnmodelledNSByReason[ObservedReasonOverflow] != 0 {
		t.Errorf("reason split = %+v", tot.UnmodelledNSByReason)
	}
	if report.Rows[0].Kernel != "kernel_a" || report.Rows[0].Model == nil {
		t.Errorf("modelled row must rank first, got %+v", report.Rows[0])
	}
	if c := observedRowNamed(t, report, "kernel_c"); c.Model != nil || !observedClose(c.TimeShare, 1.0/3.0) {
		t.Errorf("kernel_c = %+v, want unmodelled with share 1/3", c)
	}
}

// TestObservedRefusals checks each whole-join refusal with errors.Is.
func TestObservedRefusals(t *testing.T) {
	rows := observedFixture(t, "two_launches.json")
	tests := []struct {
		name string
		rows []ObservedLaunch
		edit func(in *ObservedInputs)
		want error
	}{
		{name: "empty input", rows: nil, want: ErrObservedEmpty},
		{name: "zero arch", rows: rows, edit: func(in *ObservedInputs) { in.Arch = NVArch{} }, want: ErrObservedArch},
		{name: "zero SM count", rows: rows, edit: func(in *ObservedInputs) { in.DeviceSMs = 0 }, want: ErrObservedDeviceSMs},
		{name: "negative regs", rows: rows, edit: func(in *ObservedInputs) { in.Regs["kernel_a"] = -1 }, want: ErrObservedNegativeInput},
		{name: "negative smem", rows: rows, edit: func(in *ObservedInputs) { in.SmemBytes["kernel_b"] = -4 }, want: ErrObservedNegativeInput},
		{name: "template-argument key", rows: rows, edit: func(in *ObservedInputs) { in.Threads["kernel_a<false>"] = 128 }, want: ErrObservedKeyNotBase},
		{name: "namespaced key", rows: rows, edit: func(in *ObservedInputs) { in.Regs["ns::kernel_a"] = 32 }, want: ErrObservedKeyNotBase},
		{name: "negative calls", rows: []ObservedLaunch{{Kernel: "kernel_a", GridX: 108, GridY: 1, Calls: -1, TotalNS: 1}}, want: ErrObservedRow},
		{name: "zero grid", rows: []ObservedLaunch{{Kernel: "kernel_a", GridX: 0, GridY: 1, Calls: 1, TotalNS: 1}}, want: ErrObservedRow},
		{name: "empty kernel name", rows: []ObservedLaunch{{Kernel: "", GridX: 1, GridY: 1, Calls: 1, TotalNS: 1}}, want: ErrObservedRow},
		{name: "duplicate shape", rows: []ObservedLaunch{
			{Kernel: "kernel_a", GridX: 108, GridY: 1, Calls: 1, TotalNS: 1},
			{Kernel: "kernel_a", GridX: 108, GridY: 1, Calls: 2, TotalNS: 2},
		}, want: ErrObservedDuplicate},
		{name: "total overflows int64", rows: []ObservedLaunch{
			{Kernel: "kernel_a", GridX: 108, GridY: 1, Calls: 1, TotalNS: math.MaxInt64},
			{Kernel: "kernel_b", GridX: 1728, GridY: 1, Calls: 1, TotalNS: math.MaxInt64},
		}, want: ErrObservedOverflow},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			in := observedTestInputs(t)
			if tc.edit != nil {
				tc.edit(&in)
			}
			if _, err := JoinObserved(tc.rows, in); !errors.Is(err, tc.want) {
				t.Fatalf("JoinObserved error = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestObservedParseRefusals checks the reader's refusals and the documented array form.
func TestObservedParseRefusals(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  error
	}{
		{name: "empty stream", input: "", want: nil},
		{name: "empty array", input: "[]", want: ErrObservedEmpty},
		{name: "null", input: "null", want: ErrObservedEmpty},
		{name: "missing total_ns", input: `[{"name":"kernel_a","gx":1,"gy":1,"calls":1}]`, want: ErrObservedRow},
		{name: "null field counts as missing", input: `[{"name":"kernel_a","gx":1,"gy":1,"calls":null,"total_ns":1}]`, want: ErrObservedRow},
		{name: "fractional grid", input: `[{"name":"kernel_a","gx":1.5,"gy":1,"calls":1,"total_ns":1}]`, want: nil},
		{name: "trailing array", input: `[{"name":"kernel_a","gx":1,"gy":1,"calls":1,"total_ns":1}] []`, want: nil},
		{name: "object not array", input: `{"name":"kernel_a"}`, want: nil},
		{name: "duplicate row", input: `[{"name":"k","gx":1,"gy":1,"calls":1,"total_ns":1},{"name":"k","gx":1,"gy":1,"calls":1,"total_ns":1}]`, want: ErrObservedDuplicate},
		{name: "negative time", input: `[{"name":"kernel_a","gx":1,"gy":1,"calls":1,"total_ns":-1}]`, want: ErrObservedRow},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseObservedProfile(strings.NewReader(tc.input))
			if err == nil {
				t.Fatal("parse accepted input it must refuse")
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Errorf("error = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestObservedJSONRoundTrip writes the fixture rows as JSON, parses them back, and checks the report JSON
// round-trips to an equal value.
func TestObservedJSONRoundTrip(t *testing.T) {
	rows := append(observedFixture(t, "two_launches.json"), observedFixture(t, "mixed_unmodelled.json")...)
	raw, err := json.Marshal(rows[:2])
	if err != nil {
		t.Fatal(err)
	}
	back, err := ParseObservedProfile(bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("parse round trip: %v", err)
	}
	if !reflect.DeepEqual(back, rows[:2]) {
		t.Fatalf("round trip = %+v, want %+v", back, rows[:2])
	}

	in := observedTestInputs(t)
	report, err := JoinObserved(back, in)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	var decoded ObservedReport
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded, report) {
		t.Fatalf("report JSON round trip differs")
	}
	if !strings.Contains(string(encoded), `"time_weighted_idle_ns":937`) || !strings.Contains(string(encoded), `"caveat"`) {
		t.Errorf("report JSON lacks the ranking key or caveat: %s", encoded)
	}
}

// TestObservedDeterministicOrdering checks that input order does not change the report, and that a tie on idle
// time breaks by TotalNS and then by grid.
func TestObservedDeterministicOrdering(t *testing.T) {
	rows := []ObservedLaunch{
		{Kernel: "kernel_b", GridX: 1728, GridY: 1, Calls: 1, TotalNS: 300},
		{Kernel: "kernel_b", GridX: 1728, GridY: 3, Calls: 1, TotalNS: 200},
		{Kernel: "kernel_a", GridX: 108, GridY: 1, Calls: 10, TotalNS: 1000},
		{Kernel: "kernel_c", GridX: 108, GridY: 1, Calls: 2, TotalNS: 500},
	}
	in := observedTestInputs(t)
	delete(in.Regs, "kernel_c")
	forward, err := JoinObserved(rows, in)
	if err != nil {
		t.Fatal(err)
	}
	reversed := make([]ObservedLaunch, len(rows))
	for i, r := range rows {
		reversed[len(rows)-1-i] = r
	}
	backward, err := JoinObserved(reversed, in)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(forward)
	b, _ := json.Marshal(backward)
	if !bytes.Equal(a, b) {
		t.Fatal("report depends on input order")
	}
	want := []string{"kernel_a", "kernel_b", "kernel_b", "kernel_c"}
	for i, name := range want {
		if forward.Rows[i].Kernel != name {
			t.Errorf("rank %d = %s, want %s", i, forward.Rows[i].Kernel, name)
		}
	}
	if forward.Rows[1].TotalNS != 300 || forward.Rows[2].TotalNS != 200 {
		t.Errorf("tie on idle time must break by TotalNS descending, got %d then %d",
			forward.Rows[1].TotalNS, forward.Rows[2].TotalNS)
	}
}

// TestObservedTopN checks the ranked prefix, its refusal of a non-positive n, and that unmodelled rows never
// appear in it.
func TestObservedTopN(t *testing.T) {
	in := observedTestInputs(t)
	delete(in.Regs, "kernel_c")
	report, err := JoinObserved(observedFixture(t, "mixed_unmodelled.json"), in)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := report.Top(0); !errors.Is(err, ErrObservedTopN) {
		t.Errorf("Top(0) error = %v, want ErrObservedTopN", err)
	}
	if _, err := report.Top(-1); !errors.Is(err, ErrObservedTopN) {
		t.Errorf("Top(-1) error = %v, want ErrObservedTopN", err)
	}
	top, err := report.Top(5)
	if err != nil {
		t.Fatal(err)
	}
	if len(top) != 1 || top[0].Kernel != "kernel_a" {
		t.Errorf("Top(5) = %+v, want only the modelled kernel_a", top)
	}
}

// TestObservedRankingIsIdleTime pins the ranking key to the active-cell complement across the fixture.
func TestObservedRankingIsIdleTime(t *testing.T) {
	report, err := JoinObserved(observedFixture(t, "two_launches.json"), observedTestInputs(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range report.Rows {
		if row.Model.TimeWeightedIdleNS != row.TotalNS-row.Model.CellTimeNS["active"] {
			t.Errorf("%s: idle %d is not total %d minus active %d",
				row.Kernel, row.Model.TimeWeightedIdleNS, row.TotalNS, row.Model.CellTimeNS["active"])
		}
	}
	if report.Rows[0].Model.TimeWeightedIdleNS < report.Rows[1].Model.TimeWeightedIdleNS {
		t.Errorf("rows are not ranked by idle time descending")
	}
}

func TestObservedLedgerReasonKeepsInvariantApart(t *testing.T) {
	if got := observedLedgerReason(fmt.Errorf("%w: kernel", ErrObservedInvariant)); got != ObservedReasonInvariant {
		t.Errorf("invariant failure labelled %q, want %q", got, ObservedReasonInvariant)
	}
	if got := observedLedgerReason(fmt.Errorf("%w: 256", ErrLaneRegsOverLimit)); got != ObservedReasonUnlaunchable {
		t.Errorf("registers over the ceiling labelled %q, want %q", got, ObservedReasonUnlaunchable)
	}
}
