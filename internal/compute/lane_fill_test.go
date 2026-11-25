package compute

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

// mustLaneUnit builds a caller-input unit and fails the test if the constructor refuses it.
func mustLaneUnit(t *testing.T, family string, width int) LaneUnit {
	t.Helper()
	u, err := CallerLaneUnit(family, width, "test width")
	if err != nil {
		t.Fatalf("CallerLaneUnit(%q, %d): %v", family, width, err)
	}
	return u
}

// TestLaneFillHandDerived pins each partition against arithmetic worked out by hand from the group and
// width counts. Every expected part is written out, so the test does not reuse the implementation's formulas.
func TestLaneFillHandDerived(t *testing.T) {
	tests := []struct {
		name          string
		width         int
		launch        LaneFillLaunch
		wantGroups    int64
		wantUnitsPerG int64
		wantCapacity  LaneSlots
		wantPartial   LaneSlots
		wantTailIdle  LaneSlots
		wantScalar    LaneSlots
	}{
		{
			name:  "masked width 32 group 64 items 1000",
			width: 32, launch: LaneFillLaunch{GroupSize: 64, Items: 1000, TailMode: TailMasked},
			wantGroups: 16, wantUnitsPerG: 2, wantCapacity: 1024, wantPartial: 0, wantTailIdle: 24,
		},
		{
			name:  "masked width 64 group 32 items 1000 (every group half fills one unit)",
			width: 64, launch: LaneFillLaunch{GroupSize: 32, Items: 1000, TailMode: TailMasked},
			wantGroups: 32, wantUnitsPerG: 1, wantCapacity: 2048, wantPartial: 1024, wantTailIdle: 24,
		},
		{
			name:  "masked width 32 group 48 items 100 (group not a multiple of width)",
			width: 32, launch: LaneFillLaunch{GroupSize: 48, Items: 100, TailMode: TailMasked},
			wantGroups: 3, wantUnitsPerG: 2, wantCapacity: 192, wantPartial: 48, wantTailIdle: 44,
		},
		{
			name:  "masked width 8 group 8 items 1003",
			width: 8, launch: LaneFillLaunch{GroupSize: 8, Items: 1003, TailMode: TailMasked},
			wantGroups: 126, wantUnitsPerG: 1, wantCapacity: 1008, wantPartial: 0, wantTailIdle: 5,
		},
		{
			name:  "masked exact multiple has no tail idle",
			width: 32, launch: LaneFillLaunch{GroupSize: 64, Items: 1024, TailMode: TailMasked},
			wantGroups: 16, wantUnitsPerG: 2, wantCapacity: 1024, wantPartial: 0, wantTailIdle: 0,
		},
		{
			name:  "masked items smaller than one group",
			width: 32, launch: LaneFillLaunch{GroupSize: 64, Items: 10, TailMode: TailMasked},
			wantGroups: 1, wantUnitsPerG: 2, wantCapacity: 64, wantPartial: 0, wantTailIdle: 54,
		},
		{
			name:  "scalar width 8 group 8 items 1003",
			width: 8, launch: LaneFillLaunch{GroupSize: 8, Items: 1003, TailMode: TailScalar},
			wantGroups: 126, wantUnitsPerG: 1, wantCapacity: 1024, wantPartial: 0, wantTailIdle: 0, wantScalar: 21,
		},
		{
			name:  "scalar width 32 group 64 items 1000 (F=15, R=40)",
			width: 32, launch: LaneFillLaunch{GroupSize: 64, Items: 1000, TailMode: TailScalar},
			wantGroups: 16, wantUnitsPerG: 2, wantCapacity: 2240, wantPartial: 0, wantTailIdle: 0, wantScalar: 1240,
		},
		{
			name:  "scalar items smaller than one group",
			width: 8, launch: LaneFillLaunch{GroupSize: 8, Items: 3, TailMode: TailScalar},
			wantGroups: 1, wantUnitsPerG: 1, wantCapacity: 24, wantPartial: 0, wantTailIdle: 0, wantScalar: 21,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := mustLaneUnit(t, "wavefront", tt.width).Fill(tt.launch)
			if err != nil {
				t.Fatalf("Fill: %v", err)
			}
			if f.Groups != tt.wantGroups || f.UnitsPerGroup != tt.wantUnitsPerG {
				t.Errorf("groups/units = %d/%d, want %d/%d", f.Groups, f.UnitsPerGroup, tt.wantGroups, tt.wantUnitsPerG)
			}
			if f.Capacity != tt.wantCapacity || f.Active != LaneSlots(tt.launch.Items) {
				t.Errorf("capacity/active = %d/%d, want %d/%d", f.Capacity, f.Active, tt.wantCapacity, tt.launch.Items)
			}
			if f.PartialUnit != tt.wantPartial || f.TailIdle != tt.wantTailIdle || f.ScalarTail != tt.wantScalar {
				t.Errorf("partial/tailIdle/scalar = %d/%d/%d, want %d/%d/%d",
					f.PartialUnit, f.TailIdle, f.ScalarTail, tt.wantPartial, tt.wantTailIdle, tt.wantScalar)
			}
			if !f.Conserved() {
				t.Errorf("not conserved: %+v", f)
			}
		})
	}
}

// TestLaneFillSameLaunchWave32VsWave64 asks the wave32-versus-wave64 question for one launch: the same
// group size and item count on two caller-input units of different width.
func TestLaneFillSameLaunchWave32VsWave64(t *testing.T) {
	w32, err := CallerLaneUnit("wavefront", 32, "wave32 mode for the comparison")
	if err != nil {
		t.Fatal(err)
	}
	w64, err := CallerLaneUnit("wavefront", 64, "wave64 mode for the comparison")
	if err != nil {
		t.Fatal(err)
	}
	launch := LaneFillLaunch{Kernel: "decode_gemv", GroupSize: 32, Items: 1000, TailMode: TailMasked}

	f32, err := w32.Fill(launch)
	if err != nil {
		t.Fatal(err)
	}
	f64, err := w64.Fill(launch)
	if err != nil {
		t.Fatal(err)
	}
	if f32.Capacity != 1024 || f32.PartialUnit != 0 || f32.TailIdle != 24 {
		t.Errorf("width 32 = capacity %d partial %d tailIdle %d, want 1024/0/24", f32.Capacity, f32.PartialUnit, f32.TailIdle)
	}
	if f64.Capacity != 2048 || f64.PartialUnit != 1024 || f64.TailIdle != 24 {
		t.Errorf("width 64 = capacity %d partial %d tailIdle %d, want 2048/1024/24", f64.Capacity, f64.PartialUnit, f64.TailIdle)
	}
	if !f32.Conserved() || !f64.Conserved() {
		t.Fatalf("partition not conserved: w32=%+v w64=%+v", f32, f64)
	}

	got := f64.Findings()
	if len(got) != 2 || got[0].Name != "partial_unit" || got[0].LaneSlots != 1024 || got[1].Name != "tail_idle" {
		t.Fatalf("width 64 findings = %+v, want partial_unit 1024 then tail_idle", got)
	}
	if got[0].ShareOfCapacity != 0.5 {
		t.Errorf("partial_unit share = %v, want 0.5", got[0].ShareOfCapacity)
	}
	if only := f32.Findings(); len(only) != 1 || only[0].Name != "tail_idle" {
		t.Errorf("width 32 findings = %+v, want only tail_idle", only)
	}
	for _, want := range []string{"family=wavefront", "width=64", "group_size=32", "items=1000"} {
		if !strings.Contains(got[0].Trigger, want) {
			t.Errorf("trigger %q missing %q", got[0].Trigger, want)
		}
	}
}

// TestLaneFillConservationSweep checks the identity Active+PartialUnit+TailIdle+ScalarTail==Capacity and
// the group and item counts over a grid of widths, group sizes and item counts, in both tail modes.
func TestLaneFillConservationSweep(t *testing.T) {
	widths := []int{1, 2, 4, 8, 16, 32, 64, 96}
	groups := []int{1, 3, 8, 32, 48, 64, 96, 128}
	items := []int64{1, 2, 7, 31, 32, 33, 100, 1003, 4096, 1<<20 + 7}
	for _, w := range widths {
		u := mustLaneUnit(t, "sweep", w)
		for _, g := range groups {
			for _, n := range items {
				for _, mode := range []TailMode{TailMasked, TailScalar} {
					if mode == TailScalar && g%w != 0 {
						continue
					}
					f, err := u.Fill(LaneFillLaunch{GroupSize: g, Items: n, TailMode: mode})
					if err != nil {
						t.Fatalf("w=%d g=%d n=%d %s: %v", w, g, n, mode, err)
					}
					if !f.Conserved() {
						t.Errorf("w=%d g=%d n=%d %s not conserved: %+v", w, g, n, mode, f)
					}
					sum := f.Active + f.PartialUnit + f.TailIdle + f.ScalarTail
					if sum != f.Capacity {
						t.Errorf("w=%d g=%d n=%d %s sum %d != capacity %d", w, g, n, mode, sum, f.Capacity)
					}
					if wantGroups := (n + int64(g) - 1) / int64(g); f.Groups != wantGroups {
						t.Errorf("w=%d g=%d n=%d groups = %d, want %d", w, g, n, f.Groups, wantGroups)
					}
					if f.Active != LaneSlots(n) {
						t.Errorf("w=%d g=%d n=%d active = %d", w, g, n, f.Active)
					}
				}
			}
		}
	}
}

// TestLaneFillRefusals asserts each refusal returns its sentinel, so a caller can branch on the cause.
func TestLaneFillRefusals(t *testing.T) {
	good := LaneUnit{Family: "wavefront", Width: 32, Provenance: "caller input: test"}
	tests := []struct {
		name   string
		unit   LaneUnit
		launch LaneFillLaunch
		want   error
	}{
		{"zero width", LaneUnit{Family: "f", Width: 0, Provenance: "p"}, LaneFillLaunch{GroupSize: 8, Items: 8, TailMode: TailMasked}, ErrLaneFillWidth},
		{"negative width", LaneUnit{Family: "f", Width: -4, Provenance: "p"}, LaneFillLaunch{GroupSize: 8, Items: 8, TailMode: TailMasked}, ErrLaneFillWidth},
		{"empty family", LaneUnit{Width: 32, Provenance: "p"}, LaneFillLaunch{GroupSize: 32, Items: 8, TailMode: TailMasked}, ErrLaneFillFamily},
		{"empty provenance", LaneUnit{Family: "f", Width: 32}, LaneFillLaunch{GroupSize: 32, Items: 8, TailMode: TailMasked}, ErrLaneFillProvenance},
		{"zero group size", good, LaneFillLaunch{GroupSize: 0, Items: 8, TailMode: TailMasked}, ErrLaneFillGroupSize},
		{"zero items", good, LaneFillLaunch{GroupSize: 32, Items: 0, TailMode: TailMasked}, ErrLaneFillItems},
		{"negative items", good, LaneFillLaunch{GroupSize: 32, Items: -1, TailMode: TailMasked}, ErrLaneFillItems},
		{"empty tail mode", good, LaneFillLaunch{GroupSize: 32, Items: 8}, ErrLaneFillTailMode},
		{"unknown tail mode", good, LaneFillLaunch{GroupSize: 32, Items: 8, TailMode: "wrap"}, ErrLaneFillTailMode},
		{"scalar with group not multiple of width", good, LaneFillLaunch{GroupSize: 48, Items: 100, TailMode: TailScalar}, ErrLaneFillScalarGroup},
		{"masked overflow", LaneUnit{Family: "f", Width: 64, Provenance: "p"}, LaneFillLaunch{GroupSize: 1, Items: math.MaxInt64, TailMode: TailMasked}, ErrLaneFillOverflow},
		{"scalar overflow", LaneUnit{Family: "f", Width: 64, Provenance: "p"}, LaneFillLaunch{GroupSize: 64, Items: math.MaxInt64, TailMode: TailScalar}, ErrLaneFillOverflow},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := tt.unit.Fill(tt.launch)
			if !errors.Is(err, tt.want) {
				t.Fatalf("Fill error = %v, want %v", err, tt.want)
			}
			if f.Conserved() {
				t.Errorf("refused fill reported conserved: %+v", f)
			}
		})
	}
}

// TestLaneUnitCallerRefusals asserts CallerLaneUnit refuses the inputs it cannot label.
func TestLaneUnitCallerRefusals(t *testing.T) {
	tests := []struct {
		name   string
		family string
		width  int
		note   string
		want   error
	}{
		{"empty family", "", 8, "cpu simd", ErrLaneFillFamily},
		{"zero width", "cpu_simd", 0, "cpu simd", ErrLaneFillWidth},
		{"empty note", "cpu_simd", 8, "", ErrLaneUnitNote},
		{"whitespace-only note", "cpu_simd", 8, "   \t", ErrLaneUnitNote},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			u, err := CallerLaneUnit(tt.family, tt.width, tt.note)
			if !errors.Is(err, tt.want) {
				t.Fatalf("CallerLaneUnit error = %v, want %v", err, tt.want)
			}
			if u != (LaneUnit{}) {
				t.Errorf("refused unit is not zero: %+v", u)
			}
		})
	}
}

// TestLaneUnitCallerProvenance asserts a caller unit records its note as Provenance.
func TestLaneUnitCallerProvenance(t *testing.T) {
	u, err := CallerLaneUnit("cpu_simd", 8, "AVX2 float lanes, supplied by the benchmark config")
	if err != nil {
		t.Fatal(err)
	}
	want := "caller input: AVX2 float lanes, supplied by the benchmark config"
	if u.Provenance != want || u.Family != "cpu_simd" || u.Width != 8 {
		t.Errorf("unit = %+v, want family cpu_simd width 8 provenance %q", u, want)
	}
}

// TestLaneUnitRepoConstants asserts each repo-cited unit reads its constant by identifier: the width is
// compared against the identifier itself, not a literal, and the provenance names that identifier.
func TestLaneUnitRepoConstants(t *testing.T) {
	tests := []struct {
		name       string
		unit       LaneUnit
		wantWidth  int
		wantFamily string
		identifier string
	}{
		{"StrixHaloWavefrontSize", StrixHaloWavefrontLaneUnit(), StrixHaloWavefrontSize, "wavefront", "StrixHaloWavefrontSize"},
		{"Wave32WavefrontSize", Wave32KDALaneUnit(), Wave32WavefrontSize, "wavefront", "Wave32WavefrontSize"},
		{"vulkanWave32SubgroupSz", VulkanWave32SubgroupLaneUnit(), vulkanWave32SubgroupSz, "vulkan_subgroup", "vulkanWave32SubgroupSz"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.unit.Width != tt.wantWidth || tt.unit.Family != tt.wantFamily {
				t.Errorf("unit = %+v, want family %q width %d", tt.unit, tt.wantFamily, tt.wantWidth)
			}
			want := "repo constant " + tt.identifier + " (code assumption, not a vendor citation)"
			if tt.unit.Provenance != want {
				t.Errorf("provenance = %q, want %q", tt.unit.Provenance, want)
			}
			if _, err := tt.unit.Fill(LaneFillLaunch{GroupSize: tt.wantWidth, Items: 1, TailMode: TailMasked}); err != nil {
				t.Errorf("repo unit does not fill: %v", err)
			}
		})
	}
}

// TestLaneFillFindingsOrder asserts findings are non-zero only, sorted by loss then name, and that a
// fill with no loss reports nothing.
func TestLaneFillFindingsOrder(t *testing.T) {
	u := mustLaneUnit(t, "wavefront", 8)
	tests := []struct {
		name   string
		launch LaneFillLaunch
		want   []string
	}{
		{"no loss", LaneFillLaunch{GroupSize: 8, Items: 1024, TailMode: TailMasked}, nil},
		{"tail idle only", LaneFillLaunch{GroupSize: 8, Items: 1003, TailMode: TailMasked}, []string{"tail_idle"}},
		{"scalar tail only", LaneFillLaunch{GroupSize: 8, Items: 1003, TailMode: TailScalar}, []string{"scalar_tail"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := u.Fill(tt.launch)
			if err != nil {
				t.Fatal(err)
			}
			var names []string
			for _, finding := range f.Findings() {
				names = append(names, finding.Name)
				if finding.LaneSlots <= 0 || finding.Unit != LaneFillUnit {
					t.Errorf("finding %+v must be non-zero and in %s", finding, LaneFillUnit)
				}
			}
			if !reflect.DeepEqual(names, tt.want) {
				t.Errorf("findings = %v, want %v", names, tt.want)
			}
		})
	}
}

// TestLaneFillFindingsTieBreak asserts equal losses are ordered by name, after the larger loss.
func TestLaneFillFindingsTieBreak(t *testing.T) {
	f := LaneFill{Family: "t", Width: 4, GroupSize: 4, Items: 8, Capacity: 10,
		Active: 4, PartialUnit: 2, TailIdle: 2, ScalarTail: 2}
	got := f.Findings()
	var names []string
	for _, finding := range got {
		names = append(names, finding.Name)
	}
	want := []string{"partial_unit", "scalar_tail", "tail_idle"}
	if !reflect.DeepEqual(names, want) {
		t.Errorf("findings order = %v, want %v", names, want)
	}
}

// TestLaneFillConservedRejectsZeroAndTampered asserts the zero value and a partition whose parts do not
// sum to Capacity are not conserved.
func TestLaneFillConservedRejectsZeroAndTampered(t *testing.T) {
	if (LaneFill{}).Conserved() {
		t.Error("zero LaneFill reported conserved")
	}
	f, err := mustLaneUnit(t, "wavefront", 32).Fill(LaneFillLaunch{GroupSize: 64, Items: 1000, TailMode: TailMasked})
	if err != nil {
		t.Fatal(err)
	}
	f.TailIdle++
	if f.Conserved() {
		t.Error("partition with a part one too large reported conserved")
	}
	f.TailIdle = -1
	if f.Conserved() {
		t.Error("partition with a negative part reported conserved")
	}
}

// TestLaneFillJSONRoundTrip asserts the JSON tags carry every exported field through a round trip.
func TestLaneFillJSONRoundTrip(t *testing.T) {
	f, err := mustLaneUnit(t, "wavefront", 64).Fill(LaneFillLaunch{Kernel: "k", GroupSize: 32, Items: 1000, TailMode: TailMasked})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(f)
	if err != nil {
		t.Fatal(err)
	}
	var back LaneFill
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back != f {
		t.Errorf("round trip = %+v, want %+v", back, f)
	}
	for _, key := range []string{`"partial_unit"`, `"tail_idle"`, `"scalar_tail"`, `"capacity"`, `"tail_mode"`} {
		if !strings.Contains(string(data), key) {
			t.Errorf("json %s missing key %s", data, key)
		}
	}
}
