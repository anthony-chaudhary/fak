package memgate

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestParseLinux(t *testing.T) {
	mem := ParseLinux("MemTotal: 36000000 kB\nMemFree: 4000000 kB\nMemAvailable: 28000000 kB\nCached: 12000000 kB\n")
	if mem.TotalBytes != 36000000*1024 || mem.FreeBytes != 4000000*1024 {
		t.Fatalf("mem=%+v", mem)
	}
	if mem.AvailableBytes <= 0 || mem.PurgeableBytes != 12000000*1024 {
		t.Fatalf("mem=%+v", mem)
	}
}

func TestParseDarwin(t *testing.T) {
	vm := "Pages free: 1000.\nPages purgeable: 500.\nPages wired down: 2000.\nPages occupied by compressor: 10.\n"
	mem := ParseDarwin(vm, 4096, 16_000_000)
	if mem.FreeBytes != 1000*4096 || mem.PurgeableBytes != 500*4096 || mem.WiredBytes != 2000*4096 || mem.CompressedBytes != 10*4096 {
		t.Fatalf("mem=%+v", mem)
	}
}

// TestParseDarwinCountsInactiveAsAvailable pins the steady-state Mac case where the
// standalone `fak serve --gguf` Metal admission refused every load (issue #10595):
// macOS parks most reclaimable RAM in inactive (and speculative) pages, so a
// free+purgeable-only available figure reads ~0 on a healthy 36G host and the
// fail-closed reservation always refuses. The sample is the live vm_stat captured
// at the refusal: free 0.2G, inactive 16.6G, speculative ~0, purgeable 0.2G.
func TestParseDarwinCountsInactiveAsAvailable(t *testing.T) {
	vm := strings.Join([]string{
		"Mach Virtual Memory Statistics: (page size of 16384 bytes)",
		"Pages free:                                    10279.",
		"Pages active:                                1012697.",
		"Pages inactive:                              1012324.",
		"Pages speculative:                              1215.",
		"Pages throttled:                                   0.",
		"Pages wired down:                             154812.",
		"Pages purgeable:                               13107.",
	}, "\n")
	mem := ParseDarwin(vm, 16384, 38_654_705_664)
	reclaimable := int64(10279+1012324+1215+13107) * 16384
	if mem.AvailableBytes != reclaimable-int64(SafetyMarginGB*1e9) {
		t.Fatalf("AvailableBytes=%d, want inactive+speculative counted: %d", mem.AvailableBytes, reclaimable-int64(SafetyMarginGB*1e9))
	}
}

func TestParseHolders(t *testing.T) {
	holders := ParseHolders("PID RSS COMM\n123 2500000 llama-server\n456 100 shell\n789 1500000 python worker\n")
	if len(holders) != 2 || holders[0].PID != 123 || holders[1].PID != 789 {
		t.Fatalf("holders=%+v", holders)
	}
}

func TestBuildSnapshotAndEvaluate(t *testing.T) {
	mem := Memory{TotalBytes: 10_000_000_000, FreeBytes: 1_000_000_000, AvailableBytes: 5_000_000_000, WiredBytes: 5_000_000_000}
	s := BuildSnapshot("darwin", mem, nil)
	if !s.HighWired || s.Note == "ok" {
		t.Fatalf("snapshot=%+v", s)
	}
	e := Evaluate(s, 4)
	if e.Admit == nil || *e.Admit {
		t.Fatalf("high wired should refuse: %+v", e)
	}
	mem.WiredBytes = 0
	s = BuildSnapshot("linux", mem, nil)
	e = Evaluate(s, 4)
	if e.Admit == nil || !*e.Admit || e.ShortfallGB != 0 {
		t.Fatalf("expected admit: %+v", e)
	}
	e = Evaluate(s, 8)
	if e.Admit == nil || *e.Admit || e.ShortfallGB != 3 {
		t.Fatalf("expected shortfall: %+v", e)
	}
}

func TestAdmissionSampleForFailsClosedAndClassifiesPressure(t *testing.T) {
	tests := []struct {
		name string
		mem  Memory
		want Pressure
	}{
		{name: "unknown total", mem: Memory{AvailableBytes: 8 << 30}, want: PressureUnknown},
		{name: "unknown available", mem: Memory{TotalBytes: 16 << 30}, want: PressureUnknown},
		{name: "normal", mem: Memory{TotalBytes: 16 << 30, AvailableBytes: 8 << 30}, want: PressureNormal},
		{name: "warning compressor", mem: Memory{TotalBytes: 16 << 30, AvailableBytes: 8 << 30, CompressedBytes: 2 << 30}, want: PressureWarning},
		{name: "critical compressor", mem: Memory{TotalBytes: 16 << 30, AvailableBytes: 8 << 30, CompressedBytes: 4 << 30}, want: PressureCritical},
		{name: "critical wired", mem: Memory{TotalBytes: 16 << 30, AvailableBytes: 8 << 30, WiredBytes: 7 << 30}, want: PressureCritical},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := AdmissionSampleFor(tc.mem)
			if got.Pressure != tc.want {
				t.Fatalf("pressure=%q want %q sample=%+v", got.Pressure, tc.want, got)
			}
			if got.AllocatableBytes != tc.mem.AvailableBytes {
				t.Fatalf("allocatable=%d want %d", got.AllocatableBytes, tc.mem.AvailableBytes)
			}
		})
	}
}

// TestObservedDarwinPressureOverridesOccupancyFallback pins #12701: wired and
// compressed byte counts describe occupancy, while the Darwin memorystatus level
// reports current pressure. A recognized live level therefore takes precedence;
// the old conservative occupancy rule remains the fallback when that probe is absent.
func TestObservedDarwinPressureOverridesOccupancyFallback(t *testing.T) {
	// JSON setup keeps this regression source-compatible with the pre-fix Memory
	// shape: old code ignores the two new fields and exhibits the false refusal.
	var base Memory
	if err := json.Unmarshal([]byte(`{
		"TotalBytes":20000000000,
		"AvailableBytes":5000000000,
		"WiredBytes":9000000000,
		"CompressedBytes":4000000000,
		"Pressure":"normal",
		"PressureKnown":true
	}`), &base); err != nil {
		t.Fatal(err)
	}

	t.Run("normal pressure admits fitting load despite high occupancy", func(t *testing.T) {
		if got := AdmissionSampleFor(base); got.Pressure != PressureNormal {
			t.Fatalf("admission pressure=%q, want normal", got.Pressure)
		}
		snap := BuildSnapshot("darwin", base, nil)
		var fields map[string]any
		raw, err := json.Marshal(snap)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(raw, &fields); err != nil {
			t.Fatal(err)
		}
		if fields["pressure_observed"] != true || fields["pressure"] != string(PressureNormal) {
			t.Fatalf("snapshot pressure fields=%v, want normal/true", fields)
		}
		if got := Evaluate(snap, 4); got.Admit == nil || !*got.Admit {
			t.Fatalf("fitting load refused under observed normal pressure: %+v", got)
		}
	})

	t.Run("critical pressure refuses fitting load", func(t *testing.T) {
		mem := base
		if err := json.Unmarshal([]byte(`{
			"WiredBytes":100,
			"CompressedBytes":100,
			"Pressure":"critical"
		}`), &mem); err != nil {
			t.Fatal(err)
		}
		if got := AdmissionSampleFor(mem); got.Pressure != PressureCritical {
			t.Fatalf("admission pressure=%q, want critical", got.Pressure)
		}
		if got := Evaluate(BuildSnapshot("darwin", mem, nil), 4); got.Admit == nil || *got.Admit {
			t.Fatalf("critical pressure admitted load: %+v", got)
		}
	})

	t.Run("capacity shortfall still refuses under normal pressure", func(t *testing.T) {
		got := Evaluate(BuildSnapshot("darwin", base, nil), 6)
		if got.Admit == nil || *got.Admit || got.ShortfallGB != 1 {
			t.Fatalf("short load capacity gate=%+v, want one GB refusal", got)
		}
	})

	t.Run("missing pressure probe preserves occupancy fallback", func(t *testing.T) {
		mem := base
		if err := json.Unmarshal([]byte(`{"PressureKnown":false}`), &mem); err != nil {
			t.Fatal(err)
		}
		if got := AdmissionSampleFor(mem); got.Pressure != PressureCritical {
			t.Fatalf("fallback admission pressure=%q, want critical", got.Pressure)
		}
		if got := Evaluate(BuildSnapshot("darwin", mem, nil), 4); got.Admit == nil || *got.Admit {
			t.Fatalf("missing-probe occupancy fallback admitted load: %+v", got)
		}
	})
}
