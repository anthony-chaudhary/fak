package memgate

import "testing"

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
