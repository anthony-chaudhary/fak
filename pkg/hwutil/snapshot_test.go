package hwutil

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
)

// frozenFixture is the exact wire shape from the GOAL-hw-utilization-telemetry
// schema contract, with a fully populated sample including explicit nulls.
// It pins the tag set: any added/renamed/dropped field breaks this test.
const frozenFixture = `{
  "schema": "fak.hwutil.v1",
  "host": "node-01", "arch": "amd64", "os": "linux",
  "at": "2026-10-28T12:00:00Z", "duration_seconds": 0.5, "error": "",
  "cpu": {"cores": 8, "util_pct": 12.5, "steal_pct": null, "iowait_pct": 0.4, "interval_s": 0.5, "load1": 1.2},
  "dram": {"total_bytes": 34359738368, "used_bytes": 17179869184, "available_bytes": 17179869184,
           "page_faults_per_s": 1200.5, "swap_used_bytes": 0, "bandwidth_gib_s": null},
  "disks": [{"name": "nvme0n1", "kind": "nvme",
             "read_bytes_per_s": 10485760, "write_bytes_per_s": 5242880,
             "iops_read": 2048, "iops_write": 512, "queue_depth": 4,
             "latency_ms_avg": 0.35, "free_bytes": 107374182400, "total_bytes": 205909447680}],
  "net": [{"iface": "eth0", "rx_bytes_per_s": 1048576, "tx_bytes_per_s": 524288,
           "rx_packets_per_s": 800, "tx_packets_per_s": 400,
           "rx_drops_per_s": 0, "tx_drops_per_s": 0, "errors_total": 0}],
  "links": [{"kind": "thunderbolt", "id": "br0",
             "speed_gbps": 40, "rx_bytes_per_s": null, "tx_bytes_per_s": null,
             "state": "up"}],
  "gpus": [{"uuid": "GPU-abc", "name": "NVIDIA RTX 5070 Ti", "pci": "0000:01:00.0",
            "util_pct": 87.5, "mem_total_bytes": 16106127360, "mem_used_bytes": 8053063680,
            "power_w": 245.5, "sm_clock_mhz": 2407, "max_sm_clock_mhz": 2610,
            "pcie_rx_bytes_per_s": 1073741824, "pcie_tx_bytes_per_s": 536870912,
            "throttle_reasons": ["HW Slowdown"], "ecc_uncorrected_total": 0}],
  "tiers": [{"kind": "vram", "device": "cuda:0", "model_id": "qwen3.8-14b",
             "layers_start": 0, "layers_end": 23, "bytes": 8589934592,
             "hotness_rank": 0, "last_transfer_at": "2026-10-28T11:59:00Z",
             "transfer_bytes_per_s": null}],
  "transfers": [{"from_tier": "ssd", "to_tier": "ram", "bytes": 4294967296,
                 "duration_s": 0.9, "rate_gib_s": 4.66,
                 "reason": "load"}]
}`

func TestFrozenFixtureParsesAndValidates(t *testing.T) {
	var s Snapshot
	if err := json.Unmarshal([]byte(frozenFixture), &s); err != nil {
		t.Fatalf("unmarshal frozen fixture: %v", err)
	}
	if err := s.Validate(); err != nil {
		t.Fatalf("Validate on frozen fixture: %v", err)
	}
	// Identity always set.
	if s.Schema != Schema || s.Host != "node-01" || s.Arch != "amd64" || s.Os != "linux" {
		t.Fatalf("identity fields wrong: %+v", s)
	}
	if !s.At.Equal(time.Date(2026, 10, 28, 12, 0, 0, 0, time.UTC)) {
		t.Fatalf("at = %v", s.At)
	}
	if s.DurationSeconds != 0.5 || s.Error != "" {
		t.Fatalf("duration/error = %v / %q", s.DurationSeconds, s.Error)
	}
	// CPU: filled values plus the pinned null.
	c := s.CPU
	if c == nil || c.Cores != 8 || *c.UtilPct != 12.5 || *c.IOWaitPct != 0.4 || *c.Load1 != 1.2 || c.IntervalS != 0.5 {
		t.Fatalf("cpu = %+v", c)
	}
	if c.StealPct != nil {
		t.Fatalf("steal_pct should be nil, got %v", *c.StealPct)
	}
	// DRAM: swap zero is a value; bandwidth null stays nil.
	d := s.DRAM
	if d == nil || *d.TotalBytes != 34359738368 || *d.UsedBytes != 17179869184 || *d.AvailableBytes != 17179869184 {
		t.Fatalf("dram = %+v", d)
	}
	if *d.SwapUsedBytes != 0 || d.BandwidthGiBS != nil || *d.PageFaultsPerSec != 1200.5 {
		t.Fatalf("dram metrics = %+v", d)
	}
	// Disks / net / links.
	if len(s.Disks) != 1 || s.Disks[0].Name != "nvme0n1" || s.Disks[0].Kind != DiskNVMe || *s.Disks[0].QueueDepth != 4 {
		t.Fatalf("disks = %+v", s.Disks)
	}
	if len(s.Net) != 1 || s.Net[0].Iface != "eth0" || *s.Net[0].ErrorsTotal != 0 {
		t.Fatalf("net = %+v", s.Net)
	}
	l := s.Links[0]
	if l.Kind != LinkThunderbolt || l.ID != "br0" || l.State != LinkUp || *l.SpeedGBps != 40 || l.RxBytesPerS != nil {
		t.Fatalf("links = %+v", s.Links)
	}
	// GPU row incl. extended fields.
	g := s.GPUs[0]
	if g.UUID != "GPU-abc" || g.PCI != "0000:01:00.0" || *g.UtilPct != 87.5 || *g.MemTotalBytes != 16106127360 ||
		*g.PowerW != 245.5 || *g.SMClockMHz != 2407 || *g.MaxSMClockMHz != 2610 ||
		*g.PCIERxBytesPerS != 1073741824 || *g.PCIETxBytesPerS != 536870912 || *g.ECCUncorrectedTotal != 0 {
		t.Fatalf("gpu = %+v", g)
	}
	if !reflect.DeepEqual(g.ThrottleReasons, []string{"HW Slowdown"}) {
		t.Fatalf("throttle reasons = %v", g.ThrottleReasons)
	}
	// Tier + transfer rows.
	tr := s.Tiers[0]
	if tr.Kind != TierVRAM || tr.Device != "cuda:0" || tr.ModelID != "qwen3.8-14b" ||
		tr.LayersStart != 0 || tr.LayersEnd != 23 || tr.Bytes != 8589934592 || tr.HotnessRank != 0 {
		t.Fatalf("tier = %+v", tr)
	}
	if tr.LastTransferAt == nil || !tr.LastTransferAt.Equal(time.Date(2026, 10, 28, 11, 59, 0, 0, time.UTC)) {
		t.Fatalf("last_transfer_at = %v", tr.LastTransferAt)
	}
	if tr.TransferBytesPerS != nil {
		t.Fatalf("transfer_bytes_per_s should be nil")
	}
	x := s.Transfers[0]
	if x.FromTier != "ssd" || x.ToTier != "ram" || x.Bytes != 4294967296 || x.DurationS != 0.9 || x.RateGiBS != 4.66 || x.Reason != TransferLoad {
		t.Fatalf("transfer = %+v", x)
	}
}

func TestFrozenFixtureRoundTripStable(t *testing.T) {
	var a Snapshot
	if err := json.Unmarshal([]byte(frozenFixture), &a); err != nil {
		t.Fatal(err)
	}
	raw, err := RenderJSON(&a)
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	var b Snapshot
	if err := json.Unmarshal(raw, &b); err != nil {
		t.Fatalf("re-unmarshal: %v", err)
	}
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("round trip unstable:\n%+v\nvs\n%+v", a, b)
	}
	// Null-on-error survives to the wire: pinned nulls must appear literally.
	for _, needle := range []string{`"steal_pct":null`, `"bandwidth_gib_s":null`, `"rx_bytes_per_s":null`} {
		if !strings.Contains(string(raw), needle) {
			t.Fatalf("wire lost null for %s:\n%s", needle, raw)
		}
	}
}

func TestValidateRejectsMalformedSnapshots(t *testing.T) {
	base := func() Snapshot {
		var s Snapshot
		if err := json.Unmarshal([]byte(frozenFixture), &s); err != nil {
			t.Fatal(err)
		}
		return s
	}
	cases := map[string]func(*Snapshot){
		"wrong schema":     func(s *Snapshot) { s.Schema = "fak.hwutil.v2" },
		"empty host":       func(s *Snapshot) { s.Host = "  " },
		"bad arch":         func(s *Snapshot) { s.Arch = "riscv64" },
		"bad os":           func(s *Snapshot) { s.Os = "darwin" },
		"zero at":          func(s *Snapshot) { s.At = time.Time{} },
		"negative dur":     func(s *Snapshot) { s.DurationSeconds = -1 },
		"zero cores":       func(s *Snapshot) { s.CPU.Cores = 0 },
		"util over 100":    func(s *Snapshot) { v := 101.0; s.CPU.UtilPct = &v },
		"negative load1":   func(s *Snapshot) { v := -0.5; s.CPU.Load1 = &v },
		"negative dram":    func(s *Snapshot) { s.DRAM.UsedBytes = i64(-1) },
		"bad disk kind":    func(s *Snapshot) { s.Disks[0].Kind = "scsi" },
		"empty disk name":  func(s *Snapshot) { s.Disks[0].Name = "" },
		"negative iops":    func(s *Snapshot) { v := -3.0; s.Disks[0].IopsRead = &v },
		"empty iface":      func(s *Snapshot) { s.Net[0].Iface = "" },
		"bad link kind":    func(s *Snapshot) { s.Links[0].Kind = "firewire" },
		"bad link state":   func(s *Snapshot) { s.Links[0].State = "sleeping" },
		"empty gpu uuid":   func(s *Snapshot) { s.GPUs[0].UUID = "" },
		"gpu util 101":     func(s *Snapshot) { v := 101.0; s.GPUs[0].UtilPct = &v },
		"bad tier kind":    func(s *Snapshot) { s.Tiers[0].Kind = "hbm" },
		"inverted layers":  func(s *Snapshot) { s.Tiers[0].LayersStart, s.Tiers[0].LayersEnd = 5, 2 },
		"bad transfer src": func(s *Snapshot) { s.Transfers[0].FromTier = "flash" },
		"bad reason":       func(s *Snapshot) { s.Transfers[0].Reason = "migrate" },
		"negative bytes":   func(s *Snapshot) { s.Transfers[0].Bytes = -1 },
	}
	for name, mutate := range cases {
		s := base()
		mutate(&s)
		err := s.Validate()
		if err == nil {
			t.Errorf("%s: Validate passed, want error", name)
			continue
		}
		if !errors.Is(err, ErrSnapshotMalformed) {
			t.Errorf("%s: errors.Is(ErrSnapshotMalformed) = false: %v", name, err)
		}
		if CodeOf(err) != CodeSnapshotMalformed {
			t.Errorf("%s: CodeOf = %q", name, CodeOf(err))
		}
	}
}

func TestErrorCodeVocabularyIsClosed(t *testing.T) {
	want := map[ErrorCode]string{
		CodeCollectorUnreadable: "HWUTIL_COLLECTOR_UNREADABLE",
		CodePartial:             "HWUTIL_PARTIAL",
		CodeHostUnreachable:     "HWUTIL_HOST_UNREACHABLE",
		CodeSnapshotMalformed:   "HWUTIL_SNAPSHOT_MALFORMED",
		CodeStoreFull:           "HWUTIL_STORE_FULL",
	}
	seen := map[ErrorCode]bool{}
	for code, literal := range want {
		if string(code) != literal {
			t.Fatalf("code %v does not match frozen literal %q", code, literal)
		}
		seen[code] = true
	}
	if len(seen) != 5 {
		t.Fatalf("closed vocabulary drifted: %v", seen)
	}
	// Dynamic errors wrap their sentinel so errors.Is keeps working.
	if !errors.Is(NewErr(CodePartial, "cpu: no procfs"), ErrPartial) {
		t.Fatal("NewErr(Partial) must wrap ErrPartial")
	}
	if !errors.Is(errors.New("plain"), ErrPartial) && CodeOf(errors.New("plain")) != "" {
		t.Fatal("plain errors carry no hwutil code")
	}
	if CodeOf(nil) != "" {
		t.Fatal("CodeOf(nil) must be empty")
	}
}

func TestNumericCounterCount(t *testing.T) {
	var s Snapshot
	if err := json.Unmarshal([]byte(frozenFixture), &s); err != nil {
		t.Fatal(err)
	}
	// cpu: cores+interval+util+iowait+load1 = 5 (steal nil)
	// dram: total+used+avail+faults+swap = 5 (bandwidth nil)
	// disk: all 8 present
	// net: all 7 present
	// link: speed only = 1
	// gpu: all 9 present
	// tier: 4 scalars (rate nil)
	// transfer: 3
	if got := s.NumericCounterCount(); got != 5+5+8+7+1+9+4+3 {
		t.Fatalf("NumericCounterCount = %d, want 42", got)
	}
	// An all-nil host plane counts only identity-adjacent scalars: zero here.
	empty := Snapshot{Schema: Schema, Host: "h", Arch: "amd64", Os: "linux"}
	if got := empty.NumericCounterCount(); got != 0 {
		t.Fatalf("empty snapshot count = %d, want 0", got)
	}
}
