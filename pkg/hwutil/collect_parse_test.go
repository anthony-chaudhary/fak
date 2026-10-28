package hwutil

// Fake-probe contract tests: synthetic /proc and /sys text pinned to real
// kernel formats, asserting the normalized rows the collectors must emit.
// The null-on-error edges are asserted explicitly, in the style of
// internal/metrics/device_spine_test.go.

import (
	"math"
	"testing"
)

func approx(t *testing.T, got, want float64, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Fatalf("got %v, want %v ± %v", got, want, tol)
	}
}

// Columns: user nice system idle iowait irq softirq steal.
const cpuStatA = `cpu  10000 0 5000 80000 1000 0 500 100
cpu0 1000 0 500 8000 100 0 50 10
cpu1 1000 0 500 8000 100 0 50 10
`

const cpuStatB = `cpu  10500 0 5200 80300 1010 0 510 110
cpu0 1050 0 520 8030 101 0 51 11
cpu1 1050 0 520 8030 101 0 51 11
`

func TestCPUDeltaFromFakeProcStat(t *testing.T) {
	a, ok := parseProcStat(cpuStatA)
	if !ok {
		t.Fatal("parse A failed")
	}
	b, ok := parseProcStat(cpuStatB)
	if !ok {
		t.Fatal("parse B failed")
	}
	if b.cores != 2 {
		t.Fatalf("cores = %d, want 2", b.cores)
	}
	c := cpuDelta(a, b, 0.5)
	if c == nil {
		t.Fatal("nil CPU row")
	}
	// total delta 1020 (incl. steal); idle+iowait delta 310 -> busy 710.
	approx(t, *c.UtilPct, 710.0/1020*100, 0.01)
	// iowait delta 10 -> ~0.98%; steal delta 10 -> ~0.98%.
	approx(t, *c.IOWaitPct, 10.0/1020*100, 0.01)
	approx(t, *c.StealPct, 10.0/1020*100, 0.01)
	if c.IntervalS != 0.5 || c.Cores != 2 {
		t.Fatalf("interval/cores = %v / %d", c.IntervalS, c.Cores)
	}
}

func TestCPUDeltaZeroWindowIsUnread(t *testing.T) {
	a, _ := parseProcStat(cpuStatA)
	c := cpuDelta(a, a, 0.5)
	if c == nil {
		t.Fatal("row should carry identity even on a degenerate window")
	}
	// No movement: every percentage stays nil (unread, not zero-filled).
	if c.UtilPct != nil || c.IOWaitPct != nil || c.StealPct != nil {
		t.Fatalf("degenerate window produced pcts: %+v", c)
	}
}

func TestParseProcStatRejectsGarbage(t *testing.T) {
	if _, ok := parseProcStat("garbage\n"); ok {
		t.Fatal("garbage accepted")
	}
	if _, ok := parseProcStat("cpu0 1 2 3 4\n"); ok {
		t.Fatal("per-core-only sample accepted without aggregate line")
	}
}

const meminfoText = `MemTotal:       32768000 kB
MemFree:         8000000 kB
MemAvailable:   16000000 kB
SwapTotal:       4000000 kB
SwapFree:        3500000 kB
`

func TestDramFromFakeMeminfo(t *testing.T) {
	m, ok := parseMeminfo(meminfoText)
	if !ok {
		t.Fatal("parse failed")
	}
	d := dramFromMeminfo(m)
	if d == nil {
		t.Fatal("nil DRAM row")
	}
	if *d.TotalBytes != 32768000*1024 || *d.AvailableBytes != 16000000*1024 {
		t.Fatalf("total/available = %v / %v", *d.TotalBytes, *d.AvailableBytes)
	}
	// used = total - available by construction.
	if *d.UsedBytes != 16768000*1024 {
		t.Fatalf("used = %v", *d.UsedBytes)
	}
	if *d.SwapUsedBytes != 500000*1024 {
		t.Fatalf("swap used = %v", *d.SwapUsedBytes)
	}
	// Bandwidth is never derivable from meminfo alone: stays nil.
	if d.BandwidthGiBS != nil {
		t.Fatalf("bandwidth fabricated: %v", *d.BandwidthGiBS)
	}
}

func TestVMStatCounterAndLoadavg(t *testing.T) {
	v1, ok := vmstatCounter("pgfault 1000000\npswpin 10\n", "pgfault")
	if !ok || v1 != 1000000 {
		t.Fatalf("v1 = %v ok=%v", v1, ok)
	}
	v2, ok := vmstatCounter("pgfault 1001000\n", "pgfault")
	if !ok || v2 != 1001000 {
		t.Fatalf("v2 = %v ok=%v", v2, ok)
	}
	rate := avg(float64(v2-v1), 0.5)
	if rate == nil || *rate != 2000 {
		t.Fatalf("pgfault rate = %v, want 2000/s", rate)
	}
	load, ok := parseLoadavg("0.50 1.00 1.50 2/300 12345\n")
	if !ok || load == nil || *load != 0.5 {
		t.Fatalf("loadavg = %v ok=%v", load, ok)
	}
	if _, ok := parseLoadavg(""); ok {
		t.Fatal("empty loadavg accepted")
	}
}

// Two /proc/diskstats samples 1s apart. Kernel field layout: major minor name
// reads rmerge sectors_read ms_reading writes wmerge sectors_written ms_writing
// inflight io_ms discarded.
const diskStatsA = ` 259       0 nvme0n1 1000 0 100000 500 500 0 50000 400 0 400 0
   7       0 sda 100 0 10000 50 50 0 5000 40 0 40 0
   7       1 sda1 90 0 9000 45 45 0 4500 38 0 38 0
 252       0 loop0 10 0 1000 5 0 0 0 0 0 0 0
`

const diskStatsB = ` 259       0 nvme0n1 1100 0 105120 510 550 0 55000 410 4 410 0
   7       0 sda 105 0 10256 51 55 0 5256 41 0 41 0
   7       1 sda1 95 0 9256 46 46 0 4756 39 0 39 0
 252       0 loop0 10 0 1000 5 0 0 0 0 0 0 0
`

func TestDiskDeltaFromFakeDiskstats(t *testing.T) {
	a := parseDiskstats(diskStatsA)
	b := parseDiskstats(diskStatsB)
	if len(a) != 4 || len(b) != 4 {
		t.Fatalf("parsed %d/%d devices", len(a), len(b))
	}
	rd, wr, ir, iw, lat, qd := diskDelta(a["nvme0n1"], b["nvme0n1"], 1.0)
	// 5120 read sectors * 512B = 2621440 B/s; 5000 write sectors = 2560000 B/s.
	approx(t, *rd, 2621440, 0.01)
	approx(t, *wr, 2560000, 0.01)
	approx(t, *ir, 100, 0.01)
	approx(t, *iw, 50, 0.01)
	// (510+410)-(500+400) = 20ms over 150 ios.
	approx(t, *lat, 20.0/150, 0.001)
	if qd == nil || *qd != 4 {
		t.Fatalf("queue depth = %v, want 4", qd)
	}
	// Short rows (no inflight field) keep queue depth nil.
	short := map[string]diskSample{"x": {name: "x", hasInflight: false}}
	_, _, _, _, _, qd2 := diskDelta(short["x"], short["x"], 1.0)
	if qd2 != nil {
		t.Fatalf("short row fabricated queue depth: %v", *qd2)
	}
}

func TestKeepDiskFiltersSyntheticDevices(t *testing.T) {
	all := map[string]diskSample{
		"nvme0n1": {}, "nvme0n1p1": {}, "sda": {}, "sda1": {}, "loop0": {}, "dm-0": {}, "ram0": {},
	}
	want := map[string]bool{"nvme0n1": true, "sda": true, "ram0": true}
	for name := range all {
		got := keepDisk(name, all)
		if got != want[name] {
			t.Errorf("keepDisk(%q) = %v, want %v", name, got, want[name])
		}
	}
	// A partition whose parent is absent stays (whole disk missing from view).
	orphan := map[string]diskSample{"nvme0n1p1": {}}
	if !keepDisk("nvme0n1p1", orphan) {
		t.Error("orphan partition dropped")
	}
}

func TestClassifyDiskDecisionTable(t *testing.T) {
	cases := []struct {
		name     string
		usb      bool
		modelUSB bool
		rot      float64
		hasRot   bool
		want     DiskKind
	}{
		{"nvme1n1", false, false, 0, false, DiskNVMe},
		{"ram2", false, false, 0, false, DiskRamdisk},
		{"zram0", false, false, 0, false, DiskRamdisk},
		{"sdb", true, false, 0, false, DiskUSB},
		{"sdc", false, true, 0, false, DiskUSB},
		{"sdd", false, false, 1, true, DiskSATA},
		{"mmcblk0", false, false, 0, false, DiskUnknown},
	}
	for _, c := range cases {
		if got := classifyDisk(c.name, c.usb, c.modelUSB, c.rot, c.hasRot); got != c.want {
			t.Errorf("%s: classifyDisk = %v, want %v", c.name, got, c.want)
		}
	}
}

// Two net statistics sets 1s apart on eth0: +1000 rx bytes, +500 tx bytes,
// +10 rx packets, +5 tx packets, +1 rx drop, errors 3+1 -> 4+1.
func TestNetDeltaFromFakeSysfs(t *testing.T) {
	textA := "100000\n50000\n1000\n500\n10\n2\n3\n1\n"
	textB := "101000\n50500\n1010\n505\n11\n2\n4\n1\n"
	a, _ := parseNetStatsText(textA)
	b, pb := parseNetStatsText(textB)
	x := netDelta("eth0", a, b, pb, 1.0)
	if x.Iface != "eth0" {
		t.Fatalf("iface = %q", x.Iface)
	}
	approx(t, *x.RxBytesPerS, 1000, 0.01)
	approx(t, *x.TxBytesPerS, 500, 0.01)
	approx(t, *x.RxPacketsPerS, 10, 0.01)
	approx(t, *x.TxPacketsPerS, 5, 0.01)
	approx(t, *x.RxDropsPerS, 1, 0.01)
	if x.ErrorsTotal == nil || *x.ErrorsTotal != 5 {
		t.Fatalf("errors_total = %v, want 5", x.ErrorsTotal)
	}
}

func TestNetDeltaMissingSlotsStayNil(t *testing.T) {
	// Only five statistic files exist: drops/errors slots unread.
	textA := "100\n50\n10\n5\n1\n"
	textB := "150\n60\n12\n6\n2\n"
	a, _ := parseNetStatsText(textA)
	b, pb := parseNetStatsText(textB)
	x := netDelta("wlan0", a, b, pb, 1.0)
	if x.TxDropsPerS != nil || x.ErrorsTotal != nil {
		t.Fatalf("missing slots fabricated: %+v", x)
	}
	if x.RxDropsPerS == nil {
		t.Fatalf("present slot lost: %+v", x)
	}
}

func TestThunderboltAndUSBLinkRows(t *testing.T) {
	l := thunderboltLinkRow("br0", "active", 40_000_000_000, true)
	if l.Kind != LinkThunderbolt || l.State != LinkUp || *l.SpeedGBps != 40 {
		t.Fatalf("thunderbolt row = %+v", l)
	}
	l2 := thunderboltLinkRow("br1", "inactive", 0, false)
	if l2.State != LinkDown || l2.SpeedGBps != nil {
		t.Fatalf("unread bandwidth must stay nil: %+v", l2)
	}
	u := usbLinkRow("1-2", 40000, true)
	if u.Kind != LinkUSB4 || u.ID != "usb-1-2" || u.State != LinkUp || *u.SpeedGBps != 40 {
		t.Fatalf("usb4 row = %+v", u)
	}
	u2 := usbLinkRow("1-3", 480, false)
	if u2.Kind != LinkUnknown || u2.State != LinkDown {
		t.Fatalf("slow unauthorized usb row = %+v", u2)
	}
}
