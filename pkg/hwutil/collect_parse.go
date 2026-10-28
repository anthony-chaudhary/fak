package hwutil

// Pure parsers for host counter sources. They take raw text (procfs/sysfs
// content or remote command output) and produce normalized rows, so the
// null-on-error contract and the roll-up math are testable without hardware:
// a fake source tree stands in for the real one (the device_spine
// fake-probe pattern).

import (
	"regexp"
	"strconv"
	"strings"
)

// cpuJiffies is one aggregate-CPU /proc/stat sample.
type cpuJiffies struct {
	user, nice, system, idle, iowait, steal uint64
	cores                                   int
}

// parseProcStat parses the aggregate "cpu" line plus per-core lines of
// /proc/stat. ok=false when the aggregate line is absent or malformed.
func parseProcStat(text string) (cpuJiffies, bool) {
	var out cpuJiffies
	seenAgg := false
	for _, line := range strings.Split(text, "\n") {
		f := strings.Fields(line)
		if len(f) < 2 || !strings.HasPrefix(f[0], "cpu") {
			continue
		}
		if f[0] != "cpu" { // per-core line: identity only
			out.cores++
			continue
		}
		vals := make([]uint64, 0, 8)
		ok := true
		for _, s := range f[1:] {
			v, err := strconv.ParseUint(s, 10, 64)
			if err != nil {
				ok = false
				break
			}
			vals = append(vals, v)
		}
		if !ok || len(vals) < 4 {
			return out, false
		}
		// /proc/stat columns: user nice system idle iowait irq softirq steal
		out.user, out.nice, out.system, out.idle = vals[0], vals[1], vals[2], vals[3]
		if len(vals) > 4 {
			out.iowait = vals[4]
		}
		if len(vals) > 7 {
			out.steal = vals[7]
		}
		seenAgg = true
	}
	if !seenAgg || out.cores == 0 {
		return out, false
	}
	return out, true
}

// cpuDelta turns two jiffies samples into utilization percentages over the
// measured window. A zero or backwards total delta yields all-nil pcts
// (unread, not 0).
func cpuDelta(a, b cpuJiffies, intervalS float64) *CPU {
	c := &CPU{Cores: b.cores, IntervalS: round6(intervalS)}
	totA := a.user + a.nice + a.system + a.idle + a.iowait + a.steal
	totB := b.user + b.nice + b.system + b.idle + b.iowait + b.steal
	totalD, ok := safeDelta(totA, totB)
	if !ok || totalD == 0 {
		return c
	}
	total := float64(totalD)
	idleA, idleB := a.idle+a.iowait, b.idle+b.iowait
	if idleD, ok := safeDelta(idleA, idleB); ok && idleD <= totalD {
		c.UtilPct = avg(float64(totalD-idleD)*100, total)
	}
	if d, ok := safeDelta(a.iowait, b.iowait); ok {
		c.IOWaitPct = avg(float64(d)*100, total)
	}
	if d, ok := safeDelta(a.steal, b.steal); ok {
		c.StealPct = avg(float64(d)*100, total)
	}
	return c
}

// meminfoFields is the parsed subset of /proc/meminfo we consume (kB values).
type meminfoFields struct {
	memTotal, memAvailable, swapTotal, swapFree uint64
	present                                     map[string]bool
}

// parseMeminfo parses /proc/meminfo; fields that are absent stay missing from
// present (never zero-filled).
func parseMeminfo(text string) (meminfoFields, bool) {
	out := meminfoFields{present: make(map[string]bool)}
	anyLine := false
	for _, line := range strings.Split(text, "\n") {
		kv := strings.SplitN(line, ":", 2)
		if len(kv) != 2 {
			continue
		}
		key := strings.TrimSpace(kv[0])
		fields := strings.Fields(kv[1])
		if len(fields) == 0 {
			continue
		}
		v, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			continue
		}
		anyLine = true
		switch key {
		case "MemTotal":
			out.memTotal, out.present["MemTotal"] = v, true
		case "MemAvailable":
			out.memAvailable, out.present["MemAvailable"] = v, true
		case "SwapTotal":
			out.swapTotal, out.present["SwapTotal"] = v, true
		case "SwapFree":
			out.swapFree, out.present["SwapFree"] = v, true
		}
	}
	return out, anyLine
}

const kilobyte = uint64(1024)

// dramFromMeminfo builds the DRAM row from a meminfo sample; unread inputs
// leave the matching pointer nil. used = total - available (the kernel's own
// definition of in-use memory).
func dramFromMeminfo(m meminfoFields) *DRAM {
	d := &DRAM{}
	if m.present["MemTotal"] {
		d.TotalBytes = i64(int64(m.memTotal * kilobyte))
	}
	if m.present["MemAvailable"] {
		d.AvailableBytes = i64(int64(m.memAvailable * kilobyte))
	}
	if d.TotalBytes != nil && d.AvailableBytes != nil && *d.TotalBytes >= *d.AvailableBytes {
		d.UsedBytes = i64(*d.TotalBytes - *d.AvailableBytes)
	}
	if m.present["SwapTotal"] && m.present["SwapFree"] && m.swapTotal >= m.swapFree {
		d.SwapUsedBytes = i64(int64((m.swapTotal - m.swapFree) * kilobyte))
	}
	return d
}

// vmstatCounter extracts one named counter from /proc/vmstat text.
func vmstatCounter(text, name string) (uint64, bool) {
	for _, line := range strings.Split(text, "\n") {
		f := strings.Fields(line)
		if len(f) != 2 || f[0] != name {
			continue
		}
		v, err := strconv.ParseUint(f[1], 10, 64)
		if err != nil {
			return 0, false
		}
		return v, true
	}
	return 0, false
}

// parseLoadavg returns load1 from /proc/loadavg.
func parseLoadavg(text string) (*float64, bool) {
	f := strings.Fields(text)
	if len(f) < 1 {
		return nil, false
	}
	v, err := strconv.ParseFloat(f[0], 64)
	if err != nil {
		return nil, false
	}
	return f64(v), true
}

// diskSample is one /proc/diskstats row (kernel field layout, 4.x+).
type diskSample struct {
	name            string
	readsCompleted  uint64
	sectorsRead     uint64
	msReading       uint64
	writesCompleted uint64
	sectorsWritten  uint64
	msWriting       uint64
	inflight        uint64
	hasInflight     bool
}

// parseDiskstats parses /proc/diskstats; rows with too few fields are skipped.
func parseDiskstats(text string) map[string]diskSample {
	out := make(map[string]diskSample)
	for _, line := range strings.Split(text, "\n") {
		f := strings.Fields(line)
		if len(f) < 12 {
			continue
		}
		s := diskSample{name: f[2]}
		u := func(i int) uint64 {
			v, _ := strconv.ParseUint(f[i], 10, 64)
			return v
		}
		s.readsCompleted = u(3)
		s.sectorsRead = u(5)
		s.msReading = u(6)
		s.writesCompleted = u(7)
		s.sectorsWritten = u(9)
		s.msWriting = u(10)
		if len(f) >= 14 {
			s.inflight = u(11)
			s.hasInflight = true
		}
		out[s.name] = s
	}
	return out
}

// sectorSize is the kernel's fixed 512-byte accounting unit.
const sectorSize = 512

var (
	partitionNVMe = regexp.MustCompile(`^nvme(\d+)n(\d+)p(\d+)$`)
	partitionSD   = regexp.MustCompile(`^([a-z]+d[a-z])(\d+)$`)
)

// classifyDisk is the pure name+marker decision table for disk transports,
// split out so the classification is testable without a block-device tree.
func classifyDisk(name string, usbMarker, modelHasUSB bool, rotational float64, hasRotational bool) DiskKind {
	switch {
	case strings.HasPrefix(name, "nvme"):
		return DiskNVMe
	case strings.HasPrefix(name, "ram"), strings.HasPrefix(name, "zram"):
		return DiskRamdisk
	case usbMarker || modelHasUSB:
		return DiskUSB
	case hasRotational && rotational == 1:
		return DiskSATA
	default:
		return DiskUnknown
	}
}

// keepDisk drops virtual aliases and partitions whose parent device is
// present, so each physical transport appears once. Shared across platforms
// so the filtering rule stays testable without a block-device tree.
func keepDisk(name string, all map[string]diskSample) bool {
	switch {
	case strings.HasPrefix(name, "loop"), strings.HasPrefix(name, "dm-"):
		return false
	}
	if m := partitionNVMe.FindStringSubmatch(name); m != nil {
		parent := "nvme" + m[1] + "n" + m[2]
		if _, ok := all[parent]; ok {
			return false
		}
	}
	if m := partitionSD.FindStringSubmatch(name); m != nil {
		if _, ok := all[m[1]]; ok {
			return false
		}
	}
	return true
}

// diskDelta builds rate counters between two diskstats samples over dt
// seconds; any backwards counter leaves its own field nil.
func diskDelta(a, b diskSample, dt float64) (readBps, writeBps, iopsR, iopsW, latencyMS *float64, queue *int64) {
	if d, ok := safeDelta(a.sectorsRead, b.sectorsRead); ok {
		readBps = avg(float64(d)*sectorSize, dt)
	}
	if d, ok := safeDelta(a.sectorsWritten, b.sectorsWritten); ok {
		writeBps = avg(float64(d)*sectorSize, dt)
	}
	if d, ok := safeDelta(a.readsCompleted, b.readsCompleted); ok {
		iopsR = avg(float64(d), dt)
	}
	if d, ok := safeDelta(a.writesCompleted, b.writesCompleted); ok {
		iopsW = avg(float64(d), dt)
	}
	ioA, ioB := a.readsCompleted+a.writesCompleted, b.readsCompleted+b.writesCompleted
	msA, msB := a.msReading+a.msWriting, b.msReading+b.msWriting
	if ops, ok := safeDelta(ioA, ioB); ok && ops > 0 {
		if ms, ok := safeDelta(msA, msB); ok {
			latencyMS = avg(float64(ms), float64(ops))
		}
	}
	if b.hasInflight {
		queue = i64(int64(b.inflight))
	}
	return
}

// netStats is one interface's cumulative /sys/class/net/<if>/statistics set.
type netStats struct {
	rxBytes, txBytes, rxPackets, txPackets, rxDropped, txDropped, rxErrors, txErrors uint64
}

// parseNetStatsText parses the eight statistic files' contents joined with
// newline separators in the fixed order of netStatKeys. Missing files keep
// their slot empty (nil downstream).
var netStatKeys = []string{"rx_bytes", "tx_bytes", "rx_packets", "tx_packets", "rx_dropped", "tx_dropped", "rx_errors", "tx_errors"}

func parseNetStatsText(text string) (netStats, []bool) {
	lines := strings.Split(text, "\n")
	present := make([]bool, len(netStatKeys))
	var s netStats
	vals := make([]uint64, len(netStatKeys))
	for i := range netStatKeys {
		if i >= len(lines) {
			break
		}
		v, err := strconv.ParseUint(strings.TrimSpace(lines[i]), 10, 64)
		if err != nil {
			continue
		}
		vals[i] = v
		present[i] = true
	}
	s.rxBytes, s.txBytes = vals[0], vals[1]
	s.rxPackets, s.txPackets = vals[2], vals[3]
	s.rxDropped, s.txDropped = vals[4], vals[5]
	s.rxErrors, s.txErrors = vals[6], vals[7]
	return s, present
}

// netDelta builds the NetIfc row between two samples over dt seconds; a
// missing source slot or a backwards counter leaves its field nil.
func netDelta(iface string, a, b netStats, present []bool, dt float64) *NetIfc {
	n := &NetIfc{Iface: iface}
	set := func(dst **float64, from, to uint64, idx int) {
		if !present[idx] {
			return
		}
		if d, ok := safeDelta(from, to); ok {
			*dst = avg(float64(d), dt)
		}
	}
	set(&n.RxBytesPerS, a.rxBytes, b.rxBytes, 0)
	set(&n.TxBytesPerS, a.txBytes, b.txBytes, 1)
	set(&n.RxPacketsPerS, a.rxPackets, b.rxPackets, 2)
	set(&n.TxPacketsPerS, a.txPackets, b.txPackets, 3)
	set(&n.RxDropsPerS, a.rxDropped, b.rxDropped, 4)
	set(&n.TxDropsPerS, a.txDropped, b.txDropped, 5)
	if present[6] && present[7] {
		n.ErrorsTotal = i64(int64(b.rxErrors + b.txErrors))
	}
	return n
}

// thunderboltLinkRow normalizes one Thunderbolt bridge's raw state/bandwidth
// into a wire row; unread bandwidth stays nil.
func thunderboltLinkRow(id, stateRaw string, bandwidth uint64, hasBandwidth bool) Link {
	l := Link{Kind: LinkThunderbolt, ID: id, State: LinkStateUnknown}
	switch stateRaw {
	case "active":
		l.State = LinkUp
	case "inactive":
		l.State = LinkDown
	}
	if hasBandwidth && bandwidth > 0 {
		l.SpeedGBps = f64(float64(bandwidth) / 1e9)
	}
	return l
}

// usbLinkRow normalizes one USB device's speed/authorization into a wire row.
// Speeds at or above 40 Gbps are USB4-class; slower devices stay kind
// unknown rather than mislabeled.
func usbLinkRow(name string, speedMbps uint64, authorized bool) Link {
	gbps := f64(float64(speedMbps) / 1000)
	kind := LinkUnknown
	if *gbps >= 40 {
		kind = LinkUSB4
	}
	state := LinkDown
	if authorized {
		state = LinkUp
	}
	return Link{Kind: kind, ID: "usb-" + name, SpeedGBps: gbps, State: state}
}
