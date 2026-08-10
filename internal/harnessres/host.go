package harnessres

import (
	"fmt"
	"strings"
)

// Host is the BOX the harness ran on, sampled alongside the harness slice so a raw
// "the harness used 4.2 GiB" reads against the machine that produced it: 4.2 GiB is
// most of a laptop and a rounding error on a build server. Presence bits follow the
// same rule as the Half axes — an axis the platform cannot read stays absent and
// renders "n/a", never a fabricated 0 (#2053, epic #2044).
//
// The leaf reads NOTHING itself: it is stdlib-only and imports nothing internal, so
// the host numbers arrive through SetHostProvider, which fak guard wires to the
// existing compute host readers (compute.HostSystemMemoryInfo — the same
// hostmem_{linux,darwin,windows}.go seam the serve capacity checks use). Core count
// is NOT carried here; Snapshot.NumCPU (runtime.NumCPU) stays the one source of truth
// for it.
type Host struct {
	TotalRAMBytes uint64
	AvailRAMBytes uint64
	HaveRAM       bool
	// Load1 is the best-effort 1-minute host load average. No stdlib reader exists on
	// every platform, so it is carried but only populated by a host that can supply it.
	Load1    float64
	HaveLoad bool
}

// SetHostProvider installs the pull source for the host context block (total/available
// RAM, best-effort load). fak guard wires this to compute.HostSystemMemoryInfo. The
// bool reports whether a reading was obtained at all; per-axis presence rides the
// Have* bits on the returned Host. Call before Start.
func (s *Sampler) SetHostProvider(fn func() (Host, bool)) {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.hostProvider = fn
	s.mu.Unlock()
}

// HarnessRSSBytes sums the resident bytes of BOTH halves that reported one — the whole
// harness's memory slice, which is what a host fraction must be taken against. ok is
// false when neither half could be read.
func (s Snapshot) HarnessRSSBytes() (uint64, bool) {
	var total uint64
	var ok bool
	if s.Kernel.HaveRSS {
		total, ok = total+s.Kernel.RSSBytes, true
	}
	if s.Agent.HaveRSS {
		total, ok = total+s.Agent.RSSBytes, true
	}
	return total, ok
}

// RSSPercentOfHostRAM is the harness's resident bytes as a percentage of host physical
// RAM — the portable number a budget can be written against ("80% of host RAM"), where
// a raw byte ceiling is not (#2051). ok is false when either side is unread.
func (s Snapshot) RSSPercentOfHostRAM() (pct float64, ok bool) {
	rss, haveRSS := s.HarnessRSSBytes()
	if !haveRSS || !s.Host.HaveRAM || s.Host.TotalRAMBytes == 0 {
		return 0, false
	}
	return float64(rss) / float64(s.Host.TotalRAMBytes) * 100, true
}

// HostCoreSeconds is the CPU capacity the host offered over the sampled window:
// core count x elapsed wall seconds. It is the denominator that turns "12 CPU-seconds"
// into a share of the box. ok is false when the core count or elapsed time is unknown.
func (s Snapshot) HostCoreSeconds() (float64, bool) {
	if s.NumCPU <= 0 || s.Elapsed <= 0 {
		return 0, false
	}
	return float64(s.NumCPU) * s.Elapsed.Seconds(), true
}

// HarnessCPUSeconds sums the CPU-seconds of BOTH halves that reported CPU. ok is false
// when neither half could be read.
func (s Snapshot) HarnessCPUSeconds() (float64, bool) {
	var total float64
	var ok bool
	if s.Kernel.HaveCPU {
		total, ok = total+s.Kernel.CPUSeconds(), true
	}
	if s.Agent.HaveCPU {
		total, ok = total+s.Agent.CPUSeconds(), true
	}
	return total, ok
}

// CPUPercentOfHost is the harness's CPU-seconds as a percentage of host core-seconds
// elapsed — 100% means the harness kept every core of the box busy for the whole
// session. Unlike Half.CPUPercentAvg (where 100% is ONE busy core) this is normalized
// by core count, so it is comparable across boxes. ok is false when either side is
// unread.
func (s Snapshot) CPUPercentOfHost() (pct float64, ok bool) {
	cpu, haveCPU := s.HarnessCPUSeconds()
	coreS, haveCore := s.HostCoreSeconds()
	if !haveCPU || !haveCore || coreS <= 0 {
		return 0, false
	}
	return cpu / coreS * 100, true
}

// writeHostSection renders the host-vs-harness context: what the box offered, and the
// harness's slice of it as a fraction. Each fraction is emitted only when BOTH sides
// were read, so an unreadable host axis degrades to the bare capacity rather than to a
// fabricated percentage.
func writeHostSection(b *strings.Builder, s Snapshot) {
	fmt.Fprintf(b, "; host %d cores", s.NumCPU)
	if s.GOMAXPROCS > 0 && s.GOMAXPROCS != s.NumCPU {
		fmt.Fprintf(b, " (GOMAXPROCS %d)", s.GOMAXPROCS)
	}
	if s.Host.HaveRAM {
		fmt.Fprintf(b, " / %s ram", humanBytes(s.Host.TotalRAMBytes))
		if s.Host.AvailRAMBytes > 0 {
			fmt.Fprintf(b, " (%s avail)", humanBytes(s.Host.AvailRAMBytes))
		}
	}
	if s.Host.HaveLoad {
		fmt.Fprintf(b, ", load %.2f", s.Host.Load1)
	}
	b.WriteString("; harness/host ")
	if rss, ok := s.HarnessRSSBytes(); ok {
		fmt.Fprintf(b, "rss %s", humanBytes(rss))
		if pct, ok := s.RSSPercentOfHostRAM(); ok {
			fmt.Fprintf(b, " (%s of host ram)", humanPercent(pct))
		} else {
			b.WriteString(" (host ram n/a)")
		}
	} else {
		b.WriteString("rss n/a")
	}
	if cpu, ok := s.HarnessCPUSeconds(); ok {
		fmt.Fprintf(b, ", cpu %.1fs", cpu)
		if coreS, ok := s.HostCoreSeconds(); ok {
			fmt.Fprintf(b, " of %.1f core-s", coreS)
			if pct, ok := s.CPUPercentOfHost(); ok {
				fmt.Fprintf(b, " (%s)", humanPercent(pct))
			}
		}
	} else {
		b.WriteString(", cpu n/a")
	}
}

// humanPercent keeps a small-but-real fraction legible: a harness using 0.03% of a
// build server's RAM must not round to "0%", which reads as "unmeasured". One decimal
// everywhere else, because the interesting differences between two boxes ("12.5% of
// host ram" vs "1.6%") live in that digit.
func humanPercent(pct float64) string {
	if pct < 1 {
		return fmt.Sprintf("%.2f%%", pct)
	}
	return fmt.Sprintf("%.1f%%", pct)
}

// hostJSON is the durable `host` block on a ledger row. Pointer fields + omitempty keep
// an axis the platform could not read OUT of the row rather than recording a misleading
// 0 — except Cores, which runtime.NumCPU always knows.
type hostJSON struct {
	Cores            int      `json:"cores"`
	RAMTotalBytes    *uint64  `json:"ram_total_bytes,omitempty"`
	RAMAvailBytes    *uint64  `json:"ram_avail_bytes,omitempty"`
	Load1            *float64 `json:"load1,omitempty"`
	CoreSeconds      *float64 `json:"core_seconds,omitempty"`
	HarnessRSSBytes  *uint64  `json:"harness_rss_bytes,omitempty"`
	HarnessRSSPctRAM *float64 `json:"harness_rss_pct_of_host_ram,omitempty"`
	HarnessCPUS      *float64 `json:"harness_cpu_s,omitempty"`
	HarnessCPUPct    *float64 `json:"harness_cpu_pct_of_host_core_s,omitempty"`
}

func (s Snapshot) hostToJSON() hostJSON {
	j := hostJSON{Cores: s.NumCPU}
	if s.Host.HaveRAM {
		t := s.Host.TotalRAMBytes
		j.RAMTotalBytes = &t
		if s.Host.AvailRAMBytes > 0 {
			a := s.Host.AvailRAMBytes
			j.RAMAvailBytes = &a
		}
	}
	if s.Host.HaveLoad {
		l := s.Host.Load1
		j.Load1 = &l
	}
	if coreS, ok := s.HostCoreSeconds(); ok {
		j.CoreSeconds = &coreS
	}
	if rss, ok := s.HarnessRSSBytes(); ok {
		j.HarnessRSSBytes = &rss
	}
	if pct, ok := s.RSSPercentOfHostRAM(); ok {
		j.HarnessRSSPctRAM = &pct
	}
	if cpu, ok := s.HarnessCPUSeconds(); ok {
		j.HarnessCPUS = &cpu
	}
	if pct, ok := s.CPUPercentOfHost(); ok {
		j.HarnessCPUPct = &pct
	}
	return j
}

// writeHostGauges emits the fak_harness_host_* family. Only axes actually read emit a
// sample line, so a platform with no host-memory reader omits them rather than
// reporting a fake 0.
func writeHostGauges(b *strings.Builder, s Snapshot) {
	if s.Host.HaveRAM {
		writeHelp(b, "fak_harness_host_ram_bytes", "Physical RAM of the host running the fak guard harness, by kind (total/available), so the harness RSS gauges can be read as a fraction of the box.", "gauge")
		fmt.Fprintf(b, "fak_harness_host_ram_bytes{kind=\"total\"} %s\n", promFloat(float64(s.Host.TotalRAMBytes)))
		if s.Host.AvailRAMBytes > 0 {
			fmt.Fprintf(b, "fak_harness_host_ram_bytes{kind=\"available\"} %s\n", promFloat(float64(s.Host.AvailRAMBytes)))
		}
	}
	if s.Host.HaveLoad {
		writeHelp(b, "fak_harness_host_load1", "Best-effort 1-minute load average of the host running the fak guard harness.", "gauge")
		fmt.Fprintf(b, "fak_harness_host_load1 %s\n", promFloat(s.Host.Load1))
	}
	if coreS, ok := s.HostCoreSeconds(); ok {
		writeHelp(b, "fak_harness_host_core_seconds", "CPU capacity the host offered over the sampled window (cores x elapsed wall seconds) — the denominator for fak_harness_cpu_seconds_total.", "gauge")
		fmt.Fprintf(b, "fak_harness_host_core_seconds %s\n", promFloat(coreS))
	}
}
