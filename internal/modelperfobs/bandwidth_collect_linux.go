//go:build linux

package modelperfobs

import (
	"bufio"
	"io"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

func collectHostSnapshot() (hostSnapshot, error) {
	s := hostSnapshot{at: time.Now(), collector: "procfs"}
	if f, e := os.Open("/proc/meminfo"); e == nil {
		defer f.Close()
		m := map[string]uint64{}
		scanFields(f, func(p []string) {
			if len(p) >= 2 {
				v, _ := strconv.ParseUint(p[1], 10, 64)
				m[strings.TrimSuffix(p[0], ":")] = v * 1024
			}
		})
		if m["MemTotal"] > 0 {
			total := m["MemTotal"]
			s.host.PhysicalTotalBytes = cloneU64(&total)
			a := m["MemAvailable"]
			s.host.PhysicalAvailableBytes = cloneU64(&a)
			s.availability.PhysicalMemory = true
		}
		if total, ok := m["SwapTotal"]; ok {
			free := m["SwapFree"]
			used := uint64(0)
			if total >= free {
				used = total - free
			}
			s.host.SwapTotalBytes = &total
			s.host.SwapUsedBytes = &used
			s.availability.MemoryPressure = true
		}
	}
	if f, e := os.Open("/proc/self/status"); e == nil {
		defer f.Close()
		scanFields(f, func(p []string) {
			if len(p) >= 2 && p[0] == "VmRSS:" {
				v, _ := strconv.ParseUint(p[1], 10, 64)
				v *= 1024
				s.host.ProcessResidentBytes = &v
				s.availability.ProcessMemory = true
			}
		})
	}
	if b, e := os.ReadFile("/proc/self/stat"); e == nil {
		if minor, major, ok := parseProcSelfStatFaults(string(b)); ok {
			s.host.ProcessMinorFaults = &minor
			s.host.ProcessMajorFaults = &major
			s.availability.MemoryPressure = true
		}
	}
	if b, e := os.ReadFile("/proc/pressure/memory"); e == nil {
		if parseProcPressureMemory(string(b), &s.host) {
			s.availability.MemoryPressure = true
		}
	}
	if b, e := os.ReadFile("/proc/vmstat"); e == nil {
		if parseProcVMStat(string(b), &s.host) {
			s.availability.MemoryPressure = true
		}
	}
	if f, e := os.Open("/proc/self/io"); e == nil {
		defer f.Close()
		scanFields(f, func(p []string) {
			if len(p) == 2 {
				v, _ := strconv.ParseUint(p[1], 10, 64)
				if p[0] == "read_bytes:" {
					s.host.ProcessReadBytes = &v
				}
				if p[0] == "write_bytes:" {
					s.host.ProcessWriteBytes = &v
				}
			}
		})
		if s.host.ProcessReadBytes != nil || s.host.ProcessWriteBytes != nil {
			s.host.ProcessIOScope = "process-storage-io-not-dram"
			s.availability.ProcessIO = true
		}
	}
	return s, nil
}

func parseProcSelfStatFaults(line string) (minor, major uint64, ok bool) {
	// comm is parenthesized and may contain spaces; fields after the final ')'
	// resume at field 3. minflt and majflt are fields 10 and 12.
	end := strings.LastIndex(line, ")")
	if end < 0 || end+1 >= len(line) {
		return 0, 0, false
	}
	fields := strings.Fields(line[end+1:])
	if len(fields) < 10 {
		return 0, 0, false
	}
	minor, e1 := strconv.ParseUint(fields[7], 10, 64)
	major, e2 := strconv.ParseUint(fields[9], 10, 64)
	return minor, major, e1 == nil && e2 == nil
}

func parseProcPressureMemory(input string, host *HostSignals) bool {
	if host == nil {
		return false
	}
	available := false
	sc := bufio.NewScanner(strings.NewReader(input))
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) < 2 || (fields[0] != "some" && fields[0] != "full") {
			continue
		}
		scope := fields[0]
		for _, field := range fields[1:] {
			key, raw, ok := strings.Cut(field, "=")
			if !ok {
				continue
			}
			switch key {
			case "avg10", "avg60", "avg300":
				value, err := strconv.ParseFloat(raw, 64)
				if err != nil || math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 100 {
					continue
				}
				switch {
				case scope == "some" && key == "avg10":
					host.MemoryPressureSomeAvg10Percent = cloneFloat(&value)
				case scope == "some" && key == "avg60":
					host.MemoryPressureSomeAvg60Percent = cloneFloat(&value)
				case scope == "some" && key == "avg300":
					host.MemoryPressureSomeAvg300Percent = cloneFloat(&value)
				case scope == "full" && key == "avg10":
					host.MemoryPressureFullAvg10Percent = cloneFloat(&value)
				case scope == "full" && key == "avg60":
					host.MemoryPressureFullAvg60Percent = cloneFloat(&value)
				case scope == "full" && key == "avg300":
					host.MemoryPressureFullAvg300Percent = cloneFloat(&value)
				}
				available = true
			case "total":
				value, err := strconv.ParseUint(raw, 10, 64)
				if err != nil {
					continue
				}
				if scope == "some" {
					host.MemoryPressureSomeTotalStallMicroseconds = cloneU64(&value)
				} else {
					host.MemoryPressureFullTotalStallMicroseconds = cloneU64(&value)
				}
				available = true
			}
		}
	}
	return available
}

func parseProcVMStat(input string, host *HostSignals) bool {
	if host == nil {
		return false
	}
	type reclaimCounter struct {
		destination    **uint64
		modernPresent  bool
		legacyTotal    uint64
		legacyPresent  bool
		legacyOverflow bool
	}
	reclaim := map[string]*reclaimCounter{
		"pgscan_kswapd":  {destination: &host.MemoryReclaimKswapdScannedPagesTotal},
		"pgscan_direct":  {destination: &host.MemoryReclaimDirectScannedPagesTotal},
		"pgsteal_kswapd": {destination: &host.MemoryReclaimKswapdReclaimedPagesTotal},
		"pgsteal_direct": {destination: &host.MemoryReclaimDirectReclaimedPagesTotal},
	}
	available := false
	sc := bufio.NewScanner(strings.NewReader(input))
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) != 2 {
			continue
		}
		key := fields[0]
		counter := reclaim[key]
		legacy := false
		if counter != nil {
			// Presence, rather than successful parsing, controls precedence: a
			// malformed aggregate must stay omitted instead of being concealed by
			// a legacy fallback from mixed input.
			counter.modernPresent = true
		} else if base, ok := legacyVMStatBase(key); ok {
			counter = reclaim[base]
			legacy = true
		}
		if counter != nil {
			value, err := strconv.ParseUint(fields[1], 10, 64)
			if err != nil {
				continue
			}
			if legacy {
				counter.legacyPresent = true
				if counter.legacyOverflow || value > math.MaxUint64-counter.legacyTotal {
					counter.legacyOverflow = true
					continue
				}
				counter.legacyTotal += value
				continue
			}
			*counter.destination = cloneU64(&value)
			available = true
			continue
		}
		var destination **uint64
		switch key {
		case "pswpin":
			destination = &host.MemorySwapInPagesTotal
		case "pswpout":
			destination = &host.MemorySwapOutPagesTotal
		default:
			continue
		}
		value, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			continue
		}
		*destination = cloneU64(&value)
		available = true
	}
	for _, key := range []string{"pgscan_kswapd", "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct"} {
		counter := reclaim[key]
		if counter.modernPresent || !counter.legacyPresent || counter.legacyOverflow {
			continue
		}
		*counter.destination = cloneU64(&counter.legacyTotal)
		available = true
	}
	return available
}

func legacyVMStatBase(key string) (string, bool) {
	separator := strings.LastIndexByte(key, '_')
	if separator < 0 {
		return "", false
	}
	switch key[separator+1:] {
	case "dma", "dma32", "normal", "high", "movable":
	default:
		return "", false
	}
	base := key[:separator]
	switch base {
	case "pgscan_kswapd", "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct":
		return base, true
	default:
		return "", false
	}
}

func scanFields(r io.Reader, visit func([]string)) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		visit(strings.Fields(scanner.Text()))
	}
}
