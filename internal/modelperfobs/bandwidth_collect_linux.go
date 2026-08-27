//go:build linux

package modelperfobs

import (
	"bufio"
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
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			p := strings.Fields(sc.Text())
			if len(p) >= 2 {
				v, _ := strconv.ParseUint(p[1], 10, 64)
				m[strings.TrimSuffix(p[0], ":")] = v * 1024
			}
		}
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
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			p := strings.Fields(sc.Text())
			if len(p) >= 2 && p[0] == "VmRSS:" {
				v, _ := strconv.ParseUint(p[1], 10, 64)
				v *= 1024
				s.host.ProcessResidentBytes = &v
				s.availability.ProcessMemory = true
			}
		}
	}
	if b, e := os.ReadFile("/proc/self/stat"); e == nil {
		if minor, major, ok := parseProcSelfStatFaults(string(b)); ok {
			s.host.ProcessMinorFaults = &minor
			s.host.ProcessMajorFaults = &major
			s.availability.MemoryPressure = true
		}
	}
	if f, e := os.Open("/proc/self/io"); e == nil {
		defer f.Close()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			p := strings.Fields(sc.Text())
			if len(p) == 2 {
				v, _ := strconv.ParseUint(p[1], 10, 64)
				if p[0] == "read_bytes:" {
					s.host.ProcessReadBytes = &v
				}
				if p[0] == "write_bytes:" {
					s.host.ProcessWriteBytes = &v
				}
			}
		}
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
