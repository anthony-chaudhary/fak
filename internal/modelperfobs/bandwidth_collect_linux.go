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
