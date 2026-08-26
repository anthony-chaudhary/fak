//go:build linux

package systembaseline

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const linuxClockTicks = uint64(100)

func capturePlatform() Snapshot {
	s := Snapshot{At: time.Now().UTC(), Host: readLinuxHost()}
	entries, err := os.ReadDir("/proc")
	if err != nil {
		s.ProcessNote = err.Error()
		return s
	}
	s.ProcessEnumerationOK = true
	for _, e := range entries {
		pid, err := strconv.Atoi(e.Name())
		if err != nil || !e.IsDir() {
			continue
		}
		p, ok := readLinuxProcess(pid)
		if !ok {
			s.ProcessUnreadable++
			s.AttributionIncomplete = true
			continue
		}
		s.Processes = append(s.Processes, p)
	}
	return s
}

func readLinuxHost() HostSample {
	var h HostSample
	if b, err := os.ReadFile("/proc/stat"); err == nil {
		fields := strings.Fields(strings.SplitN(string(b), "\n", 2)[0])
		if len(fields) >= 5 && fields[0] == "cpu" {
			var ticks []uint64
			for _, field := range fields[1:] {
				n, e := strconv.ParseUint(field, 10, 64)
				if e != nil {
					ticks = nil
					break
				}
				ticks = append(ticks, n)
			}
			if len(ticks) >= 4 {
				total, idle, _ := canonicalLinuxCPUTicks(ticks)
				h.CPUAvailable = true
				h.TotalCPUNS = total * uint64(time.Second) / linuxClockTicks
				h.BusyCPUNS = (total - idle) * uint64(time.Second) / linuxClockTicks
			}
		}
	}
	if f, err := os.Open("/proc/meminfo"); err == nil {
		defer f.Close()
		sc := bufio.NewScanner(f)
		for sc.Scan() {
			x := strings.Fields(sc.Text())
			if len(x) < 2 {
				continue
			}
			n, _ := strconv.ParseUint(x[1], 10, 64)
			switch x[0] {
			case "MemTotal:":
				h.MemoryTotal = n * 1024
			case "MemAvailable:":
				h.MemoryFree = n * 1024
			}
		}
		h.MemoryAvailable = h.MemoryTotal > 0 && h.MemoryFree > 0
	}
	return h
}

func readLinuxProcess(pid int) (ProcessSample, bool) {
	b, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "stat"))
	if err != nil {
		return ProcessSample{}, false
	}
	raw := string(b)
	open, close := strings.Index(raw, "("), strings.LastIndex(raw, ")")
	if open < 0 || close < open {
		return ProcessSample{}, false
	}
	fields := strings.Fields(strings.TrimSpace(raw[close+1:]))
	if len(fields) < 22 {
		return ProcessSample{}, false
	}
	ppid, e1 := strconv.Atoi(fields[1])
	user, e2 := strconv.ParseUint(fields[11], 10, 64)
	sys, e3 := strconv.ParseUint(fields[12], 10, 64)
	start, e4 := strconv.ParseUint(fields[19], 10, 64)
	rss, e5 := strconv.ParseInt(fields[21], 10, 64)
	if e1 != nil || e2 != nil || e3 != nil || e4 != nil || e5 != nil {
		return ProcessSample{}, false
	}
	if rss < 0 {
		rss = 0
	}
	return ProcessSample{PID: pid, PPID: ppid, StartID: start, Image: raw[open+1 : close], CPUAvailable: true, CPUNS: (user + sys) * uint64(time.Second) / linuxClockTicks, RSSAvailable: true, RSSBytes: uint64(rss) * uint64(os.Getpagesize())}, true
}
