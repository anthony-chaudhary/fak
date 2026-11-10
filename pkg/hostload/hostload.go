// Package hostload reads the host's 1/5/15-minute load averages without cgo.
// It is shared by fak-spawn's host-capacity gate and fak-flow's load-scaled
// scoped-validate budget, so both read the same numbers the same way.
package hostload

import (
	"math"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
)

// Read returns the load averages for the current platform; ok is false on an
// unsupported platform or when the reading fails. Windows keeps no load
// average, so there all three are one short sample of busy cores plus threads
// ready to run (see windowsLoad).
func Read() (l1, l5, l15 float64, ok bool) {
	switch runtime.GOOS {
	case "darwin":
		return ReadDarwin()
	case "linux":
		return ReadLinux()
	case "windows":
		l1, ok = readWindows()
		return l1, l1, l1, ok
	}
	return 0, 0, 0, false
}

// windowsLoad approximates a load average the way Linux counts it, running
// plus runnable threads: the busy fraction of total CPU time between two
// GetSystemTimes readings, times cores, plus the mean system processor queue
// length over that span. Times are in the same unit; total includes idle.
func windowsLoad(idle0, total0, idle1, total1 uint64, cores int, queue float64) (float64, bool) {
	if cores < 1 || total1 <= total0 || idle1 < idle0 || math.IsNaN(queue) || math.IsInf(queue, 0) || queue < 0 {
		return 0, false
	}
	dTotal, dIdle := total1-total0, idle1-idle0
	if dIdle > dTotal {
		return 0, false
	}
	busy := float64(dTotal-dIdle) / float64(dTotal)
	return busy*float64(cores) + queue, true
}

// ReadDarwin parses `sysctl -n vm.loadavg`, whose output looks like
// `{ 1.2 3.4 5.6 }`.
func ReadDarwin() (l1, l5, l15 float64, ok bool) {
	out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
	if err != nil {
		return 0, 0, 0, false
	}
	return parseDarwin(string(out))
}

func parseDarwin(s string) (l1, l5, l15 float64, ok bool) {
	return parse(strings.NewReplacer("{", " ", "}", " ").Replace(s))
}

// ReadLinux reads the first three fields of /proc/loadavg.
func ReadLinux() (l1, l5, l15 float64, ok bool) {
	data, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, 0, 0, false
	}
	return parse(string(data))
}

func parse(s string) (l1, l5, l15 float64, ok bool) {
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return 0, 0, 0, false
	}
	a, err1 := strconv.ParseFloat(fields[0], 64)
	b, err2 := strconv.ParseFloat(fields[1], 64)
	c, err3 := strconv.ParseFloat(fields[2], 64)
	if err1 != nil || err2 != nil || err3 != nil {
		return 0, 0, 0, false
	}
	return a, b, c, true
}

// Fixed perflib title indices of the System object and its Processor Queue
// Length counter, the same on every Windows install.
const (
	perfObjectSystem                = 2
	perfCounterProcessorQueueLength = 44
)

// perfDataCounter returns a 4-byte counter of a single-instance object from a
// HKEY_PERFORMANCE_DATA buffer (winperf.h PERF_DATA_BLOCK layout). Every
// offset is bounds-checked; a malformed buffer reads as ok=false.
func perfDataCounter(b []byte, object, counter uint32) (uint32, bool) {
	if len(b) < 32 || string([]byte{b[0], b[2], b[4], b[6]}) != "PERF" {
		return 0, false
	}
	headerLen, _ := le32(b, 24)
	objects, _ := le32(b, 28)
	off := uint64(headerLen)
	for i := uint32(0); i < objects; i++ {
		total, ok1 := le32(b, off)
		defLen, ok2 := le32(b, off+4)
		hdrLen, ok3 := le32(b, off+8)
		index, ok4 := le32(b, off+12)
		numCounters, ok5 := le32(b, off+32)
		instances, ok6 := le32(b, off+40)
		if !(ok1 && ok2 && ok3 && ok4 && ok5 && ok6) || total == 0 {
			return 0, false
		}
		if index == object {
			if int32(instances) != -1 {
				return 0, false
			}
			def := off + uint64(hdrLen)
			for c := uint32(0); c < numCounters; c++ {
				size, okA := le32(b, def)
				name, okB := le32(b, def+4)
				csize, okC := le32(b, def+32)
				coff, okD := le32(b, def+36)
				if !(okA && okB && okC && okD) || size == 0 {
					return 0, false
				}
				if name == counter {
					if csize != 4 {
						return 0, false
					}
					return le32(b, off+uint64(defLen)+uint64(coff))
				}
				def += uint64(size)
			}
			return 0, false
		}
		off += uint64(total)
	}
	return 0, false
}

func le32(b []byte, off uint64) (uint32, bool) {
	if off+4 > uint64(len(b)) {
		return 0, false
	}
	return uint32(b[off]) | uint32(b[off+1])<<8 | uint32(b[off+2])<<16 | uint32(b[off+3])<<24, true
}
