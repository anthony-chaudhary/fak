// Package memgate checks whether a heavy model load should proceed under memory pressure.
package memgate

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	SafetyMarginGB    = 2.0
	HighWiredFraction = 0.40
	HolderGB          = 1.0
)

type Memory struct {
	TotalBytes      int64
	FreeBytes       int64
	PurgeableBytes  int64
	WiredBytes      int64
	CompressedBytes int64
	AvailableBytes  int64
}

type Holder struct {
	PID   int     `json:"pid"`
	RSSGB float64 `json:"rss_gb"`
	Comm  string  `json:"comm"`
}

type Snapshot struct {
	Platform       string   `json:"platform"`
	TotalGB        float64  `json:"total_gb"`
	FreeGB         float64  `json:"free_gb"`
	AvailableGB    float64  `json:"available_gb"`
	PurgeableGB    float64  `json:"purgeable_gb"`
	WiredGB        float64  `json:"wired_gb"`
	CompressedGB   float64  `json:"compressed_gb"`
	SafetyMarginGB float64  `json:"safety_margin_gb"`
	HighWired      bool     `json:"high_wired"`
	Holders        []Holder `json:"holders"`
	Note           string   `json:"note"`
	RequireGB      float64  `json:"require_gb,omitempty"`
	Admit          *bool    `json:"admit,omitempty"`
	ShortfallGB    float64  `json:"shortfall_gb,omitempty"`
}

func ParseDarwin(vmStat string, pageSize, total int64) Memory {
	vals := map[string]int64{}
	for _, line := range strings.Split(vmStat, "\n") {
		line = strings.TrimSuffix(strings.TrimSpace(line), ".")
		if !strings.Contains(line, ":") {
			continue
		}
		key, rest, _ := strings.Cut(line, ":")
		fields := strings.Fields(strings.TrimSpace(rest))
		if len(fields) == 0 {
			continue
		}
		n, err := strconv.ParseInt(fields[0], 10, 64)
		if err == nil {
			vals[strings.ToLower(strings.TrimSpace(key))] = n
		}
	}
	free := firstVal(vals, "pages free", "free pages") * pageSize
	purgeable := vals["pages purgeable"] * pageSize
	wired := firstVal(vals, "pages wired down", "wired pages") * pageSize
	compressed := vals["pages occupied by compressor"] * pageSize
	return Memory{
		TotalBytes:      total,
		FreeBytes:       free,
		PurgeableBytes:  purgeable,
		WiredBytes:      wired,
		CompressedBytes: compressed,
		AvailableBytes:  max64(free+purgeable-int64(SafetyMarginGB*1e9), 0),
	}
}

func ParseLinux(meminfo string) Memory {
	vals := map[string]int64{}
	sc := bufio.NewScanner(strings.NewReader(meminfo))
	for sc.Scan() {
		key, rest, ok := strings.Cut(sc.Text(), ":")
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		n, err := strconv.ParseInt(fields[0], 10, 64)
		if err == nil {
			vals[key] = n * 1024
		}
	}
	total := vals["MemTotal"]
	free := vals["MemFree"]
	avail := vals["MemAvailable"]
	if avail == 0 {
		avail = free + vals["Cached"]
	}
	return Memory{
		TotalBytes:      total,
		FreeBytes:       free,
		AvailableBytes:  max64(avail-int64(SafetyMarginGB*1e9), 0),
		PurgeableBytes:  vals["Cached"],
		WiredBytes:      0,
		CompressedBytes: 0,
	}
}

func ParseHolders(psText string) []Holder {
	var holders []Holder
	lines := strings.Split(psText, "\n")
	for _, line := range lines[1:] {
		parts := strings.Fields(line)
		if len(parts) < 3 {
			continue
		}
		pid, err1 := strconv.Atoi(parts[0])
		rssKB, err2 := strconv.ParseFloat(parts[1], 64)
		if err1 != nil || err2 != nil {
			continue
		}
		rssGB := rssKB / 1e6
		if rssGB < HolderGB {
			continue
		}
		comm := strings.Join(parts[2:], " ")
		if len(comm) > 60 {
			comm = comm[:60]
		}
		holders = append(holders, Holder{PID: pid, RSSGB: round2(rssGB), Comm: comm})
	}
	sort.Slice(holders, func(i, j int) bool { return holders[i].RSSGB > holders[j].RSSGB })
	return holders
}

func BuildSnapshot(platform string, mem Memory, holders []Holder) Snapshot {
	totalGB := float64(mem.TotalBytes) / 1e9
	wiredGB := float64(mem.WiredBytes) / 1e9
	highWired := false
	if totalGB > 0 {
		highWired = wiredGB/totalGB > HighWiredFraction
	}
	note := "ok"
	if highWired {
		note = "wired > 40% of RAM - a Metal/GPU model is likely resident in unified memory; 'available' does NOT include wired, so a new CPU model load may still OOM against the GPU resident. Stop the GPU holder first."
	}
	return Snapshot{
		Platform:       platform,
		TotalGB:        round2(totalGB),
		FreeGB:         round2(float64(mem.FreeBytes) / 1e9),
		AvailableGB:    round2(float64(mem.AvailableBytes) / 1e9),
		PurgeableGB:    round2(float64(mem.PurgeableBytes) / 1e9),
		WiredGB:        round2(wiredGB),
		CompressedGB:   round2(float64(mem.CompressedBytes) / 1e9),
		SafetyMarginGB: SafetyMarginGB,
		HighWired:      highWired,
		Holders:        holders,
		Note:           note,
	}
}

func Evaluate(s Snapshot, requireGB float64) Snapshot {
	ok := s.AvailableGB >= requireGB && !s.HighWired
	s.RequireGB = requireGB
	s.Admit = &ok
	if requireGB > s.AvailableGB {
		s.ShortfallGB = round2(requireGB - s.AvailableGB)
	}
	return s
}

func ReadMemory() (Memory, error) {
	switch runtime.GOOS {
	case "darwin":
		vm, err := runOut("vm_stat")
		if err != nil {
			return Memory{}, err
		}
		ps, err := runOut("sysctl", "-n", "hw.pagesize")
		if err != nil {
			return Memory{}, err
		}
		total, err := runOut("sysctl", "-n", "hw.memsize")
		if err != nil {
			return Memory{}, err
		}
		pageSize, _ := strconv.ParseInt(strings.TrimSpace(ps), 10, 64)
		totalBytes, _ := strconv.ParseInt(strings.TrimSpace(total), 10, 64)
		return ParseDarwin(vm, pageSize, totalBytes), nil
	case "linux":
		b, err := os.ReadFile("/proc/meminfo")
		if err != nil {
			return Memory{}, err
		}
		return ParseLinux(string(b)), nil
	default:
		return Memory{}, fmt.Errorf("unsupported platform for memgate: %s", runtime.GOOS)
	}
}

func BigHolders() []Holder {
	out, err := runOut("ps", "-axo", "pid,rss,comm")
	if err != nil {
		return nil
	}
	return ParseHolders(out)
}

func CurrentSnapshot() (Snapshot, error) {
	mem, err := ReadMemory()
	if err != nil {
		return Snapshot{}, err
	}
	return BuildSnapshot(runtime.GOOS, mem, BigHolders()), nil
}

func runOut(name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func firstVal(m map[string]int64, keys ...string) int64 {
	for _, k := range keys {
		if v, ok := m[k]; ok {
			return v
		}
	}
	return 0
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func round2(v float64) float64 {
	if v >= 0 {
		return float64(int(v*100+0.5)) / 100
	}
	return float64(int(v*100-0.5)) / 100
}
