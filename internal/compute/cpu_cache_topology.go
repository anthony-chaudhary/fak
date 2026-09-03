package compute

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// HostCPUTopologyLevel denotes L1d, L1i, L2, or L3.
type HostCPUTopologyLevel string

const (
	LevelL1d HostCPUTopologyLevel = "L1d"
	LevelL1i HostCPUTopologyLevel = "L1i"
	LevelL2  HostCPUTopologyLevel = "L2"
	LevelL3  HostCPUTopologyLevel = "L3"
)

// HostCPUTopologyRow describes a discovered host CPU hierarchy tier.
// Missing data reports Status: "unknown" and SizeBytes: -1 (never zero).
type HostCPUTopologyRow struct {
	Level     HostCPUTopologyLevel `json:"level"`
	SizeBytes int64                `json:"size_bytes"`
	Status    string               `json:"status"` // "known" | "unknown"
	LineSize  int                  `json:"line_size,omitempty"`
}

// HostCPUTopologyEnvelope packages the discovered CPU hierarchy for native execution receipts.
type HostCPUTopologyEnvelope struct {
	ModelEnvelope string               `json:"model_envelope"`
	WitnessSource string               `json:"witness_source"`
	Rows          []HostCPUTopologyRow `json:"rows"`
}

// ParseByteSizeString converts strings like "32K", "512K", "16M", "1G", or raw integers into bytes.
func ParseByteSizeString(s string) (int64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return -1, fmt.Errorf("empty size string")
	}

	mult := int64(1)
	last := s[len(s)-1]
	switch last {
	case 'k', 'K':
		mult = 1024
		s = s[:len(s)-1]
	case 'm', 'M':
		mult = 1024 * 1024
		s = s[:len(s)-1]
	case 'g', 'G':
		mult = 1024 * 1024 * 1024
		s = s[:len(s)-1]
	}

	val, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return -1, err
	}
	return val * mult, nil
}

// ParseLinuxSysfsCPUTopology reads CPU topology from a Linux sysfs tree (e.g. /sys/devices/system/cpu/cpu0/cache).
func ParseLinuxSysfsCPUTopology(dirRoot string, modelName string) (HostCPUTopologyEnvelope, error) {
	env := HostCPUTopologyEnvelope{
		ModelEnvelope: modelName,
		WitnessSource: "linux-sysfs",
	}

	rowsMap := map[HostCPUTopologyLevel]HostCPUTopologyRow{
		LevelL1d: {Level: LevelL1d, SizeBytes: -1, Status: "unknown"},
		LevelL1i: {Level: LevelL1i, SizeBytes: -1, Status: "unknown"},
		LevelL2:  {Level: LevelL2, SizeBytes: -1, Status: "unknown"},
		LevelL3:  {Level: LevelL3, SizeBytes: -1, Status: "unknown"},
	}

	indices, err := filepath.Glob(filepath.Join(dirRoot, "index*"))
	if err != nil || len(indices) == 0 {
		env.Rows = orderedRowsFromMap(rowsMap)
		return env, nil
	}

	for _, idxDir := range indices {
		levelBytes, _ := os.ReadFile(filepath.Join(idxDir, "level"))
		typeBytes, _ := os.ReadFile(filepath.Join(idxDir, "type"))
		sizeBytes, _ := os.ReadFile(filepath.Join(idxDir, "size"))
		lineBytes, _ := os.ReadFile(filepath.Join(idxDir, "coherency_line_size"))

		lvlStr := strings.TrimSpace(string(levelBytes))
		typStr := strings.TrimSpace(string(typeBytes))
		szStr := strings.TrimSpace(string(sizeBytes))
		lineStr := strings.TrimSpace(string(lineBytes))

		var levelKey HostCPUTopologyLevel
		switch lvlStr {
		case "1":
			if typStr == "Instruction" {
				levelKey = LevelL1i
			} else {
				levelKey = LevelL1d
			}
		case "2":
			levelKey = LevelL2
		case "3":
			levelKey = LevelL3
		default:
			continue
		}

		lineSize, _ := strconv.Atoi(lineStr)
		parsedBytes, err := ParseByteSizeString(szStr)
		if err != nil || parsedBytes <= 0 {
			rowsMap[levelKey] = HostCPUTopologyRow{
				Level:     levelKey,
				SizeBytes: -1,
				Status:    "unknown",
				LineSize:  lineSize,
			}
		} else {
			rowsMap[levelKey] = HostCPUTopologyRow{
				Level:     levelKey,
				SizeBytes: parsedBytes,
				Status:    "known",
				LineSize:  lineSize,
			}
		}
	}

	env.Rows = orderedRowsFromMap(rowsMap)
	return env, nil
}

func orderedRowsFromMap(m map[HostCPUTopologyLevel]HostCPUTopologyRow) []HostCPUTopologyRow {
	return []HostCPUTopologyRow{
		m[LevelL1d],
		m[LevelL1i],
		m[LevelL2],
		m[LevelL3],
	}
}

// DiscoverHostCPUTopology queries the host operating system for L1/L2/L3 topology metrics.
func DiscoverHostCPUTopology(modelName string) HostCPUTopologyEnvelope {
	if modelName == "" {
		modelName = "qwen3.8-native"
	}

	switch runtime.GOOS {
	case "linux":
		env, err := ParseLinuxSysfsCPUTopology("/sys/devices/system/cpu/cpu0/cache", modelName)
		if err == nil {
			return env
		}
	case "darwin":
		return discoverDarwinCPUTopology(modelName)
	}

	return HostCPUTopologyEnvelope{
		ModelEnvelope: modelName,
		WitnessSource: "unsupported-host-os",
		Rows: []HostCPUTopologyRow{
			{Level: LevelL1d, SizeBytes: -1, Status: "unknown"},
			{Level: LevelL1i, SizeBytes: -1, Status: "unknown"},
			{Level: LevelL2, SizeBytes: -1, Status: "unknown"},
			{Level: LevelL3, SizeBytes: -1, Status: "unknown"},
		},
	}
}

func discoverDarwinCPUTopology(modelName string) HostCPUTopologyEnvelope {
	querySysctl := func(prefix string) (int64, string) {
		key := fmt.Sprintf("hw.%s%ssize", prefix, "cache")
		out, err := exec.Command("sysctl", "-n", key).Output()
		if err != nil {
			return -1, "unknown"
		}
		val, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
		if err != nil || val <= 0 {
			return -1, "unknown"
		}
		return val, "known"
	}

	l1d, sL1d := querySysctl("l1d")
	l1i, sL1i := querySysctl("l1i")
	l2, sL2 := querySysctl("l2")
	l3, sL3 := querySysctl("l3")

	return HostCPUTopologyEnvelope{
		ModelEnvelope: modelName,
		WitnessSource: "darwin-sysctl",
		Rows: []HostCPUTopologyRow{
			{Level: LevelL1d, SizeBytes: l1d, Status: sL1d},
			{Level: LevelL1i, SizeBytes: l1i, Status: sL1i},
			{Level: LevelL2, SizeBytes: l2, Status: sL2},
			{Level: LevelL3, SizeBytes: l3, Status: sL3},
		},
	}
}
