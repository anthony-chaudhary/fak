// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// and Strix Halo APU operational serving profiles and validation.
package amdgpu

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// StrixTarget represents a discovered AMD Strix Halo appliance target.
type StrixTarget struct {
	Mode           string  `json:"mode"`                 // "local" | "ssh" | "sim"
	Host           string  `json:"host"`                 // hostname, IP, or "localhost"
	Reachable      bool    `json:"reachable"`            // whether device responds to probe
	CPUModel       string  `json:"cpu_model"`            // e.g. "AMD Ryzen AI MAX+ 395"
	GPUName        string  `json:"gpu_name"`             // e.g. "AMD Radeon 8060S Graphics (RADV STRIX_HALO)"
	TargetISA      string  `json:"target_isa"`           // "gfx1151"
	ComputeUnits   int     `json:"compute_units"`        // 40
	TotalRAMBytes  int64   `json:"total_ram_bytes"`      // total physical UMA memory
	UMABufferBytes int64   `json:"uma_buffer_bytes"`     // usable GTT/VRAM aperture
	DPMLevel       string  `json:"power_dpm_level"`      // "high" | "auto" | "manual"
	LockupTimeout  int     `json:"lockup_timeout"`       // -1 or timeout in seconds
	VulkanICD      string  `json:"vulkan_icd"`           // path to active ICD
	LatencyMS      float64 `json:"roundtrip_latency_ms"` // probe RTT in milliseconds
	DiscoveredAt   string  `json:"discovered_at"`        // RFC3339 timestamp
	Error          string  `json:"error,omitempty"`      // probe error if unreachable
}

// StrixPresenceCache is the serialized presence cache stored in _scratch.
type StrixPresenceCache struct {
	Target     StrixTarget `json:"target"`
	Timestamp  int64       `json:"timestamp"`
	TTLSeconds int         `json:"ttl_seconds"`
}

const (
	DefaultStrixHost    = "strix1"
	FallbackStrixMDNS   = "strix-halo-fak.local"
	DefaultPresenceTTL  = 60 * time.Second
	DefaultProbeTimeout = 3 * time.Second
	StrixPresenceFile   = "_scratch/strix_presence.json"
)

// DiscoverStrixTarget finds a local or remote AMD Strix Halo appliance.
func DiscoverStrixTarget(ctx context.Context, hostOverride string) (*StrixTarget, error) {
	// 1. Check local appliance first if running on Linux and no explicit remote host requested
	if (hostOverride == "" || hostOverride == "localhost" || hostOverride == "local" || hostOverride == "127.0.0.1") && runtime.GOOS == "linux" {
		target, err := probeLocalStrix()
		if err == nil && target != nil && target.Reachable {
			return target, nil
		}
	}

	// 2. Resolve target host
	host := hostOverride
	if host == "" {
		host = os.Getenv("FAK_STRIX_HOST")
	}
	if host == "" {
		host = DefaultStrixHost
	}

	// 3. Check scratch cache
	if cached, ok := loadPresenceCache(host); ok {
		return cached, nil
	}

	// 4. Remote probe via SSH
	target, err := probeRemoteStrix(ctx, host)
	if err != nil {
		// Try fallback mDNS if default strix1 failed and no explicit override
		if hostOverride == "" && host == DefaultStrixHost {
			if fbTarget, fbErr := probeRemoteStrix(ctx, FallbackStrixMDNS); fbErr == nil && fbTarget.Reachable {
				savePresenceCache(fbTarget)
				return fbTarget, nil
			}
		}
		// Return unreached target
		unreached := &StrixTarget{
			Mode:         "ssh",
			Host:         host,
			Reachable:    false,
			TargetISA:    "gfx1151",
			ComputeUnits: 40,
			DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
			Error:        err.Error(),
		}
		return unreached, err
	}

	savePresenceCache(target)
	return target, nil
}

func probeLocalStrix() (*StrixTarget, error) {
	// Verify /dev/kfd presence
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return nil, fmt.Errorf("local: /dev/kfd not found: %w", err)
	}

	// Read CPU model from /proc/cpuinfo
	cpuData, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return nil, fmt.Errorf("local: cannot read /proc/cpuinfo: %w", err)
	}
	cpuStr := string(cpuData)
	isAPU, isStrix := DetectAPU(cpuStr)
	if !isStrix {
		return nil, fmt.Errorf("local: CPU is not an AMD Strix Halo (APU=%v)", isAPU)
	}

	target := &StrixTarget{
		Mode:         "local",
		Host:         "localhost",
		Reachable:    true,
		TargetISA:    "gfx1151",
		ComputeUnits: 40,
		DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
	}

	// Extract CPU Model
	for _, line := range strings.Split(cpuStr, "\n") {
		if strings.HasPrefix(line, "model name") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				target.CPUModel = strings.TrimSpace(parts[1])
				break
			}
		}
	}

	// Read /proc/meminfo
	if memData, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, line := range strings.Split(string(memData), "\n") {
			if strings.HasPrefix(line, "MemTotal:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					if kb, err := strconv.ParseInt(fields[1], 10, 64); err == nil {
						target.TotalRAMBytes = kb * 1024
					}
				}
				break
			}
		}
	}

	// Read DPM level and lockup timeout from sysfs if accessible
	if dpm, err := os.ReadFile("/sys/class/drm/card1/device/power_dpm_force_performance_level"); err == nil {
		target.DPMLevel = strings.TrimSpace(string(dpm))
	} else if dpm, err := os.ReadFile("/sys/class/drm/card0/device/power_dpm_force_performance_level"); err == nil {
		target.DPMLevel = strings.TrimSpace(string(dpm))
	}

	if lto, err := os.ReadFile("/sys/module/amdgpu/parameters/lockup_timeout"); err == nil {
		if v, err := strconv.Atoi(strings.TrimSpace(string(lto))); err == nil {
			target.LockupTimeout = v
		}
	}

	target.GPUName = "AMD Radeon 8060S Graphics (RADV STRIX_HALO)"
	target.UMABufferBytes = int64(float64(target.TotalRAMBytes) * 0.88)
	return target, nil
}

func probeRemoteStrix(ctx context.Context, host string) (*StrixTarget, error) {
	start := time.Now()
	probeCtx, cancel := context.WithTimeout(ctx, DefaultProbeTimeout)
	defer cancel()

	// Probe script that collects hardware facts into JSON
	probeCmd := `python3 -c '
import json, os, subprocess

info = {
    "reachable": True,
    "target_isa": "gfx1151",
    "compute_units": 40
}

# CPU
try:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
except Exception:
    info["cpu_model"] = "AMD Ryzen AI MAX+ 395"

# RAM
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                info["total_ram_bytes"] = int(line.split()[1]) * 1024
                break
except Exception:
    info["total_ram_bytes"] = 64 * 1024 * 1024 * 1024

# GPU & Sysfs
try:
    for card in ["card1", "card0"]:
        p = f"/sys/class/drm/{card}/device/power_dpm_force_performance_level"
        if os.path.exists(p):
            with open(p) as f:
                info["power_dpm_level"] = f.read().strip()
            break
except Exception:
    pass

try:
    with open("/sys/module/amdgpu/parameters/lockup_timeout") as f:
        info["lockup_timeout"] = int(f.read().strip().split(",")[0])
except Exception:
    info["lockup_timeout"] = -1

# Vulkan device name
info["gpu_name"] = "AMD Radeon 8060S Graphics (RADV STRIX_HALO)"
info["vulkan_icd"] = "/usr/share/vulkan/icd.d/radeon_icd.json"

print(json.dumps(info))
'`

	cmd := exec.CommandContext(probeCtx, "ssh",
		"-o", "BatchMode=yes",
		"-o", "ConnectTimeout=2",
		"-o", "StrictHostKeyChecking=accept-new",
		host,
		probeCmd,
	)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	rtt := time.Since(start).Seconds() * 1000.0

	if err != nil {
		return nil, fmt.Errorf("ssh probe to %s failed: %w", host, err)
	}

	var target StrixTarget
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	lastLine := lines[len(lines)-1]
	if err := json.Unmarshal([]byte(lastLine), &target); err != nil {
		return nil, fmt.Errorf("failed to parse probe output from %s: %w", host, err)
	}

	target.Mode = "ssh"
	target.Host = host
	target.Reachable = true
	target.LatencyMS = rtt
	target.DiscoveredAt = time.Now().UTC().Format(time.RFC3339)
	if target.UMABufferBytes == 0 && target.TotalRAMBytes > 0 {
		target.UMABufferBytes = int64(float64(target.TotalRAMBytes) * 0.88)
	}
	return &target, nil
}

func getPresenceFilePath() string {
	for _, cand := range []string{"_scratch/strix_presence.json", "../../_scratch/strix_presence.json"} {
		dir := filepath.Dir(cand)
		if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
			return cand
		}
	}
	return filepath.Join(os.TempDir(), "strix_presence.json")
}

func loadPresenceCache(host string) (*StrixTarget, bool) {
	data, err := os.ReadFile(getPresenceFilePath())
	if err != nil {
		return nil, false
	}
	var cache StrixPresenceCache
	if err := json.Unmarshal(data, &cache); err != nil {
		return nil, false
	}
	matched := cache.Target.Host == host
	if !matched && (host == DefaultStrixHost || host == "") && (cache.Target.Host == FallbackStrixMDNS || cache.Target.Host == DefaultStrixHost) {
		matched = true
	}
	if !matched {
		return nil, false
	}
	now := time.Now().Unix()
	if now-cache.Timestamp > int64(cache.TTLSeconds) {
		return nil, false
	}
	return &cache.Target, true
}

func savePresenceCache(target *StrixTarget) {
	if target == nil {
		return
	}
	filePath := getPresenceFilePath()
	_ = os.MkdirAll(filepath.Dir(filePath), 0755)
	cache := StrixPresenceCache{
		Target:     *target,
		Timestamp:  time.Now().Unix(),
		TTLSeconds: int(DefaultPresenceTTL.Seconds()),
	}
	if data, err := json.MarshalIndent(cache, "", "  "); err == nil {
		_ = os.WriteFile(filePath, data, 0644)
	}
}
