// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// and Strix Halo APU operational serving profiles and validation.
package amdgpu

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"regexp"
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

var strixSSHHostnameRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._:-]*$`)

var (
	errStrixSSHRetryable         = errors.New("STRIX_SSH_REMOTE_UNAVAILABLE")
	errStrixSSHRemoteUnavailable = fmt.Errorf("%w: %w", ErrStrixHostTrustRefused, errStrixSSHRetryable)
)

type strixSSHBroker interface {
	KnownHostsCommand() (string, error)
	Close() error
}

type strixSSHTransportDeps struct {
	startBroker    func() (strixSSHBroker, error)
	lookPath       func(string) (string, error)
	executable     func() (string, error)
	lstat          func(string) (os.FileInfo, error)
	sameFile       func(os.FileInfo, os.FileInfo) bool
	fileDigest     func(string) ([sha256.Size]byte, error)
	commandContext func(context.Context, string, ...string) *exec.Cmd
	combinedOutput func(*exec.Cmd) ([]byte, error)
	environ        func() []string
}

var strixSSHDeps = strixSSHTransportDeps{
	startBroker: func() (strixSSHBroker, error) {
		return StartStrixKnownHostsBroker()
	},
	lookPath:       exec.LookPath,
	executable:     os.Executable,
	lstat:          os.Lstat,
	sameFile:       os.SameFile,
	fileDigest:     digestStrixExecutable,
	commandContext: exec.CommandContext,
	combinedOutput: func(cmd *exec.Cmd) ([]byte, error) { return cmd.CombinedOutput() },
	environ:        os.Environ,
}

type strixSSHCommand struct {
	ctx      context.Context
	cmd      *exec.Cmd
	broker   strixSSHBroker
	sshPath  string
	sshInfo  os.FileInfo
	sshHash  [sha256.Size]byte
	selfPath string
	selfInfo os.FileInfo
	selfHash [sha256.Size]byte
	deps     strixSSHTransportDeps
}

func isNilStrixSSHBroker(broker strixSSHBroker) bool {
	if broker == nil {
		return true
	}
	value := reflect.ValueOf(broker)
	return (value.Kind() == reflect.Chan || value.Kind() == reflect.Func || value.Kind() == reflect.Interface || value.Kind() == reflect.Map || value.Kind() == reflect.Ptr || value.Kind() == reflect.Slice) && value.IsNil()
}

func validateStrixSSHDestination(host string) error {
	if len(host) == 0 || len(host) > 253 || strings.HasPrefix(host, "-") {
		return strixTrustRefused("invalid ssh destination")
	}
	if net.ParseIP(host) != nil || strixSSHHostnameRE.MatchString(host) {
		return nil
	}
	return strixTrustRefused("invalid ssh destination")
}

func digestStrixExecutable(path string) ([sha256.Size]byte, error) {
	var digest [sha256.Size]byte
	f, err := os.Open(path)
	if err != nil {
		return digest, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return digest, err
	}
	copy(digest[:], h.Sum(nil))
	return digest, nil
}

func observeStrixExecutable(path string, deps strixSSHTransportDeps) (os.FileInfo, [sha256.Size]byte, error) {
	var zero [sha256.Size]byte
	if deps.lstat == nil || deps.sameFile == nil || deps.fileDigest == nil || !filepath.IsAbs(path) || filepath.Clean(path) != path || strings.ContainsAny(path, "\x00\r\n%") {
		return nil, zero, strixTrustRefused("unsafe executable")
	}
	info, err := deps.lstat(path)
	if err != nil || info == nil || !info.Mode().IsRegular() {
		return nil, zero, strixTrustRefused("unsafe executable")
	}
	digest, err := deps.fileDigest(path)
	if err != nil {
		return nil, zero, strixTrustRefused("unsafe executable")
	}
	after, err := deps.lstat(path)
	if err != nil || after == nil || !after.Mode().IsRegular() || !deps.sameFile(info, after) || !sameStrixExecutableMetadata(info, after) {
		return nil, zero, strixTrustRefused("executable changed")
	}
	return after, digest, nil
}

func sameStrixExecutableMetadata(a, b os.FileInfo) bool {
	return a != nil && b != nil && a.Mode() == b.Mode() && a.Size() == b.Size() && a.ModTime().Equal(b.ModTime())
}

func strixSSHChildEnvironment(env []string) []string {
	allowed := map[string]bool{
		"COMSPEC": true, "SYSTEMROOT": true, "WINDIR": true,
		"TEMP": true, "TMP": true,
	}
	clean := make([]string, 0, len(env))
	seen := make(map[string]bool, len(allowed))
	for _, item := range env {
		name, _, ok := strings.Cut(item, "=")
		upper := strings.ToUpper(name)
		if !ok || !allowed[upper] || seen[upper] {
			continue
		}
		seen[upper] = true
		clean = append(clean, item)
	}
	return clean
}

func newStrixSSHCommand(ctx context.Context, host string, connectTimeout time.Duration, command string, stdin []byte, deps strixSSHTransportDeps) (_ *strixSSHCommand, err error) {
	if ctx == nil || deps.startBroker == nil || deps.lookPath == nil || deps.executable == nil || deps.lstat == nil || deps.sameFile == nil || deps.fileDigest == nil || deps.commandContext == nil || deps.combinedOutput == nil || deps.environ == nil {
		return nil, strixTrustRefused("invalid ssh transport")
	}
	if err := validateStrixSSHDestination(host); err != nil {
		return nil, err
	}
	sshPath, err := deps.lookPath("ssh")
	if err != nil {
		return nil, strixTrustRefused("ssh executable unavailable")
	}
	sshInfo, sshHash, err := observeStrixExecutable(sshPath, deps)
	if err != nil {
		return nil, err
	}
	selfPath, err := deps.executable()
	if err != nil {
		return nil, strixTrustRefused("unsafe executable")
	}
	selfInfo, selfHash, err := observeStrixExecutable(selfPath, deps)
	if err != nil {
		return nil, err
	}
	broker, startErr := deps.startBroker()
	if isNilStrixSSHBroker(broker) {
		return nil, strixTrustRefused("host trust unavailable")
	}
	if startErr != nil {
		if closeErr := broker.Close(); closeErr != nil {
			return nil, strixTrustRefused("broker cleanup failed")
		}
		return nil, strixTrustRefused("host trust unavailable")
	}
	keepBroker := false
	defer func() {
		if !keepBroker {
			if closeErr := broker.Close(); closeErr != nil {
				err = strixTrustRefused("broker cleanup failed")
			}
		}
	}()
	knownHostsCommand, err := broker.KnownHostsCommand()
	if err != nil || knownHostsCommand == "" || strings.ContainsAny(knownHostsCommand, "\x00\r\n%") {
		return nil, strixTrustRefused("host trust unavailable")
	}
	seconds := int64(connectTimeout / time.Second)
	if seconds < 1 || connectTimeout != time.Duration(seconds)*time.Second {
		return nil, strixTrustRefused("invalid ssh timeout")
	}
	args := []string{
		"-F", "none",
		"-o", "BatchMode=yes",
		"-o", "ConnectTimeout=" + strconv.FormatInt(seconds, 10),
		"-o", "StrictHostKeyChecking=yes",
		"-o", "UserKnownHostsFile=none",
		"-o", "GlobalKnownHostsFile=none",
		"-o", "VerifyHostKeyDNS=no",
		"-o", "HostKeyAlias=" + StrixKnownHostsAlias,
		"-o", "UpdateHostKeys=no",
		"-o", "KnownHostsCommand=" + knownHostsCommand,
		"--", host, command,
	}
	cmd := deps.commandContext(ctx, sshPath, args...)
	if cmd == nil {
		return nil, strixTrustRefused("ssh command unavailable")
	}
	cmd.Env = strixSSHChildEnvironment(deps.environ())
	if stdin != nil {
		cmd.Stdin = bytes.NewReader(stdin)
	}
	windowgate.ConfigureBackgroundCommand(cmd)
	keepBroker = true
	return &strixSSHCommand{ctx: ctx, cmd: cmd, broker: broker, sshPath: sshPath, sshInfo: sshInfo, sshHash: sshHash, selfPath: selfPath, selfInfo: selfInfo, selfHash: selfHash, deps: deps}, nil
}

func (c *strixSSHCommand) combinedOutput() (out []byte, err error) {
	if c == nil || c.cmd == nil || c.broker == nil {
		return nil, strixTrustRefused("invalid ssh command")
	}
	defer func() {
		if closeErr := c.broker.Close(); closeErr != nil {
			out = nil
			err = strixTrustRefused("broker cleanup failed")
		}
	}()
	sshNow, sshHash, sshErr := observeStrixExecutable(c.sshPath, c.deps)
	selfNow, selfHash, selfErr := observeStrixExecutable(c.selfPath, c.deps)
	if sshErr != nil || selfErr != nil || !c.deps.sameFile(c.sshInfo, sshNow) || !sameStrixExecutableMetadata(c.sshInfo, sshNow) || sshHash != c.sshHash || !c.deps.sameFile(c.selfInfo, selfNow) || !sameStrixExecutableMetadata(c.selfInfo, selfNow) || selfHash != c.selfHash {
		return nil, strixTrustRefused("executable changed")
	}
	out, err = c.deps.combinedOutput(c.cmd)
	if err != nil {
		if c.ctx.Err() != nil {
			return nil, strixTrustRefused("ssh execution canceled")
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return nil, errStrixSSHRemoteUnavailable
		}
		return nil, strixTrustRefused("ssh start failed")
	}
	return out, nil
}

func runStrixSSHCommand(ctx context.Context, host string, connectTimeout time.Duration, command string, stdin []byte) ([]byte, error) {
	invocation, err := newStrixSSHCommand(ctx, host, connectTimeout, command, stdin, strixSSHDeps)
	if err != nil {
		return nil, err
	}
	return invocation.combinedOutput()
}

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
	if err := validateStrixSSHDestination(host); err != nil {
		return nil, err
	}

	// Cached presence never substitutes for a fresh trust admission. A remote
	// probe establishes the invocation-bound host-key snapshot before use.
	target, err := probeRemoteStrix(ctx, host)
	if err != nil {
		if hostOverride == "" && host == DefaultStrixHost && errors.Is(err, errStrixSSHRetryable) {
			fbTarget, fbErr := probeRemoteStrix(ctx, FallbackStrixMDNS)
			if fbErr == nil && fbTarget.Reachable {
				savePresenceCache(fbTarget)
				return fbTarget, nil
			}
			if errors.Is(fbErr, ErrStrixHostTrustRefused) {
				err = fbErr
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

	out, err := runStrixSSHCommand(probeCtx, host, 2*time.Second, probeCmd, nil)
	rtt := time.Since(start).Seconds() * 1000.0

	if err != nil {
		return nil, err
	}

	var target StrixTarget
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	lastLine := lines[len(lines)-1]
	if err := json.Unmarshal([]byte(lastLine), &target); err != nil {
		return nil, strixTrustRefused("invalid ssh response")
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

// SavePresenceCache stores a target in the local presence cache.
func SavePresenceCache(target *StrixTarget) {
	savePresenceCache(target)
}

// ClearPresenceCache removes the cached presence file if it exists.
func ClearPresenceCache() {
	_ = os.Remove(getPresenceFilePath())
}
