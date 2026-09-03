// Package amdgpu probes AMD GPU facts on Windows and provides hardware governor
// and TTM memory ceiling configuration utilities for AMD GPUs/APUs (q38rocm borrow).
package amdgpu

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	// DefaultDPMPerformanceLevel is the q38rocm recommended performance governor setting ("high").
	// Locking to "high" forces peak GPU and memory clocks, eliminating frequency-ramp latency
	// during token generation turns.
	DefaultDPMPerformanceLevel = "high"

	// DefaultPageSize is the standard Linux page size (4 KiB = 4096 bytes).
	DefaultPageSize = 4096

	// DefaultSysfsDPMGlob is the sysfs path pattern for AMD DRM card DPM performance levels.
	DefaultSysfsDPMGlob = "/sys/class/drm/card*/device/power_dpm_force_performance_level"

	// DefaultSysfsTTMPagesLimit is the sysfs path to the Linux TTM pages_limit parameter.
	DefaultSysfsTTMPagesLimit = "/sys/module/ttm/parameters/pages_limit"

	// DefaultProcMemInfo is the Linux proc path for system memory metrics.
	DefaultProcMemInfo = "/proc/meminfo"

	// MaxSafeRAMRatio is the safety ceiling (95%) to prevent starving the host kernel/OS.
	MaxSafeRAMRatio = 0.95

	// MinRecommendedRAMRatio is the Linux kernel default allocation level (50%).
	MinRecommendedRAMRatio = 0.50
)

// ValidDPMLevels is the closed set of valid AMD power_dpm_force_performance_level options.
var ValidDPMLevels = map[string]bool{
	"auto":             true,
	"low":              true,
	"high":             true,
	"manual":           true,
	"profile_standard": true,
	"profile_min_sclk": true,
	"profile_min_mclk": true,
	"profile_peak":     true,
}

func validDPMLevelsString() string {
	keys := make([]string, 0, len(ValidDPMLevels))
	for k := range ValidDPMLevels {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return strings.Join(keys, ", ")
}

// FileSystem abstracts OS file operations for testing Linux sysfs on any host.
type FileSystem interface {
	ReadFile(path string) ([]byte, error)
	WriteFile(path string, data []byte, perm os.FileMode) error
	Glob(pattern string) ([]string, error)
	Stat(path string) (os.FileInfo, error)
}

type osFileSystem struct{}

func (osFileSystem) ReadFile(p string) ([]byte, error) { return os.ReadFile(p) }
func (osFileSystem) WriteFile(p string, d []byte, perm os.FileMode) error {
	return os.WriteFile(p, d, perm)
}
func (osFileSystem) Glob(pattern string) ([]string, error) { return filepath.Glob(pattern) }
func (osFileSystem) Stat(p string) (os.FileInfo, error)    { return os.Stat(p) }

// ElevationChecker reports whether the caller has administrator/root privileges.
type ElevationChecker func() bool

// DefaultElevationChecker checks if the current process is elevated.
func DefaultElevationChecker() bool {
	if runtime.GOOS == "windows" {
		return false
	}
	return os.Geteuid() == 0
}

type governorEnv struct {
	fs        FileSystem
	elevation ElevationChecker
	runner    Runner
	goos      string
}

func defaultEnv() *governorEnv {
	return &governorEnv{
		fs:        osFileSystem{},
		elevation: DefaultElevationChecker,
		runner:    PowerShellRunner,
		goos:      runtime.GOOS,
	}
}

// Option configures execution environment dependencies.
type Option func(*governorEnv)

// WithFS overrides the FileSystem implementation.
func WithFS(fs FileSystem) Option {
	return func(e *governorEnv) {
		if fs != nil {
			e.fs = fs
		}
	}
}

// WithElevation overrides the ElevationChecker.
func WithElevation(checker ElevationChecker) Option {
	return func(e *governorEnv) {
		if checker != nil {
			e.elevation = checker
		}
	}
}

// WithRunner overrides the PowerShell runner for Windows facts probing.
func WithRunner(runner Runner) Option {
	return func(e *governorEnv) {
		if runner != nil {
			e.runner = runner
		}
	}
}

// WithGOOS overrides the operating system name ("linux", "windows", etc.).
func WithGOOS(goos string) Option {
	return func(e *governorEnv) {
		if goos != "" {
			e.goos = goos
		}
	}
}

// GovernorConfig specifies desired hardware governor and memory parameters.
type GovernorConfig struct {
	TargetDPMLevel   string  `json:"target_dpm_level"`             // e.g. "high" (default)
	TargetRatio      float64 `json:"target_ratio,omitempty"`       // fraction of system RAM (e.g. 0.875, 0.90)
	TargetPagesLimit uint64  `json:"target_pages_limit,omitempty"` // explicit 4096-byte page count
	SysfsRoot        string  `json:"sysfs_root,omitempty"`         // override sysfs root (default "/sys")
	ProcRoot         string  `json:"proc_root,omitempty"`          // override proc root (default "/proc")
	NameFilter       string  `json:"name_filter,omitempty"`        // device filter for Windows probe
}

// CardDPMStatus describes the power DPM status of an individual AMD DRM card.
type CardDPMStatus struct {
	CardPath     string `json:"card_path"`
	CardName     string `json:"card_name,omitempty"`
	CurrentLevel string `json:"current_level"`
	TargetLevel  string `json:"target_level"`
	NeedsUpdate  bool   `json:"needs_update"`
}

// TTMStatus describes TTM graphics translation table memory ceiling and system RAM facts.
type TTMStatus struct {
	Available            bool    `json:"available"`
	SysfsPath            string  `json:"sysfs_path,omitempty"`
	CurrentPagesLimit    uint64  `json:"current_pages_limit"`
	CurrentLimitBytes    uint64  `json:"current_limit_bytes"`
	CurrentLimitGiB      float64 `json:"current_limit_gib"`
	CurrentPercentOfRAM  float64 `json:"current_percent_of_ram"`
	TargetPagesLimit     uint64  `json:"target_pages_limit"`
	TargetLimitBytes     uint64  `json:"target_limit_bytes"`
	TargetLimitGiB       float64 `json:"target_limit_gib"`
	TargetPercentOfRAM   float64 `json:"target_percent_of_ram"`
	TotalSystemRAMBytes  uint64  `json:"total_system_ram_bytes"`
	TotalSystemRAMGiB    float64 `json:"total_system_ram_gib"`
	PageSizeBytes        uint64  `json:"page_size_bytes"`
	Default50PercentMode bool    `json:"default_50_percent_mode"`
	NeedsUpdate          bool    `json:"needs_update"`
	Note                 string  `json:"note,omitempty"`
}

// WindowsStatus reports AMD GPU driver power/clock policy and VRAM allocations on Windows.
type WindowsStatus struct {
	Available         bool           `json:"available"`
	DeviceName        string         `json:"device_name,omitempty"`
	DriverVersion     string         `json:"driver_version,omitempty"`
	AdapterRAMBytes   int64          `json:"adapter_ram_bytes,omitempty"`
	VRAMUsedBytes     int64          `json:"vram_used_bytes,omitempty"`
	VRAMUsedMiB       float64        `json:"vram_used_mib,omitempty"`
	ComputeUtilPct    float64        `json:"compute_util_pct,omitempty"`
	TotalUtilPct      float64        `json:"total_util_pct,omitempty"`
	BusiestUnit       string         `json:"busiest_unit,omitempty"`
	BusiestUtilPct    float64        `json:"busiest_util_pct,omitempty"`
	PowerClockProfile string         `json:"power_clock_profile,omitempty"`
	MemoryTuningNote  string         `json:"memory_tuning_note,omitempty"`
	RawFacts          map[string]any `json:"raw_facts,omitempty"`
	Error             string         `json:"error,omitempty"`
}

// ActionItem represents a specific sysfs configuration write.
type ActionItem struct {
	Parameter string `json:"parameter"`
	Path      string `json:"path"`
	Current   string `json:"current"`
	Target    string `json:"target"`
}

// GovernorReport is the diagnostic inspection and configuration plan.
type GovernorReport struct {
	Platform         string          `json:"platform"`
	Timestamp        string          `json:"timestamp"`
	Cards            []CardDPMStatus `json:"cards,omitempty"`
	TTM              TTMStatus       `json:"ttm"`
	Windows          *WindowsStatus  `json:"windows,omitempty"`
	Actions          []ActionItem    `json:"actions,omitempty"`
	NeedsElevation   bool            `json:"needs_elevation"`
	ElevationReason  string          `json:"elevation_reason,omitempty"`
	RecommendedRunAs string          `json:"recommended_run_as,omitempty"`
	Errors           []string        `json:"errors,omitempty"`
	Warnings         []string        `json:"warnings,omitempty"`
	Ready            bool            `json:"ready"`
}

// JSON encodes the GovernorReport as indented JSON bytes.
func (p *GovernorReport) JSON() ([]byte, error) {
	return json.MarshalIndent(p, "", "  ")
}

// ApplyResult describes the outcome of applying the governor plan.
type ApplyResult struct {
	Success      bool            `json:"success"`
	AppliedCount int             `json:"applied_count"`
	Actions      []ActionItem    `json:"actions,omitempty"`
	Errors       []string        `json:"errors,omitempty"`
	Plan         *GovernorReport `json:"plan"`
	Verified     bool            `json:"verified"`
}

// JSON encodes the ApplyResult as indented JSON bytes.
func (r *ApplyResult) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// CalculateRecommendedPagesLimit calculates the recommended TTM pages_limit for APUs/Radeon GPUs.
// If targetRatio > 0, it allocates that fraction of totalRAMBytes.
// Otherwise, it reserves a safe OS floor based on system RAM:
//   - >= 64 GiB (e.g. 64GB / 128GB systems): reserves 8 GiB for OS, allocating the rest (~87.5% - 93.75%).
//     (For 64GB RAM -> ~56 GiB = 14,680,064 pages, 87.5%; for 128GB RAM -> ~120 GiB = 31,457,280 pages, 93.75%).
//   - >= 32 GiB: reserves 6 GiB for OS, allocating the rest (~81.25%).
//   - >= 16 GiB: reserves 4 GiB for OS, allocating the rest (~75.0%).
//   - Smaller systems: reserves 25% for OS (min 1 GiB).
func CalculateRecommendedPagesLimit(totalRAMBytes uint64, pageSize uint64, targetRatio float64) (pages uint64, limitBytes uint64, ratio float64, err error) {
	if totalRAMBytes == 0 {
		return 0, 0, 0, errors.New("totalRAMBytes must be greater than 0")
	}
	if pageSize == 0 {
		pageSize = DefaultPageSize
	}

	if targetRatio < 0 {
		return 0, 0, 0, fmt.Errorf("target ratio cannot be negative (got %.4f)", targetRatio)
	}
	if targetRatio > 0 {
		if targetRatio > MaxSafeRAMRatio {
			return 0, 0, 0, fmt.Errorf("target ratio %.4f exceeds maximum safe ratio %.2f (would starve host OS/kernel)", targetRatio, MaxSafeRAMRatio)
		}
		limitBytes = uint64(float64(totalRAMBytes) * targetRatio)
		ratio = targetRatio
	} else {
		reserveBytes := RecommendedOSReserveBytes(totalRAMBytes)
		if reserveBytes >= totalRAMBytes {
			return 0, 0, 0, fmt.Errorf("system RAM (%d bytes) is too small to reserve %d bytes for OS", totalRAMBytes, reserveBytes)
		}
		limitBytes = totalRAMBytes - reserveBytes
		ratio = float64(limitBytes) / float64(totalRAMBytes)
		if ratio > MaxSafeRAMRatio {
			limitBytes = uint64(float64(totalRAMBytes) * MaxSafeRAMRatio)
			ratio = MaxSafeRAMRatio
		}
	}

	pages = limitBytes / pageSize
	if pages == 0 {
		return 0, 0, 0, errors.New("calculated pages_limit is 0")
	}
	return pages, limitBytes, ratio, nil
}

// RecommendedOSReserveBytes calculates the OS reserve memory floor.
func RecommendedOSReserveBytes(totalRAMBytes uint64) uint64 {
	const GiB = 1024 * 1024 * 1024
	switch {
	case totalRAMBytes >= 64*GiB:
		return 8 * GiB
	case totalRAMBytes >= 32*GiB:
		return 6 * GiB
	case totalRAMBytes >= 16*GiB:
		return 4 * GiB
	default:
		quarter := totalRAMBytes / 4
		if quarter < 1*GiB {
			return 1 * GiB
		}
		return quarter
	}
}

// ParseMemTotalFromProcMeminfo parses MemTotal from /proc/meminfo content and returns bytes.
func ParseMemTotalFromProcMeminfo(content string) (uint64, error) {
	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "MemTotal:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				kb, err := strconv.ParseUint(fields[1], 10, 64)
				if err != nil {
					return 0, fmt.Errorf("invalid MemTotal %q: %w", fields[1], err)
				}
				return kb * 1024, nil
			}
		}
	}
	return 0, errors.New("MemTotal field not found in meminfo")
}

// ValidateGovernorConfig verifies that configuration parameters are valid and safe.
func ValidateGovernorConfig(cfg GovernorConfig) error {
	level := cfg.TargetDPMLevel
	if level == "" {
		level = DefaultDPMPerformanceLevel
	}
	if !ValidDPMLevels[level] {
		return fmt.Errorf("invalid target DPM performance level %q (valid: %s)", level, validDPMLevelsString())
	}
	if cfg.TargetRatio > MaxSafeRAMRatio {
		return fmt.Errorf("target RAM ratio %.4f exceeds maximum safe ceiling %.2f", cfg.TargetRatio, MaxSafeRAMRatio)
	}
	if cfg.TargetRatio < 0 {
		return fmt.Errorf("target RAM ratio cannot be negative (got %.4f)", cfg.TargetRatio)
	}
	return nil
}

func buildWindowsStatus(winFacts map[string]any, plan *GovernorReport) *WindowsStatus {
	ws := &WindowsStatus{
		Available: winFacts["available"] == true,
		RawFacts:  winFacts,
	}
	if ws.Available {
		if name, ok := winFacts["name"].(string); ok {
			ws.DeviceName = name
		}
		if dv, ok := winFacts["driver_version"].(string); ok {
			ws.DriverVersion = dv
		}
		ws.AdapterRAMBytes = int64(number(winFacts["adapter_ram"]))
		ws.VRAMUsedBytes = int64(number(winFacts["vram_used_bytes"]))
		if vramMiB, ok := winFacts["vram_used_mib"].(float64); ok {
			ws.VRAMUsedMiB = vramMiB
		}
		if comp, ok := winFacts["compute_util_pct"].(float64); ok {
			ws.ComputeUtilPct = comp
		}
		if tot, ok := winFacts["total_util_pct"].(float64); ok {
			ws.TotalUtilPct = tot
		}
		if bEng, ok := winFacts["busiest_engine"].(string); ok {
			ws.BusiestUnit = bEng
		}
		if bUtil, ok := winFacts["busiest_util_pct"].(float64); ok {
			ws.BusiestUtilPct = bUtil
		}
		ws.PowerClockProfile = "Windows WDDM driver manages clock scaling dynamically; set Windows Power Plan to 'High Performance' or configure AMD Adrenalin profile for sustained peak clocks"
		ws.MemoryTuningNote = "Windows WDDM automatically manages shared system memory aperture (default up to 50% RAM). Hardware UMA aperture on APUs (Strix Halo/Phoenix) is configured via UEFI/BIOS UMA Frame Buffer Size"
	} else {
		if errMsg, ok := winFacts["error"].(string); ok {
			ws.Error = errMsg
			plan.Warnings = append(plan.Warnings, fmt.Sprintf("Windows AMD GPU facts probe: %s", errMsg))
		}
	}
	return ws
}

// BuildPlan performs a diagnostic probe of the system and builds a GovernorReport.
func BuildPlan(cfg GovernorConfig, opts ...Option) (*GovernorReport, error) {
	env := defaultEnv()
	for _, opt := range opts {
		opt(env)
	}

	if cfg.TargetDPMLevel == "" {
		cfg.TargetDPMLevel = DefaultDPMPerformanceLevel
	}
	if err := ValidateGovernorConfig(cfg); err != nil {
		return nil, err
	}

	plan := &GovernorReport{
		Platform:  env.goos,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Cards:     make([]CardDPMStatus, 0),
		Actions:   make([]ActionItem, 0),
		Errors:    make([]string, 0),
		Warnings:  make([]string, 0),
	}

	// On Windows, read and report AMD GPU facts and WDDM memory/power policies.
	if env.goos == "windows" {
		winFacts := Facts(cfg.NameFilter, env.runner)
		plan.Windows = buildWindowsStatus(winFacts, plan)

		// If no sysfs override is provided on Windows, Windows diagnosis is complete.
		if cfg.SysfsRoot == "" {
			plan.Ready = true
			return plan, nil
		}
	}

	// Linux sysfs inspection (also executed when SysfsRoot is specified for cross-platform planning/tests).
	sysfsRoot := cfg.SysfsRoot
	if sysfsRoot == "" {
		sysfsRoot = "/sys"
	}
	procRoot := cfg.ProcRoot
	if procRoot == "" {
		procRoot = "/proc"
	}

	// 1. Inspect Linux DRM card DPM performance levels.
	dpmPattern := filepath.ToSlash(filepath.Join(sysfsRoot, "class", "drm", "card*", "device", "power_dpm_force_performance_level"))
	cardPaths, globErr := env.fs.Glob(dpmPattern)
	if globErr != nil {
		plan.Warnings = append(plan.Warnings, fmt.Sprintf("error scanning DRM cards at %s: %v", dpmPattern, globErr))
	} else if len(cardPaths) == 0 {
		plan.Warnings = append(plan.Warnings, fmt.Sprintf("no AMD DRM cards found matching %s", dpmPattern))
	} else {
		sort.Strings(cardPaths)
		for _, cardPath := range cardPaths {
			data, readErr := env.fs.ReadFile(cardPath)
			if readErr != nil {
				plan.Warnings = append(plan.Warnings, fmt.Sprintf("failed to read %s: %v", cardPath, readErr))
				continue
			}
			current := strings.TrimSpace(string(data))
			cardName := extractCardName(cardPath)
			needsUpdate := current != cfg.TargetDPMLevel
			status := CardDPMStatus{
				CardPath:     cardPath,
				CardName:     cardName,
				CurrentLevel: current,
				TargetLevel:  cfg.TargetDPMLevel,
				NeedsUpdate:  needsUpdate,
			}
			plan.Cards = append(plan.Cards, status)
			if needsUpdate {
				plan.Actions = append(plan.Actions, ActionItem{
					Parameter: "power_dpm_force_performance_level",
					Path:      cardPath,
					Current:   current,
					Target:    cfg.TargetDPMLevel,
				})
			}
		}
	}

	// 2. Inspect Linux TTM memory parameters and system RAM.
	var totalRAMBytes uint64
	meminfoPath := filepath.Join(procRoot, "meminfo")
	memData, memErr := env.fs.ReadFile(meminfoPath)
	if memErr != nil {
		plan.Warnings = append(plan.Warnings, fmt.Sprintf("failed to read system memory at %s: %v", meminfoPath, memErr))
	} else {
		totalRAMBytes, memErr = ParseMemTotalFromProcMeminfo(string(memData))
		if memErr != nil {
			plan.Warnings = append(plan.Warnings, fmt.Sprintf("failed to parse MemTotal from %s: %v", meminfoPath, memErr))
		}
	}

	ttmPath := filepath.Join(sysfsRoot, "module", "ttm", "parameters", "pages_limit")
	ttmData, ttmErr := env.fs.ReadFile(ttmPath)
	if ttmErr != nil {
		plan.TTM.Available = false
		plan.TTM.Note = fmt.Sprintf("TTM module parameter pages_limit not found at %s (driver not loaded or non-AMD/TTM system)", ttmPath)
		plan.Warnings = append(plan.Warnings, plan.TTM.Note)
	} else {
		plan.TTM.Available = true
		plan.TTM.SysfsPath = ttmPath
		plan.TTM.PageSizeBytes = DefaultPageSize
		plan.TTM.TotalSystemRAMBytes = totalRAMBytes
		plan.TTM.TotalSystemRAMGiB = round2(float64(totalRAMBytes) / (1024 * 1024 * 1024))

		currStr := strings.TrimSpace(string(ttmData))
		currPages, parseErr := strconv.ParseUint(currStr, 10, 64)
		if parseErr != nil {
			plan.Warnings = append(plan.Warnings, fmt.Sprintf("invalid integer in %s: %q (%v)", ttmPath, currStr, parseErr))
		} else {
			plan.TTM.CurrentPagesLimit = currPages
			if currPages == 0 {
				plan.TTM.Default50PercentMode = true
				if totalRAMBytes > 0 {
					plan.TTM.CurrentLimitBytes = totalRAMBytes / 2
					plan.TTM.CurrentLimitGiB = round2(float64(plan.TTM.CurrentLimitBytes) / (1024 * 1024 * 1024))
					plan.TTM.CurrentPercentOfRAM = 50.0
				}
			} else {
				plan.TTM.CurrentLimitBytes = currPages * DefaultPageSize
				plan.TTM.CurrentLimitGiB = round2(float64(plan.TTM.CurrentLimitBytes) / (1024 * 1024 * 1024))
				if totalRAMBytes > 0 {
					plan.TTM.CurrentPercentOfRAM = round2(float64(plan.TTM.CurrentLimitBytes) / float64(totalRAMBytes) * 100.0)
				}
			}
		}

		// Calculate target pages limit.
		var targetPages uint64
		if cfg.TargetPagesLimit > 0 {
			targetPages = cfg.TargetPagesLimit
			plan.TTM.TargetPagesLimit = targetPages
			plan.TTM.TargetLimitBytes = targetPages * DefaultPageSize
			plan.TTM.TargetLimitGiB = round2(float64(plan.TTM.TargetLimitBytes) / (1024 * 1024 * 1024))
			if totalRAMBytes > 0 {
				plan.TTM.TargetPercentOfRAM = round2(float64(plan.TTM.TargetLimitBytes) / float64(totalRAMBytes) * 100.0)
				if plan.TTM.TargetLimitBytes > uint64(float64(totalRAMBytes)*MaxSafeRAMRatio) {
					plan.Errors = append(plan.Errors, fmt.Sprintf("target pages %d (%s GiB) exceeds maximum safe memory ratio %.2f of total RAM (%s GiB)",
						targetPages, fmt.Sprintf("%.2f", plan.TTM.TargetLimitGiB), MaxSafeRAMRatio, fmt.Sprintf("%.2f", plan.TTM.TotalSystemRAMGiB)))
				}
			}
		} else if totalRAMBytes > 0 {
			p, limitBytes, ratio, calcErr := CalculateRecommendedPagesLimit(totalRAMBytes, DefaultPageSize, cfg.TargetRatio)
			if calcErr != nil {
				plan.Errors = append(plan.Errors, fmt.Sprintf("memory ceiling calculation failed: %v", calcErr))
			} else {
				targetPages = p
				plan.TTM.TargetPagesLimit = targetPages
				plan.TTM.TargetLimitBytes = limitBytes
				plan.TTM.TargetLimitGiB = round2(float64(limitBytes) / (1024 * 1024 * 1024))
				plan.TTM.TargetPercentOfRAM = round2(ratio * 100.0)
			}
		} else {
			plan.Warnings = append(plan.Warnings, "unable to determine system RAM; specify --pages explicitly for TTM ceiling")
		}

		if targetPages > 0 && currPages != targetPages {
			plan.TTM.NeedsUpdate = true
			plan.Actions = append(plan.Actions, ActionItem{
				Parameter: "pages_limit",
				Path:      ttmPath,
				Current:   fmt.Sprintf("%d", currPages),
				Target:    fmt.Sprintf("%d", targetPages),
			})
		}
	}

	// 3. Evaluate elevation and readiness.
	if len(plan.Actions) == 0 && len(plan.Errors) == 0 {
		plan.Ready = true
	} else if len(plan.Actions) > 0 {
		plan.Ready = false
		elevated := env.elevation()
		if !elevated {
			plan.NeedsElevation = true
			if env.goos == "windows" {
				plan.ElevationReason = "modifying Windows GPU policy or driver configuration requires Administrator privileges"
				plan.RecommendedRunAs = "Run PowerShell as Administrator"
			} else {
				plan.ElevationReason = "modifying Linux sysfs parameters (/sys/class/drm and /sys/module/ttm) requires root (uid 0) or sudo"
				plan.RecommendedRunAs = "sudo fak-dev amd-setup --apply"
			}
		}
	}

	return plan, nil
}

// appliedAction tracks a successfully applied sysfs configuration write for rollback.
type appliedAction struct {
	target        string
	originalValue string
}

// Apply executes the planned hardware governor and memory limit writes.
func Apply(plan *GovernorReport, opts ...Option) (*ApplyResult, error) {
	if plan == nil {
		return nil, errors.New("nil governor plan")
	}

	env := defaultEnv()
	for _, opt := range opts {
		opt(env)
	}

	if len(plan.Errors) > 0 {
		return &ApplyResult{
			Success:      false,
			AppliedCount: 0,
			Errors:       plan.Errors,
			Plan:         plan,
			Verified:     false,
		}, fmt.Errorf("plan contains validation errors: %s", strings.Join(plan.Errors, "; "))
	}

	if len(plan.Actions) == 0 {
		return &ApplyResult{
			Success:      true,
			AppliedCount: 0,
			Plan:         plan,
			Verified:     true,
		}, nil
	}

	// Guard against non-root execution when elevation is required.
	if plan.NeedsElevation && !env.elevation() {
		errText := fmt.Sprintf("elevation required: %s (run with: %s)", plan.ElevationReason, plan.RecommendedRunAs)
		return &ApplyResult{
			Success:      false,
			AppliedCount: 0,
			Errors:       []string{errText},
			Plan:         plan,
			Verified:     false,
		}, errors.New(errText)
	}

	applied := make([]appliedAction, 0, len(plan.Actions))
	appliedActions := make([]ActionItem, 0, len(plan.Actions))

	for _, action := range plan.Actions {
		// Before writing action.TargetValue, record action.TargetFile and action.Current.
		currentAction := appliedAction{
			target:        action.Path,
			originalValue: action.Current,
		}

		var failErr string

		// Write parameter with trailing newline as expected by Linux sysfs.
		payload := []byte(action.Target + "\n")
		if err := env.fs.WriteFile(action.Path, payload, 0644); err != nil {
			failErr = fmt.Sprintf("failed to write %q to %s: %v", action.Target, action.Path, err)
		} else {
			// Read back parameter immediately to verify kernel accepted the write.
			data, err := env.fs.ReadFile(action.Path)
			if err != nil {
				failErr = fmt.Sprintf("failed to read-back %s after write: %v", action.Path, err)
			} else {
				readBack := strings.TrimSpace(string(data))
				if readBack != strings.TrimSpace(action.Target) {
					failErr = fmt.Sprintf("read-back verification mismatch for %s: got %q, want %q", action.Path, readBack, action.Target)
				}
			}
		}

		if failErr != "" {
			var rollbackErrs []string
			for i := len(applied) - 1; i >= 0; i-- {
				prev := applied[i]
				if rbErr := env.fs.WriteFile(prev.target, []byte(prev.originalValue), 0644); rbErr != nil {
					rollbackErrs = append(rollbackErrs, fmt.Sprintf("failed to restore %s to %q: %v", prev.target, prev.originalValue, rbErr))
				}
			}

			errs := []string{failErr}
			var retErr error
			if len(rollbackErrs) > 0 {
				rbDetails := strings.Join(rollbackErrs, "; ")
				errs = append(errs, fmt.Sprintf("rollback failed: %s", rbDetails))
				retErr = fmt.Errorf("%s; rollback failed: %s", failErr, rbDetails)
			} else {
				errs = append(errs, "rollback succeeded")
				retErr = fmt.Errorf("%s; rollback succeeded", failErr)
			}

			return &ApplyResult{
				Success:      false,
				AppliedCount: 0,
				Actions:      nil,
				Errors:       errs,
				Plan:         plan,
				Verified:     false,
			}, retErr
		}

		applied = append(applied, currentAction)
		appliedActions = append(appliedActions, action)
	}

	return &ApplyResult{
		Success:      true,
		AppliedCount: len(appliedActions),
		Actions:      appliedActions,
		Plan:         plan,
		Verified:     true,
	}, nil
}

// RunCLI parses arguments, performs diagnostic planning or applies tweaks, and formats output.
func RunCLI(stdout, stderr io.Writer, argv []string, opts ...Option) int {
	fs := flag.NewFlagSet("amd-setup", flag.ContinueOnError)
	fs.SetOutput(stderr)

	apply := fs.Bool("apply", false, "apply proposed hardware governor and memory ceiling tweaks")
	dryRun := fs.Bool("dry-run", false, "dry-run: inspect diagnostic plan without modifying system (default)")
	jsonOut := fs.Bool("json", false, "output report or result as JSON")
	level := fs.String("level", DefaultDPMPerformanceLevel, "target DPM performance level (high, auto, low, manual)")
	ratio := fs.Float64("ratio", 0, "target fraction of system RAM for TTM memory ceiling (e.g. 0.875, 0.90)")
	pages := fs.Uint64("pages", 0, "explicit target pages_limit (4096-byte pages)")
	sysfsRoot := fs.String("sysfs-root", "", "override sysfs root directory (default /sys)")
	procRoot := fs.String("proc-root", "", "override proc root directory (default /proc)")
	name := fs.String("name", "", "filter GPU device name on Windows")

	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "amd-setup: unexpected arguments: %s\n", strings.Join(fs.Args(), " "))
		return 2
	}

	cfg := GovernorConfig{
		TargetDPMLevel:   *level,
		TargetRatio:      *ratio,
		TargetPagesLimit: *pages,
		SysfsRoot:        *sysfsRoot,
		ProcRoot:         *procRoot,
		NameFilter:       *name,
	}

	plan, err := BuildPlan(cfg, opts...)
	if err != nil {
		fmt.Fprintf(stderr, "amd-setup plan error: %v\n", err)
		return 1
	}

	if *apply && !*dryRun {
		res, applyErr := Apply(plan, opts...)
		if *jsonOut {
			data, _ := json.MarshalIndent(res, "", "  ")
			fmt.Fprintln(stdout, string(data))
		} else {
			fmt.Fprintln(stdout, "AMD GPU Hardware Governor & TTM Memory Ceiling Apply:")
			for _, action := range res.Actions {
				fmt.Fprintf(stdout, "  [OK] %s: %s -> %s\n", action.Parameter, action.Current, action.Target)
			}
			if applyErr != nil {
				fmt.Fprintf(stderr, "Apply failed: %v\n", applyErr)
			} else {
				fmt.Fprintf(stdout, "READY: applied %d tweaks successfully and verified.\n", res.AppliedCount)
			}
		}
		if applyErr != nil || !res.Success {
			return 1
		}
		return 0
	}

	// Dry-run / diagnostic mode.
	if *jsonOut {
		data, _ := plan.JSON()
		fmt.Fprintln(stdout, string(data))
		return 0
	}

	// Format human-readable diagnostic plan.
	fmt.Fprintln(stdout, "AMD GPU Hardware Governor & TTM Memory Ceiling Plan:")
	fmt.Fprintf(stdout, "Platform: %s\n", plan.Platform)

	if plan.Windows != nil {
		if plan.Windows.Available {
			fmt.Fprintf(stdout, "Windows GPU: %s (Driver: %s, VRAM: %.1f MiB)\n",
				plan.Windows.DeviceName, plan.Windows.DriverVersion, plan.Windows.VRAMUsedMiB)
			fmt.Fprintf(stdout, "  Units: busiest=%s (%.1f%%), total=%.1f%%\n",
				plan.Windows.BusiestUnit, plan.Windows.BusiestUtilPct, plan.Windows.TotalUtilPct)
			fmt.Fprintf(stdout, "  Profile: %s\n", plan.Windows.PowerClockProfile)
			fmt.Fprintf(stdout, "  Note: %s\n", plan.Windows.MemoryTuningNote)
		} else if plan.Windows.Error != "" {
			fmt.Fprintf(stdout, "Windows GPU: unavailable (%s)\n", plan.Windows.Error)
		}
	}

	if len(plan.Cards) > 0 {
		fmt.Fprintf(stdout, "DRM Cards (%d detected):\n", len(plan.Cards))
		for _, card := range plan.Cards {
			upd := ""
			if card.NeedsUpdate {
				upd = " (UPDATE NEEDED)"
			}
			fmt.Fprintf(stdout, "  - %s: current=%s, target=%s%s\n", card.CardPath, card.CurrentLevel, card.TargetLevel, upd)
		}
	}

	if plan.TTM.Available {
		fmt.Fprintln(stdout, "TTM Memory Ceiling:")
		fmt.Fprintf(stdout, "  Total System RAM: %.1f GiB\n", plan.TTM.TotalSystemRAMGiB)
		defNote := ""
		if plan.TTM.Default50PercentMode {
			defNote = " (kernel default 50%)"
		}
		fmt.Fprintf(stdout, "  Current Limit:    %.1f GiB (%d pages, %.1f%% of RAM)%s\n",
			plan.TTM.CurrentLimitGiB, plan.TTM.CurrentPagesLimit, plan.TTM.CurrentPercentOfRAM, defNote)
		upd := ""
		if plan.TTM.NeedsUpdate {
			upd = " (UPDATE NEEDED)"
		}
		fmt.Fprintf(stdout, "  Target Limit:     %.1f GiB (%d pages, %.1f%% of RAM)%s\n",
			plan.TTM.TargetLimitGiB, plan.TTM.TargetPagesLimit, plan.TTM.TargetPercentOfRAM, upd)
	}

	if len(plan.Actions) > 0 {
		fmt.Fprintf(stdout, "\nProposed Actions (%d pending):\n", len(plan.Actions))
		for _, action := range plan.Actions {
			fmt.Fprintf(stdout, "  * %s (%s): %s -> %s\n", action.Parameter, action.Path, action.Current, action.Target)
		}
		if plan.NeedsElevation {
			fmt.Fprintf(stdout, "\nElevation Required: %s\nRun with: %s\n", plan.ElevationReason, plan.RecommendedRunAs)
		} else {
			fmt.Fprintln(stdout, "\nRun with --apply to execute proposed changes.")
		}
	} else if plan.Ready {
		fmt.Fprintln(stdout, "\nStatus: READY (all parameters match target baselines)")
	}

	for _, w := range plan.Warnings {
		fmt.Fprintf(stderr, "Warning: %s\n", w)
	}
	for _, e := range plan.Errors {
		fmt.Fprintf(stderr, "Error: %s\n", e)
	}

	if len(plan.Errors) > 0 {
		return 1
	}
	return 0
}

func extractCardName(p string) string {
	parts := strings.Split(filepath.ToSlash(p), "/")
	for i, part := range parts {
		if strings.HasPrefix(part, "card") && i > 0 && parts[i-1] == "drm" {
			return part
		}
	}
	return filepath.Base(p)
}

func round2(v float64) float64 {
	return float64(int(v*100+0.5)) / 100.0
}
