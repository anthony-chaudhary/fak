// Hardware clock control mechanism. Presence does not establish serving invocation or physical qualification.
package amdgpu

import (
	"context"
	"fmt"
	"github.com/anthony-chaudhary/fak/pkg/amdgpuclock"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Clock-control defaults preserve the migrated mechanism configuration.
const (
	DefaultLockedShaderClockMHz   = 2200
	NominalMemoryBusClockMT       = 8000
	DefaultSysfsDRMPath           = "/sys/class/drm"
	DefaultPreferredGPUCard       = "card1"
	DefaultFallbackGPUCard        = "card0"
	SysfsDPMForcePerformanceLevel = "power_dpm_force_performance_level"
	SysfsPPDpmSclk                = "pp_dpm_sclk"
	SysfsPPDpmMclk                = "pp_dpm_mclk"
)

type FileReaderFunc func(path string) ([]byte, error)
type FileWriterFunc func(path string, data []byte, perm os.FileMode) error

// HardwarePlatform identifies the hardware accelerator vendor or architecture.
type HardwarePlatform string

const (
	PlatformAMD    HardwarePlatform = "amd"
	PlatformNVIDIA HardwarePlatform = "nvidia"
	PlatformApple  HardwarePlatform = "apple"

	// DefaultHysteresisWindow is the 5000ms anti-flapping hold timer preventing
	// downclocking during brief inter-token pauses (15-50ms).
	DefaultHysteresisWindow = 5000 * time.Millisecond

	// DefaultNVIDIALockedCoreClockMHz is the default locked GPU graphics core clock for NVIDIA GPUs.
	DefaultNVIDIALockedCoreClockMHz = 2100

	// DefaultNVIDIALockedMemClockMHz is the default locked memory bus clock for NVIDIA GPUs.
	DefaultNVIDIALockedMemClockMHz = 1215

	// DefaultAppleLockedCoreClockMHz is the nominal GPU frequency for Apple Silicon M-series.
	DefaultAppleLockedCoreClockMHz = 1398

	// DefaultAppleLockedMemClockMHz is the nominal unified memory clock for Apple Silicon M-series.
	DefaultAppleLockedMemClockMHz = 3200
)

// ClockGovernor defines the public interface for hardware clock and memory bandwidth governance.
type ClockGovernor interface {
	Start(ctx context.Context) error
	Stop() error
	RecordActivity()
	IsLocked() bool
	Lock() error
	Unlock() error
	Status() HardwareGovernorStatus
	Metrics() string
	Platform() HardwarePlatform
	Hysteresis() time.Duration
	SetHysteresis(d time.Duration)
	SynthesizeLockCommands() []string
	SynthesizeUnlockCommands() []string
}

// HardwareGovernor is an alias for ClockGovernor.
type HardwareGovernor = ClockGovernor

// HardwareGovernorStatus reports the real-time operational state of the hardware clock governor.
type HardwareGovernorStatus struct {
	Platform           HardwarePlatform `json:"platform"`
	Device             string           `json:"device"`
	Locked             bool             `json:"locked"`
	State              string           `json:"state"` // "LOCKED_PERFORMANCE", "QUIESCENT_IDLE", "HYSTERESIS_HOLD"
	CoreClockMHz       int              `json:"core_clock_mhz"`
	TargetCoreClockMHz int              `json:"target_core_clock_mhz"`
	MemClockMHz        int              `json:"memory_clock_mhz"`
	TargetMemClockMHz  int              `json:"target_mem_clock_mhz"`
	HysteresisMs       int64            `json:"hysteresis_ms"`
	LastActivityAgoMs  int64            `json:"last_activity_ago_ms"`
	ThrottleEvents     int64            `json:"throttle_events_total"`
	Reason             string           `json:"reason,omitempty"`
}

// HardwareGovernorConfig holds configuration for HardwareClockGovernor.
type HardwareGovernorConfig struct {
	Platform           HardwarePlatform `json:"platform"`
	Hysteresis         time.Duration    `json:"hysteresis"`
	TargetCoreClockMHz int              `json:"target_core_clock_mhz"`
	TargetMemClockMHz  int              `json:"target_mem_clock_mhz"`
	TargetMemClockMT   int              `json:"target_mem_clock_mt"`
	SysfsDRMRoot       string           `json:"sysfs_drm_root"`
	PreferredCard      string           `json:"preferred_card"`
	FallbackCard       string           `json:"fallback_card"`
	AutoLockOnActivity bool             `json:"auto_lock_on_activity"`
	PollInterval       time.Duration    `json:"poll_interval"`
}

// DefaultHardwareGovernorConfig returns production defaults for the hardware clock governor.
func DefaultHardwareGovernorConfig() HardwareGovernorConfig {
	return HardwareGovernorConfig{
		Platform:           PlatformAMD,
		Hysteresis:         DefaultHysteresisWindow,
		TargetCoreClockMHz: DefaultLockedShaderClockMHz,
		TargetMemClockMHz:  1000,
		TargetMemClockMT:   NominalMemoryBusClockMT,
		SysfsDRMRoot:       DefaultSysfsDRMPath,
		PreferredCard:      DefaultPreferredGPUCard,
		FallbackCard:       DefaultFallbackGPUCard,
		AutoLockOnActivity: true,
		PollInterval:       50 * time.Millisecond,
	}
}

// HardwareClockGovernorOption configures a HardwareClockGovernor instance.
type HardwareClockGovernorOption func(*HardwareClockGovernor)

// WithClockGovernorPlatform sets the hardware accelerator platform (amd, nvidia, apple).
func WithClockGovernorPlatform(p HardwarePlatform) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if p != "" {
			g.config.Platform = p
		}
	}
}

// WithClockGovernorHysteresis sets the anti-flapping hysteresis hold duration.
func WithClockGovernorHysteresis(d time.Duration) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if d > 0 {
			g.config.Hysteresis = d
		}
	}
}

// WithClockGovernorTargetClocks sets target compute core and memory frequencies in MHz.
func WithClockGovernorTargetClocks(coreMHz, memMHz int) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if coreMHz > 0 {
			g.config.TargetCoreClockMHz = coreMHz
		}
		if memMHz > 0 {
			g.config.TargetMemClockMHz = memMHz
		}
	}
}

// WithClockGovernorTargetMemMT sets the target memory transfer rate in MT/s.
func WithClockGovernorTargetMemMT(mt int) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if mt > 0 {
			g.config.TargetMemClockMT = mt
		}
	}
}

// WithClockGovernorSysfsDRMRoot sets a custom DRM sysfs directory for testing.
func WithClockGovernorSysfsDRMRoot(dir string) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if dir != "" {
			g.config.SysfsDRMRoot = dir
		}
	}
}

// WithClockGovernorDeviceCards sets preferred and fallback DRM card names.
func WithClockGovernorDeviceCards(preferred, fallback string) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if preferred != "" {
			g.config.PreferredCard = preferred
		}
		if fallback != "" {
			g.config.FallbackCard = fallback
		}
	}
}

// WithClockGovernorFileReader injects a custom file reader for mock testing.
func WithClockGovernorFileReader(fn FileReaderFunc) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if fn != nil {
			g.fileReader = fn
		}
	}
}

// WithClockGovernorFileWriter injects a custom file writer for mock testing.
func WithClockGovernorFileWriter(fn FileWriterFunc) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if fn != nil {
			g.fileWriter = fn
		}
	}
}

// WithClockGovernorCommandRunner injects a custom command execution runner for mock testing.
func WithClockGovernorCommandRunner(fn func(cmd string, args ...string) (string, error)) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if fn != nil {
			g.commandRunner = fn
		}
	}
}

// WithClockGovernorTimeFunc injects a virtual clock function for deterministic testing.
func WithClockGovernorTimeFunc(fn func() time.Time) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if fn != nil {
			g.nowFn = fn
		}
	}
}

// WithClockGovernorPollInterval sets the polling interval for the background hysteresis check loop.
func WithClockGovernorPollInterval(d time.Duration) HardwareClockGovernorOption {
	return func(g *HardwareClockGovernor) {
		if d > 0 {
			g.config.PollInterval = d
		}
	}
}

// Option aliases
var (
	WithGovernorPlatform      = WithClockGovernorPlatform
	WithGovernorHysteresis    = WithClockGovernorHysteresis
	WithGovernorTargetClocks  = WithClockGovernorTargetClocks
	WithGovernorTargetMemMT   = WithClockGovernorTargetMemMT
	WithGovernorSysfsDRMRoot  = WithClockGovernorSysfsDRMRoot
	WithGovernorDeviceCards   = WithClockGovernorDeviceCards
	WithGovernorCommandRunner = WithClockGovernorCommandRunner
	WithGovernorTimeFunc      = WithClockGovernorTimeFunc
	WithGovernorPollInterval  = WithClockGovernorPollInterval
)

// HardwareClockGovernor implements ClockGovernor as a cross-platform hardware clock
// and memory bandwidth saturation governor daemon.
//
// During active model serving with intermittent inter-token idle phases (15-50ms),
// OS and driver power governors aggressively downthrottle GPU compute cores and memory bus clocks:
// - On AMD: amdgpu DPM drops memory from 3750 MHz to low idle states;
// - On NVIDIA: lack of persistence mode enters P8 state;
// - On Apple Silicon: thermal QoS downgrades threads.
//
// HardwareClockGovernor locks performance states and prevents frequency dropping during
// active serving sessions with a 5000ms anti-flapping hysteresis timer (Issue #569).
type HardwareClockGovernor struct {
	// effectsMu orders external mutations independently of reader snapshots.
	effectsMu sync.Mutex
	config    HardwareGovernorConfig
	mu        sync.RWMutex

	locked             bool
	state              string // "QUIESCENT_IDLE", "LOCKED_PERFORMANCE", "HYSTERESIS_HOLD"
	lastActivity       time.Time
	throttleEvents     int64
	activeCoreClockMHz int
	activeMemClockMHz  int
	activeDevice       string

	// Platform state tracking
	nvmlPersistenceMode bool
	nvmlLockedCoreMHz   int
	nvmlLockedMemMHz    int
	appleQoSPolicy      string

	// Lifecycle
	started bool
	stopCh  chan struct{}
	doneCh  chan struct{}

	// Pluggable hooks
	nowFn         func() time.Time
	fileReader    FileReaderFunc
	fileWriter    FileWriterFunc
	commandRunner func(name string, args ...string) (string, error)
	statFn        func(name string) (os.FileInfo, error)
	readDirFn     func(name string) ([]os.DirEntry, error)
}

var _ ClockGovernor = (*HardwareClockGovernor)(nil)

// NewHardwareClockGovernor constructs a new HardwareClockGovernor daemon instance.
func NewHardwareClockGovernor(opts ...HardwareClockGovernorOption) *HardwareClockGovernor {
	cfg := DefaultHardwareGovernorConfig()
	g := &HardwareClockGovernor{
		config:     cfg,
		state:      "QUIESCENT_IDLE",
		fileReader: os.ReadFile,
		fileWriter: os.WriteFile,
		statFn:     os.Stat,
		readDirFn:  os.ReadDir,
	}
	for _, opt := range opts {
		opt(g)
	}
	if g.config.Platform == "" {
		g.config.Platform = PlatformAMD
	}
	if g.config.Hysteresis <= 0 {
		g.config.Hysteresis = DefaultHysteresisWindow
	}
	if g.config.TargetCoreClockMHz <= 0 {
		switch g.config.Platform {
		case PlatformNVIDIA:
			g.config.TargetCoreClockMHz = DefaultNVIDIALockedCoreClockMHz
		case PlatformApple:
			g.config.TargetCoreClockMHz = DefaultAppleLockedCoreClockMHz
		default:
			g.config.TargetCoreClockMHz = DefaultLockedShaderClockMHz
		}
	}
	if g.config.TargetMemClockMHz <= 0 {
		switch g.config.Platform {
		case PlatformNVIDIA:
			g.config.TargetMemClockMHz = DefaultNVIDIALockedMemClockMHz
		case PlatformApple:
			g.config.TargetMemClockMHz = DefaultAppleLockedMemClockMHz
		default:
			g.config.TargetMemClockMHz = 1000
		}
	}
	if g.config.SysfsDRMRoot == "" {
		g.config.SysfsDRMRoot = DefaultSysfsDRMPath
	}
	if g.config.PreferredCard == "" {
		g.config.PreferredCard = DefaultPreferredGPUCard
	}
	if g.config.FallbackCard == "" {
		g.config.FallbackCard = DefaultFallbackGPUCard
	}
	return g
}

func (g *HardwareClockGovernor) now() time.Time {
	if g.nowFn != nil {
		return g.nowFn()
	}
	return time.Now()
}

func (g *HardwareClockGovernor) resolveAMDDeviceDir() string {
	candidates := []string{g.config.PreferredCard, g.config.FallbackCard}
	for _, c := range candidates {
		if c != "" {
			p := filepath.Join(g.config.SysfsDRMRoot, c, "device")
			if fi, err := g.statFn(p); err == nil && fi.IsDir() {
				return p
			}
		}
	}
	if g.readDirFn != nil {
		if entries, err := g.readDirFn(g.config.SysfsDRMRoot); err == nil {
			for _, e := range entries {
				if strings.HasPrefix(e.Name(), "card") {
					p := filepath.Join(g.config.SysfsDRMRoot, e.Name(), "device")
					if fi, err := g.statFn(p); err == nil && fi.IsDir() {
						return p
					}
				}
			}
		}
	}
	return filepath.Join(g.config.SysfsDRMRoot, g.config.PreferredCard, "device")
}

// Platform returns the hardware accelerator platform.
func (g *HardwareClockGovernor) Platform() HardwarePlatform {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.config.Platform
}

// Hysteresis returns the currently configured anti-flapping hold window.
func (g *HardwareClockGovernor) Hysteresis() time.Duration {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.config.Hysteresis
}

// SetHysteresis updates the anti-flapping hold window.
func (g *HardwareClockGovernor) SetHysteresis(d time.Duration) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if d > 0 {
		g.config.Hysteresis = d
	}
}

// IsLocked reports whether the governor has locked performance frequencies.
func (g *HardwareClockGovernor) IsLocked() bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.locked
}

// SynthesizeLockCommands synthesizes vendor-specific shell / CLI commands to lock clocks.
func (g *HardwareClockGovernor) SynthesizeLockCommands() []string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	switch g.config.Platform {
	case PlatformNVIDIA:
		return []string{
			"nvidia-smi -pm 1",
			fmt.Sprintf("nvidia-smi -lgc %d,%d", g.config.TargetCoreClockMHz, g.config.TargetCoreClockMHz),
			fmt.Sprintf("nvidia-smi -lmc %d", g.config.TargetMemClockMHz),
		}
	case PlatformApple:
		return []string{
			"taskpolicy -c user_interactive -p 0",
		}
	default: // PlatformAMD
		devDir := g.resolveAMDDeviceDir()
		return []string{
			fmt.Sprintf("echo manual > %s/power_dpm_force_performance_level", devDir),
			fmt.Sprintf("echo 2 > %s/pp_dpm_sclk", devDir),
			fmt.Sprintf("echo 2 > %s/pp_dpm_mclk", devDir),
		}
	}
}

// SynthesizeUnlockCommands synthesizes vendor-specific shell / CLI commands to unlock clocks.
func (g *HardwareClockGovernor) SynthesizeUnlockCommands() []string {
	g.mu.RLock()
	defer g.mu.RUnlock()

	switch g.config.Platform {
	case PlatformNVIDIA:
		return []string{
			"nvidia-smi -rgc",
			"nvidia-smi -rmc",
			"nvidia-smi -pm 0",
		}
	case PlatformApple:
		return []string{
			"taskpolicy -c default -p 0",
		}
	default: // PlatformAMD
		devDir := g.resolveAMDDeviceDir()
		return []string{
			fmt.Sprintf("echo auto > %s/power_dpm_force_performance_level", devDir),
		}
	}
}

// applyClockTransition owns effect ordering from snapshot through publication.
// The private snapshot has its own mutex, so command synthesis cannot recursively
// take the live reader mutex while a syscall holds it.
func (g *HardwareClockGovernor) applyClockTransition(lock bool) error {
	g.effectsMu.Lock()
	defer g.effectsMu.Unlock()
	return g.applyClockTransitionOrdered(lock)
}

// applyClockTransitionOrdered requires effectsMu, including activity authority.
func (g *HardwareClockGovernor) applyClockTransitionOrdered(lock bool) error {
	g.mu.RLock()
	snapshot := &HardwareClockGovernor{
		config: g.config, fileReader: g.fileReader, fileWriter: g.fileWriter, commandRunner: g.commandRunner, statFn: g.statFn, readDirFn: g.readDirFn,
		locked: g.locked, state: g.state, activeCoreClockMHz: g.activeCoreClockMHz, activeMemClockMHz: g.activeMemClockMHz, activeDevice: g.activeDevice,
		nvmlPersistenceMode: g.nvmlPersistenceMode, nvmlLockedCoreMHz: g.nvmlLockedCoreMHz, nvmlLockedMemMHz: g.nvmlLockedMemMHz, appleQoSPolicy: g.appleQoSPolicy,
	}
	g.mu.RUnlock()
	var err error
	if lock {
		err = snapshot.lockSnapshot()
	} else {
		err = snapshot.unlockSnapshot()
	}
	if err != nil {
		return err
	}
	g.mu.Lock()
	g.locked, g.state = snapshot.locked, snapshot.state
	g.activeCoreClockMHz, g.activeMemClockMHz, g.activeDevice = snapshot.activeCoreClockMHz, snapshot.activeMemClockMHz, snapshot.activeDevice
	g.nvmlPersistenceMode, g.nvmlLockedCoreMHz, g.nvmlLockedMemMHz = snapshot.nvmlPersistenceMode, snapshot.nvmlLockedCoreMHz, snapshot.nvmlLockedMemMHz
	g.appleQoSPolicy = snapshot.appleQoSPolicy
	g.mu.Unlock()
	return nil
}

// Lock locks compute core and memory bus frequencies to peak performance states.
func (g *HardwareClockGovernor) Lock() error { return g.applyClockTransition(true) }

func (g *HardwareClockGovernor) lockSnapshot() error {

	switch g.config.Platform {
	case PlatformNVIDIA:
		cmds := []struct {
			cmd  string
			args []string
		}{
			{"nvidia-smi", []string{"-pm", "1"}},
			{"nvidia-smi", []string{"-lgc", fmt.Sprintf("%d,%d", g.config.TargetCoreClockMHz, g.config.TargetCoreClockMHz)}},
			{"nvidia-smi", []string{"-lmc", strconv.Itoa(g.config.TargetMemClockMHz)}},
		}
		if g.commandRunner != nil {
			for _, c := range cmds {
				if _, err := g.commandRunner(c.cmd, c.args...); err != nil {
					return fmt.Errorf("strix/governor: nvidia lock failed (%s %v): %w", c.cmd, c.args, err)
				}
			}
		}
		g.nvmlPersistenceMode = true
		g.nvmlLockedCoreMHz = g.config.TargetCoreClockMHz
		g.nvmlLockedMemMHz = g.config.TargetMemClockMHz
		g.activeCoreClockMHz = g.config.TargetCoreClockMHz
		g.activeMemClockMHz = g.config.TargetMemClockMHz
		g.activeDevice = "nvidia0"
		g.locked = true
		g.state = "LOCKED_PERFORMANCE"
		return nil

	case PlatformApple:
		if g.commandRunner != nil {
			if _, err := g.commandRunner("taskpolicy", "-c", "user_interactive", "-p", "0"); err != nil {
				return fmt.Errorf("strix/governor: apple taskpolicy failed: %w", err)
			}
		}
		g.appleQoSPolicy = "user_interactive"
		g.activeCoreClockMHz = g.config.TargetCoreClockMHz
		g.activeMemClockMHz = g.config.TargetMemClockMHz
		g.activeDevice = "soc0"
		g.locked = true
		g.state = "LOCKED_PERFORMANCE"
		return nil

	default: // PlatformAMD
		devDir := g.resolveAMDDeviceDir()
		g.activeDevice = filepath.Base(filepath.Dir(devDir))

		devExists := false
		if fi, err := g.statFn(devDir); err == nil && fi.IsDir() {
			devExists = true
		}

		if devExists && g.fileWriter != nil {
			// 1. power_dpm_force_performance_level = manual
			dpmFile := filepath.Join(devDir, SysfsDPMForcePerformanceLevel)
			if err := g.fileWriter(dpmFile, []byte("manual\n"), 0644); err != nil {
				return fmt.Errorf("strix/governor: failed to set manual dpm at %s: %w", dpmFile, err)
			}

			// 2. Discover or use sclk state index
			sclkIdx := 2
			sclkFile := filepath.Join(devDir, SysfsPPDpmSclk)
			if g.fileReader != nil {
				if data, err := g.fileReader(sclkFile); err == nil {
					if states, _, _, parseErr := amdgpuclock.ParseDPMSCLK(string(data)); parseErr == nil {
						if idx, findErr := amdgpuclock.FindStateForFrequency(states, g.config.TargetCoreClockMHz); findErr == nil {
							sclkIdx = idx
						}
					}
				}
			}
			if err := g.fileWriter(sclkFile, []byte(fmt.Sprintf("%d\n", sclkIdx)), 0644); err != nil {
				return fmt.Errorf("strix/governor: failed to write sclk state %d to %s: %w", sclkIdx, sclkFile, err)
			}

			// 3. Discover or use mclk state index
			mclkIdx := 2
			mclkFile := filepath.Join(devDir, SysfsPPDpmMclk)
			if g.fileReader != nil {
				if data, err := g.fileReader(mclkFile); err == nil {
					if states, _, _, _, parseErr := amdgpuclock.ParseDPMMCLK(string(data)); parseErr == nil {
						if idx, findErr := amdgpuclock.FindPeakMCLKState(states, g.config.TargetMemClockMT); findErr == nil {
							mclkIdx = idx
						}
					}
				}
			}
			if err := g.fileWriter(mclkFile, []byte(fmt.Sprintf("%d\n", mclkIdx)), 0644); err != nil {
				return fmt.Errorf("strix/governor: failed to write mclk state %d to %s: %w", mclkIdx, mclkFile, err)
			}
		}

		if g.commandRunner != nil {
			for _, cmdStr := range g.SynthesizeLockCommands() {
				parts := strings.Fields(cmdStr)
				if len(parts) > 0 {
					_, _ = g.commandRunner(parts[0], parts[1:]...)
				}
			}
		}

		g.activeCoreClockMHz = g.config.TargetCoreClockMHz
		g.activeMemClockMHz = g.config.TargetMemClockMHz
		g.locked = true
		g.state = "LOCKED_PERFORMANCE"
		return nil
	}
}

// Unlock restores default dynamic power management states.
func (g *HardwareClockGovernor) Unlock() error { return g.applyClockTransition(false) }

func (g *HardwareClockGovernor) unlockSnapshot() error {

	switch g.config.Platform {
	case PlatformNVIDIA:
		if g.commandRunner != nil {
			cmds := []struct {
				cmd  string
				args []string
			}{
				{"nvidia-smi", []string{"-rgc"}},
				{"nvidia-smi", []string{"-rmc"}},
				{"nvidia-smi", []string{"-pm", "0"}},
			}
			for _, c := range cmds {
				if _, err := g.commandRunner(c.cmd, c.args...); err != nil {
					return fmt.Errorf("strix/governor: nvidia reset failed (%s %v): %w", c.cmd, c.args, err)
				}
			}
		}
		g.nvmlPersistenceMode = false
		g.nvmlLockedCoreMHz = 0
		g.nvmlLockedMemMHz = 0
		g.locked = false
		g.state = "QUIESCENT_IDLE"
		return nil

	case PlatformApple:
		if g.commandRunner != nil {
			if _, err := g.commandRunner("taskpolicy", "-c", "default", "-p", "0"); err != nil {
				return fmt.Errorf("strix/governor: apple taskpolicy reset failed: %w", err)
			}
		}
		g.appleQoSPolicy = "default"
		g.locked = false
		g.state = "QUIESCENT_IDLE"
		return nil

	default: // PlatformAMD
		devDir := g.resolveAMDDeviceDir()
		devExists := false
		if fi, err := g.statFn(devDir); err == nil && fi.IsDir() {
			devExists = true
		}
		if devExists && g.fileWriter != nil {
			dpmFile := filepath.Join(devDir, SysfsDPMForcePerformanceLevel)
			if err := g.fileWriter(dpmFile, []byte("auto\n"), 0644); err != nil {
				return fmt.Errorf("strix/governor: failed to set auto dpm at %s: %w", dpmFile, err)
			}
		}
		if g.commandRunner != nil {
			for _, cmdStr := range g.SynthesizeUnlockCommands() {
				parts := strings.Fields(cmdStr)
				if len(parts) > 0 {
					_, _ = g.commandRunner(parts[0], parts[1:]...)
				}
			}
		}
		g.locked = false
		g.state = "QUIESCENT_IDLE"
		return nil
	}
}

// RecordActivity signals active serving work, locking clocks if quiescent and
// resetting the anti-flapping hysteresis hold timer.
func (g *HardwareClockGovernor) RecordActivity() {
	g.effectsMu.Lock()
	defer g.effectsMu.Unlock()
	g.mu.Lock()
	now := g.now()
	g.lastActivity = now
	wasLocked := g.locked
	autoLock := g.config.AutoLockOnActivity
	g.state = "LOCKED_PERFORMANCE"
	g.mu.Unlock()

	if !wasLocked && autoLock {
		_ = g.applyClockTransitionOrdered(true)
	}
}

// CheckHysteresis evaluates whether the anti-flapping hysteresis timer has expired.
// If idle time is within the hysteresis duration (default 5000ms), performance clock
// states are held without dropping. If idle duration exceeds the hysteresis window,
// clocks are automatically unlocked to quiescent mode.
// Returns true if the governor transitioned to unlocked due to expiration.
func (g *HardwareClockGovernor) CheckHysteresis(current ...time.Time) bool {
	g.effectsMu.Lock()
	defer g.effectsMu.Unlock()
	g.mu.Lock()
	if !g.locked {
		g.mu.Unlock()
		return false
	}

	var now time.Time
	if len(current) > 0 && !current[0].IsZero() {
		now = current[0]
	} else {
		now = g.now()
	}

	idleDuration := now.Sub(g.lastActivity)
	if idleDuration < g.config.Hysteresis {
		g.state = "HYSTERESIS_HOLD"
		g.mu.Unlock()
		return false
	}

	g.mu.Unlock()
	return g.applyClockTransitionOrdered(false) == nil
}

// RecordThrottleEvent increments the count of observed downthrottle events.
func (g *HardwareClockGovernor) RecordThrottleEvent() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.throttleEvents++
}

// Inspect queries the hardware backend for active core and memory clocks.
func (g *HardwareClockGovernor) Inspect() (int, int, bool, error) {
	g.effectsMu.Lock()
	defer g.effectsMu.Unlock()
	g.mu.RLock()
	snapshot := &HardwareClockGovernor{
		config: g.config, fileReader: g.fileReader, statFn: g.statFn, readDirFn: g.readDirFn,
		locked: g.locked, activeDevice: g.activeDevice,
		nvmlPersistenceMode: g.nvmlPersistenceMode, nvmlLockedCoreMHz: g.nvmlLockedCoreMHz, nvmlLockedMemMHz: g.nvmlLockedMemMHz, appleQoSPolicy: g.appleQoSPolicy,
		activeCoreClockMHz: g.activeCoreClockMHz, activeMemClockMHz: g.activeMemClockMHz,
	}
	g.mu.RUnlock()
	core, mem, locked, err := snapshot.inspectSnapshot()
	g.mu.Lock()
	g.throttleEvents += snapshot.throttleEvents
	g.activeCoreClockMHz, g.activeMemClockMHz = snapshot.activeCoreClockMHz, snapshot.activeMemClockMHz
	g.mu.Unlock()
	return core, mem, locked, err
}

func (g *HardwareClockGovernor) inspectSnapshot() (int, int, bool, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	switch g.config.Platform {
	case PlatformNVIDIA:
		core := g.nvmlLockedCoreMHz
		mem := g.nvmlLockedMemMHz
		if core == 0 {
			core = 405 // idle P8
		}
		if mem == 0 {
			mem = 405
		}
		isLocked := g.locked && g.nvmlPersistenceMode
		if g.locked && (core < g.config.TargetCoreClockMHz || mem < g.config.TargetMemClockMHz) {
			g.throttleEvents++
		}
		return core, mem, isLocked, nil

	case PlatformApple:
		isLocked := g.locked && g.appleQoSPolicy == "user_interactive"
		return g.config.TargetCoreClockMHz, g.config.TargetMemClockMHz, isLocked, nil

	default: // PlatformAMD
		devDir := g.resolveAMDDeviceDir()
		dpmFile := filepath.Join(devDir, SysfsDPMForcePerformanceLevel)
		perfLevel := "unknown"
		if g.fileReader != nil {
			if data, err := g.fileReader(dpmFile); err == nil {
				perfLevel = strings.TrimSpace(string(data))
			}
		}

		core := 0
		sclkFile := filepath.Join(devDir, SysfsPPDpmSclk)
		if g.fileReader != nil {
			if data, err := g.fileReader(sclkFile); err == nil {
				if _, _, freq, err := amdgpuclock.ParseDPMSCLK(string(data)); err == nil {
					core = freq
				}
			}
		}

		mem := 0
		mclkFile := filepath.Join(devDir, SysfsPPDpmMclk)
		if g.fileReader != nil {
			if data, err := g.fileReader(mclkFile); err == nil {
				if _, _, freq, _, err := amdgpuclock.ParseDPMMCLK(string(data)); err == nil {
					mem = freq
				}
			}
		}

		isLocked := (perfLevel == "manual" || perfLevel == "high") && core >= g.config.TargetCoreClockMHz
		if g.locked && !isLocked {
			g.throttleEvents++
		}
		if core > 0 {
			g.activeCoreClockMHz = core
		}
		if mem > 0 {
			g.activeMemClockMHz = mem
		}
		return core, mem, isLocked, nil
	}
}

// Status returns a telemetry snapshot of the hardware clock governor.
func (g *HardwareClockGovernor) Status() HardwareGovernorStatus {
	g.mu.RLock()
	defer g.mu.RUnlock()

	now := g.now()
	var lastActivityAgo int64
	if !g.lastActivity.IsZero() {
		lastActivityAgo = now.Sub(g.lastActivity).Milliseconds()
	}

	coreClock := g.activeCoreClockMHz
	memClock := g.activeMemClockMHz
	if coreClock == 0 {
		coreClock = g.config.TargetCoreClockMHz
	}
	if memClock == 0 {
		memClock = g.config.TargetMemClockMHz
	}

	dev := g.activeDevice
	if dev == "" {
		if g.config.Platform == PlatformAMD {
			dev = g.config.PreferredCard
		} else if g.config.Platform == PlatformNVIDIA {
			dev = "nvidia0"
		} else {
			dev = "soc0"
		}
	}

	return HardwareGovernorStatus{
		Platform:           g.config.Platform,
		Device:             dev,
		Locked:             g.locked,
		State:              g.state,
		CoreClockMHz:       coreClock,
		TargetCoreClockMHz: g.config.TargetCoreClockMHz,
		MemClockMHz:        memClock,
		TargetMemClockMHz:  g.config.TargetMemClockMHz,
		HysteresisMs:       g.config.Hysteresis.Milliseconds(),
		LastActivityAgoMs:  lastActivityAgo,
		ThrottleEvents:     g.throttleEvents,
		Reason:             fmt.Sprintf("governor state: %s (locked=%v, hysteresis=%s)", g.state, g.locked, g.config.Hysteresis),
	}
}

// Metrics formats real-time governor metrics in Prometheus exposition format.
func (g *HardwareClockGovernor) Metrics() string {
	st := g.Status()
	dev := st.Device
	if dev == "" {
		dev = "accelerator0"
	}
	plat := string(st.Platform)

	var sb strings.Builder
	sb.WriteString("# HELP hardware_clock_mhz Current hardware compute core clock frequency in MHz\n")
	sb.WriteString("# TYPE hardware_clock_mhz gauge\n")
	fmt.Fprintf(&sb, "hardware_clock_mhz{platform=%q,device=%q} %d\n\n", plat, dev, st.CoreClockMHz)

	sb.WriteString("# HELP memory_clock_mhz Current hardware memory bus clock frequency in MHz\n")
	sb.WriteString("# TYPE memory_clock_mhz gauge\n")
	fmt.Fprintf(&sb, "memory_clock_mhz{platform=%q,device=%q} %d\n\n", plat, dev, st.MemClockMHz)

	sb.WriteString("# HELP throttle_events_total Total hardware clock and memory downthrottle events detected\n")
	sb.WriteString("# TYPE throttle_events_total counter\n")
	fmt.Fprintf(&sb, "throttle_events_total{platform=%q,device=%q} %d\n", plat, dev, st.ThrottleEvents)

	return sb.String()
}

// Start runs the periodic hysteresis monitoring loop in the background.
func (g *HardwareClockGovernor) Start(ctx context.Context) error {
	g.mu.Lock()
	if g.started {
		g.mu.Unlock()
		return fmt.Errorf("strix/governor: HardwareClockGovernor already started")
	}
	g.started = true
	g.stopCh = make(chan struct{})
	g.doneCh = make(chan struct{})
	g.mu.Unlock()

	go func() {
		defer close(g.doneCh)
		interval := g.config.PollInterval
		if interval <= 0 {
			interval = 50 * time.Millisecond
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				_ = g.Unlock()
				return
			case <-g.stopCh:
				_ = g.Unlock()
				return
			case <-ticker.C:
				g.CheckHysteresis()
			}
		}
	}()
	return nil
}

// Stop terminates the background hysteresis monitoring loop and unlocks clocks.
func (g *HardwareClockGovernor) Stop() error {
	g.mu.Lock()
	if !g.started {
		g.mu.Unlock()
		return nil
	}
	g.started = false
	close(g.stopCh)
	g.mu.Unlock()

	<-g.doneCh
	return g.Unlock()
}
