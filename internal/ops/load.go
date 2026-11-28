// Load accounting for the autonomous operations daemon (#14474).
//
// Every maintenance tick — including one that finds nothing to do — is recorded
// as one fak-ops-load/1 cycle row. The pre-existing fak-ops-event/1 ledger only
// appends an Event when a phase MUTATED something (see engine.go), so a healthy
// daemon on a quiet repo and a daemon that died yesterday produce the same
// "zero events" read. This file is the accounting that makes a tick a witnessed
// fact rather than an inference.
package ops

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessres"
)

// LoadSchemaV1 is the schema identifier for an ops load-cycle row.
const LoadSchemaV1 = "fak-ops-load/1"

// DefaultLoadLedgerRel is the load ledger path relative to the repo root.
//
// It is deliberately a SEPARATE file from the fak-ops-event/1 ledger. The two
// rows carry different schemas, and Ledger.QueryEvents decodes the whole file
// into Event: json.Decoder would accept a load row as an Event with an empty
// ActionType, which would silently pollute both `fak ops log` and
// `fak ops status`'s recent_events count. Keeping the schemas in separate files
// makes that silent mis-decode impossible rather than merely unlikely.
const DefaultLoadLedgerRel = ".fak/ops-load.jsonl"

// LoadVerdict is the closed vocabulary for what a tick's load bought.
//
// The spine uses only the mechanically-derivable members: a tick either mutated
// something (PRODUCTIVE) or it did not but still cost host time (IDLE), or a
// phase failed (ERROR). The richer pressure-relief / skipped / unknown-host
// members and the idle-streak stretch policy belong to #14477.
type LoadVerdict string

const (
	// LoadProductive means at least one phase mutated repo or host state.
	LoadProductive LoadVerdict = "PRODUCTIVE"
	// LoadIdle means the tick cost measurable host time and mutated nothing.
	// Long idle runs are the healthy steady state for a quiet tree, NOT waste —
	// no cadence change may be licensed by a single idle tick (#14477).
	LoadIdle LoadVerdict = "IDLE"
	// LoadError means a phase was considered and failed.
	LoadError LoadVerdict = "ERROR"
	// LoadSkipped means the tick skipped all phases under an active pause.
	LoadSkipped LoadVerdict = "SKIPPED"
	// LoadPressureRelief means only pressure-relief phases ran while
	// deferrable phases were skipped under pressure (#14478).
	LoadPressureRelief LoadVerdict = "PRESSURE_RELIEF"
	// LoadPressureReliefOnly means deferrable junkwork was skipped under
	// pressure while pressure-relief phases ran (#14477).
	LoadPressureReliefOnly LoadVerdict = "PRESSURE_RELIEF_ONLY"
	// LoadUnknown means the host sample was unreadable (#14478).
	LoadUnknown LoadVerdict = "UNKNOWN"
)

// ValidLoadVerdict reports whether v is a member of the closed verdict vocabulary.
func ValidLoadVerdict(v LoadVerdict) bool {
	switch v {
	case LoadProductive, LoadIdle, LoadError, LoadSkipped, LoadPressureRelief, LoadPressureReliefOnly, LoadUnknown:
		return true
	}
	return false
}

// PhaseLoad records that one maintenance phase was considered on this tick.
//
// A phase that was due but not run and a phase that was not due are both
// recorded, because "we skipped the expensive tier" is exactly the load fact an
// operator has to be able to see. Ran therefore means "this tick considered the
// phase", and Mutated is the separate question of whether it did anything.
type PhaseLoad struct {
	Phase       string `json:"phase"`
	Ran         bool   `json:"ran"`
	WallMS      int64  `json:"wall_ms"`
	Mutated     bool   `json:"mutated"`
	Deferrable  bool   `json:"deferrable,omitempty"`
	PhaseAction string `json:"phase_action,omitempty"`
	// Skipped is each host-reaching step the phase declined because the engine
	// was not granted that reach, so a confined tick is visible, not silent.
	Skipped []ScopeSkip `json:"skipped,omitempty"`
}

// LoadCycle is one accounted maintenance tick under fak-ops-load/1.
type LoadCycle struct {
	Schema   string    `json:"schema"`
	Tick     uint64    `json:"tick"`
	Started  time.Time `json:"started"`
	Finished time.Time `json:"finished"`
	WallMS   int64     `json:"wall_ms"`
	// CPUSeconds is this process's total CPU consumption sampled around the
	// tick. HaveCPU distinguishes a measured zero from an unreadable axis.
	CPUSeconds float64 `json:"cpu_seconds"`
	HaveCPU    bool    `json:"have_cpu"`
	// SelfCPUMillis is the CPU time the tick itself consumed. Zero with
	// HaveCPU=false means "unmeasured", never "free".
	SelfCPUMillis int64            `json:"self_cpu_ms"`
	HaveSelfCPU   bool             `json:"have_self_cpu"`
	Phases        []PhaseLoad      `json:"phases"`
	Mutations     int              `json:"mutations"`
	Verdict       LoadVerdict      `json:"verdict"`
	PauseAction   string           `json:"pause_action,omitempty"`
	ReasonClass   PauseReasonClass `json:"reason_class,omitempty"`
	TriggerValue  string           `json:"trigger_value,omitempty"`
	Host          *HostLoad        `json:"host,omitempty"`
}

// HostSampleState is the closed three-way availability state for a host
// pressure reading on an ops tick (#14475).
type HostSampleState string

const (
	// HostSampleMeasured means RAM (and optionally swap/PSI) was read cleanly.
	HostSampleMeasured HostSampleState = "MEASURED"
	// HostSampleUnknown means the host sampler could not observe RAM (e.g.
	// HaveRAM=false on an unsupported platform), which is not host pressure.
	HostSampleUnknown HostSampleState = "UNKNOWN"
	// HostSampleError means the host sampler failed or returned corrupted data.
	HostSampleError HostSampleState = "ERROR"
)

// ValidHostSampleState reports whether s is a member of the closed host state vocabulary.
func ValidHostSampleState(s HostSampleState) bool {
	switch s {
	case HostSampleMeasured, HostSampleUnknown, HostSampleError:
		return true
	}
	return false
}

// HostLoad records the host-pressure snapshot sampled once per ops tick (#14475).
type HostLoad struct {
	State          HostSampleState `json:"state"`
	TotalRAMBytes  uint64          `json:"total_ram_bytes,omitempty"`
	AvailRAMBytes  uint64          `json:"avail_ram_bytes,omitempty"`
	HaveRAM        bool            `json:"have_ram,omitempty"`
	TotalSwapBytes uint64          `json:"total_swap_bytes,omitempty"`
	AvailSwapBytes uint64          `json:"avail_swap_bytes,omitempty"`
	HaveSwap       bool            `json:"have_swap,omitempty"`
	PSIAvg10       float64         `json:"psi_avg10,omitempty"`
	PSIAvg60       float64         `json:"psi_avg60,omitempty"`
	PSIAvg300      float64         `json:"psi_avg300,omitempty"`
	PSISomeTotalMS int64           `json:"psi_some_total_ms,omitempty"`
	PSIFullTotalMS int64           `json:"psi_full_total_ms,omitempty"`
	HavePSI        bool            `json:"have_psi,omitempty"`
	Error          string          `json:"error,omitempty"`
}

// HostSampler reads a point-in-time host pressure sample for an ops tick (#14475).
type HostSampler interface {
	SampleHost() (harnessres.HostSample, error)
}

// HarnessHostSampler delegates host pressure sampling to internal/harnessres.
type HarnessHostSampler struct{}

// SampleHost reads the host's RAM, swap, and PSI pressure via harnessres.ReadHostSample.
func (HarnessHostSampler) SampleHost() (harnessres.HostSample, error) {
	return harnessres.ReadHostSample(), nil
}

// classifyHostSample maps a raw harnessres.HostSample and error into the three-way
// HostLoad representation so "unknown" is never conflated with "busy" or "error".
func classifyHostSample(sample harnessres.HostSample, err error) *HostLoad {
	if err != nil {
		return &HostLoad{
			State: HostSampleError,
			Error: err.Error(),
		}
	}
	if sample.Corrupted {
		return &HostLoad{
			State: HostSampleError,
			Error: "corrupted host sample",
		}
	}
	if !sample.HaveRAM {
		return &HostLoad{
			State: HostSampleUnknown,
		}
	}
	if sample.TotalRAMBytes == 0 || sample.AvailRAMBytes > sample.TotalRAMBytes || (sample.HaveSwap && sample.AvailSwapBytes > sample.TotalSwapBytes) {
		return &HostLoad{
			State: HostSampleError,
			Error: "invalid host memory bounds",
		}
	}
	return &HostLoad{
		State:          HostSampleMeasured,
		TotalRAMBytes:  sample.TotalRAMBytes,
		AvailRAMBytes:  sample.AvailRAMBytes,
		HaveRAM:        sample.HaveRAM,
		TotalSwapBytes: sample.TotalSwapBytes,
		AvailSwapBytes: sample.AvailSwapBytes,
		HaveSwap:       sample.HaveSwap,
		PSIAvg10:       sample.PSIAvg10,
		PSIAvg60:       sample.PSIAvg60,
		PSIAvg300:      sample.PSIAvg300,
		PSISomeTotalMS: sample.PSISomeTotal.Milliseconds(),
		PSIFullTotalMS: sample.PSIFullTotal.Milliseconds(),
		HavePSI:        sample.HavePSI,
	}
}

// Validate enforces the row's own contract so a malformed cycle is refused at
// write time rather than discovered by a reader.
func (c LoadCycle) Validate() error {
	if c.Schema != LoadSchemaV1 {
		return fmt.Errorf("ops load: unexpected schema %q", c.Schema)
	}
	if c.Tick == 0 {
		return fmt.Errorf("ops load: tick ordinal must be positive")
	}
	if !ValidLoadVerdict(c.Verdict) {
		return fmt.Errorf("ops load: verdict %q is outside the closed vocabulary", c.Verdict)
	}
	if c.Host != nil && !ValidHostSampleState(c.Host.State) {
		return fmt.Errorf("ops load: host state %q is outside the closed vocabulary", c.Host.State)
	}
	if c.WallMS < 0 || c.SelfCPUMillis < 0 {
		return fmt.Errorf("ops load: negative duration")
	}
	return nil
}

// LoadLedger is the append-only fak-ops-load/1 persistence.
type LoadLedger struct {
	mu   sync.Mutex
	path string
}

// DefaultLoadLedgerPath returns the load ledger path for a repo root.
func DefaultLoadLedgerPath(root string) string {
	if root != "" {
		return filepath.Join(root, filepath.FromSlash(DefaultLoadLedgerRel))
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(os.TempDir(), "fak-ops-load.jsonl")
	}
	return filepath.Join(home, ".fak", "ops-load.jsonl")
}

// OpenLoadLedger opens or creates the load ledger, ensuring its directory.
func OpenLoadLedger(path string) (*LoadLedger, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return &LoadLedger{path: path}, nil
}

// Record appends one validated load cycle. There is no "nothing happened"
// short-circuit here: recording the cycle is the one action a tick may never
// skip.
func (l *LoadLedger) Record(c LoadCycle) error {
	if l == nil {
		return nil
	}
	if c.Schema == "" {
		c.Schema = LoadSchemaV1
	}
	if err := c.Validate(); err != nil {
		return err
	}
	data, err := json.Marshal(c)
	if err != nil {
		return err
	}

	l.mu.Lock()
	defer l.mu.Unlock()

	f, err := os.OpenFile(l.path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := f.Write(append(data, '\n')); err != nil {
		return err
	}
	return nil
}

// Read returns every accounted cycle in file order.
func (l *LoadLedger) Read() ([]LoadCycle, error) {
	if l == nil {
		return nil, nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	return readLoadCycles(l.path)
}

// Count returns the number of accounted cycles, used to derive the next
// monotonically increasing tick ordinal across daemon restarts.
func (l *LoadLedger) Count() (uint64, error) {
	cycles, err := l.Read()
	if err != nil {
		return 0, err
	}
	return uint64(len(cycles)), nil
}

// SelfCostFunc reads the cumulative CPU time this process has consumed. The
// Have bit is load-bearing: an unavailable reader must not be folded in as a
// zero cost, which would make an expensive tick look free.
type SelfCostFunc func() (time.Duration, bool)

// DefaultSelfCost reads this process's CPU time through the existing
// cross-platform harnessres reader (Getrusage on unix, GetProcessTimes on
// Windows). It reuses the shipped reader rather than adding a second one, and
// adds no module dependency: internal/harnessres is stdlib-only and imports
// nothing internal.
func DefaultSelfCost() (time.Duration, bool) {
	r, ok := harnessres.ReadProcessResource(os.Getpid())
	if !ok || !r.HaveCPU {
		return 0, false
	}
	return time.Duration(r.CPUSeconds * float64(time.Second)), true
}

func readLoadCycles(path string) ([]LoadCycle, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()

	dec := json.NewDecoder(f)
	var cycles []LoadCycle
	for {
		var c LoadCycle
		if err := dec.Decode(&c); err != nil {
			if err == io.EOF {
				break
			}
			// A torn tail row (a crash mid-append) must not make every earlier
			// cycle unreadable: return what was parsed plus the error, so a
			// reader can surface the tail damage without losing the history.
			return cycles, err
		}
		cycles = append(cycles, c)
	}
	return cycles, nil
}
