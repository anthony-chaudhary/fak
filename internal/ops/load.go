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
)

// ValidLoadVerdict reports whether v is a member of the closed verdict vocabulary.
func ValidLoadVerdict(v LoadVerdict) bool {
	switch v {
	case LoadProductive, LoadIdle, LoadError:
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
	Phase   string `json:"phase"`
	Ran     bool   `json:"ran"`
	WallMS  int64  `json:"wall_ms"`
	Mutated bool   `json:"mutated"`
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
	SelfCPUMillis int64       `json:"self_cpu_ms"`
	HaveSelfCPU   bool        `json:"have_self_cpu"`
	Phases        []PhaseLoad `json:"phases"`
	Mutations     int         `json:"mutations"`
	Verdict       LoadVerdict `json:"verdict"`
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
