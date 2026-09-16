package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ConcurrencyMechanism names the layer that serializes concurrent same-prefix
// turnkey fan-out, as measured by a device-forward phase timeline. It is the
// closed vocabulary issue #1589 asked for (lock / single-queue / no-batch):
//
//   - MechanismDeviceMutex: admitted requests reach the forward boundary
//     concurrently but enter it one at a time because one planner-scoped mutex
//     (devMu) is held across the WHOLE device forward pass (Prefill + decode
//     loop). The observed gain is overlap-of-admission, never a shared batch.
//   - MechanismSingleQueueNoBatch: requests entered the forward concurrently
//     (max_concurrent_forward > 1) and no mutex serialized them - the
//     accelerator ran them concurrently but never fused them into a shared
//     batch.
//   - MechanismUnknown: no concurrent entrant was observed, so nothing can be
//     named; callers must not treat this as a fix input.
type ConcurrencyMechanism string

const (
	// MechanismDeviceMutex is a whole-forward mutex (devMu) on the device path.
	MechanismDeviceMutex ConcurrencyMechanism = "device-mutex"
	// MechanismSingleQueueNoBatch is concurrent-but-unfused execution.
	MechanismSingleQueueNoBatch ConcurrencyMechanism = "single-queue-no-batch"
	// MechanismUnknown means the timeline could not name a mechanism.
	MechanismUnknown ConcurrencyMechanism = "unknown"
)

// ConcurrencyPhase records the timestamps of one request's admission-to-forward
// path. All timestamps are monotonic offsets in nanoseconds from the profile's
// start instant so a receipt is orderable without a wall clock.
type ConcurrencyPhase struct {
	RequestID    int   `json:"request_id"`
	AdmittedNS   int64 `json:"admitted_ns"`
	ForwardInNS  int64 `json:"forward_in_ns"`
	ForwardOutNS int64 `json:"forward_out_ns"`
	Serialized   bool  `json:"serialized"`
}

// ConcurrencyProfile is the durable phase timeline receipt for one concurrent
// fan-out cell (issue #1589). It is the input contract the batched-path child
// (#1590) consumes: it names the serialization layer with a witnessing timeline
// instead of guessing the mechanism.
type ConcurrencyProfile struct {
	Schema        string               `json:"schema"`
	GeneratedAt   time.Time            `json:"generated_at"`
	Concurrency   int                  `json:"concurrency"`
	Mechanism     ConcurrencyMechanism `json:"mechanism"`
	MaxConcurrent int                  `json:"max_concurrent_forward"`
	ForwardPath   string               `json:"forward_path,omitempty"`
	Phases        []ConcurrencyPhase   `json:"phases"`
	Witness       string               `json:"witness"`
}

// ConcurrencyProfileSchema is the schema identifier for the concurrency profile receipt.
const ConcurrencyProfileSchema = "fak.inkernel_concurrency_profile.v1"

// concurrencyProfiler is an opt-in, low-overhead recorder for the device-forward
// phase timeline. Its zero value is inert, so a planner that never opts in is
// byte-for-byte unaffected. When enabled it is safe for concurrent use: each
// request appends one phase under a short critical section, never across the
// device forward itself.
type concurrencyProfiler struct {
	mu      sync.Mutex
	start   time.Time
	enabled bool
	nextID  int
	phases  []*ConcurrencyPhase
	open    int
	maxOpen int
}

func newConcurrencyProfiler() *concurrencyProfiler {
	return &concurrencyProfiler{start: time.Now(), enabled: true}
}

// admit registers a request at the fan-out boundary and returns the phase slot
// to be completed by forwardEnter/forwardExit.
func (c *concurrencyProfiler) admit() *ConcurrencyPhase {
	if c == nil || !c.enabled {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nextID++
	ph := &ConcurrencyPhase{RequestID: c.nextID, AdmittedNS: time.Since(c.start).Nanoseconds()}
	c.phases = append(c.phases, ph)
	return ph
}

// forwardEnter marks entry into the serialized forward critical section and
// reports how many requests are concurrently inside (including this one).
func (c *concurrencyProfiler) forwardEnter(ph *ConcurrencyPhase, serialized bool) int {
	if c == nil || !c.enabled || ph == nil {
		return 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.open++
	if c.open > c.maxOpen {
		c.maxOpen = c.open
	}
	ph.ForwardInNS = time.Since(c.start).Nanoseconds()
	ph.Serialized = serialized
	return c.open
}

// forwardExit marks leaving the forward critical section.
func (c *concurrencyProfiler) forwardExit(ph *ConcurrencyPhase) {
	if c == nil || !c.enabled || ph == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.open--
	ph.ForwardOutNS = time.Since(c.start).Nanoseconds()
}

// build folds the recorded phases into a named-mechanism profile receipt. The
// naming rule is deterministic from the timeline:
//
//	max_concurrent_forward > 1                   -> single-queue-no-batch
//	max_concurrent_forward == 1 AND admitted > 1 -> device-mutex
//	otherwise                                     -> unknown
func (c *concurrencyProfiler) build(forwardPath string) *ConcurrencyProfile {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	prof := &ConcurrencyProfile{
		Schema:        ConcurrencyProfileSchema,
		GeneratedAt:   time.Now().UTC(),
		Concurrency:   len(c.phases),
		MaxConcurrent: c.maxOpen,
		ForwardPath:   forwardPath,
		Phases:        make([]ConcurrencyPhase, 0, len(c.phases)),
	}
	for _, ph := range c.phases {
		prof.Phases = append(prof.Phases, *ph)
	}
	switch {
	case len(c.phases) == 0:
		prof.Mechanism = MechanismUnknown
		prof.Witness = "no requests admitted: nothing to name"
	case c.maxOpen > 1:
		prof.Mechanism = MechanismSingleQueueNoBatch
		prof.Witness = "requests entered the forward concurrently (max_concurrent_forward > 1) and no mutex serialized them; the accelerator never fused them into a shared batch"
	case c.maxOpen == 1 && len(c.phases) > 1:
		prof.Mechanism = MechanismDeviceMutex
		prof.Witness = "admitted concurrently but max_concurrent_forward == 1: a single forward-pass mutex (devMu) serialized the whole Prefill+decode for each request"
	default:
		prof.Mechanism = MechanismUnknown
		prof.Witness = "single request or no overlap observed; mechanism not determinable"
	}
	return prof
}

// Save writes the profile receipt to disk as indented JSON (durable artifact for
// the batched-path child to consume).
func (p *ConcurrencyProfile) Save(filePath string) error {
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	if dir := filepath.Dir(filePath); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	return os.WriteFile(filePath, data, 0o644)
}
