package model

// v41_activation_trace_env.go -- the opt-in FAK_V41_ACTIVATION_TRACE=<path>
// recorder for the native V4.1 forward.
//
// computetrace.Enable was reachable only from the `fak compute-trace capture`
// toy, so a real V4.1 forward under fak serve/run could not produce a per-layer
// activation trace. With the variable set, each forwardV41 enables a recorder
// sized for that forward (every layer x every stage x up to
// v41ActivationTraceEnvMaxTokens tokens x 8 samples), runs, and writes the
// artifact -- with the consumed token ids in InputTokens -- to <path>,
// replacing the previous capture. Incremental Session.Step uses its own recorder
// and publishes its absolute-position checkpoints after committing or aborting.
// Suffix-prefill routes do not emit these activation checkpoints.
//
// The env recorder retains only v41.activation events. Other backend events
// cannot consume its event/sample budget; global observers still receive them.
//
// Unset, the cost is one cached string check per full forward or Step. Full
// forward capture is serialized because its recorder is process-global. Recorder
// installation atomically preserves an existing owner. The event budget bounds
// all recorded events; it is not an independently enforced token-coverage cap.
// Step recorders are transaction-local; only their publication is serialized.

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

const (
	v41ActivationTraceEnvVar       = "FAK_V41_ACTIVATION_TRACE"
	v41ActivationTraceEnvMaxTokens = 128
	// Five per-layer stages plus one slack slot for a stage a forward variant
	// emits twice.
	v41ActivationTraceEnvStageSlots = 6
)

var (
	v41ActivationTraceEnvPath = sync.OnceValue(func() string {
		return strings.TrimSpace(os.Getenv(v41ActivationTraceEnvVar))
	})
	v41ActivationTraceEnvMu         sync.Mutex
	v41ActivationTracePublicationMu sync.Mutex
	v41ActivationTracePublishedID   string
	v41ActivationTracePublishedPath string
)

// V41ActivationCaptureReceipt describes the session's last configured Step
// capture. Published records a successful write; Current means it remains the
// recorder's latest known publication. External file changes require artifact
// validation, and neither field proves hardware or reference authenticity.
type V41ActivationCaptureReceipt struct {
	ID               string `json:"id"`
	Token            int    `json:"token"`
	Outcome          string `json:"outcome"`
	Path             string `json:"path"`
	Published        bool   `json:"published"`
	Current          bool   `json:"current"`
	PublicationError string `json:"publication_error,omitempty"`
}

// LastV41ActivationCapture returns the last configured incremental capture.
// An overwritten cohort lane remains Published but is no longer Current.
func (s *Session) LastV41ActivationCapture() V41ActivationCaptureReceipt {
	if s == nil || s.v41Forward == nil {
		return V41ActivationCaptureReceipt{}
	}
	v41ActivationTracePublicationMu.Lock()
	defer v41ActivationTracePublicationMu.Unlock()
	r := s.v41Forward.activationCapture
	r.Current = r.Published && r.ID != "" && r.ID == v41ActivationTracePublishedID && r.Path == v41ActivationTracePublishedPath
	return r
}

type v41StepActivationCapture struct {
	state    *v41ForwardState
	recorder *computetrace.Recorder
	trace    *v41ActivationTraceState
	tokens   []int
	layers   int
	receipt  V41ActivationCaptureReceipt
	finished bool
}

func (m *Model) beginV41StepActivationCapture(id int, state *v41ForwardState) *v41StepActivationCapture {
	path := v41ActivationTraceEnvPath()
	if m == nil || state == nil || path == "" {
		return nil
	}
	c := &v41StepActivationCapture{
		state:   state,
		tokens:  append(append([]int(nil), state.history...), id),
		layers:  m.Cfg.NumLayers,
		receipt: V41ActivationCaptureReceipt{Token: len(state.history), Outcome: computetrace.ActivationCaptureAborted, Path: path},
	}
	var identity [16]byte
	if _, err := rand.Read(identity[:]); err != nil {
		c.receipt.PublicationError = fmt.Sprintf("capture identity: %v", err)
	} else {
		c.receipt.ID = fmt.Sprintf("v41-step-%x", identity[:])
		events, ok := checkedMulInt(max(1, c.layers), 5)
		samples, samplesOK := checkedMulInt(events, v41ActivationTraceMaxSamples)
		if !ok || !samplesOK {
			c.receipt.PublicationError = "capture budget overflow"
		} else {
			c.recorder = computetrace.NewWithOptions(events, computetrace.RecorderOptions{
				MaxSamples: samples,
				Operations: []string{v41ActivationTraceOperation},
			})
			c.trace = &v41ActivationTraceState{
				inputDigest:  m.v41ActivationInputIdentity(c.tokens),
				weightDigest: m.v41ActivationWeightIdentity(),
				recorder:     c.recorder,
				captureID:    c.receipt.ID,
			}
		}
	}
	v41ActivationTracePublicationMu.Lock()
	state.activationCapture = c.receipt
	v41ActivationTracePublicationMu.Unlock()
	return c
}

func (c *v41StepActivationCapture) finish(committed bool) {
	if c == nil || c.finished {
		return
	}
	c.finished = true
	coverage := c.state.forwardCoverage
	if committed && coverage != nil && coverage.Validate() == nil && coverage.LogitsSource == NativeForwardLogitsSourceV41Incremental {
		c.receipt.Outcome = computetrace.ActivationCaptureCommitted
	}
	v41ActivationTracePublicationMu.Lock()
	defer v41ActivationTracePublicationMu.Unlock()
	defer func() { c.state.activationCapture = c.receipt }()
	if c.recorder == nil {
		return
	}
	a := c.recorder.Artifact()
	a.InputTokens = c.tokens
	a.ActivationCapture = &computetrace.ActivationCapture{
		ID:             c.receipt.ID,
		Kind:           computetrace.ActivationCaptureIncrementalStep,
		Outcome:        c.receipt.Outcome,
		Token:          c.receipt.Token,
		ExpectedLayers: c.layers,
		ExpectedStages: []string{v41TraceStageQLatent, v41TraceStageKVLatent, v41TraceStageAttnOut, v41TraceStageMoESum, v41TraceStageResidual},
	}
	// A diagnostic publication failure must not turn a committed token into an
	// execution error that a caller might replay.
	defer func() {
		if failure := recover(); failure != nil {
			c.receipt.PublicationError = fmt.Sprintf("capture publication: %v", failure)
		}
	}()
	if err := writeV41ActivationTrace(c.receipt.Path, a); err != nil {
		c.receipt.PublicationError = err.Error()
		return
	}
	c.receipt.Published = true
	v41ActivationTracePublishedID, v41ActivationTracePublishedPath = c.receipt.ID, c.receipt.Path
}

// v41ActivationTraceFromEnv starts the env-selected recorder for one forward
// over seq and returns the finisher the forward defers, or nil when the
// variable is unset. The finisher writes the artifact even when the forward
// failed; a write failure becomes the forward's error only when the forward
// itself succeeded.
func (m *Model) v41ActivationTraceFromEnv(seq []int) func(error) error {
	path := v41ActivationTraceEnvPath()
	if m == nil || path == "" {
		return nil
	}
	v41ActivationTraceEnvMu.Lock()
	events := max(1, m.Cfg.NumLayers) * v41ActivationTraceEnvStageSlots * max(1, min(len(seq), v41ActivationTraceEnvMaxTokens))
	rec, disable := computetrace.TryEnableWithOptions(events, computetrace.RecorderOptions{
		MaxSamples: events * v41ActivationTraceMaxSamples,
		Operations: []string{v41ActivationTraceOperation},
	}, "v41-forward", v41ActivationTraceEnvVar)
	if rec == nil {
		v41ActivationTraceEnvMu.Unlock()
		return nil
	}
	tokens := append([]int(nil), seq...)
	return func(err error) error {
		// Publication belongs to the same critical section as capture, so an
		// older env capture cannot overwrite a later one after releasing it.
		defer v41ActivationTraceEnvMu.Unlock()
		disable()
		a := rec.Artifact()
		a.InputTokens = tokens
		werr := func() error {
			v41ActivationTracePublicationMu.Lock()
			defer v41ActivationTracePublicationMu.Unlock()
			werr := writeV41ActivationTrace(path, a)
			if werr == nil {
				v41ActivationTracePublishedID, v41ActivationTracePublishedPath = "", path
			}
			return werr
		}()
		if werr != nil && err == nil {
			return fmt.Errorf("model: %s=%s: %w", v41ActivationTraceEnvVar, path, werr)
		}
		return err
	}
}

// writeV41ActivationTrace publishes a closed, unique same-directory temporary
// file with Rename. Publication ordering is the caller's responsibility;
// filesystem/platform Rename semantics determine atomic replacement support.
// Uses the original fresh-temporary creation policy: 0644 before umask.
// The destination inode's mode is not inherited.
// Non-finite samples are clamped (ClampNonFinite) for JSON serialization.
func writeV41ActivationTrace(path string, a computetrace.Artifact) error {
	a.ClampNonFinite()
	data, err := json.Marshal(a)
	if err != nil {
		return err
	}
	f, err := createV41ActivationTraceTemp(path, rand.Reader)
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer os.Remove(tmp)
	if _, err := f.Write(append(data, '\n')); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// createV41ActivationTraceTemp retains exclusive, unpredictable same-directory
// creation while letting the kernel apply the original writer's 0644 mode and
// current umask. Chmod after CreateTemp would bypass the umask's restrictions.
// The entropy reader permits deterministic collision tests without a global hook.
func createV41ActivationTraceTemp(path string, entropy io.Reader) (*os.File, error) {
	dir, target := filepath.Dir(path), filepath.Base(path)
	for range 10000 {
		var suffix [16]byte
		if _, err := io.ReadFull(entropy, suffix[:]); err != nil {
			return nil, err
		}
		base := fmt.Sprintf(".fak-v41-trace-%x", suffix[:])
		// The configured destination may itself use our temporary naming
		// pattern. Never create it before publication, even on a case-folding
		// filesystem, or deferred cleanup could remove the final artifact.
		if strings.EqualFold(base, target) {
			continue
		}
		name := filepath.Join(dir, base)
		f, err := os.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if os.IsExist(err) {
			continue
		}
		return f, err
	}
	return nil, &os.PathError{Op: "createtemp", Path: path, Err: os.ErrExist}
}
