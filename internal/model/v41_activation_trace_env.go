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
// replacing the previous full forward's trace. Incremental Session.Step and
// suffix-prefill routes do not currently emit these activation checkpoints;
// their successful completion does not update this artifact.
//
// The env recorder retains only v41.activation events. Other backend events
// cannot consume its event/sample budget; global observers still receive them.
//
// Unset, the cost is one cached string check per full forward. Env capture and
// publication are serialized because the recorder is process-global. Recorder
// installation atomically preserves an existing owner. The event budget bounds
// all recorded events; it is not an independently enforced token-coverage cap.

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
	v41ActivationTraceEnvMu sync.Mutex
)

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
		if werr := writeV41ActivationTrace(path, a); werr != nil && err == nil {
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
