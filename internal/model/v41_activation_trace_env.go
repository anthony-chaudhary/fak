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
// replacing the previous forward's trace. A session Step recomputes its whole
// history, so the last write covers every position so far.
//
// Unset, the cost is one cached string check per forward. Env-traced forwards
// are serialized because the recorder is process-global, and the variable never
// clobbers a recorder some other caller already enabled. Tokens beyond the cap
// are dropped and counted, so a comparator reports the trace incomplete rather
// than silently short.

import (
	"encoding/json"
	"fmt"
	"os"
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
	if computetrace.Enabled() {
		v41ActivationTraceEnvMu.Unlock()
		return nil
	}
	events := max(1, m.Cfg.NumLayers) * v41ActivationTraceEnvStageSlots * max(1, min(len(seq), v41ActivationTraceEnvMaxTokens))
	rec, disable := computetrace.EnableWithSampleBudget(events, events*v41ActivationTraceMaxSamples, "v41-forward", v41ActivationTraceEnvVar)
	tokens := append([]int(nil), seq...)
	return func(err error) error {
		disable()
		v41ActivationTraceEnvMu.Unlock()
		a := rec.Artifact()
		a.InputTokens = tokens
		if werr := writeV41ActivationTrace(path, a); werr != nil && err == nil {
			return fmt.Errorf("model: %s=%s: %w", v41ActivationTraceEnvVar, path, werr)
		}
		return err
	}
}

// writeV41ActivationTrace replaces path atomically so a reader never sees a
// half-written trace. Non-finite samples are clamped (ClampNonFinite) because a
// diverging forward is exactly the one whose trace must still be written.
func writeV41ActivationTrace(path string, a computetrace.Artifact) error {
	a.ClampNonFinite()
	data, err := json.Marshal(a)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}
