package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

// ErrServeRoleUnknown marks a --role value outside the closed native P/D role
// admission set. A misspelled role must refuse startup before any weight load
// rather than silently boot an ordinary combined serve that ignores the split.
var ErrServeRoleUnknown = errors.New("unknown serve role")

// ErrServeRoleMismatch marks a role admission surface invoked on the wrong role
// host: a prefill host admits prompts only, a decode host admits imported
// sequences only. The asymmetry is the contract — a host that answered both
// would be a combined serve wearing a role flag.
var ErrServeRoleMismatch = errors.New("serve role mismatch")

// ErrServeRoleRequiresModel marks a native P/D role host without the in-kernel
// model: role serving is defined over resident weights, so without them there
// is nothing to split into prefill and decode work.
var ErrServeRoleRequiresModel = errors.New("serve role requires the in-kernel model")

// resolveServePDRole resolves the --role selector against the closed admission
// set {prefill, decode, empty}: empty keeps the ordinary combined serve. The
// normalization (lowercase, trim) matches resolveSpeculativeMode so an admitted
// value reaches the role host byte-for-byte.
func resolveServePDRole(flagValue string) (modelengine.NativePDRole, error) {
	effective := strings.ToLower(strings.TrimSpace(flagValue))
	switch effective {
	case "":
		return modelengine.NativePDRole(""), nil
	case string(modelengine.NativePDRolePrefill):
		return modelengine.NativePDRolePrefill, nil
	case string(modelengine.NativePDRoleDecode):
		return modelengine.NativePDRoleDecode, nil
	default:
		return "", fmt.Errorf("%w: %q (admitted: prefill, decode, empty)", ErrServeRoleUnknown, effective)
	}
}

// servePDRoleDefaultDecodeWorkers is the decode half of a role host's pool when
// --role decode routes imported sequences. Two workers let round-robin admission
// queue one import while an earlier lane is still decoding.
const servePDRoleDefaultDecodeWorkers = 2

// servePDRoleHost is the per-host native P/D role runtime: a NativePDCluster
// exposed through exactly one role admission surface. Keeping the two surfaces
// behind role checks is what makes --role meaningful — a prefill host never
// admits imported decode work, and a decode host never runs prompt prefill.
type servePDRoleHost struct {
	role    modelengine.NativePDRole
	cluster *modelengine.NativePDCluster

	mu     sync.Mutex
	rr     int
	closed bool
}

// newServePDRoleHost builds a role host over the in-kernel model. The guards are
// fail-loud by design: an unknown role or a missing model must refuse the host
// instead of constructing a cluster that silently serves the wrong shape.
func newServePDRoleHost(role modelengine.NativePDRole, m *fakmodel.Model, decodeWorkers int) (*servePDRoleHost, error) {
	switch role {
	case modelengine.NativePDRolePrefill, modelengine.NativePDRoleDecode:
	default:
		return nil, fmt.Errorf("%w: %q (admitted: prefill, decode)", ErrServeRoleUnknown, role)
	}
	if m == nil {
		return nil, fmt.Errorf("serve P/D role host: %w", ErrServeRoleRequiresModel)
	}
	if decodeWorkers < 1 {
		return nil, fmt.Errorf("serve P/D role host: decode workers must be >= 1, got %d", decodeWorkers)
	}
	return &servePDRoleHost{role: role, cluster: modelengine.NewNativePDCluster(m, decodeWorkers)}, nil
}

func (h *servePDRoleHost) Role() modelengine.NativePDRole { return h.role }

// Cluster returns the underlying cluster; it satisfies
// gateway.NativePDMetricsProvider via Metrics(), which is the seam the serve
// entrypoint wires onto per-role /metrics.
func (h *servePDRoleHost) Cluster() *modelengine.NativePDCluster { return h.cluster }

// AdmitPrefill is the prefill role surface: prompt prefill plus KV transfer
// through the cluster's Admit. A decode host answers with ErrServeRoleMismatch
// so misrouted work fails at the caller instead of silently re-prefilling.
func (h *servePDRoleHost) AdmitPrefill(ctx context.Context, req modelengine.NativePDRequest) (modelengine.NativePDAdmission, error) {
	if h == nil || h.role != modelengine.NativePDRolePrefill {
		return modelengine.NativePDAdmission{}, fmt.Errorf("%w: AdmitPrefill requires the prefill role", ErrServeRoleMismatch)
	}
	return h.cluster.Admit(ctx, req)
}

// AdmitImported is the decode role surface: sequences whose prompt KV was built
// elsewhere enter a decode worker's scheduler without re-prefill. Round-robin
// over the decode pool spreads imports; the mutex keeps the counter race-free
// under concurrent admissions.
func (h *servePDRoleHost) AdmitImported(ctx context.Context, call *abi.ToolCall, seq modelengine.ImportedSequence) (abi.EngineRequest, error) {
	if h == nil || h.role != modelengine.NativePDRoleDecode {
		return nil, fmt.Errorf("%w: AdmitImported requires the decode role", ErrServeRoleMismatch)
	}
	workers := h.cluster.DecodeWorkers()
	if len(workers) == 0 {
		return nil, errors.New("serve P/D role host: no decode workers")
	}
	h.mu.Lock()
	w := workers[h.rr%len(workers)]
	h.rr++
	h.mu.Unlock()
	return w.Scheduler().AdmitImported(ctx, call, seq)
}

// Close is idempotent and nil-safe: serve teardown paths may run it alongside
// deferred cleanup without coordinating, and a double close must not re-close
// live scheduler state underneath an in-flight admission.
func (h *servePDRoleHost) Close() {
	if h == nil {
		return
	}
	h.mu.Lock()
	if h.closed {
		h.mu.Unlock()
		return
	}
	h.closed = true
	h.mu.Unlock()
	h.cluster.Close()
}
