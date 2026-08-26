//go:build darwin && arm64 && cgo

package metalgemm

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework Foundation -framework CoreFoundation
#include <stdint.h>

typedef struct {
    uintptr_t command_buffer;
    int committed;
    int completed_wait;
    int encoders;
    int state_h2d_transfers;
    int state_d2h_transfers;
    int host_recurrence_steps;
    int owned_buffers;
    int private_state_buffers;
    int panel_h2d_transfers;
    int output_d2h_transfers;
    uint64_t state_bytes;
} mg_gdn_event;

int mg_gdn_state_new(int nK, int nV, int kHd, int vHd, int convKernel,
                     uint64_t *convHandle, uint64_t *recurrentHandle);
int mg_gdn_state_run(int owner,
                     const float *mixed, const float *z, const float *b, const float *a,
                     const float *conv, const float *aLog, const float *dtBias, const float *norm,
                     int tokens, int nK, int nV, int kHd, int vHd, int convKernel, float eps,
                     float *core, int injectPostSubmitFailure, mg_gdn_event *event);
int mg_gdn_state_reset(int owner);
int mg_gdn_state_snapshot(int owner, float *conv, int convElems, float *recurrent, int recurrentElems);
void mg_gdn_state_release(int owner);
int mg_gdn_live_buffers(void);
uint64_t mg_gdn_current_allocated_size(void);
*/
import "C"

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"
)

const (
	// GDNMaxPanelTokens bounds one serial token scan. Larger prompts are split by
	// the caller while the same persistent owner carries state across calls.
	GDNMaxPanelTokens  = 64
	gdnMaxKeyHeads     = 32
	gdnMaxValueHeads   = 64
	gdnMaxKeyHeadDim   = 128
	gdnMaxValueHeadDim = 256
	gdnMaxConvKernel   = 8
)

// GDNGeometry is the supported preprojected Qwen Gated-DeltaNet geometry.
type GDNGeometry struct {
	NumKeyHeads, NumValueHeads int
	KeyHeadDim, ValueHeadDim   int
	ConvKernel                 int
}

func (g GDNGeometry) keyDim() int   { return g.NumKeyHeads * g.KeyHeadDim }
func (g GDNGeometry) valueDim() int { return g.NumValueHeads * g.ValueHeadDim }
func (g GDNGeometry) convDim() int  { return 2*g.keyDim() + g.valueDim() }

func (g GDNGeometry) validate() error {
	if g.NumKeyHeads < 1 || g.NumKeyHeads > gdnMaxKeyHeads ||
		g.NumValueHeads < 1 || g.NumValueHeads > gdnMaxValueHeads ||
		g.NumValueHeads%g.NumKeyHeads != 0 {
		return fmt.Errorf("value heads must be a positive multiple of key heads within %d/%d", gdnMaxKeyHeads, gdnMaxValueHeads)
	}
	if g.KeyHeadDim < 1 || g.KeyHeadDim > gdnMaxKeyHeadDim ||
		g.ValueHeadDim < 1 || g.ValueHeadDim > gdnMaxValueHeadDim {
		return fmt.Errorf("head dimensions exceed supported key/value bounds %d/%d", gdnMaxKeyHeadDim, gdnMaxValueHeadDim)
	}
	if g.ConvKernel < 1 || g.ConvKernel > gdnMaxConvKernel {
		return fmt.Errorf("convolution kernel must be in [1,%d]", gdnMaxConvKernel)
	}
	return nil
}

// GDNPanel contains already-projected sequence operands. Input/output staging
// may cross the host boundary; the persistent convolution and recurrent state
// identified by GDNState never does during an accepted call.
type GDNPanel struct {
	Tokens                     int
	Mixed, Z, B, A             []float32
	Conv1D, ALog, DTBias, Norm []float32
	RMSNormEpsilon             float32
}

func (p GDNPanel) validate(g GDNGeometry) error {
	if err := g.validate(); err != nil {
		return err
	}
	if p.Tokens < 1 || p.Tokens > GDNMaxPanelTokens {
		return fmt.Errorf("tokens=%d outside supported panel [1,%d]", p.Tokens, GDNMaxPanelTokens)
	}
	wants := []struct {
		name string
		got  int
		want int
	}{
		{"mixed", len(p.Mixed), p.Tokens * g.convDim()},
		{"z", len(p.Z), p.Tokens * g.valueDim()},
		{"b", len(p.B), p.Tokens * g.NumValueHeads},
		{"a", len(p.A), p.Tokens * g.NumValueHeads},
		{"conv1d", len(p.Conv1D), g.convDim() * g.ConvKernel},
		{"a_log", len(p.ALog), g.NumValueHeads},
		{"dt_bias", len(p.DTBias), g.NumValueHeads},
		{"norm", len(p.Norm), g.ValueHeadDim},
	}
	for _, shape := range wants {
		if shape.got != shape.want {
			return fmt.Errorf("%s elements=%d, want %d", shape.name, shape.got, shape.want)
		}
	}
	if p.RMSNormEpsilon <= 0 {
		return fmt.Errorf("RMSNorm epsilon must be positive")
	}
	return nil
}

// GDNStateHandle is an opaque native state-buffer identity. The two handles of
// one owner are non-zero, distinct, stable across Run/Reset, and never reused
// while live.
type GDNStateHandle uint64

// GDNAccounting binds one call to native command-buffer and state-ownership facts.
type GDNAccounting struct {
	CommandBufferID                       uint64
	Committed, CompletedWait              bool
	Encoders                              int
	StateH2DTransfers, StateD2HTransfers  int
	HostRecurrenceSteps                   int
	OwnedBuffers, PrivateStateBuffers     int
	PanelH2DTransfers, OutputD2HTransfers int
	StateBytes                            uint64
}

// GDNDeclinedError is a pre-submit refusal; no persistent state was mutated.
type GDNDeclinedError struct{ Reason string }

func (e *GDNDeclinedError) Error() string {
	return "metalgemm: GDN sequence declined before mutation: " + e.Reason
}

// GDNPostSubmitError is a typed accepted-call failure. The owner has been
// released before this error is returned and must never be retried or fallen back.
type GDNPostSubmitError struct{ Reason string }

func (e *GDNPostSubmitError) Error() string {
	return "metalgemm: GDN sequence failed after submit: " + e.Reason
}

func IsGDNPostSubmit(err error) bool {
	var post *GDNPostSubmitError
	return errors.As(err, &post)
}

// GDNState owns two distinct persistent private Metal buffers.
type GDNState struct {
	mu              sync.Mutex
	owner           C.int
	geometry        GDNGeometry
	conv, recurrent GDNStateHandle
	closed          bool
}

// NewGDNState allocates zeroed convolution-window and recurrent-state buffers.
func NewGDNState(g GDNGeometry) (*GDNState, error) {
	if err := g.validate(); err != nil {
		return nil, &GDNDeclinedError{Reason: err.Error()}
	}
	if !Available() {
		return nil, &GDNDeclinedError{Reason: "Metal unavailable"}
	}
	var conv, recurrent C.uint64_t
	owner := C.mg_gdn_state_new(C.int(g.NumKeyHeads), C.int(g.NumValueHeads), C.int(g.KeyHeadDim),
		C.int(g.ValueHeadDim), C.int(g.ConvKernel), &conv, &recurrent)
	if owner < 0 || conv == 0 || recurrent == 0 || conv == recurrent {
		if owner >= 0 {
			C.mg_gdn_state_release(owner)
		}
		return nil, fmt.Errorf("metalgemm: allocate GDN auxiliary state")
	}
	return &GDNState{owner: owner, geometry: g, conv: GDNStateHandle(conv), recurrent: GDNStateHandle(recurrent)}, nil
}

// Handles returns the stable convolution and recurrent identities.
func (s *GDNState) Handles() (GDNStateHandle, GDNStateHandle) {
	if s == nil {
		return 0, 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return 0, 0
	}
	return s.conv, s.recurrent
}

func gdnF32(p []float32) *C.float { return (*C.float)(unsafe.Pointer(&p[0])) }

func accountingFromC(event C.mg_gdn_event) GDNAccounting {
	return GDNAccounting{
		CommandBufferID: uint64(event.command_buffer),
		Committed:       event.committed != 0, CompletedWait: event.completed_wait != 0,
		Encoders: int(event.encoders), StateH2DTransfers: int(event.state_h2d_transfers),
		StateD2HTransfers: int(event.state_d2h_transfers), HostRecurrenceSteps: int(event.host_recurrence_steps),
		OwnedBuffers: int(event.owned_buffers), PrivateStateBuffers: int(event.private_state_buffers),
		PanelH2DTransfers: int(event.panel_h2d_transfers), OutputD2HTransfers: int(event.output_d2h_transfers),
		StateBytes: uint64(event.state_bytes),
	}
}

// Run executes convolution, Q/K normalization, recurrent scan, and gated RMSNorm
// in one command buffer and one encoder. accepted is true once native submission
// occurs, including post-submit failures.
func (s *GDNState) Run(panel GDNPanel) (core []float32, accounting GDNAccounting, accepted bool, err error) {
	return s.run(panel, false)
}

func (s *GDNState) run(panel GDNPanel, injectPostSubmitFailure bool) ([]float32, GDNAccounting, bool, error) {
	if s == nil {
		return nil, GDNAccounting{}, false, &GDNDeclinedError{Reason: "nil owner"}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, GDNAccounting{}, false, &GDNDeclinedError{Reason: "owner is closed"}
	}
	if err := panel.validate(s.geometry); err != nil {
		return nil, GDNAccounting{OwnedBuffers: 2}, false, &GDNDeclinedError{Reason: err.Error()}
	}
	core := make([]float32, panel.Tokens*s.geometry.valueDim())
	var event C.mg_gdn_event
	inject := C.int(0)
	if injectPostSubmitFailure {
		inject = 1
	}
	status := C.mg_gdn_state_run(s.owner,
		gdnF32(panel.Mixed), gdnF32(panel.Z), gdnF32(panel.B), gdnF32(panel.A),
		gdnF32(panel.Conv1D), gdnF32(panel.ALog), gdnF32(panel.DTBias), gdnF32(panel.Norm),
		C.int(panel.Tokens), C.int(s.geometry.NumKeyHeads), C.int(s.geometry.NumValueHeads),
		C.int(s.geometry.KeyHeadDim), C.int(s.geometry.ValueHeadDim), C.int(s.geometry.ConvKernel),
		C.float(panel.RMSNormEpsilon), gdnF32(core), inject, &event)
	accounting := accountingFromC(event)
	if status == 1 {
		return core, accounting, true, nil
	}
	if status < 0 || accounting.Committed {
		s.releaseLocked()
		return nil, accounting, true, &GDNPostSubmitError{Reason: "native operation did not complete successfully"}
	}
	return nil, accounting, false, &GDNDeclinedError{Reason: "native validation refused owner or geometry"}
}

// Reset zeros both persistent buffers without changing their identities.
func (s *GDNState) Reset() error {
	if s == nil {
		return &GDNDeclinedError{Reason: "nil owner"}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return &GDNDeclinedError{Reason: "owner is closed"}
	}
	if C.mg_gdn_state_reset(s.owner) != 1 {
		s.releaseLocked()
		return &GDNPostSubmitError{Reason: "state reset failed"}
	}
	return nil
}

// Snapshot copies persistent state to host for explicit diagnostics and oracle
// witnesses. It is never called by Run and is therefore outside that operation's
// zero-state-transfer accounting boundary.
func (s *GDNState) Snapshot() (conv, recurrent []float32, err error) {
	if s == nil {
		return nil, nil, &GDNDeclinedError{Reason: "nil owner"}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, nil, &GDNDeclinedError{Reason: "owner is closed"}
	}
	conv = make([]float32, (s.geometry.ConvKernel-1)*s.geometry.convDim())
	recurrent = make([]float32, s.geometry.NumValueHeads*s.geometry.KeyHeadDim*s.geometry.ValueHeadDim)
	convPtr := (*C.float)(nil)
	if len(conv) > 0 {
		convPtr = gdnF32(conv)
	}
	if C.mg_gdn_state_snapshot(s.owner, convPtr, C.int(len(conv)), gdnF32(recurrent), C.int(len(recurrent))) != 1 {
		return nil, nil, fmt.Errorf("metalgemm: snapshot GDN state")
	}
	return conv, recurrent, nil
}

func (s *GDNState) releaseLocked() {
	if s.closed {
		return
	}
	C.mg_gdn_state_release(s.owner)
	s.closed = true
	s.owner = -1
	s.conv, s.recurrent = 0, 0
}

// Close releases both buffers exactly once.
func (s *GDNState) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releaseLocked()
}

// GDNLiveBufferCount returns the number of currently owned persistent buffers.
func GDNLiveBufferCount() int { return int(C.mg_gdn_live_buffers()) }

// gdnCurrentAllocatedBytes is a native readback for the Darwin lifetime witness.
// Keep it package-private: allocation policy remains owned by Metal, while the test
// needs to distinguish completed transient resources from persistent GDN state.
func gdnCurrentAllocatedBytes() uint64 { return uint64(C.mg_gdn_current_allocated_size()) }
