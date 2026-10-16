package model

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A cold grouped layer may retain a prefix of its checked expert publications.
// Later publications keep the ordinary host route; neither the whole prompt
// nor its first-touch expert schedule is split. Bound metadata independently
// of H, including deliberately small fixture geometries.
const (
	v41RoutedOutputMaxBytes = 30 << 20
	v41RoutedOutputMaxRows  = 1536
)

type v41RetainedRoutedOutput struct {
	row    v41ExpertRow
	tensor compute.Tensor
}

// This owner is private to one fresh grouped output table and one execution
// scope. Published tensors transfer here only after their original deferred
// read, status, width and BF16 checks, and their independent host copies.
// The host rows remain authoritative for validation and unretained slots.
type v41RoutedOutputOwner struct {
	session *Session
	model   *Model
	backend compute.Backend
	scope   *backendExecutionScope
	layer   int
	width   int
	inner   int
	limit   float32
	picks   [][]routePick // borrowed, immutable until this grouped call returns
	capRows int
	rows    []v41RetainedRoutedOutput
	index   map[v41ExpertRow]int
	buffers map[compute.Buffer]bool
	closed  bool
}

// The activation adapter must additionally prove canonical routed-sum
// authority. This constructor alone does not authorize bypassing a callback.
func newV41RoutedOutputOwner(s *Session, layer, width int, picks [][]routePick) *v41RoutedOutputOwner {
	if s == nil || s.M == nil || !s.v41DeviceSumAvailable() || !backendHasComparableIdentity(s.Backend) ||
		s.halClosed || s.halFailure != nil || s.v41RoutedOutputs != nil || width <= 0 || width > v41RoutedOutputMaxBytes/4 ||
		width != s.M.Cfg.HiddenSize || s.M.Cfg.MoEIntermediateSize <= 0 || !finite32(float32(s.M.Cfg.SwigluLimit)) ||
		layer < 0 || layer >= s.M.Cfg.NumLayers || v41PrefillExpertPanelOptedIn() {
		return nil
	}
	scope := s.backendExecutionScope
	if scope == nil || !scope.live || scope.owner == nil || backendExecutionOwner(s.Backend) != scope.owner {
		return nil
	}
	if full, err := v41ForwardGeometry(s.M.Cfg); err != nil || !full {
		return nil
	}
	// Divide before multiplying; neither prompt length nor total picks can
	// overflow this saturating count. A large prompt retains only a prefix.
	maxRows := min(v41RoutedOutputMaxRows, v41RoutedOutputMaxBytes/4/width)
	count := 0
	for _, row := range picks {
		count += min(len(row), maxRows-count)
		if count == maxRows {
			break
		}
	}
	if count == 0 {
		return nil
	}
	o := &v41RoutedOutputOwner{
		session: s, model: s.M, backend: s.Backend, scope: scope,
		layer: layer, width: width, inner: s.M.Cfg.MoEIntermediateSize, limit: float32(s.M.Cfg.SwigluLimit),
		picks: picks, capRows: count, rows: make([]v41RetainedRoutedOutput, 0, count),
		index: make(map[v41ExpertRow]int, count), buffers: make(map[compute.Buffer]bool, count),
	}
	s.v41RoutedOutputs = o
	return o
}

func (o *v41RoutedOutputOwner) live() bool {
	return o != nil && !o.closed && o.session.v41RoutedOutputs == o && o.session.M == o.model && o.session.Backend == o.backend &&
		!o.session.halClosed && o.session.halFailure == nil && o.scope.live &&
		o.session.backendExecutionScope == o.scope && o.model.Cfg.HiddenSize == o.width &&
		o.model.Cfg.MoEIntermediateSize == o.inner && float32(o.model.Cfg.SwigluLimit) == o.limit
}

func (o *v41RoutedOutputOwner) owns(buf compute.Buffer) bool {
	return o != nil && o.buffers[buf]
}

// The concrete handoff cannot run user code. Its destinations are the existing
// group's original (token, slot) positions. A full owner leaves later cohorts
// untouched, without allocating another tensor table or changing dispatch.
// Keep the owner attached even after its cap fills: later selected failures
// must still drain earlier retained rows, and may not alias their live buffers.
type v41RoutedOutputHandoff struct {
	owner *v41RoutedOutputOwner
	rows  []v41ExpertRow
}

func (o *v41RoutedOutputOwner) handoff(rows []v41ExpertRow) *v41RoutedOutputHandoff {
	if o == nil {
		return nil
	}
	return &v41RoutedOutputHandoff{owner: o, rows: rows}
}

// canCompleteUnretained authorizes only a zero-transfer continuation after the
// existing retained prefix is full. Keep this handoff attached: prior retained
// buffers remain excluded from new ownership and every failure still retires
// the original owner. A cohort crossing the cap stays on the row publication
// path. Recheck at completion; this predicate alone transfers no authority.
func (h *v41RoutedOutputHandoff) canCompleteUnretained(rows int) bool {
	return h != nil && h.owner.live() && len(h.rows) == rows &&
		h.owner.capRows > 0 && len(h.owner.rows) == h.owner.capRows
}

// completeUnretained is the explicit equivalent of adopt's n==0 branch, without
// pretending packed panels are independent retained row tensors. It neither
// changes the retained prefix nor suppresses layer/session cleanup obligations.
func (h *v41RoutedOutputHandoff) completeUnretained(rows int) error {
	if !h.canCompleteUnretained(rows) {
		return errV41ExpertResult
	}
	return nil
}

func (h *v41RoutedOutputHandoff) adopt(published []compute.Tensor) (int, error) {
	if h == nil {
		return 0, nil
	}
	o := h.owner
	if !o.live() || len(h.rows) != len(published) {
		return 0, errV41ExpertResult
	}
	n := min(len(published), o.capRows-len(o.rows))
	// Validate the entire transfer before changing either owner's ledger.
	for i, t := range published[:n] {
		row := h.rows[i]
		if row.Token < 0 || row.Token >= len(o.picks) || row.Slot < 0 || row.Slot >= len(o.picks[row.Token]) ||
			t.Buf() == nil || t.Dtype != compute.F32 || t.Layout != compute.RowMajor || t.Quant != nil ||
			len(t.Shape) != 1 || t.Shape[0] != o.width || o.buffers[t.Buf()] {
			return 0, errV41ExpertResult
		}
		if _, exists := o.index[row]; exists {
			return 0, errV41ExpertResult
		}
		for j := 0; j < i; j++ {
			if h.rows[j] == row || published[j].Buf() == t.Buf() {
				return 0, errV41ExpertResult
			}
		}
	}
	for i, t := range published[:n] {
		row := h.rows[i]
		o.index[row], o.buffers[t.Buf()] = len(o.rows), true
		o.rows = append(o.rows, v41RetainedRoutedOutput{row, t})
	}
	return n, nil
}

// release attempts every matching Free once, in first-touch publication order.
// Remove ownership before Free so a panic cannot cause a double release.
// Published Free timing intentionally moves from cohort return to token sum
// completion (or layer abort); no tensor remains live into the next layer.
type v41RoutedOutputCleanup struct {
	value   any
	present bool
}

func (o *v41RoutedOutputOwner) release(token int, all bool) (failure v41RoutedOutputCleanup) {
	if o == nil || o.closed {
		return failure
	}
	if all {
		o.closed = true
		if o.session.v41RoutedOutputs == o {
			o.session.v41RoutedOutputs = nil
		}
	}
	completed := false
	defer func() {
		if all {
			o.rows, o.index, o.buffers, o.picks = nil, nil, nil, nil
		}
		if !completed {
			// Goexit cannot be recovered. The scheduled releases still drain,
			// and an incomplete cleanup can never leave a reusable session.
			o.retire(errors.New("model: incomplete V4.1 routed-output cleanup"))
		}
	}()
	func() {
		// Schedule every release before invoking one. Goexit in a backend Free
		// still runs the remaining defers; reverse registration keeps the
		// original first-touch publication order.
		for i := len(o.rows) - 1; i >= 0; i-- {
			entry := &o.rows[i]
			if entry.tensor.Buf() == nil || !all && entry.row.Token != token {
				continue
			}
			t := entry.tensor
			entry.tensor = compute.Tensor{}
			delete(o.index, entry.row)
			delete(o.buffers, t.Buf())
			defer func(t compute.Tensor) {
				freed := false
				defer func() {
					p := recover()
					if !freed && !failure.present {
						failure = v41RoutedOutputCleanup{value: p, present: true}
					}
				}()
				o.backend.Free(t)
				freed = true
			}(t)
		}
	}()
	completed = true
	return failure
}

func (o *v41RoutedOutputOwner) retire(cause error) error {
	var forward *BackendForwardOperationError
	if !errors.As(cause, &forward) {
		cause = &BackendForwardOperationError{Backend: o.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-output", Layer: o.layer, Stage: "cleanup", Cause: cause}
	}
	if o.session.halFailure == nil {
		o.session.halFailure = cause
	}
	if !o.session.halClosed {
		func() { defer func() { o.session.halClosed = true; _ = recover() }(); o.session.closeBackendOwned() }()
	}
	return cause
}

// finish is deferred by the fresh grouped caller. Existing compute/validation
// errors and panics stay primary; any cleanup fault still retires the session.
// A cleanup-only known backend fault is returned typed; an unknown panic stays
// a panic. Every owned row has been attempted before either outcome escapes.
func (o *v41RoutedOutputOwner) finish(err *error, completed *bool) {
	primary := recover()
	cleanup := o.release(0, true)
	if !*completed {
		// A later cohort may terminate without a normal return even when all
		// retained frees succeed. Retire before that incomplete body escapes.
		cause := *err
		if cause == nil {
			if primary != nil {
				var known bool
				cause, known = v41RoutedCohortPanicCause(primary)
				if !known {
					cause = v41RoutedCohortUnknownPanic(primary)
				}
			} else {
				cause = errors.New("model: incomplete V4.1 grouped routed contraction")
			}
		}
		cause = o.retire(cause)
		if primary != nil {
			panic(primary)
		}
		// Goexit continues unwinding. Legacy panic(nil) instead returns this
		// typed fatal error, never an apparently successful grouped result.
		*err = cause
		return
	}
	if cleanup.present {
		cause, known := v41RoutedCohortPanicCause(cleanup.value)
		if !known {
			cause = v41RoutedCohortUnknownPanic(cleanup.value)
		}
		cause = o.retire(cause)
		if *err == nil {
			if !known {
				panic(cleanup.value)
			}
			*err = cause
		}
	}
}

// sum keeps all ordinary host checks and the exact stable ascending-expert
// F32 addition order. Only Upload of retained slots disappears. Both the zero
// accumulator upload and final checked Read/copy remain.
func (o *v41RoutedOutputOwner) sum(token int, picks []routePick, outputs [][]float32) (result []float32, cause error) {
	s, width, layer := o.session, o.width, o.layer
	invalid := v41ExpertOperationErr(layer, "sum", errV41ExpertResult)
	operationComplete, callbackComplete := false, false
	failureStage := "payload"
	incomplete := func() error {
		return &BackendForwardOperationError{Backend: o.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-sum", Layer: layer, Stage: failureStage, Cause: errors.New("model: incomplete V4.1 routed sum")}
	}
	defer func() {
		primary := recover()
		if !operationComplete {
			if cause == nil {
				cause = incomplete()
			}
			result, cause = nil, o.retire(cause)
		}
		if primary != nil {
			panic(primary)
		}
		// An active Goexit continues; a legacy nil panic cannot publish success.
	}()
	var unknownCleanup *v41RoutedOutputCleanup
	result, cause = v41SelectedDeviceCall(s, layer, "v41-routed-sum", func(stage *string) (result []float32, cause error) {
		completed := false
		// Drain before v41SelectedDeviceCall retires the backend. Cleanup does
		// not replace an earlier payload/add/read failure or panic.
		defer func() {
			failureStage = *stage
			cleanup := o.release(token, cause != nil || !completed)
			if cleanup.present && completed && cause == nil {
				// Drain the remaining layer before retirement. Return a real
				// failure through the legacy selected-call adapter, then restore
				// an unknown panic outside it (including legacy panic(nil)).
				o.release(0, true)
				var known bool
				cause, known = v41RoutedCohortPanicCause(cleanup.value)
				if !known {
					cause = v41RoutedCohortUnknownPanic(cleanup.value)
					unknownCleanup = &cleanup
				}
				result = nil
			}
		}()
		result, cause = func() (result []float32, cause error) {
			if !o.live() || len(picks) != len(outputs) || width <= 0 {
				return nil, invalid
			}
			for _, row := range outputs {
				if len(row) != width {
					return nil, invalid
				}
				for _, v := range row {
					if !finite32(v) {
						return nil, invalid
					}
				}
			}
			*stage = "accumulator upload"
			acc := s.uploadHostF32([]int{width}, make([]float32, width), compute.MemoryActivation, "V4.1 routed-sum accumulator")
			defer o.backend.Free(acc)
			rows := make([]compute.Tensor, len(outputs))
			*stage = "expert upload"
			for slot, row := range outputs {
				if index, found := o.index[v41ExpertRow{token, slot}]; found {
					rows[slot] = o.rows[index].tensor
				} else {
					rows[slot] = s.uploadHostF32([]int{width}, row, compute.MemoryActivation, "V4.1 routed-sum expert output")
					defer o.backend.Free(rows[slot])
				}
			}
			*stage = "routed-sum"
			v41DeviceAccumulate(o.backend, acc, rows, v41RoutedSumOrder(picks))
			*stage = "readback"
			result, cause = s.v41DeviceSumReadback(acc, width, invalid)
			return result, cause
		}()
		completed, callbackComplete = true, true
		return result, cause
	})
	if !callbackComplete && cause == nil {
		result, cause = nil, o.retire(incomplete())
	}
	if unknownCleanup != nil {
		panic(unknownCleanup.value)
	}
	operationComplete = true
	return result, cause
}

// All selected operations, including ordinary scalar fallback in a later
// group, retire through closeBackendOwned. Drain this active layer before that
// teardown can recycle device storage. No per-session result is cached here.
func (s *Session) drainV41RoutedOutputs() v41RoutedOutputCleanup {
	o := s.v41RoutedOutputs
	if o == nil {
		return v41RoutedOutputCleanup{}
	}
	cleanup := o.release(0, true)
	if !cleanup.present || s.halFailure != nil {
		return v41RoutedOutputCleanup{} // an existing selected primary remains fatal and authoritative
	}
	cause, known := v41RoutedCohortPanicCause(cleanup.value)
	if !known {
		cause = v41RoutedCohortUnknownPanic(cleanup.value)
	}
	closed := &BackendForwardOperationError{Backend: o.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-routed-output", Layer: o.layer, Stage: "cleanup", Cause: cause}
	s.halFailure = closed
	if known {
		cleanup.value = closed
	}
	return cleanup
}
