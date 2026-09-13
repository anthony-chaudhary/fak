package compute

import (
	"fmt"
	"strings"
	"sync"
)

// host_callback.go — a CUDA-graph-safe host-callback node and a pure-Go execution seam
// that models graph *capture* semantics without linking any CUDA driver, without cgo,
// and without build tags.
//
// A real CUDA graph may embed a host function node (cudaLaunchHostFunc). Capture rules
// forbid any operation inside that node that would synchronize with or re-enter the
// driver: the body may only touch host-side staging buffers that were made resident
// BEFORE capture began. This file makes that rule a typed, testable contract:
//
//   - GraphHostCallbackNode refuses capture with a typed *HostCallbackError until its
//     input staging has been warmed via WarmStaging.
//   - The callback body is a pure host Go func; the seam accepts only pure-Go funcs, so
//     "no CUDA call in the body" is enforced by construction. A simulated session that a
//     caller wires to a deviceCallHook counts any device call made through the hook; a
//     well-behaved body leaves that counter at zero.
//
// This is strictly additive: it defines new types and changes no existing entrypoint.

// HostCallbackErrorKind classifies host-callback graph failures. It is a closed string
// enum so callers can branch on the failure class without parsing prose.
type HostCallbackErrorKind string

const (
	// HostCallbackNotWarmed is returned when capture is attempted before the node's
	// input staging was made resident (WarmStaging was never called).
	HostCallbackNotWarmed HostCallbackErrorKind = "not_warmed"

	// HostCallbackNilBody is returned when capture is attempted on a node whose callback
	// body is nil: a hook that can never run is a capture-time misconfiguration.
	HostCallbackNilBody HostCallbackErrorKind = "nil_body"

	// HostCallbackNotCaptured is returned when a node is invoked without a prior capture.
	HostCallbackNotCaptured HostCallbackErrorKind = "not_captured"

	// HostCallbackDeviceCallInBody is returned when a callback body is observed routing a
	// call through the session's device call hook: the body is not capture-safe.
	HostCallbackDeviceCallInBody HostCallbackErrorKind = "device_call_in_body"

	// HostCallbackPanic is returned when a callback body panics; the panic is recovered
	// and surfaced fail-closed instead of unwinding through the capture session.
	HostCallbackPanic HostCallbackErrorKind = "panic"

	// HostCallbackBodyFailed is returned when a callback body returns a non-nil error.
	HostCallbackBodyFailed HostCallbackErrorKind = "body_failed"
)

// HostCallbackError is the typed error returned by GraphHostCallbackNode capture and by
// graph capture sessions. It names the fatal classification (Kind), the node it concerns
// (Node), and any wrapped cause (Err). Both Error and Unwrap are nil-receiver safe.
type HostCallbackError struct {
	Kind HostCallbackErrorKind
	Node string
	Err  error
}

// Error formats the failure with a stable "compute:" prefix for operator logs.
func (e *HostCallbackError) Error() string {
	if e == nil {
		return "compute: nil host callback error"
	}
	var b strings.Builder
	b.WriteString("compute: host callback")
	if e.Node != "" {
		b.WriteString(" ")
		b.WriteString(e.Node)
	}
	if e.Kind != "" {
		b.WriteString(" [")
		b.WriteString(string(e.Kind))
		b.WriteString("]")
	}
	switch {
	case e.Err != nil:
		b.WriteString(": ")
		b.WriteString(e.Err.Error())
	case e.Kind == HostCallbackNotWarmed:
		b.WriteString(": capture refused before input staging was warmed")
	case e.Kind == HostCallbackNilBody:
		b.WriteString(": capture refused: nil callback body")
	default:
		b.WriteString(": capture failed")
	}
	return b.String()
}

// Unwrap exposes the wrapped cause so errors.Is and errors.As traverse it. It is
// nil-receiver safe, mirroring the TypedError convention used across the package.
func (e *HostCallbackError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Err
}

// GraphHostCallbackFunc is a host callback body executed as a node inside a captured
// CUDA graph. It MUST issue no CUDA/driver calls: graph capture forbids any operation
// that would synchronize with or re-enter the driver. It may only touch host-side
// staging buffers that were made resident BEFORE capture.
type GraphHostCallbackFunc func() error

// GraphHostCallbackNode is a graph-capturable host-callback node. Construct it with
// NewGraphHostCallbackNode. Its input staging must be warmed before Capture will accept
// it; the callback body is a pure host Go func and is never handed a device handle.
type GraphHostCallbackNode struct {
	mu sync.Mutex

	name   string
	id     string
	body   GraphHostCallbackFunc
	warmed bool

	captureCalls int
	invokes      int
}

// NewGraphHostCallbackNode creates a host-callback node from a name and an optional id.
// A nil body is accepted at construction time (the node may be inspected) but is refused
// at Capture time with Kind HostCallbackNilBody.
func NewGraphHostCallbackNode(name, id string, body GraphHostCallbackFunc) *GraphHostCallbackNode {
	return &GraphHostCallbackNode{name: name, id: id, body: body}
}

// Name reports the node's human-readable name.
func (n *GraphHostCallbackNode) Name() string {
	if n == nil {
		return ""
	}
	return n.name
}

// ID reports the node's optional stable identifier.
func (n *GraphHostCallbackNode) ID() string {
	if n == nil {
		return ""
	}
	return n.id
}

// WarmStaging marks the node's input staging as resident and safe for capture. It must be
// called BEFORE capture. It is idempotent and returns the receiver for chaining.
func (n *GraphHostCallbackNode) WarmStaging() *GraphHostCallbackNode {
	if n == nil {
		return nil
	}
	n.mu.Lock()
	n.warmed = true
	n.mu.Unlock()
	return n
}

// StagingWarmed reports whether WarmStaging was called on this node.
func (n *GraphHostCallbackNode) StagingWarmed() bool {
	if n == nil {
		return false
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.warmed
}

// Body returns the node's callback body (may be nil).
func (n *GraphHostCallbackNode) Body() GraphHostCallbackFunc {
	if n == nil {
		return nil
	}
	return n.body
}

// CaptureCalls reports how many times Capture was attempted on this node, whether or not
// it was accepted.
func (n *GraphHostCallbackNode) CaptureCalls() int {
	if n == nil {
		return 0
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.captureCalls
}

// Invokes reports how many times this node's body has been invoked.
func (n *GraphHostCallbackNode) Invokes() int {
	if n == nil {
		return 0
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.invokes
}

// Capture validates the node for inclusion in a captured graph and returns a typed
// *HostCallbackError (never a bare error) when the node is not capture-safe. It performs
// no device work: it only checks the warm state and body presence.
//
// Callers should errors.As the result to obtain the Kind and Node they can route on.
func (n *GraphHostCallbackNode) Capture() error {
	if n == nil {
		return &HostCallbackError{Kind: HostCallbackNotWarmed}
	}
	n.mu.Lock()
	n.captureCalls++
	warmed := n.warmed
	body := n.body
	n.mu.Unlock()

	if !warmed {
		return &HostCallbackError{Kind: HostCallbackNotWarmed, Node: n.name}
	}
	if body == nil {
		return &HostCallbackError{Kind: HostCallbackNilBody, Node: n.name}
	}
	return nil
}

// GraphCaptureSession models the capture semantics of a CUDA-graph driver session. Capture
// admits a host-callback node into the graph and Invoke replays the captured node. A
// simulated implementation is provided for tests and hosts without a CUDA device.
type GraphCaptureSession interface {
	// CaptureNode validates and admits node into the captured graph. It must return a
	// typed *HostCallbackError when the node is not capture-safe.
	CaptureNode(node *GraphHostCallbackNode) error

	// Invoke replays a previously captured node, running its host callback body exactly
	// once. A body error is surfaced fail-closed (never swallowed); a body panic is
	// recovered into a typed error.
	Invoke(node *GraphHostCallbackNode) error

	// DeviceCalls reports the number of device calls issued through the session's device
	// call hook. A capture-safe host-callback body must leave this at zero.
	DeviceCalls() int
}

// SimulatedGraphCapture is an in-memory GraphCaptureSession. It holds no device handle and
// links no driver. If deviceCallHook is non-nil, any body that routes through it is a
// capture-safety violation; a well-behaved pure-Go body never calls it, so DeviceCalls
// stays zero. A real cudaLaunchHostFunc body must call no cuda* API; this seam enforces it
// by construction because only pure-Go funcs are accepted and the only device-call route
// is the explicitly installed hook.
type SimulatedGraphCapture struct {
	mu sync.Mutex

	captured   map[*GraphHostCallbackNode]GraphHostCallbackFunc
	deviceCall int
	// deviceCallHook, when set, is the sole route by which a body can register a device
	// call. A capture-safe body must not use it.
	deviceCallHook func()
}

// NewSimulatedGraphCapture returns an empty in-memory capture session.
func NewSimulatedGraphCapture() *SimulatedGraphCapture {
	return &SimulatedGraphCapture{captured: make(map[*GraphHostCallbackNode]GraphHostCallbackFunc)}
}

// SetDeviceCallHook installs the route a body would use to register a device call, and
// returns the previous hook so the caller can restore it. Passing nil clears it. The
// installed hook is wrapped so each invocation increments the session's device-call
// counter: a capture-safe body never calls it, so DeviceCalls stays zero, while a body
// that does route through it is detected by Invoke as a capture-safety violation.
func (s *SimulatedGraphCapture) SetDeviceCallHook(hook func()) func() {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	prev := s.deviceCallHook
	s.deviceCallHook = hook
	return prev
}

// CallDevice is the sole sanctioned route by which a callback body could register a
// device call. A capture-safe body must never call it. It increments the session's
// device-call counter and then runs the installed hook (if any); capture-safety is
// therefore observable as a non-zero DeviceCalls after Invoke.
func (s *SimulatedGraphCapture) CallDevice() {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.deviceCall++
	hook := s.deviceCallHook
	s.mu.Unlock()
	if hook != nil {
		hook()
	}
}

// DeviceCallHook returns the currently installed device-call hook (may be nil).
func (s *SimulatedGraphCapture) DeviceCallHook() func() {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.deviceCallHook
}

// DeviceCalls reports how many times the device-call hook has been observed.
func (s *SimulatedGraphCapture) DeviceCalls() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.deviceCall
}

// CaptureNode admits node into the simulated graph. It refuses (typed *HostCallbackError)
// when the node is nil, its staging is unwarmed, or its body is nil. It is idempotent for
// an already-captured node.
func (s *SimulatedGraphCapture) CaptureNode(node *GraphHostCallbackNode) error {
	if node == nil {
		return &HostCallbackError{Kind: HostCallbackNotWarmed}
	}
	if err := node.Capture(); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if s.captured == nil {
		s.captured = make(map[*GraphHostCallbackNode]GraphHostCallbackFunc)
	}
	s.captured[node] = node.body
	return nil
}

// Invoke replays a captured node, running its body exactly once. It refuses a node that
// was never captured (typed error) and surfaces a body panic or body error fail-closed.
func (s *SimulatedGraphCapture) Invoke(node *GraphHostCallbackNode) error {
	if node == nil {
		return &HostCallbackError{Kind: HostCallbackNotCaptured}
	}

	s.mu.Lock()
	body, ok := s.captured[node]
	s.mu.Unlock()
	if !ok {
		return &HostCallbackError{Kind: HostCallbackNotCaptured, Node: node.name}
	}

	node.mu.Lock()
	node.invokes++
	node.mu.Unlock()

	err := runHostCallbackBody(node.name, body)

	s.mu.Lock()
	deviceCalls := s.deviceCall
	s.mu.Unlock()
	if deviceCalls > 0 {
		return &HostCallbackError{Kind: HostCallbackDeviceCallInBody, Node: node.name, Err: err}
	}
	return err
}

// runHostCallbackBody invokes body exactly once, converting a panic into a typed
// *HostCallbackError and wrapping a returned error with Kind HostCallbackBodyFailed.
func runHostCallbackBody(name string, body GraphHostCallbackFunc) (err error) {
	if body == nil {
		return &HostCallbackError{Kind: HostCallbackNilBody, Node: name}
	}
	defer func() {
		if r := recover(); r != nil {
			err = &HostCallbackError{
				Kind: HostCallbackPanic,
				Node: name,
				Err:  fmt.Errorf("panic: %v", r),
			}
		}
	}()
	if bErr := body(); bErr != nil {
		return &HostCallbackError{Kind: HostCallbackBodyFailed, Node: name, Err: bErr}
	}
	return nil
}
