package compute

import (
	"errors"
	"testing"
)

// TestGraphHostCallbackCaptureGate is the table-driven core of the host-callback capture
// contract: which (warmed, body, invoke) shapes are admitted, which are refused with which
// typed Kind, and whether the body ran.
func TestGraphHostCallbackCaptureGate(t *testing.T) {
	sentinel := errors.New("staging read failed")

	cases := []struct {
		name       string
		warmed     bool
		body       GraphHostCallbackFunc
		captureErr HostCallbackErrorKind // "" means capture must succeed
		invokeErr  HostCallbackErrorKind // "" means invoke must succeed; only checked if captured
		wantRuns   int
	}{
		{
			name:       "unwarmed_refused",
			warmed:     false,
			body:       func() error { return nil },
			captureErr: HostCallbackNotWarmed,
		},
		{
			name:       "nil_body_refused",
			warmed:     true,
			body:       nil,
			captureErr: HostCallbackNilBody,
		},
		{
			name:     "warmed_ok",
			warmed:   true,
			body:     func() error { return nil },
			wantRuns: 1,
		},
		{
			name:      "body_error_propagates",
			warmed:    true,
			body:      func() error { return sentinel },
			invokeErr: HostCallbackBodyFailed,
			wantRuns:  1,
		},
		{
			name:      "body_panic_recovered",
			warmed:    true,
			body:      func() error { panic("boom") },
			invokeErr: HostCallbackPanic,
			wantRuns:  1,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runs := 0
			body := tc.body
			if body != nil {
				inner := body
				body = func() error {
					runs++
					return inner()
				}
			}

			node := NewGraphHostCallbackNode(tc.name, tc.name+"-id", body)
			if tc.warmed {
				node.WarmStaging()
			}
			sess := NewSimulatedGraphCapture()

			err := sess.CaptureNode(node)
			if tc.captureErr != "" {
				var hce *HostCallbackError
				if !errors.As(err, &hce) {
					t.Fatalf("CaptureNode error = %T (%v), want *HostCallbackError", err, err)
				}
				if hce.Kind != tc.captureErr {
					t.Fatalf("CaptureNode Kind = %q, want %q", hce.Kind, tc.captureErr)
				}
				if hce.Node != tc.name {
					t.Fatalf("CaptureNode Node = %q, want %q", hce.Node, tc.name)
				}
				// A refused node must not be captured: invoking it is itself refused.
				var invErr *HostCallbackError
				if !errors.As(sess.Invoke(node), &invErr) || invErr.Kind != HostCallbackNotCaptured {
					t.Fatalf("Invoke on refused node = %v, want Kind=%q", sess.Invoke(node), HostCallbackNotCaptured)
				}
				if runs != 0 {
					t.Fatalf("body ran %d times after refused capture, want 0", runs)
				}
				return
			}

			if err != nil {
				t.Fatalf("CaptureNode: %v", err)
			}
			invErr := sess.Invoke(node)
			if tc.invokeErr != "" {
				var hce *HostCallbackError
				if !errors.As(invErr, &hce) {
					t.Fatalf("Invoke error = %T (%v), want *HostCallbackError", invErr, invErr)
				}
				if hce.Kind != tc.invokeErr {
					t.Fatalf("Invoke Kind = %q, want %q", hce.Kind, tc.invokeErr)
				}
				if tc.invokeErr == HostCallbackBodyFailed && !errors.Is(invErr, sentinel) {
					t.Fatal("errors.Is(err, sentinel) = false; wrapped chain does not reach the cause")
				}
			} else if invErr != nil {
				t.Fatalf("Invoke: %v", invErr)
			}
			if runs != tc.wantRuns {
				t.Fatalf("body ran %d times, want %d", runs, tc.wantRuns)
			}
			if node.Invokes() != tc.wantRuns {
				t.Fatalf("Invokes = %d, want %d", node.Invokes(), tc.wantRuns)
			}
		})
	}
}

// TestGraphHostCallbackStagingWarmedLifecycle asserts WarmStaging is idempotent, chainable,
// and observable through StagingWarmed.
func TestGraphHostCallbackStagingWarmedLifecycle(t *testing.T) {
	node := NewGraphHostCallbackNode("lifecycle", "lc-1", func() error { return nil })
	if node.StagingWarmed() {
		t.Fatal("fresh node reports staging warmed")
	}
	if node.WarmStaging() != node {
		t.Fatal("WarmStaging did not return the receiver for chaining")
	}
	node.WarmStaging()
	if !node.StagingWarmed() {
		t.Fatal("StagingWarmed false after WarmStaging")
	}
}

// TestGraphHostCallbackNoDeviceCalls asserts a capture-safe body leaves the simulated
// session's device-call counter at zero across capture and invoke.
func TestGraphHostCallbackNoDeviceCalls(t *testing.T) {
	node := NewGraphHostCallbackNode("pure", "pure-1", func() error { return nil })
	node.WarmStaging()
	sess := NewSimulatedGraphCapture()

	// Install a hook a misbehaving body could call; a pure-Go body never does.
	prev := sess.SetDeviceCallHook(func() {})
	defer sess.SetDeviceCallHook(prev)

	if err := sess.CaptureNode(node); err != nil {
		t.Fatalf("CaptureNode: %v", err)
	}
	if err := sess.Invoke(node); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if got := sess.DeviceCalls(); got != 0 {
		t.Fatalf("DeviceCalls = %d, want 0 for a capture-safe body", got)
	}
}

// TestGraphHostCallbackNilBodyNoOpNoInvoke asserts a nil-body node is refused at capture,
// so it can never be invoked through the session.
func TestGraphHostCallbackNilBodyNoOpNoInvoke(t *testing.T) {
	node := NewGraphHostCallbackNode("noop", "noop-1", nil)
	node.WarmStaging()
	sess := NewSimulatedGraphCapture()
	if err := sess.CaptureNode(node); err == nil {
		t.Fatal("CaptureNode with nil body returned nil, want refusal")
	}
	if err := sess.Invoke(node); err == nil {
		t.Fatal("Invoke on un-captured nil-body node returned nil, want refusal")
	}
}

// TestGraphHostCallbackDetectsDeviceCallInBody proves the device-call route is live: a
// body that reaches for a device call through the session hook is caught by Invoke as a
// capture-safety violation, while a pure-Go body leaves the counter at zero.
func TestGraphHostCallbackDetectsDeviceCallInBody(t *testing.T) {
	sess := NewSimulatedGraphCapture()

	node := NewGraphHostCallbackNode("bad-body", "bad-1", func() error {
		sess.CallDevice()
		return nil
	})
	node.WarmStaging()

	if err := sess.CaptureNode(node); err != nil {
		t.Fatalf("CaptureNode: %v", err)
	}
	err := sess.Invoke(node)
	var hce *HostCallbackError
	if !errors.As(err, &hce) {
		t.Fatalf("Invoke error = %T (%v), want *HostCallbackError", err, err)
	}
	if hce.Kind != HostCallbackDeviceCallInBody {
		t.Fatalf("Invoke Kind = %q, want %q", hce.Kind, HostCallbackDeviceCallInBody)
	}
	if got := sess.DeviceCalls(); got != 1 {
		t.Fatalf("DeviceCalls = %d, want 1 after an in-body device call", got)
	}
}

// TestGraphHostCallbackNilReceiverSafe pins the nil-receiver safety contract on the typed
// error so callers may defensively format an error they only hold a pointer to.
func TestGraphHostCallbackNilReceiverSafe(t *testing.T) {
	var hce *HostCallbackError
	if got := hce.Error(); got == "" {
		t.Fatal("nil *HostCallbackError.Error() returned empty string")
	}
	if err := hce.Unwrap(); err != nil {
		t.Fatalf("nil *HostCallbackError.Unwrap() = %v, want nil", err)
	}
}
