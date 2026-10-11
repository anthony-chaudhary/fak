package model

import (
	"errors"
	"testing"
)

// fak-test:runtime medium est=3s lane=default
func TestV41DeviceOnlyDeclinedSeamsRefuseTyped(t *testing.T) {
	newSession := func(t *testing.T) (*Model, *Session) {
		m := v41FullStepPatchedNorms(t)
		s := v41DenseTestSession(t, m, newV41MHCTestBackend(true))
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		return m, s
	}
	refused := func(t *testing.T, s *Session, what string, err error) {
		t.Helper()
		if !errors.Is(err, errV41DeviceOnlyHostOp) || errors.Is(err, ErrV41ForwardStage) || !s.BackendSessionClosed() || !s.HostFallbackObserved() {
			t.Fatalf("%s: err=%v closed=%t observed=%t, want the typed device-only refusal", what, err, s.BackendSessionClosed(), s.HostFallbackObserved())
		}
	}
	declineDense := func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionDeclined, nil
	}
	t.Run("unbound and declined wrappers", func(t *testing.T) {
		for name, call := range map[string]func(*Session) error{
			"dense-declined": func(s *Session) error {
				_, o, err := v41DeviceOnlyDense(s, declineDense)(0, "attn.wq_a.weight", nil, 1, 1, 1)
				if o != v41ProjectionError {
					t.Fatalf("dense outcome %v", o)
				}
				return err
			},
			"grouped-unbound": func(s *Session) error {
				_, o, err := v41DeviceOnlyGrouped(s, nil)(0, nil, 1, 1, 1, 1, 1)
				if o != v41ProjectionError {
					t.Fatalf("grouped outcome %v", o)
				}
				return err
			},
			"engram-unbound": func(s *Session) error { _, _, err := v41DeviceOnlyEngram(s, nil)(1, nil, 1, 1); return err },
			"mhc-unbound": func(s *Session) error {
				_, _, err := v41DeviceOnlyMHCProjection(s, nil)(0, nil, 1, 1e-6, true, false)
				return err
			},
			"activation-unbound": func(s *Session) error {
				_, _, err := v41DeviceOnlySharedActivation(s, nil)(0, nil, nil, 0)
				return err
			},
			"gate-up-declined": func(s *Session) error {
				_, o, err := v41DeviceOnlyGateUp(s, func(int, string, []float32) ([]float32, v41ExpertGateUpOutcome, error) {
					return nil, v41GateUpDeclined, nil
				})(0, "ffn.experts.0", nil)
				if o != v41GateUpError {
					t.Fatalf("gate/up outcome %v", o)
				}
				return err
			},
			"down-unbound": func(s *Session) error {
				_, o, err := v41DeviceOnlyDown(s, nil)(0, "ffn.experts.0", nil)
				if o != v41DownError {
					t.Fatalf("down outcome %v", o)
				}
				return err
			},
		} {
			t.Run(name, func(t *testing.T) {
				_, s := newSession(t)
				refused(t, s, name, call(s))
			})
		}
	})
	t.Run("handled passes through", func(t *testing.T) {
		_, s := newSession(t)
		y, o, err := v41DeviceOnlyDense(s, func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			return []float32{7}, v41ProjectionHandled, nil
		})(0, "attn.wq_a.weight", nil, 1, 1, 1)
		if err != nil || o != v41ProjectionHandled || len(y) != 1 || y[0] != 7 || s.BackendSessionClosed() {
			t.Fatalf("handled call changed: y=%v o=%v err=%v", y, o, err)
		}
	})
	t.Run("bound state refuses through the forward consumers", func(t *testing.T) {
		m, s := newSession(t)
		st := s.v41State()
		for name, bound := range map[string]bool{
			"dense": st.denseProjection != nil, "grouped": st.groupedOutput != nil, "engram": st.engramProjection != nil,
			"mhc": st.mhcProjection != nil && st.mhcFFNProjection != nil, "activation": st.sharedActivation != nil,
			"gate-up": st.expertGateUp != nil, "down": st.expertDown != nil,
		} {
			if !bound {
				t.Fatalf("device-only state left %s unbound, so its host arm would run", name)
			}
		}
		st.denseProjection = v41DeviceOnlyDense(s, declineDense)
		H := m.Cfg.HiddenSize
		_, err := m.v41ProjMatRowsWithProjection(0, "attn.wq_a.weight", make([]float32, H), m.Cfg.QLoraRank, H, st.denseProjection)
		refused(t, s, "dense consumer", err)
	})
}
