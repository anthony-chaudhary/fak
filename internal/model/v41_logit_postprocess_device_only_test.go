package model

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

var v41LogitPostprocessOps = []string{"logit-scale", "logit-softcap", "logit-token-suppression"}

// The official V4.1 Flash config sets no logit scale, softcap or suppressed
// tokens, so its head postprocess is the identity and no logit op is device-only
// debt. A config that sets one is refused typed unless a device seam binds it;
// softcap and suppression have none. cpu-ref recorders only; no device claim.
// fak-test:runtime fast est=1s lane=default
func TestV41DeviceOnlyLogitPostprocessBindings(t *testing.T) {
	for _, op := range v41DeviceOnlyHostOps {
		if slices.Contains(v41LogitPostprocessOps, op.name) {
			t.Fatalf("static inventory still lists %s; the binding rows own it", op.name)
		}
	}
	var rows []v41DeviceOnlyBinding
	for _, b := range v41DeviceOnlyBindings {
		if slices.Contains(v41LogitPostprocessOps, b.op.name) {
			if b.uses == nil {
				t.Fatalf("%s binding is unconditional; the identity postprocess would refuse", b.op.name)
			}
			rows = append(rows, b)
		}
	}
	if got := len(rows); got != len(v41LogitPostprocessOps) {
		t.Fatalf("logit postprocess binding rows = %d, want %d", got, len(v41LogitPostprocessOps))
	}
	savedOps, savedBindings := v41DeviceOnlyHostOps, v41DeviceOnlyBindings
	t.Cleanup(func() { v41DeviceOnlyHostOps, v41DeviceOnlyBindings = savedOps, savedBindings })
	v41DeviceOnlyHostOps, v41DeviceOnlyBindings = nil, rows

	_, flash := readDeepSeekV41Config(t)
	if flash.LogitScale != 0 || flash.LogitSoftcap != 0 || len(flash.SuppressTokens) != 0 {
		t.Fatalf("official config sets logit postprocess scale=%g softcap=%g suppress=%v", flash.LogitScale, flash.LogitSoftcap, flash.SuppressTokens)
	}
	for _, tc := range []struct {
		name   string
		edit   func(*Config)
		seam   bool
		refuse []string
	}{
		{name: "flash", edit: func(*Config) {}},
		{name: "flash-with-seam", edit: func(*Config) {}, seam: true},
		{name: "scale-one", edit: func(c *Config) { c.LogitScale = 1 }},
		{name: "scale-rounds-to-one", edit: func(c *Config) { c.LogitScale = math.Nextafter(1, 2) }},
		{name: "scale-unbound", edit: func(c *Config) { c.LogitScale = 0.0625 }, refuse: []string{"logit-scale"}},
		{name: "scale-bound", edit: func(c *Config) { c.LogitScale = 0.0625 }, seam: true},
		{name: "scale-overflows-binary32", edit: func(c *Config) { c.LogitScale = 1e300 }, seam: true, refuse: []string{"logit-scale"}},
		{name: "scale-nan", edit: func(c *Config) { c.LogitScale = math.NaN() }, seam: true, refuse: []string{"logit-scale"}},
		{name: "softcap", edit: func(c *Config) { c.LogitSoftcap = 30 }, seam: true, refuse: []string{"logit-softcap"}},
		{name: "softcap-nan", edit: func(c *Config) { c.LogitSoftcap = math.NaN() }, refuse: []string{"logit-softcap"}},
		{name: "softcap-underflows-binary32", edit: func(c *Config) { c.LogitSoftcap = 1e-50 }},
		{name: "softcap-negative", edit: func(c *Config) { c.LogitSoftcap = -1 }},
		{name: "suppress", edit: func(c *Config) { c.SuppressTokens = []int{1} }, seam: true, refuse: []string{"logit-token-suppression"}},
		{name: "all-bound-scale", edit: func(c *Config) { c.LogitScale, c.LogitSoftcap, c.SuppressTokens = 0.0625, 0.75, []int{1} }, seam: true, refuse: []string{"logit-softcap", "logit-token-suppression"}},
		{name: "all-unbound", edit: func(c *Config) { c.LogitScale, c.LogitSoftcap, c.SuppressTokens = 0.0625, 0.75, []int{1} }, refuse: []string{"logit-scale", "logit-softcap", "logit-token-suppression"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := flash
			tc.edit(&cfg)
			var be compute.Backend = newV41DenseTestBackend()
			if tc.seam {
				be = newV41LogitScaleRecorder()
			}
			s := &Session{M: &Model{Cfg: cfg}, Backend: be}
			if tc.seam != (s.v41LogitScaleFunc() != nil) {
				t.Fatalf("fixture seam bound=%t want %t", s.v41LogitScaleFunc() != nil, tc.seam)
			}
			err := s.v41DeviceOnlyAdmissionRefusal()
			if len(tc.refuse) == 0 {
				if err != nil {
					t.Fatalf("admission refused %v; the postprocess needs no host arm", err)
				}
				return
			}
			var typed *v41DeviceOnlyHostOpError
			if !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &typed) {
				t.Fatalf("admission = %v, want typed host-op refusal", err)
			}
			if got := v41HostOpNames(typed.ops); !slices.Equal(got, tc.refuse) {
				t.Fatalf("refused ops = %v, want %v", got, tc.refuse)
			}
		})
	}
}

// Under device-only no host arm of the head postprocess runs: each used op
// without a device seam refuses before any device work and leaves the row
// untouched; the identity returns the projected row itself.
// fak-test:runtime fast est=1s lane=default
func TestV41ScaleHeadLogitsDeviceOnlyRefusesHostArms(t *testing.T) {
	type outcome struct {
		out     []float32
		err     error
		refused []string
		calls   int
	}
	refusal := errors.New("refused")
	run := func(cfg Config, row []float32, seam, deviceOnly bool) outcome {
		var o outcome
		var scale v41LogitScaleFunc
		if seam {
			scale = func(values []float32, factor float32) ([]float32, error) {
				o.calls++
				out := make([]float32, len(values))
				for i, v := range values {
					out[i] = float32(v * factor)
				}
				return out, nil
			}
		}
		var refuse v41LogitHostRefusal
		if deviceOnly {
			refuse = func(op string) error { o.refused = append(o.refused, op); return refusal }
		}
		o.out, o.err = v41ScaleHeadLogits(row, cfg, scale, refuse)
		return o
	}
	finite := func() []float32 { return []float32{1, -2, 0.75, float32(math.Copysign(0, -1))} }
	host := func(cfg Config, row []float32) []float32 {
		want := append([]float32(nil), row...)
		logitScaleInPlace(want, cfg)
		return want
	}

	t.Run("identity-returns-row", func(t *testing.T) {
		for _, factor := range []float64{0, 1, math.Nextafter(1, 2)} {
			row := append(finite(), float32(math.Inf(1)))
			want := append([]float32(nil), row...)
			o := run(Config{LogitScale: factor}, row, false, true)
			if o.err != nil || len(o.refused) != 0 || o.calls != 0 || &o.out[0] != &row[0] {
				t.Fatalf("factor %g: err=%v refused=%v calls=%d; identity must return the row", factor, o.err, o.refused, o.calls)
			}
			v41LogitScaleCheckBits(t, o.out, want)
		}
	})
	for _, tc := range []struct {
		name string
		cfg  Config
		row  []float32
		seam bool
		op   string
	}{
		{"scale-unbound", Config{LogitScale: 0.0625}, finite(), false, "logit-scale"},
		{"scale-nonfinite-row", Config{LogitScale: 0.0625}, append(finite(), float32(math.Inf(1))), true, "logit-scale"},
		{"scale-nonfinite-factor", Config{LogitScale: 1e300}, finite(), true, "logit-scale"},
		{"softcap-before-selection", Config{LogitScale: 0.0625, LogitSoftcap: 0.75}, finite(), true, "logit-softcap"},
		{"softcap-identity-scale", Config{LogitSoftcap: 30}, finite(), false, "logit-softcap"},
		{"suppress", Config{SuppressTokens: []int{1}}, finite(), true, "logit-token-suppression"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := append([]float32(nil), tc.row...)
			o := run(tc.cfg, tc.row, tc.seam, true)
			if o.err != refusal || o.out != nil || !slices.Equal(o.refused, []string{tc.op}) || o.calls != 0 {
				t.Fatalf("err=%v out=%v refused=%v calls=%d, want one %s refusal before device work", o.err, o.out, o.refused, o.calls, tc.op)
			}
			v41LogitScaleCheckBits(t, tc.row, want)
		})
	}
	t.Run("bound-scale-dispatches", func(t *testing.T) {
		cfg := Config{LogitScale: 0.0625}
		row := finite()
		o := run(cfg, row, true, true)
		if o.err != nil || len(o.refused) != 0 || o.calls != 1 {
			t.Fatalf("err=%v refused=%v calls=%d, want one device multiply", o.err, o.refused, o.calls)
		}
		v41LogitScaleCheckBits(t, o.out, host(cfg, finite()))
	})
	t.Run("host-policy-keeps-host-arms", func(t *testing.T) {
		cfg := Config{LogitScale: 0.0625, LogitSoftcap: 0.75, SuppressTokens: []int{1}}
		o := run(cfg, finite(), false, false)
		if o.err != nil || o.calls != 0 {
			t.Fatalf("err=%v calls=%d", o.err, o.calls)
		}
		v41LogitScaleCheckBits(t, o.out, host(cfg, finite()))
	})
}

// The head consumer carries the owning session's live policy: under
// device-only it refuses typed and retires the session; a model-only state or
// a host session keeps the host postprocess. As in the other device-only
// consumer tests, cpu recorders cannot run a whole device-only forward (the
// embedding gather refuses first), so this drives the head with the session's
// own bound state. The physical whole-forward run is the Vulkan witness.
// fak-test:runtime medium est=2s lane=default
func TestV41DeviceOnlyLogitPostprocessHeadRefusesTyped(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("software witness requires cpu-ref")
	}
	if (&v41ForwardState{}).logitHostRefusal() != nil {
		t.Fatal("model-only state refuses the host postprocess")
	}
	head := func(m *Model, s *Session) ([]float32, error) {
		st := s.v41State()
		x := make([]float32, m.Cfg.HiddenSize)
		for i := range x {
			x[i] = float32(i%5-2) * 0.25
		}
		return m.v41HeadWithLogitScale(x, st.denseProjection(), st.finalNorm, st.bf16Boundary, st.logitScale, st.logitHostRefusal())
	}
	session := func(t *testing.T, m *Model, b compute.Backend, deviceOnly bool) *Session {
		s := v41DenseTestSession(t, m, b)
		if deviceOnly {
			s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		}
		return s
	}
	refused := func(t *testing.T, s *Session, err error, op string) {
		t.Helper()
		var closed *BackendForwardOperationError
		var host *v41DeviceOnlyHostOpError
		if !errors.As(err, &closed) || closed.Path != "device-only" || !errors.Is(err, errV41DeviceOnlyHostOp) || !errors.As(err, &host) || errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("err=%v, want the typed device-only refusal", err)
		}
		if !slices.Equal(v41HostOpNames(host.ops), []string{op}) || !s.BackendSessionClosed() || !s.HostFallbackObserved() {
			t.Fatalf("refusal %v did not name %s or retire the session", v41HostOpNames(host.ops), op)
		}
	}
	t.Run("unbound-scale", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		m.Cfg.LogitScale = 0.0625
		s := session(t, m, newV41DenseTestBackend(), true)
		got, err := head(m, s)
		if got != nil {
			t.Fatal("refused head returned logits")
		}
		refused(t, s, err, "logit-scale")
	})
	t.Run("softcap-before-selection", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		m.Cfg.LogitScale, m.Cfg.LogitSoftcap = 0.0625, 0.75
		b := newV41LogitScaleRecorder()
		s := session(t, m, b, true)
		_, err := head(m, s)
		refused(t, s, err, "logit-softcap")
		if b.calls != 0 || b.uploads != 0 {
			t.Fatalf("scale calls=%d uploads=%d; the refusal must precede device selection", b.calls, b.uploads)
		}
	})
	t.Run("suppression", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		m.Cfg.SuppressTokens = []int{1}
		s := session(t, m, newV41DenseTestBackend(), true)
		_, err := head(m, s)
		refused(t, s, err, "logit-token-suppression")
	})
	t.Run("bound-scale-dispatches", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		m.Cfg.LogitScale = 0.0625
		b := newV41LogitScaleRecorder()
		s := session(t, m, b, true)
		got, err := head(m, s)
		if err != nil || b.calls != 1 || s.BackendSessionClosed() || s.HostFallbackObserved() {
			t.Fatalf("err=%v calls=%d closed=%t, want one device multiply", err, b.calls, s.BackendSessionClosed())
		}
		want := append([]float32(nil), b.lastInput...)
		logitScaleInPlace(want, m.Cfg)
		v41LogitScaleCheckBits(t, got, want)
	})
	t.Run("identity-matches-host-session", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		want, err := head(m, session(t, m, newV41DenseTestBackend(), false))
		if err != nil {
			t.Fatal(err)
		}
		s := session(t, m, newV41DenseTestBackend(), true)
		got, err := head(m, s)
		if err != nil || s.BackendSessionClosed() || s.HostFallbackObserved() {
			t.Fatalf("identity postprocess refused: %v", err)
		}
		v41LogitScaleCheckBits(t, got, want)
	})
	t.Run("host-policy-keeps-host-arms", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		m.Cfg.LogitScale, m.Cfg.LogitSoftcap, m.Cfg.SuppressTokens = 0.0625, 0.75, []int{1}
		s := session(t, m, newV41DenseTestBackend(), false)
		got, err := head(m, s)
		if err != nil || s.BackendSessionClosed() || !math.IsInf(float64(got[1]), -1) {
			t.Fatalf("host session lost its postprocess: err=%v", err)
		}
	})
	t.Run("policy-follows-owner", func(t *testing.T) {
		m := v41IncrementalPlainModel(t, 1)
		m.Cfg.LogitScale = 0.0625
		s := session(t, m, newV41DenseTestBackend(), false)
		if s.v41State().logitHostRefusal() != nil {
			t.Fatal("host session refuses the host postprocess")
		}
		s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
		_, err := head(m, s)
		refused(t, s, err, "logit-scale")
	})
}
