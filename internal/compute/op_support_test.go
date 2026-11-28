package compute

import "testing"

type fakeOpSupportBackend struct {
	Backend
	caps    Caps
	support map[string]bool
	offload map[string]bool
}

func (f *fakeOpSupportBackend) Caps() Caps {
	return f.caps
}

func (f *fakeOpSupportBackend) SupportsOp(op string) bool {
	return f.support[op]
}

func (f *fakeOpSupportBackend) OffloadOp(op string) bool {
	return f.offload[op]
}

var (
	_ OpSupport   = (*fakeOpSupportBackend)(nil)
	_ OpOffloader = (*fakeOpSupportBackend)(nil)
)

// fak-test:runtime fast est=1s lane=default
func TestBackendCanOffloadRequiresSupportAndOffload(t *testing.T) {
	t.Parallel()

	cpu := Default()
	if cpu == nil {
		t.Fatal("Default() backend is nil")
	}

	t.Run("cpu default is supported but not offloaded", func(t *testing.T) {
		if _, ok := cpu.(OpSupport); ok {
			t.Fatalf("cpu-ref unexpectedly implements OpSupport")
		}
		if !BackendSupportsOp(cpu, "matmul") {
			t.Fatal("BackendSupportsOp(cpu, matmul) = false, want true")
		}
		if BackendCanOffload(cpu, "matmul") {
			t.Fatal("BackendCanOffload(cpu, matmul) = true, want false")
		}
		if CanOffload(cpu, "matmul") {
			t.Fatal("CanOffload(cpu, matmul) = true, want false")
		}
		dec := InspectOpOffload(cpu, "matmul")
		if !dec.Supported || dec.Offloadable || dec.VetoReason != "not-offloader" {
			t.Fatalf("InspectOpOffload(cpu, matmul) = %+v, want Supported=true Offloadable=false VetoReason=not-offloader", dec)
		}
	})

	t.Run("nil backend and empty op fail closed", func(t *testing.T) {
		if BackendSupportsOp(nil, "matmul") || BackendCanOffload(nil, "matmul") || CanOffload(nil, "matmul") {
			t.Fatal("nil backend must return false for support and offload")
		}
		if BackendSupportsOp(cpu, "") || BackendCanOffload(cpu, "") || CanOffload(cpu, "") {
			t.Fatal("empty op must return false for support and offload")
		}
		if dec := InspectOpOffload(nil, "matmul"); dec.VetoReason != "nil-backend" {
			t.Fatalf("InspectOpOffload(nil) = %+v, want nil-backend", dec)
		}
		if dec := InspectOpOffload(cpu, ""); dec.VetoReason != "empty-op" {
			t.Fatalf("InspectOpOffload(cpu, empty) = %+v, want empty-op", dec)
		}
	})

	t.Run("supports and offload truth table enforces offload veto", func(t *testing.T) {
		be := &fakeOpSupportBackend{
			Backend: cpu,
			caps:    Caps{DeviceMemory: true},
			support: map[string]bool{
				"allow_both":     true,
				"veto_offload":   true,
				"unsupported_on": false,
				"neither":        false,
			},
			offload: map[string]bool{
				"allow_both":     true,
				"veto_offload":   false,
				"unsupported_on": true,
				"neither":        false,
			},
		}

		cases := []struct {
			op          string
			wantSupport bool
			wantOffload bool
			wantVeto    string
		}{
			{"allow_both", true, true, ""},
			{"veto_offload", true, false, "offload-vetoed"},
			{"unsupported_on", false, false, "unsupported-op"},
			{"neither", false, false, "unsupported-op"},
			{"unknown", false, false, "unsupported-op"},
		}
		for _, tc := range cases {
			if got := BackendSupportsOp(be, tc.op); got != tc.wantSupport {
				t.Errorf("%s: BackendSupportsOp = %v, want %v", tc.op, got, tc.wantSupport)
			}
			if got := BackendCanOffload(be, tc.op); got != tc.wantOffload {
				t.Errorf("%s: BackendCanOffload = %v, want %v", tc.op, got, tc.wantOffload)
			}
			if got := CanOffload(be, tc.op); got != tc.wantOffload {
				t.Errorf("%s: CanOffload = %v, want %v", tc.op, got, tc.wantOffload)
			}
			dec := InspectOpOffload(be, tc.op)
			if dec.Supported != tc.wantSupport || dec.Offloadable != tc.wantOffload || !dec.DeviceMemory || dec.VetoReason != tc.wantVeto {
				t.Errorf("%s: InspectOpOffload = %+v, want sup=%v off=%v dev=true veto=%q",
					tc.op, dec, tc.wantSupport, tc.wantOffload, tc.wantVeto)
			}
		}
	})
}
