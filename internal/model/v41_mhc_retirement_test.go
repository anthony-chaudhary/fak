package model

import (
	"errors"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type mhcRetirementBackend struct {
	compute.Backend
	calls  map[compute.Buffer]int
	faults map[compute.Buffer]func()
	order  []compute.Buffer
}

func (b *mhcRetirementBackend) Free(t compute.Tensor) {
	b.calls[t.Buf()]++
	b.order = append(b.order, t.Buf())
	if fault := b.faults[t.Buf()]; fault != nil {
		fault()
	}
	b.Backend.Free(t)
}

func mhcRetirementFixture(t *testing.T) (*v41MHCResident, *mhcRetirementBackend, []compute.Tensor) {
	t.Helper()
	b := &mhcRetirementBackend{Backend: compute.Default(), calls: make(map[compute.Buffer]int), faults: make(map[compute.Buffer]func())}
	panel := func() compute.Tensor { return compute.NewF32(b, []int{1, 4, 1}, make([]float32, 4)) }
	owned := []compute.Tensor{panel(), panel(), panel(), panel(), panel(), panel()}
	streams, err := newV41DeviceStreams(b, owned[4])
	if err != nil {
		t.Fatal(err)
	}
	updated, err := newV41DeviceStreams(b, owned[3])
	if err != nil {
		t.Fatal(err)
	}
	r := &v41MHCResident{d: &v41MHCDevice{s: &Session{Backend: b}}, streams: streams, carry: owned[5], layer: v41MHCResidentLayer{attn: owned[0], ffn: owned[1], next: owned[2], updated: updated}}
	return r, b, owned
}

// fak-test:runtime fast est=10ms lane=default
func TestV41MHCResidentRetirementAttemptsEveryOwner(t *testing.T) {
	for _, mode := range []string{"panic", "panic-nil", "goexit"} {
		t.Run(mode, func(t *testing.T) {
			r, b, owned := mhcRetirementFixture(t)
			primary, secondary := errors.New("first free"), errors.New("later free")
			b.faults[owned[0].Buf()] = func() { panic(primary) }
			if mode == "panic-nil" {
				b.faults[owned[0].Buf()] = func() { panic(nil) }
			}
			if mode == "goexit" {
				b.faults[owned[0].Buf()] = runtime.Goexit
			}
			b.faults[owned[1].Buf()] = func() { panic(secondary) }
			done := make(chan struct{})
			var recovered any
			var returned bool
			go func() { defer close(done); defer func() { recovered = recover() }(); r.free(); returned = true }()
			<-done
			if returned {
				t.Fatal("cleanup returned success after failing release")
			}
			if mode == "panic" && recovered != primary {
				t.Fatalf("panic = %v, want first", recovered)
			}
			if mode == "goexit" && recovered != nil {
				t.Fatalf("secondary panic replaced Goexit: %v", recovered)
			}
			if r.streams != nil || r.carry.Buf() != nil || r.layer.attn.Buf() != nil || r.layer.ffn.Buf() != nil || r.layer.next.Buf() != nil || r.layer.updated != nil {
				t.Fatal("ownership retained after failed cleanup")
			}
			for i, x := range owned {
				if b.calls[x.Buf()] != 1 {
					t.Fatalf("owner %d attempts=%d", i, b.calls[x.Buf()])
				}
			}
			r.free()
			for i, x := range owned {
				if b.calls[x.Buf()] != 1 {
					t.Fatalf("retry owner %d attempts=%d", i, b.calls[x.Buf()])
				}
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41DeviceStreamsFreeDetachesBeforePanic(t *testing.T) {
	r, b, owned := mhcRetirementFixture(t)
	panel := r.streams
	fault := errors.New("stream free")
	b.faults[owned[4].Buf()] = func() { panic(fault) }
	func() {
		defer func() {
			if recover() != fault {
				t.Error("lost free panic")
			}
		}()
		panel.Free()
	}()
	panel.Free()
	if panel.Tensor().Buf() != nil || b.calls[owned[4].Buf()] != 1 {
		t.Fatal("failed release remained retryable")
	}
	delete(b.faults, owned[4].Buf())
	r.free()
}

// fak-test:runtime fast est=10ms lane=default
func TestV41MHCPublishTransfersNextBeforeOldFreePanic(t *testing.T) {
	r, b, owned := mhcRetirementFixture(t)
	fault := errors.New("predecessor free")
	b.faults[owned[4].Buf()] = func() { panic(fault) }
	func() {
		defer func() {
			if recover() != fault {
				t.Error("lost predecessor panic")
			}
		}()
		_ = r.publish(0)
	}()
	if r.streams.Tensor().Buf() != owned[2].Buf() || r.layer.next.Buf() != nil {
		t.Fatal("successor has two owners after handoff panic")
	}
	delete(b.faults, owned[4].Buf())
	r.free()
	for i, x := range owned {
		if b.calls[x.Buf()] != 1 {
			t.Fatalf("owner %d freed %d times", i, b.calls[x.Buf()])
		}
	}
}
