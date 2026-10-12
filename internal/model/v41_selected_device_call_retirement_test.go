package model

import (
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This recorder exercises retirement through the existing selected-call helper
// and its actual Engram/pool factories. Its healthy outputs test ownership only.
type v41SelectedRetirementBackend struct {
	*v41DenseTestBackend
	primary       error
	panicPrimary  bool
	closeFault    bool
	calls, closes int
}

func (b *v41SelectedRetirementBackend) Recycle() {
	b.closes++
	if b.closeFault {
		panic(errors.New("secondary selected-call Close failure"))
	}
}

func (b *v41SelectedRetirementBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	if dt != compute.F32 {
		return b.v41DenseTestBackend.Upload(x, dt)
	}
	return compute.NewF32(b, append([]int(nil), x.Shape...), append([]float32(nil), b.Backend.Read(x)...))
}

func (b *v41SelectedRetirementBackend) SupportsV41CompressorPool() bool { return true }
func (b *v41SelectedRetirementBackend) SupportsV41EngramGate() bool     { return true }

func (b *v41SelectedRetirementBackend) selectedFault() error {
	b.calls++
	if b.panicPrimary {
		panic(b.primary)
	}
	return b.primary
}

func (b *v41SelectedRetirementBackend) V41CompressorPool(kv, scores compute.Tensor, ratio, width int) (compute.Tensor, error) {
	if err := b.selectedFault(); err != nil {
		return compute.Tensor{}, err
	}
	return compute.NewF32(b, []int{width}, append([]float32(nil), b.Backend.Read(kv)[:width]...)), nil
}

func (b *v41SelectedRetirementBackend) V41EngramGate(streams, projected, norms compute.Tensor, rows, hidden int, eps float32) (compute.Tensor, compute.Tensor, error) {
	if err := b.selectedFault(); err != nil {
		return compute.Tensor{}, compute.Tensor{}, err
	}
	return compute.NewF32(b, []int{rows, 4, hidden}, append([]float32(nil), b.Backend.Read(streams)...)),
		compute.NewF32(b, []int{rows, 4}, make([]float32, rows*4)), nil
}

func v41SelectedRetirementCPU(t *testing.T) {
	t.Helper()
	if b := compute.Default(); b == nil || b.Name() != "cpu-ref" || b.Caps().DeviceMemory {
		t.Fatal("selected-call retirement witness requires cpu-ref")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41SelectedDeviceCallRetirement(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, mode := range []string{"incoming-wrapped-forward", "returned-error", "backend-panic", "plain-panic"} {
		t.Run(mode, func(t *testing.T) {
			b := &v41SelectedRetirementBackend{v41DenseTestBackend: newV41DenseTestBackend(), closeFault: true}
			s := &Session{M: &Model{}, Backend: b}
			peer := &Session{M: s.M, Backend: b}
			defer peer.Close()
			defer func() { b.closeFault = false; s.Close() }()
			calls := 0
			good, err := v41SelectedDeviceCall(s, 2, "healthy", func(stage *string) ([]float32, error) {
				calls++
				return []float32{1}, nil
			})
			if err != nil || !slices.Equal(good, []float32{1}) || s.BackendSessionClosed() || b.closes != 0 {
				t.Fatal("healthy helper control failed", err)
			}
			var primary error = errors.New("primary selected fault")
			if mode == "incoming-wrapped-forward" {
				primary = fmt.Errorf("original wrapper: %w", &BackendForwardOperationError{
					Path: "another-owner", Stage: "incoming", Cause: ErrV41ForwardStage,
				})
			} else if mode == "backend-panic" {
				primary = &compute.BackendError{Backend: "cpu-ref", Class: compute.VulkanClassExecutionFailed, Err: primary}
			}
			var result []float32
			var returned error
			got := recoverError(func() {
				result, returned = v41SelectedDeviceCall(s, 2, "selected-test", func(stage *string) ([]float32, error) {
					calls++
					*stage = "dispatch"
					if mode != "returned-error" {
						panic(primary)
					}
					return []float32{99}, primary
				})
			})
			if mode == "incoming-wrapped-forward" || mode == "plain-panic" {
				if got != primary {
					t.Fatal("selected panic identity changed", got)
				}
			} else {
				if got != nil || !errors.Is(returned, primary) {
					t.Fatal("returned selected cause changed", got, returned)
				}
				var selected *BackendForwardOperationError
				if !errors.As(returned, &selected) || selected.Path != "selected-test" || selected.Stage != "dispatch" {
					t.Fatal("returned selected attribution changed", returned)
				}
			}
			if mode == "incoming-wrapped-forward" && s.halFailure != primary {
				t.Fatal("original typed wrapper was not latched")
			}
			if result != nil || !s.BackendSessionClosed() || !errors.Is(s.halFailure, primary) ||
				b.closes != 1 || calls != 2 || peer.BackendSessionClosed() || peer.halFailure != nil {
				t.Fatal("selected fault published, replayed, or retired the wrong owner")
			}
			retry := recoverError(func() {
				_, _ = v41SelectedDeviceCall(s, 2, "selected-test", func(*string) ([]float32, error) {
					calls++
					return []float32{7}, nil
				})
			})
			if retry != s.halFailure || calls != 2 || b.closes != 1 {
				t.Fatal("retired helper retried or changed the primary", retry)
			}
		})
	}
}

// Both landed seams use the common helper. Invoke their real factories and
// completed-operation boundaries, rather than replacing their callback with a
// direct test call to the helper. Existing end-to-end tests cover token rollback.
// fak-test:runtime medium est=3s lane=default
func TestV41SelectedDeviceConsumersRetireCurrentOwner(t *testing.T) {
	v41SelectedRetirementCPU(t)
	for _, op := range []string{"engram", "pool"} {
		for _, panicking := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/panic=%t", op, panicking), func(t *testing.T) {
				m, _ := v41IncrementalEngramFixture(t, false)
				b := &v41SelectedRetirementBackend{v41DenseTestBackend: newV41DenseTestBackend()}
				s := v41DenseTestSession(t, m, b)
				peer := v41DenseTestSession(t, m, b)
				t.Cleanup(func() { b.primary, b.closeFault, b.panicPrimary = nil, false, false })
				var call func() error
				var panel *v41DeviceStreams
				var panelBefore []float32
				if op == "pool" {
					fn := s.v41CompressorPoolFunc()
					if fn == nil {
						t.Fatal("pool factory did not bind")
					}
					call = func() error {
						out, err := fn(0, []float32{1, 2, 3, 4}, []float32{0, 0, 0, 0}, 2, 2)
						if err != nil && out != nil {
							t.Fatal("failed pool published an output")
						}
						return err
					}
				} else {
					layer := m.Cfg.DeepSeekV41.EngramLayerIDs[0]
					geometry, ids, err := m.v41EngramLayerRows(layer, []int{1})
					if err != nil {
						t.Fatal(err)
					}
					row, err := m.v41EngramGatherRows(layer, 1, ids, float32(m.Cfg.RMSNormEps), geometry)
					if err != nil {
						t.Fatal(err)
					}
					panel, err = newV41DeviceStreams(b, compute.NewF32(b, []int{1, 4, geometry.H}, make([]float32, 4*geometry.H)))
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(panel.Free)
					fn := s.v41EngramGateFunc()
					if fn == nil {
						t.Fatal("Engram factory did not bind")
					}
					call = func() error {
						handled, err := fn(layer, panel, row, 1, geometry, float32(m.Cfg.RMSNormEps))
						if !handled {
							t.Fatal("valid Engram geometry declined")
						}
						return err
					}
				}
				if err := call(); err != nil || b.calls != 1 || s.BackendSessionClosed() {
					t.Fatal("healthy consumer control failed", err)
				}
				if panel != nil {
					panelBefore = append([]float32(nil), b.Backend.Read(panel.Tensor())...)
				}
				primary := fmt.Errorf("consumer original wrapper: %w", &BackendForwardOperationError{
					Path: "another-owner", Stage: "incoming", Cause: ErrV41ForwardStage,
				})
				b.primary, b.panicPrimary, b.closeFault = primary, panicking, true
				var returned error
				got := recoverError(func() { returned = call() })
				if panicking {
					if got != primary || s.halFailure != primary {
						t.Fatal("consumer lost the incoming primary wrapper", got)
					}
				} else if got != nil || !errors.Is(returned, primary) {
					t.Fatal("consumer lost the returned primary to Close", got, returned)
				}
				if !s.BackendSessionClosed() || !errors.Is(s.halFailure, primary) || b.calls != 2 ||
					b.closes != 1 || peer.BackendSessionClosed() || peer.halFailure != nil {
					t.Fatal("consumer failed to retire exactly its current owner")
				}
				if panel != nil && !slices.Equal(panelBefore, b.Backend.Read(panel.Tensor())) {
					t.Fatal("failed Engram call changed the published panel")
				}
				if retry := recoverError(func() { _ = call() }); retry != s.halFailure || b.calls != 2 || b.closes != 1 {
					t.Fatal("retired consumer replayed or changed the latch", retry)
				}
			})
		}
	}
}
