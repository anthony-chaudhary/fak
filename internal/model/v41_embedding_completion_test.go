package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type embeddingCompletionBackend struct {
	compute.Backend
	owner                                           sync.Mutex
	raw                                             []byte
	fault                                           string
	primary                                         error
	checkOwner                                      func()
	table, panel                                    compute.Buffer
	uploads, gathers, reads, panelFrees, tableFrees int
	lastRead                                        []float32
}

func (b *embeddingCompletionBackend) ForwardExecutionOwner() *sync.Mutex { return &b.owner }
func (b *embeddingCompletionBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b *embeddingCompletionBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.F32 || dt == compute.Q2_K
}
func (b *embeddingCompletionBackend) V41EmbeddingGatherAvailable() bool    { return true }
func (b *embeddingCompletionBackend) V41Q2KEmbeddingGatherAvailable() bool { return true }
func (b *embeddingCompletionBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.checkOwner()
	b.uploads++
	var out compute.Tensor
	if dt == compute.Q2_K {
		out = compute.NewQ2K(b, x.Shape, append([]byte(nil), b.raw...))
	} else {
		out = compute.NewF32(b, x.Shape, append([]float32(nil), b.Backend.Read(x)...))
	}
	b.table = out.Buf()
	return out
}
func (b *embeddingCompletionBackend) V41EmbeddingGather(table compute.Tensor, ids []int, hidden int) (compute.Tensor, error) {
	return b.gather(table, ids, hidden, false)
}
func (b *embeddingCompletionBackend) V41Q2KEmbeddingGather(table compute.Tensor, ids []int, hidden int) (compute.Tensor, error) {
	return b.gather(table, ids, hidden, true)
}
func (b *embeddingCompletionBackend) gather(table compute.Tensor, ids []int, hidden int, packed bool) (compute.Tensor, error) {
	b.checkOwner()
	b.gathers++
	if table.Backend() != b || table.Buf() != b.table || (table.Dtype == compute.Q2_K) != packed {
		panic("embedding fixture lost the selected resident table")
	}
	switch b.fault {
	case "gather-nil":
		panic(nil)
	case "nonnil", "backend":
		panic(b.primary)
	case "goexit":
		runtime.Goexit()
	}
	var values []float32
	if packed {
		values = v41Q2KEmbeddingShaderOracle(b.raw, hidden, ids)
	} else {
		rows := b.Backend.Read(table)
		for _, id := range ids {
			values = append(values, rows[id*hidden:(id+1)*hidden]...)
		}
	}
	out := compute.NewF32(b, []int{len(ids), hidden}, values)
	b.panel = out.Buf()
	if b.fault == "returned" {
		return out, b.primary
	}
	return out, nil
}
func (b *embeddingCompletionBackend) Read(x compute.Tensor) []float32 {
	b.checkOwner()
	if x.Buf() != b.panel || x.Backend() != b {
		panic("embedding read is not its fresh owned panel")
	}
	b.reads++
	if b.fault == "read-nil" {
		panic(nil)
	}
	b.lastRead = b.Backend.Read(x)
	return b.lastRead
}
func (b *embeddingCompletionBackend) Free(x compute.Tensor) {
	if b.checkOwner != nil {
		b.checkOwner()
	}
	if x.Buf() == b.table {
		b.tableFrees++
	} else if x.Buf() == b.panel {
		b.panelFrees++
		b.Backend.Free(x)
		if b.fault == "cleanup-nil" {
			panic(nil)
		}
		// A successful callback must own a copy before releasing this storage.
		for i := range b.lastRead {
			b.lastRead[i] = 999
		}
		return
	}
	b.Backend.Free(x)
}

// The real embedding consumer already rejects a missing panel. This witness
// requires the selected primary and closed-session latch, rather than crediting
// that later width refusal as preservation of the original failure.
// fak-test:runtime fast est=80ms lane=default
func TestV41EmbeddingCompletion(t *testing.T) {
	v41SelectedRetirementCPU(t)
	allJoined := true
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
			for _, encoding := range []string{"f32", "q2k"} {
				t.Run(encoding, func(t *testing.T) {
					for _, fault := range []string{"gather-nil", "nonnil", "goexit", "backend", "returned", "read-nil", "cleanup-nil", "healthy"} {
						t.Run(fault, func(t *testing.T) {
							const vocab, hidden = 3, 256
							m := &Model{Cfg: Config{VocabSize: vocab, HiddenSize: hidden}}
							b := &embeddingCompletionBackend{Backend: compute.Default(), fault: fault, primary: errors.New("embedding selected primary")}
							if encoding == "q2k" {
								b.raw = v41Q2KEmbeddingTestRaw(vocab, hidden, 137)
								var err error
								m.Q2KEmbedding, err = NewQ2KEmbedding(b.raw, vocab, hidden)
								if err != nil {
									t.Fatal(err)
								}
							} else {
								m.raw = make([]byte, 4*vocab*hidden)
								for i := 0; i < vocab*hidden; i++ {
									binary.LittleEndian.PutUint32(m.raw[4*i:], math.Float32bits(float32(i%17-8)/8))
								}
								m.manifest = map[string]tensorMeta{v41EmbeddingTableName: {Dtype: "F32", Shape: []int{vocab, hidden}, Nbytes: len(m.raw)}}
							}
							if fault == "backend" || fault == "returned" {
								b.primary = &compute.BackendError{Backend: "embedding-witness", Class: compute.VulkanClassExecutionFailed, Err: b.primary}
							}
							s := &Session{M: m, Backend: b, halW: map[string]compute.Tensor{}}
							t.Cleanup(func() {
								if allJoined {
									b.fault, b.checkOwner = "", nil
									s.Close()
									if err := m.CloseWeights(); err != nil {
										t.Error(err)
									}
								}
							})
							ids := []int{2, 0, 2}
							want, err := m.v41EmbeddingPanel(ids)
							if err != nil || len(want) != len(ids)*hidden {
								t.Fatalf("host oracle: %d, %v", len(want), err)
							}
							fn := s.v41EmbeddingGatherFunc()
							if fn == nil {
								t.Fatal("embedding factory declined the admitted table")
							}
							ops := &v41DeviceGatherOps{owner: s, embedGather: fn}
							badOwner := false
							b.checkOwner = func() {
								scope := s.backendExecutionScope
								if scope == nil || !scope.live || scope.owner != &b.owner {
									badOwner = true
								} else {
									found := false
									for _, member := range scope.sessions {
										found = found || member == s
									}
									badOwner = badOwner || !found
								}
								if b.owner.TryLock() {
									b.owner.Unlock()
									badOwner = true
								}
							}
							var got []float32
							var cause error
							var payload any
							returned, boundaryReturned := false, false
							done := make(chan struct{})
							allJoined = false
							go func() {
								defer close(done)
								func() {
									defer func() { payload = recover() }()
									func() {
										scope := s.beginBackendExecution()
										complete := false
										defer scope.finishCompleted(&complete)
										got, cause = m.v41EmbeddingPanelOn(ids, ops)
										complete = true
									}()
									returned = true
								}()
								boundaryReturned = true
							}()
							select {
							case <-done:
								allJoined = true
							case <-time.After(5 * time.Second):
								t.Fatal("embedding worker did not join; cleanup withheld")
							}
							ordinary := fault == "healthy" || fault == "backend" || fault == "returned"
							if returned != ordinary || boundaryReturned == (fault == "goexit") {
								t.Fatalf("embedding %s fault=%q returned=%t boundaryReturned=%t output=%d width_refusal=%t closed=%t cause=%v panic=%v gathers=%d reads=%d", encoding, fault, returned, boundaryReturned, len(got), errors.Is(cause, errV41EmbeddingGatherResult), s.BackendSessionClosed(), cause, payload, b.gathers, b.reads)
							}
							if badOwner || s.backendExecutionScope != nil || !b.owner.TryLock() {
								t.Fatal("embedding execution owner was not held and released")
							}
							b.owner.Unlock()
							wantRead, wantFree := 0, 0
							if fault == "healthy" || fault == "read-nil" || fault == "cleanup-nil" {
								wantRead, wantFree = 1, 1
							}
							if fault == "returned" {
								wantFree = 1
							}
							if b.uploads != 1 || b.gathers != 1 || b.reads != wantRead || b.panelFrees != wantFree || b.tableFrees != 0 {
								t.Fatalf("uploads/gathers/reads/panel/table frees=%d/%d/%d/%d/%d", b.uploads, b.gathers, b.reads, b.panelFrees, b.tableFrees)
							}
							closed := fault != "healthy" && fault != "goexit"
							if s.BackendSessionClosed() != closed || (s.halFailure != nil) != closed {
								t.Fatalf("closed/latch mismatch: %t/%v", s.BackendSessionClosed(), s.halFailure)
							}
							switch fault {
							case "healthy":
								if cause != nil || payload != nil {
									t.Fatalf("healthy error: %v/%v", cause, payload)
								}
								v41TailRoPECheckBits(t, "owned embedding copy", got, want)
							case "backend", "returned":
								var selected *BackendForwardOperationError
								if !errors.As(cause, &selected) || !errors.Is(cause, b.primary) || selected.Path != "v41-embedding-gather" || selected.Layer != -1 || selected.Stage != "embedding-gather" {
									t.Fatalf("typed cause lost: %v", cause)
								}
							case "nonnil":
								if payload != b.primary {
									t.Fatalf("primary changed: %v", payload)
								}
							case "gather-nil", "read-nil", "cleanup-nil":
								if mode == "0" {
									if _, ok := payload.(*runtime.PanicNilError); !ok {
										t.Fatalf("nil panic changed: %T", payload)
									}
								} else if payload != nil {
									t.Fatalf("legacy nil panic changed: %v", payload)
								}
							case "goexit":
								if payload != nil {
									t.Fatalf("Goexit became panic: %v", payload)
								}
							}
							if fault != "healthy" && got != nil {
								t.Fatal("failed embedding published a panel")
							}
							if closed {
								before := fmt.Sprint(b.uploads, b.gathers, b.reads, b.panelFrees, b.tableFrees)
								var retry any
								func() {
									defer func() { retry = recover() }()
									scope := s.beginBackendExecution()
									complete := false
									defer scope.finishCompleted(&complete)
									_, _ = m.v41EmbeddingPanelOn(ids, ops)
									complete = true
								}()
								if retry != s.halFailure || before != fmt.Sprint(b.uploads, b.gathers, b.reads, b.panelFrees, b.tableFrees) {
									t.Fatal("closed embedding retried backend work or changed its selected latch")
								}
							}
						})
						if !allJoined {
							return
						}
					}
				})
				if !allJoined {
					return
				}
			}
		})
		if !allJoined {
			return
		}
	}
}
