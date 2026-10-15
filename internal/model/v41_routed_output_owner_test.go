package model

import (
	"errors"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This is a CPU recorder of actual model ownership/control flow, not hardware
// timing. Poisoning publication storage on Free detects a stale borrowed row.
type v41RoutedOutputOwnerWitness struct {
	*v41ScalarCohortWitness
	mu                  sync.Mutex
	freed               map[compute.Buffer]bool
	routed              map[compute.Buffer]bool
	sumUploads          int
	borrowedAdds        int
	failRetainedCleanup bool
	goexitFree          bool
	panicNilFree        bool
	sumExit             string
	ownerQuery          func()
	uploads             int
	adds                int
	freeRows            int
	failFree            bool
	failAdd             bool
	failRead            bool
	badDeferred         string
}

func (b *v41RoutedOutputOwnerWitness) ForwardExecutionOwner() *sync.Mutex {
	if b.ownerQuery != nil {
		b.ownerQuery()
	}
	return &b.mu
}
func (b *v41RoutedOutputOwnerWitness) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	b.uploads++
	return b.Backend.Upload(x, dt)
}
func (b *v41RoutedOutputOwnerWitness) UploadClass(x compute.Tensor, dt compute.Dtype, _ compute.MemoryClass, site string) compute.Tensor {
	if site == "V4.1 routed-sum expert output" {
		b.sumUploads++
	}
	return b.Upload(x, dt)
}
func (b *v41RoutedOutputOwnerWitness) V41BF16PublishDeferred(x compute.Tensor, site uint32) (compute.Tensor, compute.V41DeferredStatus, error) {
	out, status, err := b.v41ScalarCohortWitness.V41BF16PublishDeferred(x, site)
	if site == compute.V41BF16PublishRoutedDown && out.Buf() != nil {
		if b.routed == nil {
			b.routed = make(map[compute.Buffer]bool)
		}
		b.routed[out.Buf()] = true
	}
	return out, status, err
}
func (b *v41RoutedOutputOwnerWitness) AddInPlace(dst, src compute.Tensor) {
	if b.sumExit == "sum-add-nil" {
		panic(nil)
	}
	if b.sumExit == "sum-add-goexit" {
		runtime.Goexit()
	}
	if b.routed[src.Buf()] {
		b.borrowedAdds++
		if b.failRetainedCleanup {
			b.failFree = true
		}
	}
	if b.freed[src.Buf()] {
		panic("routed sum borrowed a freed publication")
	}
	b.adds++
	if b.failAdd {
		panic(&compute.BackendError{Backend: "routed-output-test", Site: "AddInPlace", Err: errors.New("sum add failure")})
	}
	b.Backend.AddInPlace(dst, src)
}
func (b *v41RoutedOutputOwnerWitness) Read(x compute.Tensor) []float32 {
	if b.sumExit == "sum-read-nil" {
		panic(nil)
	}
	if b.sumExit == "sum-read-goexit" {
		runtime.Goexit()
	}
	if b.freed[x.Buf()] {
		panic("read after publication release")
	}
	if b.failRead {
		panic(&compute.BackendError{Backend: "routed-output-test", Site: "Read", Err: errors.New("sum read failure")})
	}
	return b.v41ScalarCohortWitness.Read(x)
}
func (b *v41RoutedOutputOwnerWitness) Free(x compute.Tensor) {
	if b.routed[x.Buf()] {
		if b.freed[x.Buf()] {
			panic("publication released twice")
		}
		b.freed[x.Buf()] = true
		b.freeRows++
		values := b.Backend.Read(x)
		for i := range values {
			values[i] = float32(math.NaN())
		}
		if b.goexitFree {
			runtime.Goexit()
		}
		if b.panicNilFree {
			panic(nil)
		}
		if b.failFree {
			panic(&compute.BackendError{Backend: "routed-output-test", Site: "Free", Err: errors.New("publication cleanup failure")})
		}
	}
	b.v41ScalarCohortWitness.Free(x)
}
func (b *v41RoutedOutputOwnerWitness) V41ReadDeferred(statuses []compute.V41DeferredStatus, rows []compute.Tensor) ([]error, [][]float32, error) {
	verdicts, values, err := b.v41ScalarCohortWitness.V41ReadDeferred(statuses, rows)
	if len(values) != 0 {
		switch b.badDeferred {
		case "short":
			values[0] = values[0][:len(values[0])-1]
		case "nonfinite":
			values[0][0] = float32(math.NaN())
		}
	}
	return verdicts, values, err
}

// fak-test:runtime fast est=250ms lane=default
func TestV41RoutedOutputOwner(t *testing.T) {
	t.Setenv(v41PrefillExpertPanelEnv, "0")
	newOwner := func(width int, picks [][]routePick) (*Session, *v41RoutedOutputOwnerWitness, *v41RoutedOutputOwner, *backendExecutionScope) {
		b := &v41RoutedOutputOwnerWitness{
			v41ScalarCohortWitness: &v41ScalarCohortWitness{v41RoutedLayerTestBackend: newV41RoutedLayerTestBackend(), enabled: true},
			freed:                  make(map[compute.Buffer]bool),
		}
		s := &Session{M: &Model{Cfg: Config{HiddenSize: width, MoEIntermediateSize: 3, HeadDim: v41KVLoraRank, NumLayers: 1}}, Backend: b}
		scope := s.beginBackendExecution()
		return s, b, newV41RoutedOutputOwner(s, 0, width, picks), scope
	}
	picks := [][]routePick{{{expert: 2, weight: .25}, {expert: 0, weight: .5}}, {{expert: 0, weight: .75}, {expert: 2, weight: 1}}}
	destinations := []v41ExpertRow{{0, 0}, {1, 1}, {0, 1}, {1, 0}}
	inputs := [][]float32{{1, .5, -.25, 0}, {-.5, .25, 1, 0}, {1, .5, -.25, 0}, {-.5, .25, 1, 0}}
	weights := []float32{.25, 1, .5, .75}
	run := func(retain bool, mode string) (result [][]float32, uploads, adds, fences, freed int, closed bool, err error) {
		s, b, owner, scope := newOwner(4, picks)
		defer scope.finish()
		defer s.closeBackendOwned()
		if owner == nil {
			t.Fatal("owned fixture declined")
		}
		// One cohort straddles the retained-prefix boundary. Larger prompts
		// use the same mixed slot route once the hard cap is reached.
		if mode == "mixed" {
			owner.capRows = 3
		}
		func() {
			completed := false
			defer owner.finish(&err, &completed)
			func() {
				var handoff *v41RoutedOutputHandoff
				if retain {
					handoff = owner.handoff(destinations)
				}
				b.badDeferred = mode
				if mode == "read_failure" {
					b.readFailure = errors.New("cohort read failure")
				}
				gate := compute.NewF32(compute.Default(), []int{3, 4}, []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0})
				up := compute.NewF32(compute.Default(), []int{3, 4}, []float32{.5, 0, 0, 0, 0, .5, 0, 0, 0, 0, .5, 0})
				down := compute.NewF32(compute.Default(), []int{4, 3}, []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1})
				stage, recordingFailed := "gate/up", false
				var rows [][]float32
				rows, err = s.v41RunExpertCohortRetained(b, gate, up, down, inputs, weights, 0, 0, &stage, &recordingFailed, nil, handoff)
				if err != nil {
					if len(owner.rows) != 0 {
						t.Fatal("failed read/validation transferred ownership")
					}
					return
				}
				before := b.uploads
				b.failFree, b.failAdd, b.failRead = mode == "cleanup" || mode == "primary_cleanup", mode == "add", mode == "sum_read"
				if mode == "primary_cleanup" {
					err = errors.New("later cohort primary")
					return
				}
				if mode == "later_selected" {
					b.failFree = true
					_, err = v41SelectedDeviceCall(s, 0, "later scalar group", func(_ *string) (int, error) {
						return 0, errors.New("later selected primary")
					})
					if s.v41RoutedOutputs != nil || len(owner.rows) != 0 {
						t.Fatal("selected fallback retired before draining retained rows")
					}
					return
				}
				host := [][][]float32{{rows[0], rows[2]}, {rows[3], rows[1]}}
				out := make([][]float32, 2)
				for token := range out {
					if retain {
						out[token], err = owner.sum(token, picks[token], host[token])
					} else {
						out[token], err = s.v41RoutedSumFunc()(0, picks[token], host[token], 4)
					}
					if err != nil {
						return
					}
				}
				result, uploads = out, b.uploads-before
			}()
			completed = true
		}()
		if err != nil {
			result = nil
		}
		if s.v41RoutedOutputs != nil || len(owner.index) != 0 || len(owner.buffers) != 0 || !owner.closed || len(b.pending) != 0 {
			t.Fatal("owner or deferred-status records survived terminal cleanup")
		}
		return result, uploads, b.adds, b.fences, b.freeRows, s.BackendSessionClosed(), err
	}
	t.Run("mixed_parity_and_lifetime", func(t *testing.T) {
		want, wu, wa, wf, wr, _, we := run(false, "mixed")
		got, gu, ga, gf, gr, _, ge := run(true, "mixed")
		if we != nil || ge != nil || !v41GroupedFreshBitsEqual(got, want) || wu-gu != 3 || wa != ga || wf != gf || wr != 4 || gr != 4 {
			t.Fatalf("mixed route: errors=%v/%v uploads=%d/%d adds=%d/%d fences=%d/%d releases=%d/%d", we, ge, wu, gu, wa, ga, wf, gf, wr, gr)
		}
	})
	for _, mode := range []string{"short", "nonfinite", "read_failure", "add", "sum_read", "cleanup", "primary_cleanup", "later_selected"} {
		t.Run(mode, func(t *testing.T) {
			out, _, _, _, freed, closed, err := run(true, mode)
			if out != nil || err == nil || freed != 4 {
				t.Fatalf("failure published output or leaked rows: output=%v err=%v releases=%d", out, err, freed)
			}
			if mode == "add" || mode == "sum_read" || mode == "cleanup" || mode == "primary_cleanup" || mode == "later_selected" {
				if !closed {
					t.Fatal("selected sum/cleanup failure left session usable")
				}
			}
			if mode == "primary_cleanup" && err.Error() != "later cohort primary" {
				t.Fatalf("cleanup masked logical primary: %v", err)
			}
		})
	}
	for _, panicnil := range []string{"0", "1"} {
		t.Run("cleanup_completion_panicnil="+panicnil, func(t *testing.T) {
			t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+panicnil)
			for _, mode := range []string{"goexit", "nil-cleanup", "nil-primary", "body-goexit", "sum-add-nil", "sum-add-goexit", "sum-read-nil", "sum-read-goexit"} {
				t.Run(mode, func(t *testing.T) {
					type observed struct {
						releases                                    int
						closed, fatal, detached, returned, boundary bool
					}
					done := make(chan observed, 1)
					go func() {
						s, b, owner, scope := newOwner(4, picks)
						returned, boundary := false, false
						var got observed
						defer func() { done <- got }()
						defer scope.finish()
						defer func() {
							var fatal *BackendForwardOperationError
							// Capture before any ordinary Close, so cleanup cannot mask
							// a selected operation that forgot to retire its Session.
							got = observed{b.freeRows, s.BackendSessionClosed(), errors.As(s.halFailure, &fatal), s.v41RoutedOutputs == nil, returned, boundary}
							s.closeBackendOwned()
						}()
						rows := make([]compute.Tensor, len(destinations))
						b.routed = make(map[compute.Buffer]bool)
						for i := range rows {
							rows[i] = compute.NewF32(compute.Default(), []int{4}, make([]float32, 4))
							b.routed[rows[i].Buf()] = true
						}
						if n, err := owner.handoff(destinations).adopt(rows); err != nil || n != len(rows) {
							t.Errorf("completion fixture transfer: n=%d err=%v", n, err)
							return
						}
						b.goexitFree, b.panicNilFree = mode == "goexit", mode == "nil-cleanup"
						func() {
							defer func() { _ = recover() }()
							var err error
							if strings.HasPrefix(mode, "sum-") {
								b.sumExit = mode
								_, err = owner.sum(0, picks[0], [][]float32{make([]float32, 4), make([]float32, 4)})
							} else {
								func() {
									completed := false
									defer owner.finish(&err, &completed)
									if mode == "nil-primary" {
										panic(nil)
									}
									if mode == "body-goexit" {
										runtime.Goexit()
									}
									completed = true
								}()
							}
							returned = err == nil
						}()
						boundary = true
					}()
					got := <-done
					if got.releases != 4 || !got.closed || !got.fatal || !got.detached || got.returned || got.boundary == strings.HasSuffix(mode, "goexit") {
						t.Fatalf("incomplete cleanup reported success or retained ownership: %+v", got)
					}
				})
			}
		})
	}
	t.Run("actual_prompt_geometry_cap", func(t *testing.T) {
		large := make([][]routePick, 512)
		for i := range large {
			large[i] = make([]routePick, 6)
		}
		for _, width := range []int{1, 5120, 10240, v41RoutedOutputMaxBytes/4 + 1} {
			s, _, owner, scope := newOwner(width, large)
			if width > v41RoutedOutputMaxBytes/4 {
				if owner != nil {
					t.Fatal("over-cap row admitted")
				}
			} else if owner == nil || owner.capRows != min(1536, v41RoutedOutputMaxBytes/4/width) || owner.capRows*width*4 > v41RoutedOutputMaxBytes {
				t.Fatalf("wrong actual geometry cap at H=%d: %+v", width, owner)
			}
			owner.release(0, true)
			s.closeBackendOwned()
			scope.finish()
		}
	})
}
