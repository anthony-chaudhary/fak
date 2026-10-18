package agent

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

type dflash2ScriptRequest struct {
	rounds                               [][]int
	committed                            [][]int
	prepareErr                           error
	commitErr                            error
	closeFault                           func() error
	abortFault                           func() error
	closes, aborts, bootstraps, prepares int
}

func (r *dflash2ScriptRequest) Bootstrap(context.Context, []int) error { r.bootstraps++; return nil }
func (r *dflash2ScriptRequest) Close() error {
	r.closes++
	if r.closeFault != nil {
		return r.closeFault()
	}
	return nil
}
func (r *dflash2ScriptRequest) Prepare(_ context.Context, n int) (dflash2Prepared, error) {
	r.prepares++
	if r.prepareErr != nil {
		return nil, r.prepareErr
	}
	if len(r.rounds) == 0 {
		return nil, errors.New("exhausted script")
	}
	tokens := r.rounds[0]
	r.rounds = r.rounds[1:]
	if len(tokens) > n {
		tokens = tokens[:n]
	}
	return &dflash2ScriptRound{parent: r, tokens: tokens}, nil
}

type dflash2ScriptRound struct {
	parent *dflash2ScriptRequest
	tokens []int
}

func (r *dflash2ScriptRound) Tokens() []int { return append([]int(nil), r.tokens...) }
func (r *dflash2ScriptRound) Commit(_ context.Context, n int) ([]int, error) {
	if r.parent.commitErr != nil {
		return nil, r.parent.commitErr
	}
	out := append([]int(nil), r.tokens[:n]...)
	r.parent.committed = append(r.parent.committed, out)
	return out, nil
}
func (r *dflash2ScriptRound) Abort() error {
	r.parent.aborts++
	if r.parent.abortFault != nil {
		return r.parent.abortFault()
	}
	return nil
}

func dflash2ScriptPlanner(t *testing.T, r *dflash2ScriptRequest) *InKernelPlanner {
	t.Helper()
	p, target := dflash2ServingFixture(t)
	target.factory = func(_ context.Context, l *dflash2Lease) error { l.request = r; return nil }
	return p
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2DriverCommitBeforePublication(t *testing.T) {
	for _, tc := range []struct {
		name    string
		tokens  []int
		stopIDs map[int]bool
		pieces  map[int]string
		want    []int
		stop    string
		visible string
	}{
		{"EOS", []int{1, 3, 2}, map[int]bool{3: true}, map[int]string{1: "a"}, []int{1}, "", "a"},
		{"EOS-first", []int{3, 1}, map[int]bool{3: true}, nil, nil, "", ""},
		{"string", []int{1, 2, 3}, nil, map[int]string{1: "a", 2: "END", 3: "lost"}, []int{1, 2}, "END", "a"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &dflash2ScriptRequest{rounds: [][]int{tc.tokens}}
			p := dflash2ScriptPlanner(t, r)
			scanner := newIncrementalStopScanner(len(tc.stop), []string{tc.stop})
			var visible strings.Builder
			res, err := p.generateDFlash2(context.Background(), []int{0}, 8, func(ids []int) (dflash2StopPreview, error) {
				return previewDFlash2Stop(ids, tc.stopIDs, scanner, func(ids []int) (string, error) { return tc.pieces[ids[0]], nil })
			}, func(plan dflash2StopPreview) error {
				if len(r.committed) != 1 {
					t.Fatal("publication before commit")
				}
				for _, s := range plan.pieces {
					visible.WriteString(s)
				}
				if plan.trim > 0 {
					text := visible.String()
					visible.Reset()
					visible.WriteString(text[:len(text)-plan.trim])
				}
				return nil
			})
			if err != nil || !res.stopped || res.gen != len(tc.want) || !reflect.DeepEqual(r.committed[0], tc.want) || visible.String() != tc.visible {
				t.Fatalf("result=%+v err=%v commits=%v text=%q", res, err, r.committed, visible.String())
			}
			if r.closes != 1 || r.aborts != 0 || r.bootstraps != 1 {
				t.Fatalf("lifecycle %+v", r)
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2DriverNoPublishOnFailure(t *testing.T) {
	cause := errors.New("sentinel")
	for _, stage := range []string{"prepare", "decode", "commit"} {
		t.Run(stage, func(t *testing.T) {
			r := &dflash2ScriptRequest{rounds: [][]int{{1, 2}}}
			if stage == "prepare" {
				r.prepareErr = cause
			}
			if stage == "commit" {
				r.commitErr = cause
			}
			p := dflash2ScriptPlanner(t, r)
			published := 0
			_, err := p.generateDFlash2(context.Background(), []int{0}, 2, func(ids []int) (dflash2StopPreview, error) {
				return previewDFlash2Stop(ids, nil, newIncrementalStopScanner(0, nil), func([]int) (string, error) {
					if stage == "decode" {
						return "", cause
					}
					return "a", nil
				})
			}, func(dflash2StopPreview) error { published++; return nil })
			if !errors.Is(err, cause) || published != 0 || len(r.committed) != 0 || r.closes != 1 {
				t.Fatalf("err=%v publication=%d request=%+v", err, published, r)
			}
			wantAbort := 1
			if stage == "prepare" {
				wantAbort = 0
			}
			if r.aborts != wantAbort {
				t.Fatalf("abort=%d", r.aborts)
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2DriverCrossBlockStreamStop(t *testing.T) {
	r := &dflash2ScriptRequest{rounds: [][]int{{1, 2}, {3, 4}}}
	p := dflash2ScriptPlanner(t, r)
	scanner := newIncrementalStopScanner(len("界END"), []string{"界END"})
	pieces := map[int]string{1: "a", 2: "界E", 3: "ND", 4: "never"}
	var visible strings.Builder
	projector := newInKernelStreamProjector(func(s string) error { visible.WriteString(s); return nil }, []string{"界END"})
	res, err := p.generateDFlash2(context.Background(), []int{0}, 8, func(ids []int) (dflash2StopPreview, error) {
		return previewDFlash2Stop(ids, nil, scanner, func(ids []int) (string, error) { return pieces[ids[0]], nil })
	}, func(plan dflash2StopPreview) error {
		for _, piece := range plan.pieces {
			if err := projector.feed(piece); err != nil {
				return err
			}
		}
		scanner = plan.scanner
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := projector.finish("a", false); err != nil {
		t.Fatal(err)
	}
	if visible.String() != "a" || res.gen != 3 || !res.stopped || !reflect.DeepEqual(r.committed, [][]int{{1, 2}, {3}}) {
		t.Fatalf("text=%q result=%+v commits=%v", visible.String(), res, r.committed)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2DriverSinkInterruptionAfterCommit(t *testing.T) {
	for _, fault := range []string{"error", "cancel", "panic", "nil-panic", "goexit"} {
		t.Run(fault, func(t *testing.T) {
			if fault == "nil-panic" {
				t.Setenv("GODEBUG", "panicnil=1")
			}
			r := &dflash2ScriptRequest{rounds: [][]int{{1, 2}, {3}}}
			p := dflash2ScriptPlanner(t, r)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			done := make(chan struct{})
			returned := false
			var got error
			go func() {
				defer close(done)
				defer func() { _ = recover() }()
				_, got = p.generateDFlash2(ctx, []int{0}, 3, func(ids []int) (dflash2StopPreview, error) {
					return previewDFlash2Stop(ids, nil, newIncrementalStopScanner(0, nil), func([]int) (string, error) { return "a", nil })
				}, func(dflash2StopPreview) error {
					if len(r.committed) != 1 {
						t.Error("sink ran before commit")
					}
					switch fault {
					case "error":
						return errors.New("sink")
					case "cancel":
						cancel()
						return ctx.Err()
					case "panic":
						panic("sink")
					case "nil-panic":
						panic(nil)
					default:
						runtime.Goexit()
					}
					return nil
				})
				returned = true
			}()
			<-done
			if r.closes != 1 || len(r.committed) != 1 || r.aborts != 0 || len(p.dflash2.leases) != 0 {
				t.Fatalf("lost request retirement %+v", r)
			}
			if (fault == "error" || fault == "cancel") && (!returned || got == nil) {
				t.Fatal("sink failure hidden")
			}
			if fault == "cancel" && !errors.Is(got, context.Canceled) {
				t.Fatalf("cancel %v", got)
			}
			if fault == "goexit" && returned {
				t.Fatal("Goexit swallowed")
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2RequestConstraints(t *testing.T) {
	p, _ := dflash2ServingFixture(t)
	one := 1.0
	for _, sp := range []SampleParams{{TopP: &one}, {Temperature: &one}, {NativeInferenceReceipt: true}, {RawPrompt: true}, {DecodeTrace: true}, {LogitBias: map[int]float64{1: 1}}} {
		temp := 0.0
		if sp.Temperature != nil {
			temp = *sp.Temperature
		}
		if err := p.validateDFlash2Request(sp, temp, 0, 0); err == nil {
			t.Fatalf("constraint admitted %+v", sp)
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2PrimaryUnwindSurvivesCleanupPanic(t *testing.T) {
	for _, mode := range []string{"0", "1"} {
		t.Run("panicnil="+mode, func(t *testing.T) {
			t.Setenv("GODEBUG", "panicnil="+mode)
			for _, stage := range []string{"preview-abort", "preview-close", "sink-close"} {
				for _, primaryKind := range []string{"panic", "nil-panic", "goexit"} {
					for _, cleanupKind := range []string{"panic", "nil-panic"} {
						t.Run(stage+"/"+primaryKind+"/"+cleanupKind, func(t *testing.T) {
							primary, secondary := errors.New("primary"), errors.New("secondary")
							fail := func() {
								switch primaryKind {
								case "panic":
									panic(primary)
								case "nil-panic":
									panic(nil)
								default:
									runtime.Goexit()
								}
							}
							cleanupCalls := 0
							cleanup := func() error {
								cleanupCalls++
								if cleanupKind == "panic" {
									panic(secondary)
								}
								panic(nil)
							}
							r := &dflash2ScriptRequest{rounds: [][]int{{1, 2}}}
							if stage == "preview-abort" {
								r.abortFault = cleanup
							} else {
								r.closeFault = cleanup
							}
							p := dflash2ScriptPlanner(t, r)
							done := make(chan struct{})
							bodyReturned, outerReturned := false, false
							var recovered any
							go func() {
								defer close(done)
								func() {
									defer func() { recovered = recover() }()
									_, _ = p.generateDFlash2(context.Background(), []int{0}, 2, func(ids []int) (dflash2StopPreview, error) {
										if strings.HasPrefix(stage, "preview") {
											fail()
										}
										return previewDFlash2Stop(ids, nil, newIncrementalStopScanner(0, nil), func([]int) (string, error) { return "a", nil })
									}, func(dflash2StopPreview) error { fail(); return nil })
									bodyReturned = true
								}()
								outerReturned = true
							}()
							<-done
							if bodyReturned || cleanupCalls != 1 || r.closes != 1 || p.dflash2.domain.failure == nil {
								t.Fatalf("returned=%v cleanup=%d closes=%d quarantine=%v", bodyReturned, cleanupCalls, r.closes, p.dflash2.domain.failure)
							}
							if primaryKind == "goexit" {
								if outerReturned || recovered != nil {
									t.Fatalf("Goexit replaced: returned=%v panic=%v", outerReturned, recovered)
								}
							} else {
								if !outerReturned {
									t.Fatal("primary panic turned into Goexit")
								}
								if primaryKind == "panic" && recovered != primary {
									t.Fatalf("primary replaced by %v", recovered)
								}
								if primaryKind == "nil-panic" && mode == "1" && recovered != nil {
									t.Fatalf("legacy nil replaced by %v", recovered)
								}
								if primaryKind == "nil-panic" && mode == "0" {
									if _, ok := recovered.(*runtime.PanicNilError); !ok {
										t.Fatalf("nil panic type %T", recovered)
									}
								}
							}
						})
					}
				}
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2NormalBodyCleanupFailureReturnsError(t *testing.T) {
	r := &dflash2ScriptRequest{rounds: [][]int{{1}}, closeFault: func() error { panic("close") }}
	p := dflash2ScriptPlanner(t, r)
	_, err := p.generateDFlash2(context.Background(), []int{0}, 1, func(ids []int) (dflash2StopPreview, error) {
		return previewDFlash2Stop(ids, nil, newIncrementalStopScanner(0, nil), func([]int) (string, error) { return "a", nil })
	}, func(dflash2StopPreview) error { return nil })
	if err == nil || r.closes != 1 || p.dflash2.domain.failure == nil {
		t.Fatalf("cleanup error=%v closes=%d", err, r.closes)
	}
}

func dflash2WiredPlanner(t *testing.T, r *dflash2ScriptRequest) *InKernelPlanner {
	t.Helper()
	p := dflash2ScriptPlanner(t, r)
	// Standard byte-level vocabulary, plus native ChatML controls, so the actual
	// Complete renderer and tokenizer execute without a prompt hook.
	tokens := make([]string, 256)
	types := make([]int32, 256)
	next := 256
	for b := 0; b < 256; b++ {
		code := b
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			code = next
			next++
		}
		tokens[b] = string(rune(code))
		types[b] = 1
	}
	tokens = append(tokens, "<|im_start|>", "<|im_end|>", "<|endoftext|>")
	types = append(types, 3, 3, 3)
	tok, err := tokenizer.FromGGML(tokens, nil, types, "qwen2")
	if err != nil {
		t.Fatal(err)
	}
	base, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Fatal("cpu-ref required")
	}
	p.backend.(*dflash2GateBackend).Backend = base
	p.tok = tok
	p.dflash2.tok = tok
	p.dflash2.spec.TokenizerID = tok.Identity()
	m := model.NewSynthetic(model.Config{HiddenSize: 32, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 32, IntermediateSize: 64, VocabSize: len(tokens), EOSTokenID: 257, MaxPositionEmbeddings: 4096, RMSNormEps: 1e-5, RopeTheta: 10000})
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	})
	p.m = m
	p.dflash2.m = m
	p.maxNew = 2
	p.contextTokens = 4096
	p.modelID = "dflash2-driver-fixture"
	p.dflash2.fingerprint, err = dflash2TargetFingerprint(p.m, tok)
	if err != nil {
		t.Fatal(err)
	}
	p.dflash2.stops = dflash2StopIDs(tok, p.m.Cfg)
	return p
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2CompleteAndStreamUseRequestOwner(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprint(stream), func(t *testing.T) {
			r := &dflash2ScriptRequest{rounds: [][]int{{int('a'), int('b')}}}
			p := dflash2WiredPlanner(t, r)
			var comp *Completion
			var err error
			var text strings.Builder
			if stream {
				comp, err = p.CompleteStream(context.Background(), func(s string) error { text.WriteString(s); return nil }, []Message{{Role: "user", Content: "hello"}}, nil, WithPerTokenStream(true))
			} else {
				comp, err = p.Complete(context.Background(), []Message{{Role: "user", Content: "hello"}}, nil)
			}
			if err != nil {
				t.Fatal(err)
			}
			if comp == nil || comp.Message.Content != "ab" || r.bootstraps != 1 || r.closes != 1 || len(r.committed) != 1 || comp.Timings != nil || comp.NativeDecode.Path != NativeDecodePathSpeculative {
				t.Fatalf("completion=%+v request=%+v", comp, r)
			}
			if stream && text.String() != "ab" {
				t.Fatalf("stream=%q", text.String())
			}
			release, err := p.acquireDFlash2Domain(context.Background())
			if err != nil {
				t.Fatalf("gate retained: %v", err)
			}
			release()
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2CompleteStreamSinkCancelAndEarlyStopBound(t *testing.T) {
	r := &dflash2ScriptRequest{rounds: [][]int{{int('a'), int('b')}, {int('c'), int('d')}}}
	p := dflash2WiredPlanner(t, r)
	p.maxNew = 4
	cause := errors.New("sink failure")
	calls := 0
	_, err := p.CompleteStream(context.Background(), func(string) error {
		calls++
		if r.closes != 0 || len(r.committed) != 1 || r.prepares != 1 {
			t.Fatalf("sink was buffered past owner retirement: %+v", r)
		}
		return cause
	}, []Message{{Role: "user", Content: "hello"}}, nil, WithPerTokenStream(true))
	if !errors.Is(err, cause) || calls != 1 || r.closes != 1 || len(r.committed) != 1 || r.prepares != 1 || len(r.rounds) != 1 {
		t.Fatalf("sink err=%v calls=%d request=%+v", err, calls, r)
	}
	release, err := p.acquireDFlash2Domain(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	release()
	// A deliberately incomplete planner proves refusal occurs before renderer,
	// tokenizer, projector or request allocation.
	early := &InKernelPlanner{dflash2: &DFlash2Target{}}
	oversized := strings.Repeat("x", (64<<10)+1)
	if _, err := early.CompleteStream(context.Background(), func(string) error { t.Fatal("sink called"); return nil }, nil, nil, func(sp *SampleParams) { sp.Stop = []string{oversized} }); err == nil || !strings.Contains(err.Error(), "stop byte budget") {
		t.Fatalf("early stop refusal %v", err)
	}
	if err := validateDFlash2Stops(make([]string, 1025)); err == nil {
		t.Fatal("stop count budget omitted")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestDFlash2CleanupGoexitStillRetiresAndReleasesGate(t *testing.T) {
	for _, stage := range []string{"abort-clean-close", "abort-failed-close", "close"} {
		t.Run(stage, func(t *testing.T) {
			r := &dflash2ScriptRequest{rounds: [][]int{{1}}}
			if stage == "close" {
				r.closeFault = func() error { runtime.Goexit(); return nil }
			} else {
				r.abortFault = func() error { runtime.Goexit(); return nil }
				if stage == "abort-failed-close" {
					r.closeFault = func() error { panic("close during abort Goexit") }
				}
			}
			p := dflash2ScriptPlanner(t, r)
			done := make(chan struct{})
			returned := false
			var recovered any
			go func() {
				defer close(done)
				defer func() { recovered = recover() }()
				release, err := p.acquireDFlash2Domain(context.Background())
				if err != nil {
					t.Error(err)
					return
				}
				defer release()
				_, _ = p.generateDFlash2(context.Background(), []int{0}, 1, func(ids []int) (dflash2StopPreview, error) {
					if stage != "close" {
						return dflash2StopPreview{}, errors.New("preview error")
					}
					return previewDFlash2Stop(ids, nil, newIncrementalStopScanner(0, nil), func([]int) (string, error) { return "a", nil })
				}, func(dflash2StopPreview) error { return nil })
				returned = true
			}()
			<-done
			if returned || recovered != nil || r.closes != 1 || len(p.dflash2.domain.slot) != 0 {
				t.Fatalf("returned=%v panic=%v closes=%d gate=%d", returned, recovered, r.closes, len(p.dflash2.domain.slot))
			}
			release, err := p.acquireDFlash2Domain(context.Background())
			if stage == "abort-clean-close" {
				if err != nil {
					t.Fatalf("complete outer retirement not reusable: %v", err)
				}
				release()
				if r.aborts != 1 || len(p.dflash2.leases) != 0 {
					t.Fatal("abort Goexit lost outer owner")
				}
			} else {
				if !errors.Is(err, ErrDFlash2Quarantined) || len(p.dflash2.leases) != 1 {
					t.Fatalf("incomplete close escaped: %v", err)
				}
			}
		})
	}
}
