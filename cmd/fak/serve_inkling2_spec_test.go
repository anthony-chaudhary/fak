package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

type inkling2ServeSpecFakeSession struct {
	depth       int
	plainVerify bool
	nextTok     func(pos, tok int) int
	draftFn     func(history []int, k int) []int
	verifyErr   error

	pos        int
	startPos   int
	ckPos      int
	ckHorizon  int
	hasCk      bool
	warmed     int
	startCount int
	verifies   int
	steps      int
	widthOnes  int
	accepted   int
	closed     bool
}

func (s *inkling2ServeSpecFakeSession) Prefill(_ context.Context, prompt []int) (int, error) {
	var out int
	for _, tok := range prompt {
		out = s.nextTok(s.pos, tok)
		s.pos++
	}
	return out, nil
}

func (s *inkling2ServeSpecFakeSession) Step(tok int) (int, error) {
	s.steps++
	out := s.nextTok(s.pos, tok)
	s.pos++
	return out, nil
}

func (s *inkling2ServeSpecFakeSession) LastLogits() ([]float32, error) {
	return []float32{0.1, 0.2, 0.9}, nil
}

func (s *inkling2ServeSpecFakeSession) Close() error { s.closed = true; return nil }
func (s *inkling2ServeSpecFakeSession) Pos() int     { return s.pos }

func (s *inkling2ServeSpecFakeSession) Checkpoint(horizon int) error {
	s.ckPos, s.ckHorizon, s.hasCk = s.pos, max(horizon, 1), true
	return nil
}

func (s *inkling2ServeSpecFakeSession) Restore() error {
	if !s.hasCk || s.pos-s.ckPos > s.ckHorizon {
		s.hasCk = false
		return ggufload.ErrInkling2NoCheckpoint
	}
	s.pos = s.ckPos
	return nil
}

func (s *inkling2ServeSpecFakeSession) Reset() error {
	s.hasCk = false
	s.pos = 0
	return nil
}

func (s *inkling2ServeSpecFakeSession) SpecMaxDraft() int         { return s.depth }
func (s *inkling2ServeSpecFakeSession) SpecPlainUsesVerify() bool { return s.plainVerify }
func (s *inkling2ServeSpecFakeSession) SpecStart([]int) error {
	s.startPos = s.pos
	s.startCount++
	return nil
}
func (s *inkling2ServeSpecFakeSession) SpecWarm(int) error { s.warmed++; return nil }

func (s *inkling2ServeSpecFakeSession) SpecDraft(history []int, k int) []int {
	if s.draftFn != nil {
		return s.draftFn(history, k)
	}
	return nil
}

func (s *inkling2ServeSpecFakeSession) SpecVerify(tokens []int) ([]int, error) {
	s.verifies++
	if len(tokens) == 1 {
		s.widthOnes++
	}
	if s.verifyErr != nil {
		return nil, s.verifyErr
	}
	out := make([]int, len(tokens))
	pos := s.pos
	for i, tok := range tokens {
		out[i] = s.nextTok(pos, tok)
		pos++
	}
	s.pos = pos
	return out, nil
}

func (s *inkling2ServeSpecFakeSession) SpecCommitVerify(T, keep int) error {
	if keep < 0 || keep > T {
		return fmt.Errorf("invalid keep=%d T=%d", keep, T)
	}
	s.accepted += max(keep-1, 0)
	s.pos -= T - keep
	return nil
}

func (s *inkling2ServeSpecFakeSession) SpecObserve([]int, int) error       { return nil }
func (s *inkling2ServeSpecFakeSession) SpecObserveCost(int, time.Duration) {}

func runServeCollect(t *testing.T, gen agent.Inkling2LoopGenerator, prompt []int, maxNew int, stops map[int]bool, emitStopAt int) (agent.Inkling2LoopGeneration, []int) {
	t.Helper()
	var toks []int
	g, err := gen.Generate(context.Background(), prompt, maxNew, stops, func(tok int) bool {
		toks = append(toks, tok)
		return !(emitStopAt > 0 && len(toks) == emitStopAt)
	})
	if err != nil {
		t.Fatalf("Generate(maxNew=%d): %v", maxNew, err)
	}
	return g, toks
}

func TestInkling2ServeSpecParityVsPlainStep(t *testing.T) {
	nextTok := func(pos, tok int) int {
		return 100 + ((pos*17 + tok*31 + 7) % 89)
	}
	plainEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		return &inkling2ServeSpecFakeSession{nextTok: nextTok}, nil
	})
	plainEng.prefixReuse = inkling2PrefixReuseOff
	defer plainEng.Close()

	// Exercise mixed accepts, rejects, nil fallback, empty-non-nil width-1 verify, and over-proposal clamping.
	prompt := []int{5, 6, 7}
	var lastSess *inkling2ServeSpecFakeSession
	var totalVerifies, totalSteps, totalWidthOnes, totalAccepted int
	specEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		lastSess = &inkling2ServeSpecFakeSession{
			depth:   4,
			nextTok: nextTok,
			draftFn: func(history []int, k int) []int {
				step := len(history)
				switch step % 5 {
				case 0:
					totalSteps++
					return nil // falls back to Step
				case 1:
					return []int{} // empty non-nil runs width-1 SpecVerify
				case 2:
					// over-propose (>k) exact match on first 2, mismatch on 3rd
					pos := len(prompt) + len(history) - 1
					tok := history[len(history)-1]
					out := make([]int, k+2)
					for i := range out {
						tok = nextTok(pos+i, tok)
						if i == 2 {
							out[i] = tok + 1
						} else {
							out[i] = tok
						}
					}
					return out
				default:
					// exact full match
					pos := len(prompt) + len(history) - 1
					tok := history[len(history)-1]
					out := make([]int, k)
					for i := range out {
						tok = nextTok(pos+i, tok)
						out[i] = tok
					}
					return out
				}
			},
		}
		return lastSess, nil
	})
	specEng.prefixReuse = inkling2PrefixReuseOff
	defer specEng.Close()

	for _, tc := range []struct {
		name       string
		maxNew     int
		stops      map[int]bool
		emitStopAt int
	}{
		{name: "maxNew-1", maxNew: 1},
		{name: "maxNew-2", maxNew: 2},
		{name: "maxNew-25", maxNew: 25},
		{name: "emit-false-inside-draft", maxNew: 16, emitStopAt: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pg, pt := runServeCollect(t, plainEng, prompt, tc.maxNew, tc.stops, tc.emitStopAt)
			sg, st := runServeCollect(t, specEng, prompt, tc.maxNew, tc.stops, tc.emitStopAt)
			if !slices.Equal(st, pt) {
				t.Fatalf("token mismatch:\nspec = %v\nplain= %v", st, pt)
			}
			if sg.Generated != pg.Generated || sg.Stopped != pg.Stopped {
				t.Fatalf("stats mismatch spec=%+v plain=%+v", sg, pg)
			}
			if lastSess != nil {
				totalVerifies += lastSess.verifies
				totalWidthOnes += lastSess.widthOnes
				totalAccepted += lastSess.accepted
			}
		})
	}
	if totalVerifies == 0 || totalWidthOnes == 0 || totalSteps == 0 || totalAccepted == 0 {
		t.Fatalf("expected verifies, width-1 verifies, plain fallback steps, and accepted drafts; verifies=%d widthOnes=%d nilFallbacks=%d accepted=%d",
			totalVerifies, totalWidthOnes, totalSteps, totalAccepted)
	}

	// Stop ID inside an accepted draft prefix (seq[2] from case 2) must trigger identical stop and count.
	_, seq := runServeCollect(t, plainEng, prompt, 10, nil, 0)
	stopTok := seq[2]
	pg, pt := runServeCollect(t, plainEng, prompt, 10, map[int]bool{stopTok: true}, 0)
	sg, st := runServeCollect(t, specEng, prompt, 10, map[int]bool{stopTok: true}, 0)
	if !slices.Equal(st, pt) || sg.Generated != pg.Generated || !sg.Stopped {
		t.Fatalf("stop-inside-draft mismatch: spec=(%+v %v) plain=(%+v %v)", sg, st, pg, pt)
	}

	// Non-positive maxNew must return cleanly without panicking, and ctx cancel during draft emit must return ctx.Err().
	var directRes agent.Inkling2LoopGeneration
	if fault, err := inkling2ServeSpecDecode(&inkling2LoopRequest{ctx: context.Background(), maxNew: -1}, lastSess, lastSess, 0, &directRes); fault || err != nil || directRes.Generated != 0 {
		t.Fatalf("maxNew=-1 spec decode = (%v, %v, %+v), want clean no-op", fault, err, directRes)
	}
	cctx, ccancel := context.WithCancel(context.Background())
	emitted := 0
	_, cerr := specEng.Generate(cctx, prompt, 16, nil, func(int) bool {
		emitted++
		if emitted == 3 {
			ccancel()
			return false
		}
		return true
	})
	if !errors.Is(cerr, context.Canceled) {
		t.Fatalf("emit-abort with canceled ctx err = %v, want context.Canceled", cerr)
	}
}

func TestInkling2ServeSpecSkipsSamplingFallback(t *testing.T) {
	nextTok := func(pos, tok int) int { return 10 + pos + tok }
	var sess *inkling2ServeSpecFakeSession
	eng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		sess = &inkling2ServeSpecFakeSession{
			depth:   4,
			nextTok: nextTok,
			draftFn: func(history []int, k int) []int { return make([]int, k) },
		}
		return sess, nil
	})
	defer eng.Close()

	var out []int
	_, err := eng.GenerateLogits(context.Background(), []int{1, 2}, 4, nil, func([]float32) (int, error) {
		return 7, nil
	}, func(tok int) bool {
		out = append(out, tok)
		return true
	})
	if err != nil {
		t.Fatalf("GenerateLogits err = %v", err)
	}
	if len(out) != 4 || sess.verifies != 0 || sess.steps != 3 {
		t.Fatalf("sampled fallback unexpectedly entered spec path: out=%v verifies=%d steps=%d", out, sess.verifies, sess.steps)
	}
}

func TestInkling2ServeSpecMultiTurnPrefixRewindParity(t *testing.T) {
	nextTok := func(pos, tok int) int { return 200 + ((pos*13 + tok*29 + 3) % 71) }
	plainEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		return &inkling2ServeSpecFakeSession{nextTok: nextTok}, nil
	})
	plainEng.prefixReuse = inkling2PrefixReuseOff
	defer plainEng.Close()

	var specSess *inkling2ServeSpecFakeSession
	specEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		specSess = &inkling2ServeSpecFakeSession{
			depth:   4,
			nextTok: nextTok,
			draftFn: func(history []int, k int) []int {
				// Drafts deliberately alternate accepts/rejects; position bounds must hold on rewind.
				out := make([]int, k)
				for i := range out {
					out[i] = history[len(history)-1] + i + 1
				}
				return out
			},
		}
		return specSess, nil
	})
	specEng.prefixReuse = inkling2PrefixReuseOn
	specEng.slots = 1
	defer specEng.Close()

	// Turn 1 (miss, checkpoint at len-1) -> Turn 2 (prefix hit, greedy) ->
	// Turn 3 (sampled, shares prefix) -> Turn 4 (prefix hit, greedy).
	p1 := []int{1, 2, 3, 4}
	p2 := []int{1, 2, 3, 9, 10}
	p3 := []int{1, 2, 3, 11}
	p4 := []int{1, 2, 3, 12, 13, 14}

	pg1, pt1 := runServeCollect(t, plainEng, p1, 6, nil, 0)
	sg1, st1 := runServeCollect(t, specEng, p1, 6, nil, 0)
	if !slices.Equal(st1, pt1) || sg1.Generated != pg1.Generated {
		t.Fatalf("turn 1 mismatch: spec=%v plain=%v", st1, pt1)
	}
	if specSess.Pos() > specSess.ckPos+specSess.ckHorizon {
		t.Fatalf("turn 1 exceeded checkpoint horizon: pos=%d ck=%d+%d", specSess.Pos(), specSess.ckPos, specSess.ckHorizon)
	}

	pg2, pt2 := runServeCollect(t, plainEng, p2, 5, nil, 0)
	sg2, st2 := runServeCollect(t, specEng, p2, 5, nil, 0)
	if !slices.Equal(st2, pt2) || sg2.CachedPromptTokens == 0 || sg2.Generated != pg2.Generated {
		t.Fatalf("turn 2 prefix spec mismatch: spec=(%+v %v) plain=(%+v %v)", sg2, st2, pg2, pt2)
	}

	var sampledOut []int
	_, err := specEng.GenerateLogits(context.Background(), p3, 3, nil, func([]float32) (int, error) {
		return 42, nil
	}, func(tok int) bool {
		sampledOut = append(sampledOut, tok)
		return true
	})
	if err != nil || len(sampledOut) != 3 {
		t.Fatalf("turn 3 sampled err=%v out=%v", err, sampledOut)
	}

	pg4, pt4 := runServeCollect(t, plainEng, p4, 7, nil, 0)
	sg4, st4 := runServeCollect(t, specEng, p4, 7, nil, 0)
	if !slices.Equal(st4, pt4) || sg4.CachedPromptTokens == 0 || sg4.Generated != pg4.Generated {
		t.Fatalf("turn 4 post-sampled prefix spec mismatch: spec=(%+v %v) plain=(%+v %v)", sg4, st4, pg4, pt4)
	}

	// Turn 5 (prefix miss -> Reset) -> Turn 6 (prefix hit extending Turn 5).
	p5 := []int{77, 88, 99, 100}
	p6 := []int{77, 88, 99, 100, 101, 102}
	pg5, pt5 := runServeCollect(t, plainEng, p5, 6, nil, 0)
	sg5, st5 := runServeCollect(t, specEng, p5, 6, nil, 0)
	if !slices.Equal(st5, pt5) || sg5.CachedPromptTokens != 0 || sg5.Generated != pg5.Generated {
		t.Fatalf("turn 5 miss spec mismatch: spec=(%+v %v) plain=(%+v %v)", sg5, st5, pg5, pt5)
	}
	pg6, pt6 := runServeCollect(t, plainEng, p6, 5, nil, 0)
	sg6, st6 := runServeCollect(t, specEng, p6, 5, nil, 0)
	if !slices.Equal(st6, pt6) || sg6.CachedPromptTokens == 0 || sg6.Generated != pg6.Generated {
		t.Fatalf("turn 6 post-miss prefix spec mismatch: spec=(%+v %v) plain=(%+v %v)", sg6, st6, pg6, pt6)
	}
}

func TestInkling2ServeSpecWarmupHorizonAndBatchClose(t *testing.T) {
	var sess *inkling2ServeSpecFakeSession
	scalar := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		sess = &inkling2ServeSpecFakeSession{
			depth:   4,
			nextTok: func(pos, tok int) int { return 50 + pos + tok },
		}
		return sess, nil
	})
	specClosed := false
	scalar.closeSpec = func() error { specClosed = true; return nil }

	_, _ = runServeCollect(t, scalar, []int{1, 2}, 2, nil, 0)
	if sess.warmed != 0 {
		t.Fatalf("SpecWarm called on maxNew=2 < depth+2 (%d times)", sess.warmed)
	}
	_, _ = runServeCollect(t, scalar, []int{1, 2}, 5, nil, 0)
	if sess.warmed != 0 {
		t.Fatalf("SpecWarm called on maxNew=5 == depth+1 (exceeds checkpoint horizon): %d times", sess.warmed)
	}
	gWarm, err := scalar.Generate(context.Background(), []int{1, 2}, 6, nil, func(tok int) bool {
		return true
	})
	if err != nil {
		t.Fatalf("Generate(maxNew=6): %v", err)
	}
	if sess.warmed != 1 {
		t.Fatalf("SpecWarm calls = %d on maxNew=6 >= depth+2, want 1", sess.warmed)
	}
	if gWarm.FirstEmitSeconds < gWarm.PrefillSeconds {
		t.Fatalf("FirstEmitSeconds = %v < PrefillSeconds %v under spec", gWarm.FirstEmitSeconds, gWarm.PrefillSeconds)
	}

	// Verify batch wrapper with specActive+loneCohort routes lone greedy requests to scalar and drains e.run() on Close.
	cohortsClosed := false
	batchCalled := false
	adm := inkling2Admission{gap: 0, maxWait: time.Millisecond, max: 4, clock: newFakeInkling2Clock()}
	batchEng := newInkling2LoopBatchEngineWith(scalar.Generate, func([]ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
		batchCalled = true
		return nil, nil
	}, adm, 1024)
	batchEng.scalarGenerateLogits = scalar.GenerateLogits
	batchEng.loneCohort = true
	batchEng.specActive = true
	batchEng.scalarClose = scalar.Close
	batchEng.cohortsClose = func() { cohortsClosed = true }
	go batchEng.run()

	g, out := runServeCollect(t, batchEng, []int{1, 2}, 3, nil, 0)
	if batchCalled || len(out) != 3 || g.Width != 1 {
		t.Fatalf("lone greedy under specActive did not route to scalar: batchCalled=%v g=%+v out=%v", batchCalled, g, out)
	}
	if err := batchEng.Close(); err != nil {
		t.Fatalf("batchEng.Close err = %v", err)
	}
	if err := batchEng.Close(); err != nil {
		t.Fatalf("second batchEng.Close err = %v", err)
	}
	if !cohortsClosed || !specClosed || !sess.closed {
		t.Fatalf("Close missing cleanup: cohorts=%v spec=%v sess=%v", cohortsClosed, specClosed, sess.closed)
	}
	if _, err := batchEng.Generate(context.Background(), []int{1, 2}, 2, nil, func(int) bool { return true }); err == nil {
		t.Fatal("Generate after Close succeeded, want error")
	}
	if _, err := batchEng.GenerateLogits(context.Background(), []int{1, 2}, 2, nil, func([]float32) (int, error) { return 0, nil }, func(int) bool { return true }); err == nil {
		t.Fatal("GenerateLogits after Close succeeded, want error")
	}
}

func TestInkling2ServeSpecPlainVerifyCancelAndFaultDrop(t *testing.T) {
	nextTok := func(pos, tok int) int { return 300 + ((pos*7 + tok*19 + 5) % 61) }

	// 1) plainVerify=true must route nil proposals through width-1 SpecVerify instead of plain Step.
	var pvSess *inkling2ServeSpecFakeSession
	pvEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		pvSess = &inkling2ServeSpecFakeSession{depth: 4, plainVerify: true, nextTok: nextTok}
		return pvSess, nil
	})
	defer pvEng.Close()
	_, got := runServeCollect(t, pvEng, []int{4, 5}, 4, nil, 0)
	if pvSess.widthOnes != 3 || pvSess.steps != 0 || len(got) != 4 {
		t.Fatalf("plainVerify routing: widthOnes=%d steps=%d out=%v", pvSess.widthOnes, pvSess.steps, got)
	}

	// 2) Mid-decode ctx cancel preserves slot checkpoint (fault=false), and next hit reuses prefix.
	ctx, cancel := context.WithCancel(context.Background())
	var cancelSess *inkling2ServeSpecFakeSession
	eng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		cancelSess = &inkling2ServeSpecFakeSession{
			depth:   4,
			nextTok: nextTok,
			draftFn: func(history []int, k int) []int {
				cancel()
				return make([]int, k)
			},
		}
		return cancelSess, nil
	})
	eng.prefixReuse = inkling2PrefixReuseOn
	eng.slots = 1
	defer eng.Close()

	_, err := eng.Generate(ctx, []int{1, 2, 3, 4}, 6, nil, func(int) bool { return true })
	if err != context.Canceled {
		t.Fatalf("mid-decode cancel err = %v, want context.Canceled", err)
	}
	if cancelSess.closed {
		t.Fatal("mid-decode ctx cancel closed reusable slot")
	}
	firstSess := cancelSess
	firstSess.draftFn = nil
	gen2, _ := runServeCollect(t, eng, []int{1, 2, 3, 9}, 3, nil, 0)
	if gen2.CachedPromptTokens == 0 || firstSess.closed {
		t.Fatalf("post-cancel prefix reuse failed: gen2=%+v closed=%v", gen2, firstSess.closed)
	}

	// 3) Verify device error drops faulted slot (fault=true).
	firstSess.verifyErr = fmt.Errorf("verify fault")
	firstSess.draftFn = func(history []int, k int) []int { return make([]int, k) }
	if _, err := eng.Generate(context.Background(), []int{1, 2, 3, 4}, 6, nil, func(int) bool { return true }); err == nil {
		t.Fatal("verify fault returned nil error")
	}
	if !firstSess.closed {
		t.Fatal("verify fault did not close faulted slot")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInkling2ServeSpecPreservesScalarPrefixAndDropsOnCohort(t *testing.T) {
	nextTok := func(pos, tok int) int { return 400 + ((pos*11 + tok*17 + 1) % 53) }
	var sessions []*inkling2ServeSpecFakeSession
	scalar := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		var s *inkling2ServeSpecFakeSession
		s = &inkling2ServeSpecFakeSession{
			depth:   4,
			nextTok: nextTok,
			draftFn: func(history []int, k int) []int {
				pos := s.startPos + len(history) - 1
				return []int{nextTok(pos, history[len(history)-1])}
			},
		}
		sessions = append(sessions, s)
		return s, nil
	})
	scalar.prefixReuse = inkling2PrefixReuseOn
	scalar.slots = 1

	var splitSawClosed, cohortSawClosed bool
	clk := newFakeInkling2Clock()
	adm := inkling2Admission{gap: 40 * time.Millisecond, maxWait: 100 * time.Millisecond, max: 4, clock: clk}
	batchEng := newInkling2LoopBatchEngineWith(scalar.Generate, func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
		cohortSawClosed = len(sessions) > 0 && sessions[0].closed
		out := make([]ggufload.Inkling2LoopBatchResult, len(reqs))
		for i, req := range reqs {
			for j := 0; j < req.MaxTokens; j++ {
				if req.Emit != nil {
					req.Emit(100 + j)
				}
			}
			out[i] = ggufload.Inkling2LoopBatchResult{Generated: req.MaxTokens}
		}
		return out, nil
	}, adm, 1024)
	batchEng.specActive = true
	batchEng.loneCohort = true
	batchEng.scalarClose = scalar.Close
	batchEng.scalarDropSlots = scalar.dropCachedSlots
	batchEng.split = func(positions []int) ([][]int, error) {
		splitSawClosed = len(sessions) > 0 && sessions[0].closed
		rows := make([]int, len(positions))
		for i := range positions {
			rows[i] = i
		}
		return [][]int{rows}, nil
	}
	go batchEng.run()
	defer batchEng.Close()

	collectFake := func(prompt []int, maxNew int) agent.Inkling2LoopGeneration {
		t.Helper()
		type out struct {
			g   agent.Inkling2LoopGeneration
			err error
		}
		ch := make(chan out, 1)
		go func() {
			g, err := batchEng.Generate(context.Background(), prompt, maxNew, nil, func(int) bool { return true })
			ch <- out{g, err}
		}()
		clk.waitArmed(t)
		clk.advance(40 * time.Millisecond)
		res := <-ch
		if res.err != nil {
			t.Fatalf("Generate(%v): %v", prompt, res.err)
		}
		return res.g
	}

	// Lone greedy turns keep scalar prefix slot open and hit cached tokens.
	g1 := collectFake([]int{10, 20, 30, 40}, 4)
	g2 := collectFake([]int{10, 20, 30, 40, 50}, 4)
	if len(sessions) != 1 || g1.Generated != 4 || g2.CachedPromptTokens == 0 || sessions[0].closed || sessions[0].accepted == 0 {
		t.Fatalf("spec lone greedy failed to preserve scalar prefix or accept drafts: opened=%d g1=%+v g2=%+v closed=%v accepted=%d",
			len(sessions), g1, g2, sessions[0].closed, sessions[0].accepted)
	}

	// Concurrent 2-request cohort must drop cached scalar slot before splitGroup/batchGenerate run.
	errCh := make(chan error, 2)
	sendCohortReq := func(tok int) {
		go func() {
			_, err := batchEng.Generate(context.Background(), []int{1, 2, tok}, 2, nil, func(int) bool { return true })
			errCh <- err
		}()
	}
	sendCohortReq(1)
	clk.waitArmed(t)
	sendCohortReq(2)
	clk.waitArmed(t)
	clk.advance(40 * time.Millisecond)
	for i := 0; i < 2; i++ {
		if err := <-errCh; err != nil {
			t.Fatalf("cohort Generate %d err = %v", i, err)
		}
	}
	if !splitSawClosed || !cohortSawClosed || !sessions[0].closed {
		t.Fatalf("cohort ran without dropping scalar slot before split/batch: splitClosed=%v batchClosed=%v slotClosed=%v",
			splitSawClosed, cohortSawClosed, sessions[0].closed)
	}

	// Next lone greedy opens a clean replacement scalar slot.
	g3 := collectFake([]int{7, 8, 9}, 3)
	if len(sessions) != 2 || g3.Generated != 3 || sessions[1].closed {
		t.Fatalf("post-cohort lone greedy did not reopen scalar slot: opened=%d g3=%+v", len(sessions), g3)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestInkling2ServeBatchCloseConcurrentEnqueueRace(t *testing.T) {
	for iter := 0; iter < 5; iter++ {
		scalar := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
			return &inkling2ServeSpecFakeSession{
				depth:   4,
				nextTok: func(pos, tok int) int { return pos + tok + 1 },
			}, nil
		})
		adm := inkling2Admission{gap: 0, maxWait: time.Millisecond, max: 8, clock: newFakeInkling2Clock()}
		batchEng := newInkling2LoopBatchEngineWith(scalar.Generate, func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			out := make([]ggufload.Inkling2LoopBatchResult, len(reqs))
			for i, req := range reqs {
				if req.Emit != nil {
					req.Emit(1)
				}
				out[i] = ggufload.Inkling2LoopBatchResult{Generated: 1}
			}
			return out, nil
		}, adm, 1024)
		batchEng.specActive = true
		batchEng.loneCohort = true
		batchEng.scalarClose = scalar.Close
		go batchEng.run()

		var wg sync.WaitGroup
		errs := make(chan error, 8)
		for g := 0; g < 8; g++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, err := batchEng.Generate(context.Background(), []int{1, 2, 3}, 1, nil, func(int) bool { return true })
				errs <- err
			}()
		}
		if err := batchEng.Close(); err != nil {
			t.Fatalf("iter %d Close err = %v", iter, err)
		}
		wg.Wait()
		close(errs)
		for err := range errs {
			if err != nil && !errors.Is(err, errInkling2LoopEngineClosed) {
				t.Fatalf("iter %d unexpected Generate err = %v", iter, err)
			}
		}
	}
}
