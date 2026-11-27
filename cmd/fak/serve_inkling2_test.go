package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// inkling2FakeTokenizer has one id per printable ASCII byte plus the Inkling-2
// specials, so any canned turn encodes and decodes exactly.
func inkling2FakeTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{"Ġ": 0, "Ċ": 1}
	for c := 33; c < 127; c++ {
		vocab[string(rune(c))] = c
	}
	var added []map[string]any
	for i, s := range []string{"<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
		"<|content_text|>", "<|content_thinking|>", "<|content_invoke_tool_json|>", "<|content_xml|>",
		"<|end_message|>", "<|content_model_end_sampling|>"} {
		added = append(added, map[string]any{"id": 200 + i, "content": s, "special": true})
	}
	raw, err := json.Marshal(map[string]any{
		"model":        map[string]any{"type": "BPE", "vocab": vocab, "merges": []string{}},
		"decoder":      map[string]any{"type": "ByteLevel"},
		"added_tokens": added,
	})
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.ParseJSON(raw)
	if err != nil {
		t.Fatalf("parse fake tokenizer: %v", err)
	}
	return tok
}

// inkling2FakeSession replays a canned token stream: the Prefill that ends a
// request's prompt returns the first token, each Step the next. It keeps its
// history across Prefill calls, implements the rewinder, and records what it was
// fed and which rewinder calls ran.
type inkling2FakeSession struct {
	canned   []int
	i        int
	prompt   []int // every token prefilled since the last Reset/Restore point
	fed      []int
	pos      int
	ckPos    int
	ckLen    int
	hasCk    bool
	closed   bool
	stepErr  error
	ckErr    error
	resetErr error
	horizon  int
	held     int64 // bytes HeldBytes reports on top of one per position

	prefills [][]int
	calls    []string
	busy     atomic.Int32
	overlap  atomic.Bool
}

func (s *inkling2FakeSession) enter() func() {
	if s.busy.Add(1) != 1 {
		s.overlap.Store(true)
	}
	return func() { s.busy.Add(-1) }
}

func (s *inkling2FakeSession) Prefill(_ context.Context, prompt []int) (int, error) {
	defer s.enter()()
	s.prompt = append(s.prompt, prompt...)
	s.prefills = append(s.prefills, append([]int(nil), prompt...))
	s.pos += len(prompt)
	s.i, s.fed = 0, nil
	return s.canned[0], nil
}

func (s *inkling2FakeSession) Step(token int) (int, error) {
	defer s.enter()()
	if s.stepErr != nil {
		return 0, s.stepErr
	}
	s.fed = append(s.fed, token)
	s.pos++
	s.i++
	return s.canned[s.i], nil
}

func (s *inkling2FakeSession) Pos() int { return s.pos }

func (s *inkling2FakeSession) Checkpoint(horizon int) error {
	s.calls = append(s.calls, "checkpoint")
	if s.ckErr != nil {
		s.hasCk = false
		return s.ckErr
	}
	s.ckPos, s.ckLen, s.hasCk, s.horizon = s.pos, len(s.prompt), true, horizon
	return nil
}

func (s *inkling2FakeSession) Restore() error {
	s.calls = append(s.calls, "restore")
	if !s.hasCk || s.pos-s.ckPos > s.horizon {
		return ggufload.ErrInkling2NoCheckpoint
	}
	s.pos, s.prompt = s.ckPos, s.prompt[:s.ckLen]
	return nil
}

func (s *inkling2FakeSession) Reset() error {
	s.calls = append(s.calls, "reset")
	if s.resetErr != nil {
		return s.resetErr
	}
	s.pos, s.prompt, s.hasCk = 0, nil, false
	return nil
}

func (s *inkling2FakeSession) Close() error { s.closed = true; return nil }

// HeldBytes grows with the history so the cache sees a session's footprint change.
func (s *inkling2FakeSession) HeldBytes() int64 { return s.held + int64(s.pos) }

var _ inkling2Rewinder = (*inkling2FakeSession)(nil)

// inkling2PlainSession hides the rewinder so the engine must run it fresh per request.
type inkling2PlainSession struct{ s *inkling2FakeSession }

func (p inkling2PlainSession) Prefill(ctx context.Context, prompt []int) (int, error) {
	return p.s.Prefill(ctx, prompt)
}
func (p inkling2PlainSession) Step(token int) (int, error) { return p.s.Step(token) }
func (p inkling2PlainSession) Close() error                { return p.s.Close() }

func inkling2FakeOpener(canned []int, plain bool, opened *[]*inkling2FakeSession) inkling2LoopSessionOpener {
	return func() (ggufload.Inkling2LoopSession, error) {
		s := &inkling2FakeSession{canned: canned}
		*opened = append(*opened, s)
		if plain {
			return inkling2PlainSession{s}, nil
		}
		return s, nil
	}
}

// inkling2SlotEngine is a fake-session engine caching up to slots prefixes within
// capBytes, each session reporting held bytes plus one per position.
func inkling2SlotEngine(canned []int, slots int, capBytes, held int64) (*inkling2LoopEngine, *[]*inkling2FakeSession) {
	var opened []*inkling2FakeSession
	eng := &inkling2LoopEngine{slots: slots, capBytes: capBytes, open: func() (ggufload.Inkling2LoopSession, error) {
		s := &inkling2FakeSession{canned: canned, held: held}
		opened = append(opened, s)
		return s, nil
	}}
	return eng, &opened
}

func inkling2FakeEngine(canned []int) (*inkling2LoopEngine, *[]*inkling2FakeSession) {
	var opened []*inkling2FakeSession
	return &inkling2LoopEngine{open: inkling2FakeOpener(canned, false, &opened)}, &opened
}

func inkling2Gen(t *testing.T, eng *inkling2LoopEngine, prompt []int, maxNew int) (agent.Inkling2LoopGeneration, []int) {
	t.Helper()
	var got []int
	res, err := eng.Generate(context.Background(), prompt, maxNew, nil, func(id int) bool {
		got = append(got, id)
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	return res, got
}

func TestInkling2AdapterStopsOnEndOfTurnNotEndMessage(t *testing.T) {
	const endMessage, endTurn = 208, 209
	eng, opened := inkling2FakeEngine([]int{201, 204, 'h', 'i', endMessage, 201, 204, 'x', endMessage, endTurn, 'z', 'z'})
	var got []int
	res, err := eng.Generate(context.Background(), []int{200, 'q'}, 100, map[int]bool{endTurn: true}, func(id int) bool {
		got = append(got, id)
		return true
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []int{201, 204, 'h', 'i', endMessage, 201, 204, 'x', endMessage}
	if !res.Stopped || res.Generated != len(want) || !inkling2EqualInts(got, want) {
		t.Fatalf("emitted %v (stopped=%v n=%d), want %v stopped at the end-of-turn marker", got, res.Stopped, res.Generated, want)
	}
	if err := eng.Close(); err != nil {
		t.Fatal(err)
	}
	if s := (*opened)[0]; !s.closed || !inkling2EqualInts(s.prompt, []int{200, 'q'}) {
		t.Fatalf("session prompt %v closed=%v", s.prompt, s.closed)
	}
	if _, err := eng.Generate(context.Background(), []int{'p'}, 1, nil, func(int) bool { return true }); !errors.Is(err, errInkling2LoopEngineClosed) {
		t.Fatalf("Generate after Close = %v", err)
	}
}

func TestInkling2AdapterMaxTokensAndPrefixReuse(t *testing.T) {
	eng, opened := inkling2FakeEngine([]int{'a', 'b', 'c', 'd', 'e'})
	defer eng.Close()
	first := []int{'s', 'y', 's', 'u'}
	res, got := inkling2Gen(t, eng, first, 3)
	if res.Stopped || res.Generated != 3 || !inkling2EqualInts(got, []int{'a', 'b', 'c'}) || res.CachedPromptTokens != 0 {
		t.Fatalf("first request: emitted %v res=%+v, want a b c on the length limit, nothing cached", got, res)
	}
	s := (*opened)[0]
	if want := [][]int{{'s', 'y', 's'}, {'u'}}; len(s.prefills) != 2 || !inkling2EqualInts(s.prefills[0], want[0]) || !inkling2EqualInts(s.prefills[1], want[1]) {
		t.Fatalf("first request prefills %v, want P[:n-1] then the last token", s.prefills)
	}

	// extending the conversation reuses everything the checkpoint covers
	second := append(append([]int(nil), first...), 'a', 'b', 'c', 'n', 'x')
	s.prefills, s.calls = nil, nil
	res, got = inkling2Gen(t, eng, second, 2)
	if len(*opened) != 1 {
		t.Fatalf("opened %d sessions, want the persistent one", len(*opened))
	}
	if !inkling2EqualInts(got, []int{'a', 'b'}) || res.CachedPromptTokens != len(first)-1 {
		t.Fatalf("second request: emitted %v cached=%d, want a b with %d cached", got, res.CachedPromptTokens, len(first)-1)
	}
	if s.calls[0] != "restore" || len(s.prefills) != 2 || !inkling2EqualInts(s.prefills[0], second[len(first)-1:len(second)-1]) || !inkling2EqualInts(s.prefills[1], []int{'x'}) {
		t.Fatalf("second request calls %v prefills %v, want restore then only the suffix", s.calls, s.prefills)
	}
	if !inkling2EqualInts(s.prompt, second) || s.pos != len(second)+1 {
		t.Fatalf("session history %v pos %d after second request", s.prompt, s.pos)
	}

	// an identical repeat re-prefills only the last token
	s.prefills, s.calls = nil, nil
	res, _ = inkling2Gen(t, eng, second, 1)
	if res.CachedPromptTokens != len(second)-1 || s.calls[0] != "restore" || len(s.prefills) != 1 || !inkling2EqualInts(s.prefills[0], []int{'x'}) {
		t.Fatalf("repeat: cached=%d calls %v prefills %v", res.CachedPromptTokens, s.calls, s.prefills)
	}

	// divergence resets and prefills everything
	diverged := []int{'o', 't', 'h', 'e', 'r'}
	s.prefills, s.calls = nil, nil
	res, _ = inkling2Gen(t, eng, diverged, 1)
	if res.CachedPromptTokens != 0 || s.calls[0] != "reset" || len(s.prefills) != 2 || !inkling2EqualInts(s.prefills[0], diverged[:4]) {
		t.Fatalf("divergence: cached=%d calls %v prefills %v", res.CachedPromptTokens, s.calls, s.prefills)
	}
	if len(*opened) != 1 || s.closed {
		t.Fatalf("divergence reopened or closed the session: opened=%d closed=%v", len(*opened), s.closed)
	}

	// a one-token prompt checkpoints at position 0
	s.prefills, s.calls = nil, nil
	res, _ = inkling2Gen(t, eng, []int{'z'}, 1)
	if res.CachedPromptTokens != 0 || len(s.prefills) != 1 || !inkling2EqualInts(s.prefills[0], []int{'z'}) {
		t.Fatalf("one-token prompt: cached=%d prefills %v", res.CachedPromptTokens, s.prefills)
	}
}

// A checkpoint the device cannot hold costs only the next request's reuse, and a
// checkpoint overrun by more tokens than its horizon is a miss, not a fault.
func TestInkling2AdapterCheckpointUnavailableOrOverrunFallsBackToFullPrefill(t *testing.T) {
	eng, opened := inkling2FakeEngine([]int{'a', 'b', 'c', 'd', 'e'})
	defer eng.Close()
	first := []int{'s', 'y', 's', 'u'}
	inkling2Gen(t, eng, first, 3)
	s := (*opened)[0]
	if s.horizon != 3 {
		t.Fatalf("checkpoint horizon %d, want max_tokens 3", s.horizon)
	}
	s.ckErr = fmt.Errorf("wrapped: %w", ggufload.ErrInkling2CheckpointUnavailable)
	second := append(append([]int(nil), first...), 'a', 'b', 'c', 'n', 'x')
	res, got := inkling2Gen(t, eng, second, 2)
	if !inkling2EqualInts(got, []int{'a', 'b'}) || res.CachedPromptTokens != len(first)-1 {
		t.Fatalf("request with an unavailable checkpoint: emitted %v cached=%d", got, res.CachedPromptTokens)
	}
	s.ckErr, s.prefills, s.calls = nil, nil, nil
	if res, _ = inkling2Gen(t, eng, second, 2); res.CachedPromptTokens != 0 || s.calls[0] != "reset" || !inkling2EqualInts(s.prefills[0], second[:len(second)-1]) {
		t.Fatalf("after an unavailable checkpoint: cached=%d calls %v prefills %v, want a full prefill", res.CachedPromptTokens, s.calls, s.prefills)
	}
	s.horizon = 0
	s.prefills, s.calls = nil, nil
	if res, _ = inkling2Gen(t, eng, second, 2); res.CachedPromptTokens != 0 || len(s.calls) < 2 || s.calls[0] != "restore" || s.calls[1] != "reset" {
		t.Fatalf("overrun checkpoint: cached=%d calls %v, want restore miss then reset", res.CachedPromptTokens, s.calls)
	}
	if len(*opened) != 1 || s.closed {
		t.Fatalf("fallback reopened or closed the session: opened=%d closed=%v", len(*opened), s.closed)
	}

	// a reset that cannot get its zero fill reopens a fresh session instead of failing
	s.resetErr = ggufload.ErrInkling2CheckpointUnavailable
	if res, got := inkling2Gen(t, eng, []int{'o', 't', 'h'}, 2); !inkling2EqualInts(got, []int{'a', 'b'}) || res.CachedPromptTokens != 0 {
		t.Fatalf("reset without memory: emitted %v cached=%d", got, res.CachedPromptTokens)
	}
	if len(*opened) != 2 || !s.closed || !inkling2EqualInts((*opened)[1].prompt, []int{'o', 't', 'h'}) {
		t.Fatalf("reset without memory: opened=%d first closed=%v, want the request on a fresh session", len(*opened), s.closed)
	}
}

// inkling2GatedSession holds its first Prefill until the caller cancels or the test
// releases it, standing in for a long prompt on the device.
type inkling2GatedSession struct {
	*inkling2FakeSession
	block   bool
	entered chan struct{}
}

func (g *inkling2GatedSession) Prefill(ctx context.Context, prompt []int) (int, error) {
	if g.block {
		select {
		case g.entered <- struct{}{}:
		default:
		}
		<-ctx.Done()
		return 0, ctx.Err()
	}
	return g.inkling2FakeSession.Prefill(ctx, prompt)
}

func TestInkling2ServeGenerateCancelsDuringPrefill(t *testing.T) {
	entered := make(chan struct{}, 1)
	var opened []*inkling2GatedSession
	eng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		g := &inkling2GatedSession{
			inkling2FakeSession: &inkling2FakeSession{canned: []int{'a', 'b', 'c', 'd', 'e'}},
			block:               len(opened) == 0,
			entered:             entered,
		}
		opened = append(opened, g)
		return g, nil
	})
	defer eng.Close()

	prompt := []int{'s', 'y', 's', 'u'}
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := eng.Generate(ctx, prompt, 3, nil, func(int) bool { return true })
		errc <- err
	}()
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the first prefill never started")
	}
	cancel()
	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled Generate returned %v, want context.Canceled", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Generate still blocked 5s after the caller cancelled during prefill")
	}
	if !opened[0].closed {
		t.Fatal("the session whose prefill was cancelled was not dropped")
	}

	// the next request runs a full prefill on a fresh session, not on the partial state
	res, got := inkling2Gen(t, eng, prompt, 3)
	if !inkling2EqualInts(got, []int{'a', 'b', 'c'}) || res.CachedPromptTokens != 0 {
		t.Fatalf("request after a cancelled prefill: emitted %v cached=%d, want a b c with nothing reused", got, res.CachedPromptTokens)
	}
	if len(opened) != 2 {
		t.Fatalf("opened %d sessions, want a fresh one after the cancelled prefill", len(opened))
	}
	if s := opened[1]; len(s.prefills) != 2 || !inkling2EqualInts(s.prefills[0], prompt[:3]) || !inkling2EqualInts(s.prefills[1], prompt[3:]) {
		t.Fatalf("request after a cancelled prefill prefilled %v, want the whole prompt in order", s.prefills)
	}
}

func TestInkling2AdapterReuseOffOpensFreshSessionPerRequest(t *testing.T) {
	t.Setenv("FAK_INKLING2_PREFIX_REUSE", "0")
	var opened []*inkling2FakeSession
	eng := newInkling2LoopEngineFrom(inkling2FakeOpener([]int{'a', 'b', 'c', 'd'}, false, &opened))
	defer eng.Close()
	for req := 0; req < 2; req++ {
		if res, got := inkling2Gen(t, eng, []int{'p', 'q', 'r'}, 3); !inkling2EqualInts(got, []int{'a', 'b', 'c'}) || res.CachedPromptTokens != 0 {
			t.Fatalf("request %d: emitted %v res=%+v", req, got, res)
		}
	}
	if len(opened) != 2 {
		t.Fatalf("opened %d sessions, want one fresh session per request", len(opened))
	}
	for i, s := range opened {
		if !s.closed || len(s.calls) != 0 || len(s.prefills) != 2 || len(s.prefills[0]) != 2 || len(s.fed) != 2 {
			t.Fatalf("session %d: closed=%v calls=%v prefills=%v fed=%v, want closed, no rewinder calls, same split", i, s.closed, s.calls, s.prefills, s.fed)
		}
	}
}

func TestInkling2AdapterNonRewinderOpensFreshSessionPerRequest(t *testing.T) {
	var opened []*inkling2FakeSession
	eng := &inkling2LoopEngine{open: inkling2FakeOpener([]int{'a', 'b'}, true, &opened)}
	defer eng.Close()
	inkling2Gen(t, eng, []int{'p', 'q'}, 1)
	inkling2Gen(t, eng, []int{'p', 'q', 'r'}, 1)
	if len(opened) != 2 || !opened[0].closed || !opened[1].closed || len(opened[1].prefills) != 1 || len(opened[1].prefills[0]) != 3 {
		t.Fatalf("opened %d sessions, want two closed fresh sessions each prefilling the whole prompt once", len(opened))
	}
}

func TestInkling2AdapterEmitFalseAndStepErrorEndTheTurn(t *testing.T) {
	eng, _ := inkling2FakeEngine([]int{'a', 'b', 'c'})
	defer eng.Close()
	res, err := eng.Generate(context.Background(), []int{'p'}, 10, nil, func(int) bool { return false })
	if err != nil || !res.Stopped || res.Generated != 1 {
		t.Fatalf("emit=false: res=%+v err=%v", res, err)
	}
	boom := errors.New("device fault")
	var opened []*inkling2FakeSession
	failing := &inkling2LoopEngine{open: inkling2FakeOpener([]int{'a', 'b', 'c'}, false, &opened)}
	defer failing.Close()
	inkling2Gen(t, failing, []int{'p', 'q'}, 1)
	opened[0].stepErr = boom
	if _, err := failing.Generate(context.Background(), []int{'p', 'q', 'r'}, 10, nil, func(int) bool { return true }); !errors.Is(err, boom) {
		t.Fatalf("step error = %v, want %v", err, boom)
	}
	if !opened[0].closed {
		t.Fatal("faulted session was not closed")
	}
	res, _ = inkling2Gen(t, failing, []int{'p', 'q', 'r', 's'}, 1)
	if len(opened) != 2 || res.CachedPromptTokens != 0 || len(opened[1].prefills[0]) != 3 {
		t.Fatalf("after a fault: opened=%d cached=%d, want a fresh session with full prefill", len(opened), res.CachedPromptTokens)
	}
}

func TestInkling2AdapterEmitPanicFailsOnlyThatRequest(t *testing.T) {
	var opened []*inkling2FakeSession
	eng := &inkling2LoopEngine{open: inkling2FakeOpener([]int{'a', 'b', 'c'}, false, &opened)}
	defer eng.Close()
	_, err := eng.Generate(context.Background(), []int{'p'}, 10, nil, func(int) bool { panic("decode blew up") })
	if err == nil || !strings.Contains(err.Error(), "panicked") {
		t.Fatalf("emit panic: err = %v, want a panicked error", err)
	}
	if !opened[0].closed {
		t.Fatal("session used by the panicking request was not dropped")
	}
	if _, got := inkling2Gen(t, eng, []int{'p', 'q'}, 1); len(got) != 1 || len(opened) != 2 {
		t.Fatalf("after a panic: got=%v opened=%d, want the worker alive with a fresh session", got, len(opened))
	}
}

func TestInkling2AdapterSerializesConcurrentCallers(t *testing.T) {
	eng, opened := inkling2FakeEngine([]int{'a', 'b', 'c', 'd', 'e', 'f'})
	defer eng.Close()
	var wg sync.WaitGroup
	errs := make(chan error, 16)
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			prompt := []int{'p', 'q', 'a' + g%3}
			if _, err := eng.Generate(context.Background(), prompt, 4, nil, func(int) bool { return true }); err != nil {
				errs <- err
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}
	if len(*opened) != 1 || (*opened)[0].overlap.Load() {
		t.Fatalf("opened=%d overlap=%v, want one session used by one request at a time", len(*opened), (*opened)[0].overlap.Load())
	}
}

func TestInkling2AdapterQueuedCallerHonoursCancel(t *testing.T) {
	eng, _ := inkling2FakeEngine([]int{'a', 'b', 'c'})
	defer eng.Close()
	inFlight, release := make(chan struct{}), make(chan struct{})
	go func() {
		_, _ = eng.Generate(context.Background(), []int{'p'}, 2, nil, func(int) bool {
			close(inFlight)
			<-release
			return false
		})
	}()
	<-inFlight
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := eng.Generate(ctx, []int{'q'}, 1, nil, func(int) bool { return true })
		done <- err
	}()
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("queued cancel = %v, want context.Canceled", err)
	}
	close(release)
	inkling2Gen(t, eng, []int{'r'}, 1)
}

func TestInkling2AdapterPlannerRendersInklingAndLiftsToolCall(t *testing.T) {
	tok := inkling2FakeTokenizer(t)
	turn := `<|message_model|><|content_text|>ok<|end_message|><|message_model|>bash<|content_invoke_tool_json|>{"name":"bash","args":{"command":"pwd"}}<|end_message|><|content_model_end_sampling|>trailing`
	canned, err := tok.Encode(turn)
	if err != nil {
		t.Fatal(err)
	}
	eng, opened := inkling2FakeEngine(canned)
	p, err := newServeInkling2LoopPlanner(eng, tok, "inkling2-mini", 4096)
	if err != nil {
		t.Fatal(err)
	}
	tools := []agent.ToolDef{{Type: "function", Function: agent.ToolDefFunction{Name: "bash", Description: "run a command",
		Parameters: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}`)}}}
	comp, err := p.Complete(context.Background(), []agent.Message{{Role: "user", Content: "where am I"}}, tools)
	if err != nil {
		t.Fatal(err)
	}
	prompt, err := tok.Decode((*opened)[0].prompt)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prompt, "<|message_user|>") || strings.Contains(prompt, "<|im_start|>") || !strings.Contains(prompt, "tool_declare") {
		t.Fatalf("prompt did not render through the Inkling-2 template:\n%s", prompt)
	}
	if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].Function.Name != "bash" ||
		comp.Message.ToolCalls[0].Function.Arguments != `{"command":"pwd"}` || comp.FinishReason != "tool_calls" {
		t.Fatalf("completion = %+v, want one bash tool call", comp)
	}
	if strings.Contains(comp.Message.Content, "trailing") || strings.Contains(comp.Message.Content, "<|") {
		t.Fatalf("content leaked markers or post-stop text: %q", comp.Message.Content)
	}
	if comp.Model != "inkling2-mini" || comp.Usage.PromptTokens != len((*opened)[0].prompt) {
		t.Fatalf("model %q usage %+v", comp.Model, comp.Usage)
	}
}

func TestInkling2AdapterPlannerRefusesPromptPastContext(t *testing.T) {
	tok := inkling2FakeTokenizer(t)
	eng, opened := inkling2FakeEngine([]int{'a'})
	p, err := newServeInkling2LoopPlanner(eng, tok, "inkling2", 8)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Complete(context.Background(), []agent.Message{{Role: "user", Content: "a long enough prompt"}}, nil); err == nil {
		t.Fatal("prompt past the context window was accepted")
	}
	if len(*opened) != 0 {
		t.Fatal("refused request still opened a device session")
	}
}

func TestInkling2PlannerNeedsEndOfTurnToken(t *testing.T) {
	tok, err := tokenizer.ParseJSON([]byte(`{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"decoder":{"type":"ByteLevel"}}`))
	if err != nil {
		t.Fatal(err)
	}
	eng, _ := inkling2FakeEngine([]int{0})
	if _, err := newServeInkling2LoopPlanner(eng, tok, "inkling2", 64); err == nil {
		t.Fatal("tokenizer without the end-of-turn marker accepted")
	}
}

// writeInkling2HeaderGGUF writes a tensor-less GGUF v3 header carrying only
// general.architecture.
func writeInkling2HeaderGGUF(t *testing.T, arch string) string {
	t.Helper()
	var b bytes.Buffer
	w := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { w(uint64(len(s))); b.WriteString(s) }
	b.WriteString("GGUF")
	w(uint32(3))
	w(uint64(0))
	w(uint64(1))
	str("general.architecture")
	w(uint32(8))
	str(arch)
	path := filepath.Join(t.TempDir(), arch+".gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestInkling2GGUFSniffAndGenericLoaderRefusal(t *testing.T) {
	path := writeInkling2HeaderGGUF(t, serveInkling2LoopArch)
	if !serveIsInkling2GGUF(path) {
		t.Fatal("inkling2_loop header not recognised")
	}
	if serveIsInkling2GGUF(writeInkling2HeaderGGUF(t, "llama")) || serveIsInkling2GGUF(t.TempDir()) || serveIsInkling2GGUF("") {
		t.Fatal("non-looped input recognised as inkling2_loop")
	}
	_, _, _, _, _, err := loadServeInKernelModelPlacedErr(path, nil, false, false, 0, nil, 1, nil)
	var refusal *fakmodel.UnsupportedInkling2LoopExecutionError
	if !errors.As(err, &refusal) || refusal.Backend != "cpu" || !strings.Contains(err.Error(), "--backend cuda") {
		t.Fatalf("CPU load of an inkling2_loop GGUF = %v, want the typed cpu refusal naming the cuda route", err)
	}
	if err := serveRefuseInkling2LoopGeneric(nil); err == nil || strings.Contains(err.Error(), "not implemented") {
		t.Fatalf("generic refusal = %v", err)
	}
}

func inkling2EqualInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// The route verdict follows the build: a darwin/arm64 cgo binary admits the
// resident Metal forward, and every other build returns a typed refusal.
func TestServeInkling2MetalRouteFollowsBuild(t *testing.T) {
	path := writeInkling2HeaderGGUF(t, serveInkling2LoopArch)
	routed, err := serveInkling2MetalRoute(path, nil, true)
	if !routed {
		t.Fatal("inkling2_loop GGUF with Metal selected not routed to the resident Metal path")
	}
	if inkling2MetalResidentAvailable {
		if err != nil {
			t.Fatalf("available resident Metal target refused: %v", err)
		}
	} else {
		var refusal *inkling2MetalServeRefusal
		if !errors.As(err, &refusal) || !strings.Contains(err.Error(), "not available in this build") {
			t.Fatalf("unavailable resident Metal target = %v, want the typed build refusal", err)
		}
		if plan, err := serveInkling2MetalWeights(path, nil, true); plan != nil || !errors.As(err, &refusal) {
			t.Fatalf("admission plan on an unavailable target = %v, %v", plan, err)
		}
	}
	for name, off := range map[string]func() (bool, error){
		"metal not selected": func() (bool, error) { return serveInkling2MetalRoute(path, nil, false) },
		"other family": func() (bool, error) {
			return serveInkling2MetalRoute(writeInkling2HeaderGGUF(t, "llama"), nil, true)
		},
		"no path": func() (bool, error) { return serveInkling2MetalRoute("", nil, true) },
	} {
		if routed, err := off(); routed || err != nil {
			t.Fatalf("%s: routed=%v err=%v, want the route left alone", name, routed, err)
		}
	}
	if plan, err := serveInkling2MetalWeights(path, nil, false); plan != nil || err != nil {
		t.Fatalf("admission plan off the Metal route = %v, %v", plan, err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	first := filepath.Join(t.TempDir(), "m-00001-of-00002.gguf")
	if err := os.WriteFile(first, data, 0o644); err != nil {
		t.Fatal(err)
	}
	routed, err = serveInkling2MetalRoute(first, nil, true)
	var refusal *inkling2MetalServeRefusal
	if !routed || !errors.As(err, &refusal) {
		t.Fatalf("split GGUF: routed=%v err=%v, want a typed refusal", routed, err)
	}
	if inkling2MetalResidentAvailable && !strings.Contains(err.Error(), "split into 2 shards") {
		t.Fatalf("split refusal does not name the shard count: %v", err)
	}
}

func TestInkling2WarmUpPrefillsOnceAndReturnsTheSession(t *testing.T) {
	eng, opened := inkling2FakeEngine([]int{0})
	if err := warmInkling2Session(context.Background(), eng.open, inkling2FakeTokenizer(t)); err != nil {
		t.Fatal(err)
	}
	if len(*opened) != 1 || len((*opened)[0].prompt) == 0 || len((*opened)[0].fed) != 0 || !(*opened)[0].closed {
		t.Fatalf("warm-up sessions = %+v, want one prefilled, unstepped, closed session", *opened)
	}
	failing := func() (ggufload.Inkling2LoopSession, error) { return nil, errors.New("no device") }
	if err := warmInkling2Session(context.Background(), failing, inkling2FakeTokenizer(t)); err == nil {
		t.Fatal("warm-up ignored a session that would not open")
	}
}

// TestInkling2PrefillTimerIncludesSlotAcquire pins that PrefillSeconds, the prompt
// time the receipts report as prefill tok/s and TTFT, covers the wait to acquire a
// slot. A fresh slot opens a session, which is the slow part of acquire here.
func TestInkling2PrefillTimerIncludesSlotAcquire(t *testing.T) {
	const acquireWait = 200 * time.Millisecond
	var opened []*inkling2FakeSession
	eng := &inkling2LoopEngine{open: func() (ggufload.Inkling2LoopSession, error) {
		time.Sleep(acquireWait)
		s := &inkling2FakeSession{canned: []int{'a'}}
		opened = append(opened, s)
		return s, nil
	}}
	defer eng.Close()

	res, _ := inkling2Gen(t, eng, []int{'s', 'y', 's', 'u'}, 1)
	if len(opened) != 1 {
		t.Fatalf("opened %d sessions, want the one fresh slot", len(opened))
	}
	if res.PrefillSeconds < acquireWait.Seconds() {
		t.Fatalf("PrefillSeconds = %.3f, want at least the %.3fs slot acquire wait", res.PrefillSeconds, acquireWait.Seconds())
	}
}

type inkling2ServeMemBackend struct {
	compute.Backend
	name  string
	total int64
	free  int64
}

func (b inkling2ServeMemBackend) Name() string       { return b.name }
func (b inkling2ServeMemBackend) Caps() compute.Caps { return compute.Caps{CapacityProbe: true} }
func (b inkling2ServeMemBackend) DeviceMemory() (int64, int64, bool) {
	return b.total, b.free, true
}

func TestResolveServeInkling2ContextFlags(t *testing.T) {
	local := make([]bool, 24)
	for i := range local {
		local[i] = i%4 != 3
	}
	m := &ggufload.Inkling2LoopModel{
		HP: fakmodel.Inkling2LoopHParams{
			Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3,
			Hidden: 2048, Heads: 24, KVHeads: 8, HeadDim: 128,
			SlidingWindow: 1024, ConvKernel: 4, LocalLayer: local,
		},
	}
	cuda16G := inkling2ServeMemBackend{name: "cuda", total: 16 << 30, free: 2560 << 20}
	ambient := func(k string) (string, bool) {
		if k == ggufload.Inkling2ResidentCtxEnv {
			return "8192", true
		}
		return "", false
	}
	unset := func(string) (string, bool) { return "", false }

	t.Run("native-context-tokens overrides env", func(t *testing.T) {
		nctx, fit := 16384, "off"
		req, fitAuto, lk := resolveServeInkling2ContextFlags(&serveFlags{nativeContextTokens: &nctx, fit: &fit}, ambient)
		if req != 16384 || fitAuto {
			t.Fatalf("req=%d fitAuto=%v, want 16384 false", req, fitAuto)
		}
		if v, ok := lk(ggufload.Inkling2ResidentCtxEnv); !ok || v != "16384" {
			t.Fatalf("lookup FAK_INKLING2_CTX = %q (%v), want 16384", v, ok)
		}
		if got := ggufload.Inkling2ResolveServeCtx(m, cuda16G, req, lk); got != 16384 {
			t.Fatalf("resolved ctx = %d, want 16384", got)
		}
	})

	t.Run("fit auto clears ambient env and resolves above 4096", func(t *testing.T) {
		nctx, fit := 0, "auto"
		req, fitAuto, lk := resolveServeInkling2ContextFlags(&serveFlags{nativeContextTokens: &nctx, fit: &fit}, ambient)
		if req != 0 || !fitAuto {
			t.Fatalf("req=%d fitAuto=%v, want 0 true", req, fitAuto)
		}
		if v, ok := lk(ggufload.Inkling2ResidentCtxEnv); ok || v != "" {
			t.Fatalf("fit auto kept ambient FAK_INKLING2_CTX = %q (%v)", v, ok)
		}
		if got := ggufload.Inkling2ResolveServeCtx(m, cuda16G, req, lk); got <= ggufload.Inkling2ResidentCtxDefault {
			t.Fatalf("fit auto resolved ctx = %d, want > %d", got, ggufload.Inkling2ResidentCtxDefault)
		}
	})

	t.Run("fit auto honors explicit native-context-tokens pin", func(t *testing.T) {
		nctx, fit := 12288, "auto"
		req, fitAuto, lk := resolveServeInkling2ContextFlags(&serveFlags{nativeContextTokens: &nctx, fit: &fit}, ambient)
		if req != 12288 || !fitAuto {
			t.Fatalf("req=%d fitAuto=%v, want 12288 true", req, fitAuto)
		}
		if got := ggufload.Inkling2ResolveServeCtx(m, cuda16G, req, lk); got != 12288 {
			t.Fatalf("pinned fit auto ctx = %d, want 12288", got)
		}
	})

	t.Run("default unset resolves above 4096 on 16 GiB CUDA", func(t *testing.T) {
		nctx, fit := 0, "off"
		req, fitAuto, lk := resolveServeInkling2ContextFlags(&serveFlags{nativeContextTokens: &nctx, fit: &fit}, unset)
		if req != 0 || fitAuto {
			t.Fatalf("req=%d fitAuto=%v, want 0 false", req, fitAuto)
		}
		if got := ggufload.Inkling2ResolveServeCtx(m, cuda16G, req, lk); got <= ggufload.Inkling2ResidentCtxDefault {
			t.Fatalf("default auto ctx = %d, want > %d", got, ggufload.Inkling2ResidentCtxDefault)
		}
	})
}
