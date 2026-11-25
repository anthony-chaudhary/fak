package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/inklingmetal"
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

// inkling2FakeSession replays a canned token stream: Prefill returns the first
// token, each Step the next, and it records what it was fed.
type inkling2FakeSession struct {
	canned  []int
	i       int
	prompt  []int
	fed     []int
	closed  bool
	stepErr error
}

func (s *inkling2FakeSession) Prefill(prompt []int) (int, error) {
	s.prompt = append([]int(nil), prompt...)
	return s.canned[0], nil
}

func (s *inkling2FakeSession) Step(token int) (int, error) {
	if s.stepErr != nil {
		return 0, s.stepErr
	}
	s.fed = append(s.fed, token)
	s.i++
	return s.canned[s.i], nil
}

func (s *inkling2FakeSession) Close() error { s.closed = true; return nil }

func inkling2FakeEngine(canned []int) (*inkling2LoopEngine, *[]*inkling2FakeSession) {
	var opened []*inkling2FakeSession
	return &inkling2LoopEngine{open: func() (ggufload.Inkling2LoopSession, error) {
		s := &inkling2FakeSession{canned: canned}
		opened = append(opened, s)
		return s, nil
	}}, &opened
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
	if s := (*opened)[0]; !s.closed || !inkling2EqualInts(s.prompt, []int{200, 'q'}) {
		t.Fatalf("session prompt %v closed=%v", s.prompt, s.closed)
	}
}

func TestInkling2AdapterMaxTokensAndFreshSessionPerRequest(t *testing.T) {
	eng, opened := inkling2FakeEngine([]int{'a', 'b', 'c', 'd', 'e'})
	for req := 0; req < 2; req++ {
		var got []int
		res, err := eng.Generate(context.Background(), []int{'p'}, 3, nil, func(id int) bool {
			got = append(got, id)
			return true
		})
		if err != nil {
			t.Fatal(err)
		}
		if res.Stopped || res.Generated != 3 || !inkling2EqualInts(got, []int{'a', 'b', 'c'}) {
			t.Fatalf("request %d: emitted %v stopped=%v, want a b c on the length limit", req, got, res.Stopped)
		}
	}
	if len(*opened) != 2 || len((*opened)[1].fed) != 2 {
		t.Fatalf("opened %d sessions, want one fresh session per request", len(*opened))
	}
}

func TestInkling2AdapterEmitFalseAndStepErrorEndTheTurn(t *testing.T) {
	eng, _ := inkling2FakeEngine([]int{'a', 'b', 'c'})
	res, err := eng.Generate(context.Background(), []int{'p'}, 10, nil, func(int) bool { return false })
	if err != nil || !res.Stopped || res.Generated != 1 {
		t.Fatalf("emit=false: res=%+v err=%v", res, err)
	}
	boom := errors.New("device fault")
	failing := &inkling2LoopEngine{open: func() (ggufload.Inkling2LoopSession, error) {
		return &inkling2FakeSession{canned: []int{'a', 'b'}, stepErr: boom}, nil
	}}
	if _, err := failing.Generate(context.Background(), []int{'p'}, 10, nil, func(int) bool { return true }); !errors.Is(err, boom) {
		t.Fatalf("step error = %v, want %v", err, boom)
	}
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

func TestServeInkling2MetalAdmission(t *testing.T) {
	if err := serveAdmitInkling2Metal("m.gguf", serveInkling2LoopArch, 1, true); err != nil {
		t.Fatalf("single-file inkling2_loop on an available target refused: %v", err)
	}
	for name, tc := range map[string]struct {
		arch      string
		shards    int
		available bool
		want      string
		stub      bool
	}{
		"stub build":  {serveInkling2LoopArch, 1, false, "cgo", true},
		"split GGUF":  {serveInkling2LoopArch, 3, true, "split into 3 shards", false},
		"base arch":   {serveInkling2ArchBase, 1, true, "not " + serveInkling2ArchBase, false},
		"stub+split":  {serveInkling2LoopArch, 2, false, "cgo", true},
		"other+split": {serveInkling2ArchBase, 2, true, "split into 2 shards", false},
	} {
		err := serveAdmitInkling2Metal("m.gguf", tc.arch, tc.shards, tc.available)
		var refusal *inkling2MetalServeRefusal
		if !errors.As(err, &refusal) || !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "m.gguf") {
			t.Fatalf("%s: %v, want a typed refusal naming the file and %q", name, err, tc.want)
		}
		if errors.Is(err, inklingmetal.ErrUnavailable) != tc.stub {
			t.Fatalf("%s: ErrUnavailable wrap = %v, want %v", name, !tc.stub, tc.stub)
		}
	}
}

// The route verdict follows the build: a darwin/arm64 cgo binary with a
// supported GPU admits, every other build returns the stub's typed refusal.
func TestServeInkling2MetalRouteFollowsBuild(t *testing.T) {
	path := writeInkling2HeaderGGUF(t, serveInkling2LoopArch)
	routed, err := serveInkling2MetalRoute(path, nil, true)
	if !routed {
		t.Fatal("inkling2_loop GGUF with Metal selected not routed to the native target")
	}
	if inklingmetal.Available() {
		if err != nil {
			t.Fatalf("available native target refused: %v", err)
		}
	} else {
		var refusal *inkling2MetalServeRefusal
		if !errors.As(err, &refusal) || !errors.Is(err, inklingmetal.ErrUnavailable) {
			t.Fatalf("unavailable native target = %v, want the typed refusal wrapping ErrUnavailable", err)
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
	if inklingmetal.Available() && !strings.Contains(err.Error(), "split into 2 shards") {
		t.Fatalf("split refusal does not name the shard count: %v", err)
	}
}

type inkling2FakeMetalTarget struct {
	next     []int32
	prefill  []int32
	fed      []int32
	commits  []int
	rows     int
	closed   bool
	blockErr error
}

func (f *inkling2FakeMetalTarget) PrefillID(ids []int32) (int32, error) {
	f.prefill = append([]int32(nil), ids...)
	return f.next[0], nil
}

func (f *inkling2FakeMetalTarget) BeginBlockIDs(ids []int32) ([]int32, error) {
	if f.blockErr != nil {
		return nil, f.blockErr
	}
	f.fed = append(f.fed, ids...)
	if f.rows > 0 {
		return make([]int32, f.rows), nil
	}
	return []int32{f.next[len(f.fed)]}, nil
}

func (f *inkling2FakeMetalTarget) Commit(keep int) error {
	f.commits = append(f.commits, keep)
	return nil
}

func (f *inkling2FakeMetalTarget) Close() error { f.closed = true; return nil }

func TestInkling2MetalSessionDrivesOneCommittedRowPerToken(t *testing.T) {
	target := &inkling2FakeMetalTarget{next: []int32{11, 12, 13, 14}}
	eng := &inkling2LoopEngine{open: func() (ggufload.Inkling2LoopSession, error) {
		return inkling2MetalSession{s: target}, nil
	}}
	var out []int
	res, err := eng.Generate(context.Background(), []int{5, 6}, 8, map[int]bool{14: true}, func(id int) bool {
		out = append(out, id)
		return true
	})
	if err != nil || !res.Stopped || res.Generated != 3 || !inkling2EqualInts(out, []int{11, 12, 13}) {
		t.Fatalf("generate = %+v, %v, out %v", res, err, out)
	}
	if len(target.prefill) != 2 || target.prefill[0] != 5 || target.prefill[1] != 6 {
		t.Fatalf("prefill ids = %v", target.prefill)
	}
	if len(target.fed) != 3 || target.fed[0] != 11 || target.fed[2] != 13 {
		t.Fatalf("stepped ids = %v, want each emitted token fed back once", target.fed)
	}
	if len(target.commits) != 3 || target.commits[0] != 1 || target.commits[2] != 1 {
		t.Fatalf("commits = %v, want one retained row per step", target.commits)
	}
	if !target.closed {
		t.Fatal("session left open after the request")
	}

	s := inkling2MetalSession{s: &inkling2FakeMetalTarget{next: []int32{1}}}
	if _, err := s.Prefill(nil); err == nil {
		t.Fatal("empty prompt accepted")
	}
	if _, err := s.Prefill([]int{inklingmetal.Vocabulary}); err == nil {
		t.Fatal("prompt token past the vocabulary accepted")
	}
	if _, err := s.Step(-1); err == nil {
		t.Fatal("negative token accepted")
	}
	bad := &inkling2FakeMetalTarget{next: []int32{1}, rows: 2}
	if _, err := (inkling2MetalSession{s: bad}).Step(1); err == nil || len(bad.commits) != 0 {
		t.Fatalf("mis-sized target reply = %v with commits %v, want an error before any commit", err, bad.commits)
	}
	failing := &inkling2FakeMetalTarget{next: []int32{1}, blockErr: errors.New("device lost")}
	if _, err := (inkling2MetalSession{s: failing}).Step(1); err == nil || len(failing.commits) != 0 {
		t.Fatalf("failed step = %v with commits %v", err, failing.commits)
	}
}

func TestInkling2WarmUpPrefillsOnceAndReturnsTheSession(t *testing.T) {
	eng, opened := inkling2FakeEngine([]int{0})
	if err := warmInkling2Session(eng.open, inkling2FakeTokenizer(t)); err != nil {
		t.Fatal(err)
	}
	if len(*opened) != 1 || len((*opened)[0].prompt) == 0 || len((*opened)[0].fed) != 0 || !(*opened)[0].closed {
		t.Fatalf("warm-up sessions = %+v, want one prefilled, unstepped, closed session", *opened)
	}
	failing := func() (ggufload.Inkling2LoopSession, error) { return nil, errors.New("no device") }
	if err := warmInkling2Session(failing, inkling2FakeTokenizer(t)); err == nil {
		t.Fatal("warm-up ignored a session that would not open")
	}
}
