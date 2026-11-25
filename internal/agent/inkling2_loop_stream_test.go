package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// inkling2StreamTokenizer has one id per printable ASCII byte plus the Inkling-2
// specials, so a canned turn encodes and decodes exactly.
func inkling2StreamTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{"Ġ": 0, "Ċ": 1}
	for c := 33; c < 127; c++ {
		vocab[string(rune(c))] = c
	}
	var added []map[string]any
	for i, s := range []string{inkling2RoleUser, inkling2RoleModel, inkling2RoleSystem, inkling2RoleTool,
		inkling2ContentText, inkling2ContentThinking, inkling2InvokeToolJSON, inkling2ContentXML,
		inkling2EndMessage, Inkling2EndOfTurnMarker} {
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
		t.Fatalf("parse tokenizer: %v", err)
	}
	return tok
}

// inkling2StreamGen replays canned ids. afterEmit runs after each emitted id, on
// the generator's goroutine, so a test can hold the turn open.
type inkling2StreamGen struct {
	ids       []int
	afterEmit func(i int)
}

func (g *inkling2StreamGen) Generate(ctx context.Context, _ []int, maxNew int, stops map[int]bool, emit func(int) bool) (Inkling2LoopGeneration, error) {
	for i, id := range g.ids {
		if err := ctx.Err(); err != nil {
			return Inkling2LoopGeneration{Generated: i}, err
		}
		if i >= maxNew {
			return Inkling2LoopGeneration{Generated: i}, nil
		}
		if stops[id] {
			return Inkling2LoopGeneration{Generated: i, Stopped: true}, nil
		}
		if !emit(id) {
			return Inkling2LoopGeneration{Generated: i + 1, Stopped: true}, nil
		}
		if g.afterEmit != nil {
			g.afterEmit(i)
		}
	}
	return Inkling2LoopGeneration{Generated: len(g.ids)}, nil
}

func inkling2StreamPlanner(t *testing.T, gen Inkling2LoopGenerator) *Inkling2LoopPlanner {
	t.Helper()
	p, err := NewInkling2LoopPlanner(gen, inkling2StreamTokenizer(t), "inkling2", model.Config{ModelType: "inkling2_loop"}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// inkling2StreamIDs encodes a turn; spell spells the listed markers byte by byte,
// so a marker arrives split across tokens.
func inkling2StreamIDs(t *testing.T, turn string, spell bool) []int {
	t.Helper()
	tok := inkling2StreamTokenizer(t)
	if !spell {
		ids, err := tok.Encode(turn)
		if err != nil {
			t.Fatal(err)
		}
		return ids
	}
	var ids []int
	for i := 0; i < len(turn); i++ {
		c := turn[i]
		switch c {
		case ' ':
			ids = append(ids, 0)
		case '\n':
			ids = append(ids, 1)
		default:
			ids = append(ids, int(c))
		}
	}
	return ids
}

type inkling2StreamCapture struct {
	mu     sync.Mutex
	deltas []string
	got    chan struct{}
}

func newInkling2StreamCapture() *inkling2StreamCapture {
	return &inkling2StreamCapture{got: make(chan struct{}, 1024)}
}

func (c *inkling2StreamCapture) sink(delta string) error {
	c.mu.Lock()
	c.deltas = append(c.deltas, delta)
	c.mu.Unlock()
	select {
	case c.got <- struct{}{}:
	default:
	}
	return nil
}

func (c *inkling2StreamCapture) joined() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return strings.Join(c.deltas, "")
}

var inkling2StreamUser = []Message{{Role: RoleUser, Content: "hi"}}

func inkling2StreamTools() []ToolDef {
	return []ToolDef{{Type: "function", Function: ToolDefFunction{Name: "Bash", Parameters: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}}}`)}}}
}

func TestInkling2StreamEmitsTextBeforeTurnEnds(t *testing.T) {
	turn := inkling2RoleModel + inkling2ContentThinking + "think" + inkling2EndMessage +
		inkling2RoleModel + inkling2ContentText + "Hello world, this is a long answer." + inkling2EndMessage + Inkling2EndOfTurnMarker
	ids := inkling2StreamIDs(t, turn, false)
	capture := newInkling2StreamCapture()
	var early bool
	gen := &inkling2StreamGen{ids: ids, afterEmit: func(i int) {
		if i != len(ids)-3 { // the answer is decoded; end_message and the end marker are not
			return
		}
		select {
		case <-capture.got:
			early = true
		case <-time.After(5 * time.Second):
		}
	}}
	comp, err := inkling2StreamPlanner(t, gen).CompleteStream(context.Background(), capture.sink, inkling2StreamUser, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !early {
		t.Fatal("no content delta reached the sink before the turn finished decoding")
	}
	if got := capture.joined(); got != comp.Message.Content || got != "Hello world, this is a long answer." {
		t.Fatalf("stream %q, content %q", got, comp.Message.Content)
	}
}

func TestInkling2StreamHoldsToolCallTurn(t *testing.T) {
	call := inkling2RoleModel + "Bash" + inkling2InvokeToolJSON + `{"name":"Bash","args":{"command":"ls"}}` + inkling2EndMessage + Inkling2EndOfTurnMarker
	for _, turn := range []string{
		call,
		inkling2RoleModel + inkling2ContentText + `{"name":"Bash","arguments":{"command":"ls"}}` + inkling2EndMessage + Inkling2EndOfTurnMarker,
		inkling2RoleModel + inkling2ContentText + `<tool_call>{"name":"Bash","arguments":{"command":"ls"}}</tool_call>` + inkling2EndMessage + Inkling2EndOfTurnMarker,
	} {
		capture := newInkling2StreamCapture()
		gen := &inkling2StreamGen{ids: inkling2StreamIDs(t, turn, false)}
		comp, err := inkling2StreamPlanner(t, gen).CompleteStream(context.Background(), capture.sink, inkling2StreamUser, inkling2StreamTools())
		if err != nil {
			t.Fatal(err)
		}
		if len(comp.Message.ToolCalls) != 1 || comp.Message.Content != "" {
			t.Fatalf("%q: completion %+v", turn, comp.Message)
		}
		if got := capture.joined(); got != "" {
			t.Fatalf("%q: tool-call text streamed as content: %q", turn, got)
		}
	}
}

// Every released byte must be a prefix of the final content whatever arrives next,
// whether the markers and openers come as one token or byte by byte.
func TestInkling2StreamProjectionIsAlwaysAPrefix(t *testing.T) {
	for _, c := range inkling2StreamCases() {
		_, content := SplitInkling2Output(c.raw)
		want := content
		if c.tools {
			want = normalizeInKernelToolCalls(&Completion{Message: Message{Role: RoleAssistant, Content: content}}, NewOfferedTools(inkling2StreamTools()), nil, inkling2StreamTools(), nil).Message.Content
		}
		for _, chunk := range []int{1, 2, 7, len(c.raw)} {
			s := newInkling2ContentStream(c.tools)
			for i := 0; i < len(c.raw); i += chunk {
				s.feed([]byte(c.raw[i:min(i+chunk, len(c.raw))]))
				if got := s.emitted.String(); !strings.HasPrefix(want, got) {
					t.Fatalf("%s chunk %d: released %q is not a prefix of %q", c.name, chunk, got, want)
				}
			}
			if c.streams && s.emitted.Len() == 0 {
				t.Errorf("%s chunk %d: nothing streamed", c.name, chunk)
			}
			if !c.streams && s.emitted.Len() != 0 {
				t.Errorf("%s chunk %d: streamed %q", c.name, chunk, s.emitted.String())
			}
		}
	}
}

type inkling2StreamCase struct {
	name    string
	raw     string
	tools   bool
	streams bool // some content must be released before the turn ends
}

func inkling2StreamCases() []inkling2StreamCase {
	text := func(s string) string { return inkling2RoleModel + inkling2ContentText + s + inkling2EndMessage }
	think := inkling2RoleModel + inkling2ContentThinking + "plan <|end" + inkling2EndMessage
	call := inkling2RoleModel + "Bash" + inkling2InvokeToolJSON + `{"name":"Bash","args":{"command":"ls"}}` + inkling2EndMessage
	unoffered := inkling2RoleModel + "Delete" + inkling2InvokeToolJSON + `{"name":"Delete","args":{}}` + inkling2EndMessage
	eot := Inkling2EndOfTurnMarker
	return []inkling2StreamCase{
		{"plain text", text("Hello there.") + eot, false, true},
		{"thinking then text", think + text("Answer.") + eot, false, true},
		{"two text blocks", text("one") + "\n" + text("two") + eot, false, true},
		{"whitespace trimmed", text("  \n lead and trail  \n") + eot, false, true},
		{"inner whitespace kept", text("a  \n  b") + eot, false, true},
		{"empty first block", text("") + text("x") + eot, false, true},
		{"text then tool call", text("Listing.") + call + eot, true, true},
		{"text, tool call, text", text("Before.") + call + text("After.") + eot, true, true},
		{"unoffered tool call stays content", text("See:") + unoffered + eot, true, true},
		{"tool call only", call + eot, true, false},
		{"bare json is a call", text(`  {"name":"Bash","arguments":{"command":"ls"}}`) + eot, true, false},
		{"bare json without tools streams", text(`{"a":1}`) + eot, false, true},
		{"hermes call after prose", text(`Run it: <tool_call>{"name":"Bash","arguments":{"command":"ls"}}</tool_call>`) + eot, true, true},
		{"fenced call after prose", text("Run it:\n```json\n{\"name\":\"Bash\",\"arguments\":{\"command\":\"ls\"}}\n```") + eot, true, true},
		{"hermes text without tools streams", text(`<tool_call>{"name":"Bash"}</tool_call> done`) + eot, false, true},
		{"marker-like prose", text("a <| b <|end c") + eot, false, true},
		{"unclosed text block", text("cut off mid")[:len(text("cut off mid"))-len(inkling2EndMessage)], false, true},
		{"unclosed partial end marker", inkling2RoleModel + inkling2ContentText + "abc<|end_mes", false, true},
		{"headerless text block", inkling2ContentText + "bare header" + inkling2EndMessage + eot, false, true},
		{"marker-free output", "plain answer without markers", false, false},
		{"marker inside text", text("x"+inkling2RoleModel+"y") + eot, false, true},
		{"invoke marker inside text", text(`x`+inkling2InvokeToolJSON+`{"name":"Bash","args":{}}`) + eot, true, true},
		{"unknown block", inkling2RoleModel + "odd" + inkling2EndMessage + text("t") + eot, false, false},
		{"text after end of turn ignored", text("kept") + eot + text("dropped"), false, true},
		{"thinking only", think + eot, false, false},
	}
}

// The concatenated stream equals the content the non-streamed Complete returns,
// for every case, with markers arriving whole or spelled byte by byte.
func TestInkling2StreamConcatenationMatchesComplete(t *testing.T) {
	for _, c := range inkling2StreamCases() {
		for _, spell := range []bool{false, true} {
			var tools []ToolDef
			if c.tools {
				tools = inkling2StreamTools()
			}
			ids := inkling2StreamIDs(t, c.raw, spell)
			want, err := inkling2StreamPlanner(t, &inkling2StreamGen{ids: ids}).Complete(context.Background(), inkling2StreamUser, tools)
			if err != nil {
				t.Fatal(err)
			}
			capture := newInkling2StreamCapture()
			got, err := inkling2StreamPlanner(t, &inkling2StreamGen{ids: ids}).CompleteStream(context.Background(), capture.sink, inkling2StreamUser, tools)
			if err != nil {
				t.Fatalf("%s spell=%v: %v", c.name, spell, err)
			}
			if capture.joined() != want.Message.Content {
				t.Errorf("%s spell=%v: stream %q, Complete content %q", c.name, spell, capture.joined(), want.Message.Content)
			}
			wantJSON, _ := json.Marshal(want)
			gotJSON, _ := json.Marshal(got)
			if string(wantJSON) != string(gotJSON) {
				t.Errorf("%s spell=%v: stream completion %s, Complete %s", c.name, spell, gotJSON, wantJSON)
			}
		}
	}
}

// A stop string straddling the release boundary is never streamed, and a partial
// stop that turns out to be prose is released once it diverges.
func TestInkling2StreamStopStraddlesReleaseBoundary(t *testing.T) {
	cases := []struct {
		name, body string
		stops      []string
		want       string
	}{
		{"stop straddles", "Hello XYZ tail", []string{"XYZ"}, "Hello"},
		{"partial stop diverges", "Hello XYa XYZ tail", []string{"XYZ"}, "Hello XYa"},
		{"stop spans end_message", "Hello", []string{"o" + inkling2EndMessage}, "Hell"},
		{"ordered stops", "abcab", []string{"b", "ab"}, "a"},
	}
	for _, c := range cases {
		turn := inkling2RoleModel + inkling2ContentText + c.body + inkling2EndMessage + Inkling2EndOfTurnMarker
		ids := inkling2StreamIDs(t, turn, false)
		want, err := inkling2StreamPlanner(t, &inkling2StreamGen{ids: ids}).Complete(context.Background(), inkling2StreamUser, nil, WithStop(c.stops))
		if err != nil {
			t.Fatal(err)
		}
		capture := newInkling2StreamCapture()
		var leaked string
		gen := &inkling2StreamGen{ids: ids, afterEmit: func(int) {
			if got := capture.joined(); !strings.HasPrefix(c.want, got) {
				leaked = got
			}
		}}
		got, err := inkling2StreamPlanner(t, gen).CompleteStream(context.Background(), capture.sink, inkling2StreamUser, nil, WithStop(c.stops))
		if err != nil {
			t.Fatal(err)
		}
		if leaked != "" {
			t.Errorf("%s: streamed %q past the stop", c.name, leaked)
		}
		if want.Message.Content != c.want || got.Message.Content != c.want || capture.joined() != c.want {
			t.Errorf("%s: Complete %q, stream completion %q, stream %q; want %q", c.name, want.Message.Content, got.Message.Content, capture.joined(), c.want)
		}
	}
}

// A sink failure ends generation and is returned.
func TestInkling2StreamSinkErrorStopsDecode(t *testing.T) {
	turn := inkling2RoleModel + inkling2ContentText + strings.Repeat("word ", 64) + inkling2EndMessage + Inkling2EndOfTurnMarker
	ids := inkling2StreamIDs(t, turn, false)
	boom := errors.New("client gone")
	failed := make(chan struct{})
	var once sync.Once
	emitted := 0
	gen := &inkling2StreamGen{ids: ids, afterEmit: func(i int) {
		emitted++
		if i == 2 { // the first content byte is queued for the sink
			select {
			case <-failed:
			case <-time.After(5 * time.Second):
			}
		}
	}}
	sink := func(string) error {
		once.Do(func() { close(failed) })
		return boom
	}
	_, err := inkling2StreamPlanner(t, gen).CompleteStream(context.Background(), sink, inkling2StreamUser, nil)
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want the sink error", err)
	}
	if emitted > 4 {
		t.Fatalf("decode emitted %d of %d ids after the sink failed", emitted, len(ids))
	}
}

// A sink panic surfaces on the caller's goroutine, where request recovery applies.
func TestInkling2StreamSinkPanicReachesCaller(t *testing.T) {
	turn := inkling2RoleModel + inkling2ContentText + "hello there" + inkling2EndMessage + Inkling2EndOfTurnMarker
	gen := &inkling2StreamGen{ids: inkling2StreamIDs(t, turn, false)}
	got := func() (r any) {
		defer func() { r = recover() }()
		_, _ = inkling2StreamPlanner(t, gen).CompleteStream(context.Background(), func(string) error { panic("sink boom") }, inkling2StreamUser, nil)
		return nil
	}()
	if got != "sink boom" {
		t.Fatalf("recovered %v, want the sink panic", got)
	}
}

// A forced tool choice may rewrite content, so its turn is delivered once decoded.
func TestInkling2StreamForcedToolChoiceBuffers(t *testing.T) {
	turn := inkling2RoleModel + inkling2ContentText + "prose first" + inkling2EndMessage + Inkling2EndOfTurnMarker
	ids := inkling2StreamIDs(t, turn, false)
	capture := newInkling2StreamCapture()
	streamed := false
	gen := &inkling2StreamGen{ids: ids, afterEmit: func(int) {
		if capture.joined() != "" {
			streamed = true
		}
	}}
	want, err := inkling2StreamPlanner(t, &inkling2StreamGen{ids: ids}).Complete(context.Background(), inkling2StreamUser, inkling2StreamTools(), WithToolChoice(json.RawMessage(`"required"`)))
	if err != nil {
		t.Fatal(err)
	}
	_, err = inkling2StreamPlanner(t, gen).CompleteStream(context.Background(), capture.sink, inkling2StreamUser, inkling2StreamTools(), WithToolChoice(json.RawMessage(`"required"`)))
	if err != nil {
		t.Fatal(err)
	}
	if streamed {
		t.Fatal("forced tool choice streamed content during decode")
	}
	if capture.joined() != want.Message.Content {
		t.Fatalf("stream %q, Complete %q", capture.joined(), want.Message.Content)
	}
}
