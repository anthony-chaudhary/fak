package agent

import (
	"bytes"
	"errors"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

const (
	inkling2StreamBetween = iota
	inkling2StreamHeader
	inkling2StreamText
	inkling2StreamThinking
	inkling2StreamHeld // nothing further is provably content; the final turn decides
)

// inkling2StreamMarkers are the Inkling-2 control markers. Inside a text block only
// end_message continues the stream; any other one may change how the turn splits.
var inkling2StreamMarkers = []string{
	inkling2EndMessage, Inkling2EndOfTurnMarker, inkling2RoleSystem, inkling2RoleUser, inkling2RoleModel,
	inkling2RoleTool, inkling2ContentText, inkling2ContentThinking, inkling2ContentXML, inkling2InvokeToolJSON,
}

// inkling2StreamToolOpeners is where each toolCallDialects extractor's lifted span can
// begin inside prose. bare_json lifts only a whole message, so it is checked at the
// first content rune instead.
var inkling2StreamToolOpeners = []string{
	"<tool_call>", "<function_call>", "<function=", "<|python_tag|>", "[TOOL_CALLS]",
	deepSeekDSMLCallsOpen, "```",
}

// inkling2ContentStream projects raw decoded Inkling-2 output onto the content that
// SplitInkling2Output and the tool-call lift produce for the finished turn. It
// releases a byte only when no continuation can drop, move or trim it: text-block
// bytes outside any possible marker or tool-call opener, with leading whitespace
// dropped and inner whitespace held until a later visible rune. Anything it cannot
// classify stops the projection; the final content then supplies the remainder.
type inkling2ContentStream struct {
	liftable bool // the request offers tools, so text dialects may lift calls
	state    int
	buf      []byte // raw bytes not yet classified
	started  bool   // a visible content rune was released
	sep      bool   // a text block closed; the next one is joined by "\n"
	space    []byte // whitespace held until a visible rune follows it
	emitted  strings.Builder
}

func newInkling2ContentStream(liftable bool) *inkling2ContentStream {
	return &inkling2ContentStream{liftable: liftable}
}

// feed consumes raw output that no stop string can cut and returns the newly
// provable content.
func (s *inkling2ContentStream) feed(raw []byte) string {
	if s.state == inkling2StreamHeld {
		return ""
	}
	s.buf = append(s.buf, raw...)
	before := s.emitted.Len()
	for s.state != inkling2StreamHeld && s.step() {
	}
	if s.state == inkling2StreamHeld {
		s.buf = nil
	}
	return s.emitted.String()[before:]
}

// step advances one state transition and reports whether another may follow.
func (s *inkling2ContentStream) step() bool {
	switch s.state {
	case inkling2StreamBetween:
		s.buf = bytes.TrimLeft(s.buf, " \t\r\n")
		return s.open(inkling2RoleModel, inkling2ContentText, inkling2ContentThinking, Inkling2EndOfTurnMarker)
	case inkling2StreamHeader:
		return s.open(inkling2ContentText, inkling2ContentThinking)
	case inkling2StreamThinking:
		return s.skipThinking()
	case inkling2StreamText:
		return s.text()
	}
	return false
}

// open matches the marker that starts the next block or its content.
func (s *inkling2ContentStream) open(markers ...string) bool {
	if len(s.buf) == 0 {
		return false
	}
	for _, m := range markers {
		if bytes.HasPrefix(s.buf, []byte(m)) {
			s.buf = s.buf[len(m):]
			switch m {
			case inkling2RoleModel:
				s.state = inkling2StreamHeader
			case inkling2ContentText:
				s.state = inkling2StreamText
				if s.sep {
					s.sep = false
					s.put("\n")
				}
			case inkling2ContentThinking:
				s.state = inkling2StreamThinking
			default:
				s.state = inkling2StreamHeld
			}
			return true
		}
	}
	for _, m := range markers {
		if len(s.buf) < len(m) && strings.HasPrefix(m, string(s.buf)) {
			return false
		}
	}
	// A tool block, an unknown block, or text outside any block.
	s.state = inkling2StreamHeld
	return false
}

func (s *inkling2ContentStream) skipThinking() bool {
	end := bytes.Index(s.buf, []byte(inkling2EndMessage))
	eot := bytes.Index(s.buf, []byte(Inkling2EndOfTurnMarker))
	switch {
	case eot >= 0 && (end < 0 || eot < end):
		s.state = inkling2StreamHeld
		return false
	case end >= 0:
		s.buf = s.buf[end+len(inkling2EndMessage):]
		s.state = inkling2StreamBetween
		return true
	}
	if keep := len(Inkling2EndOfTurnMarker) - 1; len(s.buf) > keep {
		s.buf = s.buf[len(s.buf)-keep:]
	}
	return false
}

func (s *inkling2ContentStream) guards() [][]string {
	if s.liftable {
		return [][]string{inkling2StreamMarkers, inkling2StreamToolOpeners}
	}
	return [][]string{inkling2StreamMarkers}
}

func (s *inkling2ContentStream) text() bool {
	at, which := -1, ""
	for _, set := range s.guards() {
		for _, g := range set {
			if i := bytes.Index(s.buf, []byte(g)); i >= 0 && (at < 0 || i < at) {
				at, which = i, g
			}
		}
	}
	if at >= 0 {
		s.put(string(s.buf[:at]))
		if s.state == inkling2StreamHeld {
			return false
		}
		if which != inkling2EndMessage {
			s.state = inkling2StreamHeld
			return false
		}
		s.buf = s.buf[at+len(which):]
		s.sep = true
		s.state = inkling2StreamBetween
		return true
	}
	end := len(s.buf) - s.heldSuffix()
	end = completeRunePrefix(s.buf[:end])
	s.put(string(s.buf[:end]))
	s.buf = append(s.buf[:0], s.buf[end:]...)
	return false
}

// heldSuffix is the length of the longest suffix of buf that may still grow into a guard.
func (s *inkling2ContentStream) heldSuffix() int {
	held := 0
	for _, set := range s.guards() {
		for _, g := range set {
			for n := min(len(g)-1, len(s.buf)); n > held; n-- {
				if bytes.HasSuffix(s.buf, []byte(g[:n])) {
					held = n
					break
				}
			}
		}
	}
	return held
}

// completeRunePrefix trims a trailing UTF-8 sequence that later bytes may complete.
func completeRunePrefix(b []byte) int {
	for back := 1; back <= utf8.UTFMax-1 && back <= len(b); back++ {
		if utf8.RuneStart(b[len(b)-back]) {
			if !utf8.FullRune(b[len(b)-back:]) {
				return len(b) - back
			}
			break
		}
	}
	return len(b)
}

// put applies the final TrimSpace and, when tools are offered, the bare-JSON lift:
// a message whose first visible rune opens an object or array may be lifted whole.
func (s *inkling2ContentStream) put(text string) {
	for len(text) > 0 {
		r, size := utf8.DecodeRuneInString(text)
		switch {
		case unicode.IsSpace(r):
			if s.started {
				s.space = append(s.space, text[:size]...)
			}
		case !s.started && s.liftable && (r == '{' || r == '['):
			s.state = inkling2StreamHeld
			return
		default:
			s.emitted.Write(s.space)
			s.space = s.space[:0]
			s.emitted.WriteString(text[:size])
			s.started = true
		}
		text = text[size:]
	}
}

// inkling2StreamPump hands deltas from the generator's emit callback to the
// request's sink on another goroutine, so a slow client never stalls the decode
// that emit runs inside (a cohort worker serves every row from one loop).
type inkling2StreamPump struct {
	mu       sync.Mutex
	pending  strings.Builder
	err      error
	panicked any
	wake     chan struct{}
	done     chan struct{}
	exited   chan struct{}
	stopOnce sync.Once
}

func startInkling2StreamPump(sink StreamSink, cancel func()) *inkling2StreamPump {
	q := &inkling2StreamPump{wake: make(chan struct{}, 1), done: make(chan struct{}), exited: make(chan struct{})}
	go func() {
		defer close(q.exited)
		for {
			select {
			case <-q.wake:
				q.deliver(sink, cancel)
			case <-q.done:
				q.deliver(sink, cancel)
				return
			}
		}
	}()
	return q
}

// put queues delta and reports whether the sink is still accepting.
func (q *inkling2StreamPump) put(delta string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.err != nil {
		return false
	}
	if delta != "" {
		q.pending.WriteString(delta)
		select {
		case q.wake <- struct{}{}:
		default:
		}
	}
	return true
}

func (q *inkling2StreamPump) deliver(sink StreamSink, cancel func()) {
	q.mu.Lock()
	delta := q.pending.String()
	q.pending.Reset()
	failed := q.err != nil
	q.mu.Unlock()
	if delta == "" || failed {
		return
	}
	// A sink panic is re-raised on the request goroutine by stop, where the
	// server's per-request recovery applies; here it would end the process.
	defer func() {
		if r := recover(); r != nil {
			q.fail(errors.New("inkling2_loop planner: stream sink panicked"), r)
			cancel()
		}
	}()
	if err := sink(delta); err != nil {
		q.fail(err, nil)
		cancel()
	}
}

func (q *inkling2StreamPump) fail(err error, panicked any) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.err, q.panicked = err, panicked
}

// stop delivers what is queued and returns the first sink error. It is idempotent.
func (q *inkling2StreamPump) stop() error {
	q.stopOnce.Do(func() {
		close(q.done)
		<-q.exited
	})
	q.mu.Lock()
	err, panicked := q.err, q.panicked
	q.panicked = nil
	q.mu.Unlock()
	if panicked != nil {
		panic(panicked)
	}
	return err
}
