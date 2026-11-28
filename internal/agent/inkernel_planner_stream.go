package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"hash"
	"strings"
)

// StreamingSupported enables the gateway's semantic SSE path for in-kernel runs.
// The backend projects each completed turn as one content delta; tool lifecycle
// progress still arrives independently from the owned loop.
func (p *InKernelPlanner) StreamingSupported() bool { return true }

// streamResolvedSampleOpts transfers resolved controls through Complete's
// existing option API. Mutable referents remain shared; only the struct is copied.
func streamResolvedSampleOpts(sp SampleParams) []SampleOpt {
	return []SampleOpt{func(out *SampleParams) { *out = sp }}
}

// CompleteStream emits assistant PROSE to sink incrementally while the turn decodes,
// holding every tool-call span through post-decode lifting and never exposing tool-token
// text before adjudication can see it (the StreamingPlanner contract, stream.go).
// Mechanically it binds the sink into the decode path's existing per-token emit seam
// (generateReusedWithOOMRetry → the per-token emit closure in Complete), so each decoded
// piece of assistant prose is forwarded as it is produced — the same live token flow
// cmd/fakchat's streamDecode gives the direct process path, not one post-hoc delta.
// A toolSpanGuard sits between that raw seam and the caller's sink: it holds text that
// could open an explicit tool-call span, drops the span once confirmed, and forwards
// only prose, so raw <tool_call>/<function_call>/<|python_tag|>/[TOOL_CALLS] markup is
// never delivered as content before adjudication (toolSpanGuard, below).
//
// This per-token forwarding is OPT-IN via FAK_STREAM_INKERNEL_PER_TOKEN (default off) or
// a per-call WithPerTokenStream(true), which the turnkey gateway passes so `fak up`
// streams by default; flag OFF is the buffered projection — the whole turn decodes with
// the plain Complete (no observer, no forced DecodeTrace) and reaches the sink as AT MOST
// one post-hoc content delta, byte-identical to the pre-seam trunk. When no sink is
// provided (or the decode runs one of the speculative paths, whose draft-verify rounds
// batch tokens), the stream likewise degrades to the buffered projection and the returned
// Completion is unchanged.
//
// The sink observes the RAW incrementally decoded model text; the final Completion
// carries the POST-PROCESSED turn (reasoning split, tool-call lift) exactly as the
// buffered Complete produces it for the same prompt + seed. runArmStream wraps this raw
// sink with the post-decode projection (loop_project.go), so its emitted deltas land in
// the client's view of Content.
func (p *InKernelPlanner) CompleteStream(ctx context.Context, sink StreamSink, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	if err := p.validateV41EngramLookAhead(); err != nil {
		return nil, err
	}
	sp := applySampleOpts(opts...)
	if sp.NativeBeamSearch != nil {
		return nil, fmt.Errorf("%w: streaming is not supported", ErrNativeBeamSearchUnsupported)
	}
	if err := p.ValidateSamplingSeed(sp.Seed); err != nil {
		return nil, err
	}
	if err := refuseNativeGuidedCarriers(sp.GuidedDecode); err != nil {
		return nil, err
	}
	if err := validateNativeSamplingParams(sp); err != nil {
		return nil, err
	}
	if p.dflash2 != nil {
		if err := validateDFlash2Stops(sp.Stop); err != nil {
			return nil, err
		}
	}
	// Reuse the controls already validated above instead of replaying caller
	// option closures in Complete. Appended stream observers still override the
	// copied controls in their original order. This is not a deep input freeze.
	opts = streamResolvedSampleOpts(sp)

	if sink == nil {
		return p.Complete(ctx, messages, tools, opts...)
	}
	// A per-call override (WithPerTokenStream) wins over the package-level env
	// gate so the turnkey server streams by default while every other CompleteStream
	// caller keeps the historical env-gated default exactly.
	perToken := inKernelPerTokenStreamEnabled()
	if sp.PerTokenStream != nil {
		perToken = *sp.PerTokenStream
	}
	if !perToken {
		// Default: the buffered projection. Decode the whole turn with the plain
		// Complete (no observer, no forced DecodeTrace), then emit the finished
		// content as one delta — byte-identical to the pre-seam trunk.
		comp, err := p.Complete(ctx, messages, tools, opts...)
		if err != nil {
			return comp, err
		}
		if comp != nil && comp.Message.Content != "" {
			return comp, sink(comp.Message.Content)
		}
		return comp, nil
	}
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var sinkErr error
	if nativeStructuredRequested(sp) {
		pending := ""
		comp, err := p.Complete(streamCtx, messages, tools, append(opts, WithDecodeTokenObserver(func(tokenPiece, rawText string) {
			if sinkErr != nil || tokenPiece == "" {
				return
			}
			var piece string
			if piece, pending = nativeStructuredRunePrefix(pending + tokenPiece); piece == "" {
				return
			}
			if sinkErr = sink(piece); sinkErr != nil {
				cancel()
			}
		}))...)
		if sinkErr == nil && err == nil && pending != "" {
			sinkErr = sink(pending)
		}
		if sinkErr != nil {
			return comp, sinkErr
		}
		return comp, err
	}
	reasoningSink := streamReasoningSink(ctx)
	newProjector := func(rendered string) *inKernelStreamProjector {
		startsInReasoning := strings.HasSuffix(rendered, qwenThinkAssistantSeed)
		projector := newInKernelStreamProjector(sink, sp.Stop, startsInReasoning)
		projector.spans.deferTools = true
		projector.spans.reasoning = reasoningSink
		projector.spans.reasoningAtClose = startsInReasoning && (sp.ThinkingBudget == nil || *sp.ThinkingBudget <= 0)
		return projector
	}
	var projector *inKernelStreamProjector
	if p.nativeChatTemplate != nil {
		// Seed projection from the actual once-prepared/shrunk prompt, not a
		// second rendering that could diverge or repeat compaction side effects.
		var preparedIdentity [32]byte
		streamCtx = context.WithValue(streamCtx, nativeChatTemplatePreparedKey{}, func(rendered string) error {
			identity := sha256.Sum256([]byte(rendered))
			if projector != nil {
				if identity != preparedIdentity {
					return fmt.Errorf("%w: retry prompt changed", ErrNativeChatTemplate)
				}
				return nil
			}
			preparedIdentity = identity
			projector = newProjector(rendered)
			return nil
		})
	} else {
		rendered := inKernelRenderPrompt(messages, tools, p.m.Cfg, sp)
		if raw, selected := p.spmRawStreamPrompt(messages, sp); selected {
			rendered = raw
		}
		projector = newProjector(rendered)
	}
	comp, err := p.Complete(streamCtx, messages, tools, append(opts, WithDecodeTokenObserver(func(tokenPiece, rawText string) {
		if sinkErr != nil || tokenPiece == "" {
			return
		}
		if projector == nil {
			sinkErr = fmt.Errorf("%w: missing prepared stream identity", ErrNativeChatTemplate)
			cancel()
			return
		}
		if sinkErr = projector.feed(tokenPiece); sinkErr != nil {
			cancel()
			return
		}
	}))...)
	// Do not flush withheld stop prefixes or ambiguous control delimiters after a
	// decode error: those bytes were never established as safe visible content.
	if sinkErr == nil && err == nil {
		if projector == nil {
			sinkErr = fmt.Errorf("%w: missing prepared stream identity", ErrNativeChatTemplate)
		} else {
			sinkErr = projector.finish(comp.Message.Content, len(comp.Message.ToolCalls) > 0 || hasTextToolCandidate(comp.Message.Content))
		}
	}
	if sinkErr != nil {
		return comp, sinkErr
	}
	return comp, err
}

// inKernelStreamProjector keeps raw decode control text off the content stream.
// The span guard suppresses reasoning and tool-call regions while stopSuffixGuard
// retains a possible stop suffix until it is known to be ordinary prose. Both run
// before the caller's sink, so a sink failure cancels decode without another delivery.
type inKernelStreamProjector struct {
	emitted         int
	emittedHash     hash.Hash
	reasoningClosed bool
	afterReasoning  int
	emit            StreamSink
	spans           *toolSpanGuard
	stops           *stopSuffixGuard
}

func newInKernelStreamProjector(emit StreamSink, stops []string, startsInReasoning ...bool) *inKernelStreamProjector {
	p := &inKernelStreamProjector{emit: emit, emittedHash: sha256.New()}
	seededReasoning := len(startsInReasoning) > 0 && startsInReasoning[0]
	p.spans = newToolSpanGuardState(func(piece string) error {
		if err := emit(piece); err != nil {
			return err
		}
		p.emitted += len(piece)
		writeStreamPrefixHash(p.emittedHash, piece)
		if p.reasoningClosed {
			p.afterReasoning += len(piece)
		}
		return nil
	}, seededReasoning)
	p.spans.onReasoningClose = func() { p.reasoningClosed = true; p.afterReasoning = 0 }
	p.stops = newStopSuffixGuard(p.spans.feed, stops)
	return p
}

// Once a tool opener is observed, only the final request-bound lift verdict can
// tell whether its bytes are content. Do not emit later prose out of order.
// hasTextToolCandidate scopes strict reconciliation to text that can participate
// in a lift. Ordinary late reasoning retains its historical stream behavior.
func hasTextToolCandidate(content string) bool {
	for _, marker := range []string{"<tool_call>", "<function_call>", "<function=", "<|python_tag|>", "[TOOL_CALLS]", deepSeekDSMLCallsOpen} {
		if strings.Contains(content, marker) {
			return true
		}
	}
	// A fence or JSON value alone is not a tool candidate. Use the same existing
	// dialect recognizers, so ordinary code and nameless arrays keep legacy status.
	for _, dialect := range toolCallDialects {
		if len(dialect.extract(content)) > 0 {
			return true
		}
	}
	return false
}

func writeStreamPrefixHash(h hash.Hash, text string) {
	const chunk = 4096
	for len(text) > 0 {
		n := len(text)
		if n > chunk {
			n = chunk
		}
		_, _ = h.Write([]byte(text[:n]))
		text = text[n:]
	}
}

func (p *inKernelStreamProjector) finish(content string, toolVerdict bool) error {
	if err := p.stops.flush(); err != nil {
		return err
	}
	if tail := p.spans.utf8Tail; tail != "" {
		p.spans.utf8Tail = ""
		if err := p.spans.feedText(tail); err != nil {
			return err
		}
	}
	// The final completion is the existing authoritative transcript. Keep only a
	// byte count and digest of successful sink writes, never a second text copy.
	if p.emitted > len(content) {
		if !toolVerdict {
			return p.finishLegacyReasoning(content)
		}
		return fmt.Errorf("inkernel: streamed content exceeds final tool verdict")
	}
	want := sha256.New()
	writeStreamPrefixHash(want, content[:p.emitted])
	if !bytes.Equal(p.emittedHash.Sum(nil), want.Sum(nil)) {
		if !toolVerdict {
			return p.finishLegacyReasoning(content)
		}
		return fmt.Errorf("inkernel: streamed content is not a prefix of final tool verdict")
	}
	if rest := content[p.emitted:]; rest != "" {
		return p.emit(rest)
	}
	return nil
}

// Preserve the preexisting late-reasoning stream prefix on ordinary answers.
// Only complete bytes newly delayed by this patch; never replay its prior prefix.
func (p *inKernelStreamProjector) finishLegacyReasoning(content string) error {
	if p.reasoningClosed && p.afterReasoning <= len(content) {
		if rest := content[p.afterReasoning:]; rest != "" {
			return p.emit(rest)
		}
		return nil
	}
	return p.flush()
}

func (p *inKernelStreamProjector) feed(piece string) error { return p.stops.feed(piece) }

func (p *inKernelStreamProjector) flush() error {
	if err := p.stops.flush(); err != nil {
		return err
	}
	return p.spans.flush()
}

// partialOpenerSuffix returns the length of the longest suffix of s that is a proper
// (non-empty, shorter-than-full) prefix of any guarded opener — the bytes that must be
// held because the next piece may complete an opener.
func partialOpenerSuffix(s string) int {
	best := 0
	for _, o := range toolSpanOpeners {
		max := len(o.open) - 1
		if max > len(s) {
			max = len(s)
		}
		for n := max; n > best; n-- {
			if strings.HasPrefix(o.open, s[len(s)-n:]) {
				best = n
				break
			}
		}
	}
	return best
}

// incrementalStopScanner is the decode loop's per-turn delta accrual state for the
// string-suffix Stop probe. Instead of materializing the whole accumulated text every
// token (the old per-token sb.String() was O(N) per token => O(N^2) per completion;
// issue #922 witnessed ~19x ns/token growth from 1024 to 8192 tokens), it accrues the
// exact bytes appended to the accumulator into a bounded tail window of the longest
// stop length and fingerprints only that window per token — O(maxStop) per token,
// O(N) total string work per completion. Fires identically to checkStop: non-empty
// stops only, suffix match, longest match wins, and only once len(stop) bytes have
// actually been appended (no false fire on a short window). maxStop <= 0 (no stop
// strings at all) degenerates to a no-op so the default no-stop decode path pays zero
// probe cost; the caller-level checkStop (hoisted empty-set early-out) stays the
// authoritative trim, re-derived once per completion on the rare fire path.
type incrementalStopScanner struct {
	// tail is the last min(total, maxStop) bytes of the accumulator — the window the
	// per-token probe fingerprints (len(tail) <= maxStop, the invariant tests pin).
	tail []byte
	// hist holds up to maxStop bytes immediately PRECEDING tail. It exists for the fire
	// path: a trim removes bytes from the END of the accumulator, so the window must
	// move BACK by len(matched) to remain the true suffix. Trimming "\n\n" from "\n\n\n"
	// leaves "\n" whose window is one byte OLDER than the pre-trim window; without hist
	// that byte is unrecoverable and the next "\n" would miss the "\n\n" fire. hist is
	// never part of the visible window, so the bounded-tail invariant still holds.
	hist  []byte
	stops []string
	total int
}

// newIncrementalStopScanner builds a scanner over the non-empty stop strings.
// maxStop is the length of the longest non-empty stop string (the caller derives it
// from the same request); it bounds the retained tail window and the per-token scan.
func newIncrementalStopScanner(maxStop int, stop []string) *incrementalStopScanner {
	s := &incrementalStopScanner{}
	if maxStop <= 0 {
		return s
	}
	s.tail = make([]byte, 0, maxStop)
	s.hist = make([]byte, 0, maxStop)
	for _, str := range stop {
		if str != "" {
			s.stops = append(s.stops, str)
		}
	}
	return s
}

// pushTail appends b to the logical sequence (hist+tail) while keeping tail at exactly
// the last min(logicalLen, cap(tail)) bytes and hist at up to cap(tail) bytes before it.
// Only the last 2*cap bytes can matter (a trim removes at most cap bytes from the end),
// so the work is O(maxStop) per call regardless of piece length.
func (s *incrementalStopScanner) pushTail(b []byte) {
	need := cap(s.tail)
	histLen, tailLen := len(s.hist), len(s.tail)
	total := histLen + tailLen + len(b)
	start := total - 2*need
	if start < 0 {
		start = 0
	}
	n := total - start
	buf := make([]byte, n)
	for i := 0; i < n; i++ {
		j := start + i
		switch {
		case j < histLen:
			buf[i] = s.hist[j]
		case j < histLen+tailLen:
			buf[i] = s.tail[j-histLen]
		default:
			buf[i] = b[j-histLen-tailLen]
		}
	}
	if n <= need {
		s.hist = s.hist[:0]
		s.tail = append(s.tail[:0], buf...)
		return
	}
	s.hist = append(s.hist[:0], buf[:n-need]...)
	s.tail = append(s.tail[:0], buf[n-need:]...)
}

// appendPiece accrues the exact bytes appended to the decode accumulator this token
// into the bounded tail window and re-fingerprints it (bytes.HasSuffix over at most
// maxStop bytes). Callers pass everything that was written to the accumulator — the
// token piece plus any budget-forced reasoning close — so the window sees precisely
// what checkStop would see over the whole buffer. The retained tail never exceeds
// maxStop, yet every fire the whole-buffer probe would report is still reported:
// a stop match lives entirely inside the last maxStop bytes once enough of them exist.
//
// On a fire the scanner trims the LONGEST matching stop from its own logical tail,
// mirroring checkStop's maximal trim of the accumulator. The caller never calls reset()
// on the ordinary fire path, so a scanner that kept the matched bytes would fire again
// on the next piece although checkStop (over the now-trimmed accumulator) would not.
// Refilling the window from hist keeps it an exact suffix of the caller's trimmed
// accumulator, so every future fire agrees and no retained byte is lost.
func (s *incrementalStopScanner) appendPiece(piece string) bool {
	if len(s.stops) == 0 {
		return false
	}
	s.total += len(piece)
	s.pushTail([]byte(piece))
	best := ""
	for _, str := range s.stops {
		if len(str) <= len(s.tail) && len(str) > len(best) && bytes.HasSuffix(s.tail, []byte(str)) {
			best = str
		}
	}
	if best == "" {
		return false
	}
	// Trim the matched stop from the tail, then refill from hist so the window is again
	// the last maxStop bytes of the trimmed accumulator.
	drop := len(best)
	if drop >= len(s.tail) {
		drop -= len(s.tail)
		s.tail = s.tail[:0]
		if drop >= len(s.hist) {
			s.hist = s.hist[:0]
		} else {
			s.hist = s.hist[:len(s.hist)-drop]
		}
	} else {
		s.tail = s.tail[:len(s.tail)-drop]
	}
	need := cap(s.tail) - len(s.tail)
	if need > len(s.hist) {
		need = len(s.hist)
	}
	if need > 0 {
		refill := s.hist[len(s.hist)-need:]
		s.tail = append(s.tail, refill...)
		s.hist = s.hist[:len(s.hist)-need]
	}
	return true
}

// reset re-initializes the scanner for a start-over retry turn, mirroring the
// accumulator's own reset (same fire decisions as a freshly constructed scanner).
func (s *incrementalStopScanner) reset() {
	s.tail = s.tail[:0]
	s.hist = s.hist[:0]
	s.total = 0
}

// splitNativeBudgetReasoning accounts for an opening marker already in the
// prompt. A length/stop cutoff before a real close remains reasoning; projection
// never invents an emitted closing token or extends the generation limit.
func splitNativeBudgetReasoning(raw string, seeded bool) (string, string) {
	if seeded {
		return splitReasoning(thinkOpen + raw)
	}
	return splitReasoning(raw)
}
