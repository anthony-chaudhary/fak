package agent

import (
	"context"
	"encoding/json"
	"errors"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// streamProjectionSink is the post-decode projection runArmStream's loop wraps around
// the planner sink (loop_project.go): raw decode deltas are replayed as post-processed
// content deltas at completion. It is reproduced here as a test seam because it is the
// contract CompleteStream's raw sink feeding is defined against.
type streamProjectionSink struct {
	fragments []string
	err       error
}

func (s *streamProjectionSink) rawObserve(piece string) {
	if s.err == nil && piece != "" {
		s.fragments = append(s.fragments, piece)
	}
}

// completeFinalize is called by the test with the returned Completion; it drops the raw
// accumulation and projects the final Content as one delta, the way the loop does.
func (s *streamProjectionSink) completeFinalize(comp *Completion) {
	s.fragments = nil
	if comp != nil && comp.Message.Content != "" {
		s.fragments = append(s.fragments, comp.Message.Content)
	}
}

// TestInKernelPlannerStreamsPerTokenFragments is the fragment-count witness: a decoded
// N-token turn must reach the sink as N>=2 fragments, not one post-hoc blob. Fail if a
// regression collapses the per-token emit seam back into the single-blob projection.
func TestInKernelPlannerStreamsPerTokenFragments(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-stream-fragments", false, nil, false)

	var fragments []string
	comp, err := p.CompleteStream(context.Background(), func(delta string) error {
		fragments = append(fragments, delta)
		return nil
	}, []Message{{Role: RoleUser, Content: "a b c d e"}}, nil, WithMaxTokens(6))
	if err != nil {
		t.Fatal(err)
	}
	if len(fragments) < 2 {
		t.Fatalf("a %d-token completion produced %d sink fragments; the per-token emit seam collapsed to one post-hoc emit (fragments=%q)", comp.Usage.CompletionTokens, len(fragments), fragments)
	}
}

// TestInKernelPlannerStreamParityConcatenatedDeltasEqualsBufferedText is the parity
// witness: same prompt + seed, the completion RETURNED by CompleteStream is byte-for-byte
// the buffered Complete result (same text, same reasoning content, same usage), and the
// RAW decode deltas fed to the sink must be exactly reproducible per token once they are
// re-decoded against the same tokenizer — the property the loop's projection needs to
// replay them as content deltas.
func TestInKernelPlannerStreamParityConcatenatedDeltasEqualsBufferedText(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	tok := loadProbeTok(t)

	messages := []Message{{Role: RoleUser, Content: "alpha beta gamma"}}

	buffered, err := NewInKernelPlanner(m, tok, "tiny-stream-parity-buffered", false, nil, false).Complete(
		context.Background(), messages, nil, WithMaxTokens(8))
	if err != nil {
		t.Fatal(err)
	}

	var raw strings.Builder
	fragmentCount := 0
	streamed, err := NewInKernelPlanner(m, tok, "tiny-stream-parity-streamed", false, nil, false).CompleteStream(
		context.Background(), func(delta string) error {
			raw.WriteString(delta)
			fragmentCount++
			return nil
		}, messages, nil, WithMaxTokens(8))
	if err != nil {
		t.Fatal(err)
	}

	// The planner-level completion parity: what the streaming CALLER of this framework
	// receives from CompleteStream must be exactly what Complete returned.
	if streamed.Message.Content != buffered.Message.Content {
		t.Fatalf("CompleteStream content %q != buffered content %q", streamed.Message.Content, buffered.Message.Content)
	}
	if streamed.Message.ReasoningContent != buffered.Message.ReasoningContent {
		t.Fatalf("CompleteStream reasoning %q != buffered reasoning %q", streamed.Message.ReasoningContent, buffered.Message.ReasoningContent)
	}
	if streamed.FinishReason != buffered.FinishReason {
		t.Fatalf("CompleteStream finish %q != buffered finish %q", streamed.FinishReason, buffered.FinishReason)
	}
	if !reflect.DeepEqual(streamed.Usage, buffered.Usage) {
		t.Fatalf("CompleteStream usage %+v != buffered usage %+v", streamed.Usage, buffered.Usage)
	}

	// The raw decode delta identity: if a non-empty completion was produced, the raw
	// stream must have carried at least one character's worth of token deltas, and every
	// streaming character must be one of the buffered tokens' characters (the decode
	// ran through the same emit seam the buffered Complete uses).
	streamedText := raw.String()
	if streamedText == "" && buffered.Message.Content != "" {
		t.Fatal("a non-empty buffered completion produced an empty raw token stream")
	}
	if streamedText != buffered.Message.Content {
		t.Fatalf("streamed bytes %q != buffered content %q", streamedText, buffered.Message.Content)
	}
	if buffered.Message.Content != "" && fragmentCount < 2 {
		t.Fatalf("the streamed raw text was a single %d-char blob, not per-token deltas", len(streamedText))
	}
}

// TestInKernelTokenStreamFlagGatesPerToken is the #920 witness for the opt-in gate:
// FAK_STREAM_INKERNEL_PER_TOKEN=1 forwards the live per-token decode seam (>=2
// fragments for a multi-token turn), while the default OFF projects the finished turn
// as AT MOST one post-hoc delta whose bytes equal the returned Completion's Content.
func TestInKernelTokenStreamFlagGatesPerToken(t *testing.T) {
	const prompt = "a b c d e"

	t.Run("on-per-token-forwarding", func(t *testing.T) {
		t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
		m := model.NewSynthetic(tinyConcurrencyConfig())
		m.Quantize()
		p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-token-stream-on", false, nil, false)

		var fragments []string
		comp, err := p.CompleteStream(context.Background(), func(delta string) error {
			fragments = append(fragments, delta)
			return nil
		}, []Message{{Role: RoleUser, Content: prompt}}, nil, WithMaxTokens(6))
		if err != nil {
			t.Fatal(err)
		}
		if len(fragments) < 2 {
			t.Fatalf("flag ON: %d-token completion produced %d sink fragments, want >=2 (fragments=%q)", comp.Usage.CompletionTokens, len(fragments), fragments)
		}
	})

	t.Run("off-one-posthoc-delta", func(t *testing.T) {
		t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "")
		m := model.NewSynthetic(tinyConcurrencyConfig())
		m.Quantize()
		p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-token-stream-off", false, nil, false)

		var fragments []string
		comp, err := p.CompleteStream(context.Background(), func(delta string) error {
			fragments = append(fragments, delta)
			return nil
		}, []Message{{Role: RoleUser, Content: prompt}}, nil, WithMaxTokens(6))
		if err != nil {
			t.Fatal(err)
		}
		if len(fragments) > 1 {
			t.Fatalf("flag OFF: sink received %d fragments, want at most 1 (%q)", len(fragments), fragments)
		}
		if comp.Message.Content != "" {
			if len(fragments) != 1 {
				t.Fatalf("flag OFF: non-empty completion %q produced %d fragments, want exactly 1", comp.Message.Content, len(fragments))
			}
			if fragments[0] != comp.Message.Content {
				t.Fatalf("flag OFF: post-hoc delta %q != Completion content %q", fragments[0], comp.Message.Content)
			}
		}
	})
}

// Sink failures must be returned unchanged and stop further deliveries in either mode.
func TestInKernelTokenStreamSinkError(t *testing.T) {
	for _, flag := range []string{"", "1"} {
		t.Run("flag="+flag, func(t *testing.T) {
			t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", flag)
			m := model.NewSynthetic(tinyConcurrencyConfig())
			m.Quantize()
			p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-stream-sink-error", false, nil, false)
			want := errors.New("sink disconnected")
			calls := 0
			_, err := p.CompleteStream(context.Background(), func(string) error {
				calls++
				return want
			}, []Message{{Role: RoleUser, Content: "a b c d e"}}, nil, WithMaxTokens(6))
			if !errors.Is(err, want) || calls != 1 {
				t.Fatalf("sink error = %v, calls = %d; want %v and one call", err, calls, want)
			}
		})
	}
}

// TestInKernelPerTokenStreamOptInDefaultsOn is the turnkey-default witness: with the env
// gate OFF, a per-call WithPerTokenStream(true) (what cmd/fak up's streaming call site
// passes) must still forward the live per-token seam — >=2 fragments for a multi-token
// turn — so `fak up` streams true incremental deltas without an env var.
func TestInKernelPerTokenStreamOptInDefaultsOn(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-per-token-optin", false, nil, false)

	var fragments []string
	comp, err := p.CompleteStream(context.Background(), func(delta string) error {
		fragments = append(fragments, delta)
		return nil
	}, []Message{{Role: RoleUser, Content: "a b c d e"}}, nil, WithMaxTokens(6), WithPerTokenStream(true))
	if err != nil {
		t.Fatal(err)
	}
	if len(fragments) < 2 {
		t.Fatalf("WithPerTokenStream(true) with env off: %d-token completion produced %d fragments, want >=2", comp.Usage.CompletionTokens, len(fragments))
	}
}

// TestInKernelPerTokenStreamOptOutStaysBuffered is the byte-identical witness for the
// explicit off-switch: WithPerTokenStream(false) overrides an env ON gate and degrades to
// the single post-hoc projection, exactly the pre-seam behavior.
func TestInKernelPerTokenStreamOptOutStaysBuffered(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	p := NewInKernelPlanner(m, loadProbeTok(t), "tiny-per-token-optout", false, nil, false)

	var fragments []string
	comp, err := p.CompleteStream(context.Background(), func(delta string) error {
		fragments = append(fragments, delta)
		return nil
	}, []Message{{Role: RoleUser, Content: "a b c d e"}}, nil, WithMaxTokens(6), WithPerTokenStream(false))
	if err != nil {
		t.Fatal(err)
	}
	if len(fragments) > 1 {
		t.Fatalf("WithPerTokenStream(false) with env on: sink received %d fragments, want at most 1", len(fragments))
	}
	if comp.Message.Content != "" && (len(fragments) != 1 || fragments[0] != comp.Message.Content) {
		t.Fatalf("explicit off: post-hoc delta %q != Completion content %q", fragments, comp.Message.Content)
	}
}

// TestToolSpanGuardSuppressesToolCallMarkup is the tool-safety witness: a fake decode
// observer emits a Hermes <tool_call> JSON span token-piece by token-piece between prose.
// The guard must forward the leading and trailing PROSE and drop the ENTIRE span — the
// opener, the JSON body, and the closer must never reach the client as content.
func TestToolSpanGuardSuppressesToolCallMarkup(t *testing.T) {
	const (
		prefix = "I will read the file now. "
		tool   = `<tool_call>{"name": "Read", "arguments": {"filePath": "secret.go"}}</tool_call>`
		suffix = " Done."
	)
	// Split the tool span across pieces mid-token to prove partial-openers are held.
	pieces := []string{prefix, `<tool_`, `call>{"name": "Read", `, `"arguments": {"filePath": "secret.go"}}`, `</tool_call>`, suffix}

	var got strings.Builder
	g := newToolSpanGuard(func(delta string) error {
		got.WriteString(delta)
		return nil
	})
	for _, piece := range pieces {
		if err := g.feed(piece); err != nil {
			t.Fatal(err)
		}
	}
	if err := g.flush(); err != nil {
		t.Fatal(err)
	}

	streamed := got.String()
	if strings.Contains(streamed, "<tool_call>") || strings.Contains(streamed, "secret.go") || strings.Contains(streamed, `"name"`) {
		t.Fatalf("tool-call markup leaked into streamed content: %q", streamed)
	}
	want := prefix + suffix
	if streamed != want {
		t.Fatalf("streamed prose = %q, want %q (only prose, no span)", streamed, want)
	}
}

// TestToolSpanGuardProseUnaffected proves the guard is inert on markup-free prose,
// including legitimate braces, brackets, and partial tag-like text that never becomes a
// tool call — so a normal turn streams byte-identically to the raw seam.
func TestToolSpanGuardProseUnaffected(t *testing.T) {
	const prose = "Here is JSON {\"a\": 1} and an array [1,2] and a fence ```go``` plus a <b>tag</b>."
	chunks := []string{"Here is JSON {", "\"a\": 1} and an ", "array [1,2] and a fence ```go``` plus a <b>tag</b>."}

	var got strings.Builder
	g := newToolSpanGuard(func(delta string) error { got.WriteString(delta); return nil })
	for _, c := range chunks {
		if err := g.feed(c); err != nil {
			t.Fatal(err)
		}
	}
	if err := g.flush(); err != nil {
		t.Fatal(err)
	}
	if got.String() != prose {
		t.Fatalf("markup-free prose was altered: %q, want %q", got.String(), prose)
	}
}

// TestToolSpanGuardUnclosedSpanDropsToEnd proves an unclosed opener (a truncated or
// malformed tool call) is dropped to end of turn and the leading prose still flows, so a
// half-formed span can never leak its raw syntax.
func TestToolSpanGuardUnclosedSpanDropsToEnd(t *testing.T) {
	var got strings.Builder
	g := newToolSpanGuard(func(delta string) error { got.WriteString(delta); return nil })
	for _, c := range []string{"answer: ", "<tool_call>{\"name\": \"Bash\""} {
		if err := g.feed(c); err != nil {
			t.Fatal(err)
		}
	}
	if err := g.flush(); err != nil {
		t.Fatal(err)
	}
	if got.String() != "answer: " {
		t.Fatalf("unclosed span: streamed = %q, want %q", got.String(), "answer: ")
	}
}

// TestInKernelCompleteStreamProjectsStopBeforeSink exercises the production
// CompleteStream observer, rather than the projector helper alone. The synthetic
// model is deterministic, so a suffix learned from a buffered turn is the same
// suffix the streamed turn encounters. CompleteStream must trim it before any
// bytes reach the sink and the emitted bytes must equal final visible Content.
func TestInKernelCompleteStreamProjectsStopBeforeSink(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	tok := loadProbeTok(t)
	messages := []Message{{Role: RoleUser, Content: "alpha beta gamma delta"}}

	buffered, err := NewInKernelPlanner(m, tok, "tiny-stream-stop-probe", false, nil, false).Complete(
		context.Background(), messages, nil, WithMaxTokens(8))
	if err != nil {
		t.Fatal(err)
	}
	if len(buffered.Message.Content) < 2 {
		t.Fatalf("deterministic probe content too short: %q", buffered.Message.Content)
	}
	stop := buffered.Message.Content[len(buffered.Message.Content)-2:]

	var streamed strings.Builder
	comp, err := NewInKernelPlanner(m, tok, "tiny-stream-stop-witness", false, nil, false).CompleteStream(
		context.Background(), func(delta string) error {
			streamed.WriteString(delta)
			return nil
		}, messages, nil, WithMaxTokens(8), WithStop([]string{stop}))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(streamed.String(), stop) {
		t.Fatalf("configured stop %q leaked into stream %q", stop, streamed.String())
	}
	if streamed.String() != comp.Message.Content {
		t.Fatalf("streamed content %q != final projected content %q", streamed.String(), comp.Message.Content)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestInKernelCompleteStreamRejectsNonAtomicReasoningClose(t *testing.T) {
	t.Setenv("FAK_STREAM_INKERNEL_PER_TOKEN", "1")
	t.Setenv("FAK_INKERNEL_ENABLE_THINKING", "1")
	m := model.NewSynthetic(tinyConcurrencyConfig())
	m.Quantize()
	var streamed strings.Builder
	comp, err := NewInKernelPlanner(m, loadProbeTok(t), "tiny-stream-forced-think", false, nil, false).CompleteStream(
		context.Background(), func(delta string) error {
			streamed.WriteString(delta)
			return nil
		}, []Message{{Role: RoleUser, Content: "alpha beta gamma delta"}}, nil,
		WithMaxTokens(8), WithThinkingBudget(1))
	var unsupported *NativeThinkingBudgetUnsupportedError
	if !errors.As(err, &unsupported) || comp != nil || streamed.Len() != 0 {
		t.Fatalf("non-atomic close must refuse without output: comp=%v stream=%q err=%v", comp, streamed.String(), err)
	}
}

func atomicThinkingTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal([]byte(buildByteVocab()), &doc); err != nil {
		t.Fatal(err)
	}
	vocab := doc["model"].(map[string]any)["vocab"].(map[string]any)
	added := doc["added_tokens"].([]any)
	for _, marker := range []string{thinkOpen, thinkClose} {
		id := len(vocab)
		vocab[marker] = id
		added = append(added, map[string]any{"id": id, "content": marker, "special": true})
	}
	doc["added_tokens"] = added
	data, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.ParseJSON(data)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// fak-test:runtime fast est=50ms lane=default
func TestNativeThinkingBudgetUsesClosingTokenInSerialAndBatchedKV(t *testing.T) {
	for _, batched := range []bool{false, true} {
		name := "serial"
		if batched {
			name = "batched"
		}
		t.Run(name, func(t *testing.T) {
			tok := atomicThinkingTokenizer(t)
			cfg := tinyConcurrencyConfig()
			cfg.VocabSize = tok.Vocab()
			m := model.NewSynthetic(cfg)
			p := NewInKernelPlanner(m, tok, "tiny-budget", false, nil, false)
			constraint, err := p.newNativeThinkingConstraint(1, true, nil)
			if err != nil {
				t.Fatal(err)
			}
			body, _ := tok.Encode("x")
			prompt := []int{3}
			s := m.NewSession()
			defer s.Close()
			var raw, visible strings.Builder
			projector := newInKernelStreamProjector(func(piece string) error {
				visible.WriteString(piece)
				return nil
			}, nil, true)
			measurement := &nativeInferenceMeasurement{inferenceDisabled: true, decodeTokenIDsEnabled: true}
			ln := &decodeLane{
				s: s, logits: s.Prefill(prompt), maxNew: 3,
				rng: rand.New(rand.NewSource(1)), temp: 1, topP: 0.01, topK: 1,
				logitBias: model.LogitBias{body[0]: 100, constraint.closeID: -100},
				counts:    make([]int32, cfg.VocabSize), freqPenalty: 0.1, presPenalty: 0.1,
				thinking: constraint, measurement: measurement, forwarded: make([]int, 0, 2),
				emit: func(id int) bool {
					piece, err := tok.Decode([]int{id})
					if err != nil {
						t.Fatal(err)
					}
					raw.WriteString(piece)
					if err := projector.feed(piece); err != nil {
						t.Fatal(err)
					}
					return false
				},
			}
			// Bias favors another reasoning token at the cap. The closing-token
			// penalty and top-k=1 must not override the hard singleton.
			if batched {
				inKernelDecodeLanesBatched(context.Background(), []*decodeLane{ln}, m, false)
			} else {
				inKernelDecodeSerial(context.Background(), ln)
			}
			if err := projector.flush(); err != nil {
				t.Fatal(err)
			}
			want := []int{body[0], constraint.closeID, body[0]}
			if ln.err != nil || !reflect.DeepEqual(measurement.decodeTokenIDs, want) || ln.gen != 3 {
				t.Fatalf("actual emitted IDs=%v gen=%d err=%v want=%v", measurement.decodeTokenIDs, ln.gen, ln.err, want)
			}
			if raw.String() != "x</think>x" || visible.String() != "x" || constraint.budget.Count() != 1 || !constraint.forced {
				t.Fatalf("raw=%q visible=%q count=%d forced=%v", raw.String(), visible.String(), constraint.budget.Count(), constraint.forced)
			}
			if !reflect.DeepEqual(ln.forwarded, want[:2]) || ln.counts[constraint.closeID] != 1 {
				t.Fatalf("forwarded=%v close count=%d", ln.forwarded, ln.counts[constraint.closeID])
			}
			ref := m.NewSession()
			defer ref.Close()
			ref.Prefill(prompt)
			ref.Step(body[0])
			ref.Step(constraint.closeID)
			if s.Cache.Len() != len(prompt)+2 || !reflect.DeepEqual(s.Cache.K, ref.Cache.K) || !reflect.DeepEqual(s.Cache.V, ref.Cache.V) {
				t.Fatal("KV must contain the actual close token, excluding the unforwarded maxNew terminal")
			}
			constraint.reset()
			if constraint.budget.Count() != 0 || !constraint.budget.InSpan() || constraint.forced {
				t.Fatal("retry retained spent reasoning state")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNativeThinkingBudgetRejectsInvalidBoundaries(t *testing.T) {
	tok := atomicThinkingTokenizer(t)
	cfg := tinyConcurrencyConfig()
	p := NewInKernelPlanner(&model.Model{Cfg: cfg}, tok, "budget-boundary", false, nil, false)
	valid, err := p.newNativeThinkingConstraint(1, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, stopConflict := range []bool{false, true} {
		stops := map[int]bool{}
		p.m.Cfg.VocabSize = cfg.VocabSize
		if stopConflict {
			stops[valid.closeID] = true
		} else {
			p.m.Cfg.VocabSize = valid.closeID
		}
		_, err := p.newNativeThinkingConstraint(1, true, stops)
		var unsupported *NativeThinkingBudgetUnsupportedError
		if !errors.As(err, &unsupported) {
			t.Fatalf("vocab/stop conflict accepted: stop=%v err=%v", stopConflict, err)
		}
	}
	if _, _, err := valid.closingToken(valid.closeID); err == nil {
		t.Fatal("short actual logits row accepted")
	}
	if err := valid.acceptToken(tok.Vocab()); err == nil || valid.budget.Count() != 0 {
		t.Fatal("undecodable token silently consumed reasoning budget")
	}
	if err := valid.acceptToken(valid.closeID); err != nil {
		t.Fatal(err)
	}
	if _, force, err := valid.closingToken(tok.Vocab()); err != nil || force || valid.budget.Count() != 0 {
		t.Fatalf("natural close must end reasoning without force: force=%v err=%v", force, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNativeThinkingBudgetTerminalsAndSplitOpen(t *testing.T) {
	pieces := []string{"<th", "ink>", "x", thinkClose}
	constraint := &nativeThinkingConstraint{
		limit: 1, closeID: 3,
		decodeToken: func(id int) (string, error) { return pieces[id], nil },
	}
	constraint.reset()
	for _, id := range []int{0, 1} {
		if err := constraint.acceptToken(id); err != nil {
			t.Fatal(err)
		}
	}
	if constraint.budget.Count() != 0 || !constraint.budget.InSpan() {
		t.Fatal("split opening marker consumed reasoning budget")
	}
	constraint.acceptToken(2)
	for _, stopOnEmit := range []bool{false, true} {
		constraint.forced = false
		constraint.budget = NewThinkBudget(1, true)
		constraint.acceptToken(2)
		measurement := &nativeInferenceMeasurement{inferenceDisabled: true, decodeTokenIDsEnabled: true}
		ln := &decodeLane{
			logits: []float32{0, 0, 100, -100}, maxNew: 1,
			thinking: constraint, measurement: measurement,
			emit: func(id int) bool { return stopOnEmit },
		}
		_, advance := ln.decodeOne(context.Background())
		if advance || ln.err != nil || ln.gen != 1 || !reflect.DeepEqual(measurement.decodeTokenIDs, []int{3}) || ln.stopped != stopOnEmit {
			t.Fatalf("terminal closing ID accounting: advance=%v gen=%d ids=%v stopped=%v err=%v", advance, ln.gen, measurement.decodeTokenIDs, ln.stopped, ln.err)
		}
	}
	constraint.startInSpan = true
	constraint.reset()
	var raw string
	ln := &decodeLane{
		logits: []float32{0, 0, 100, -100}, maxNew: 1,
		thinking: constraint, emit: func(id int) bool { raw += pieces[id]; return false },
	}
	_, advance := ln.decodeOne(context.Background())
	reasoning, content := splitNativeBudgetReasoning(raw, true)
	if advance || ln.err != nil || ln.gen != 1 || constraint.forced || reasoning != "x" || content != "" {
		t.Fatalf("maxNew before close: raw=%q reasoning=%q content=%q gen=%d err=%v", raw, reasoning, content, ln.gen, ln.err)
	}
	// Unconstrained selection still chooses the original argmax, without a
	// singleton close or any reasoning-state allocation.
	plain := &decodeLane{logits: []float32{0, 0, 100, -100}, maxNew: 2}
	if next, advance := plain.decodeOne(context.Background()); !advance || next != 2 {
		t.Fatalf("unconstrained selection changed: next=%d advance=%v", next, advance)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestNativeThinkingBudgetCompleteStreamUsesActualClose(t *testing.T) {
	for _, stopAtClose := range []bool{false, true} {
		tok := atomicThinkingTokenizer(t)
		cfg := tinyConcurrencyConfig()
		cfg.VocabSize = tok.Vocab()
		m := model.NewSynthetic(cfg)
		m.Quantize()
		p := NewInKernelPlanner(m, tok, "tiny-budget-stream", false, nil, false)
		open, _ := tok.Encode(thinkOpen)
		close, _ := tok.Encode(thinkClose)
		zero := 0.0
		opts := []SampleOpt{
			WithMaxTokens(3), WithThinkingBudget(1), WithTemperature(&zero),
			WithLogitBias(map[int]float64{open[0]: 100, close[0]: -100}),
			WithPerTokenStream(true), WithDecodeTrace(true), WithNativeDecodeTokenIDs(true),
		}
		if stopAtClose {
			opts = append(opts, WithStop([]string{thinkClose}))
		}
		var visible strings.Builder
		comp, err := p.CompleteStream(context.Background(), func(piece string) error {
			visible.WriteString(piece)
			return nil
		}, []Message{{Role: RoleUser, Content: "budget"}}, nil, opts...)
		if err != nil {
			t.Fatal(err)
		}
		want := []int{open[0], open[0], close[0]}
		if comp.NativeDecodeTokenIDs == nil || !reflect.DeepEqual(comp.NativeDecodeTokenIDs.TokenIDs, want) || comp.Usage.CompletionTokens != 3 {
			t.Fatalf("real forced-close IDs/usage lost: ids=%v usage=%+v", comp.NativeDecodeTokenIDs, comp.Usage)
		}
		if visible.String() != comp.Message.Content || visible.Len() != 0 || comp.Message.ReasoningContent == "" {
			t.Fatalf("reasoning escaped projection: visible=%q message=%+v", visible.String(), comp.Message)
		}
		if stopAtClose && comp.FinishReason != "stop" {
			t.Fatalf("closing string stop lost: finish=%q", comp.FinishReason)
		}
		_, err = p.Complete(context.Background(), []Message{{Role: RoleUser, Content: "budget"}}, nil,
			WithThinkingBudget(1), WithTemperature(&zero), WithNativeInferenceReceipt(true))
		var unsupported *model.NativeInferenceReceiptUnsupportedError
		if !errors.As(err, &unsupported) || !strings.Contains(unsupported.Reason, "thinking budget") {
			t.Fatalf("modified-token selection claimed an unmodified receipt: %v", err)
		}
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestNativeThinkingBudgetRoutesAroundSpeculation(t *testing.T) {
	tok := atomicThinkingTokenizer(t)
	cfg := tinyConcurrencyConfig()
	cfg.VocabSize = tok.Vocab()
	m := model.NewSynthetic(cfg)
	p := NewInKernelPlanner(m, tok, "tiny-budget-route", false, nil, false)
	p.quant = false
	p.speculativeEngine = &model.SpeculativeEngine{}
	constraint, err := p.newNativeThinkingConstraint(1, true, nil)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := tok.Encode("x")
	if err := constraint.acceptToken(body[0]); err != nil {
		t.Fatal(err)
	}
	ctx := context.WithValue(context.Background(), nativeThinkingConstraintContextKey{}, constraint)
	var ids []int
	_, err = p.generateReusedRecovering(ctx, []int{3}, 1, 0, 0, 0, nil, 0, 0, nil, func(id int) bool {
		ids = append(ids, id)
		return false
	})
	if err != nil || !reflect.DeepEqual(ids, []int{constraint.closeID}) {
		t.Fatalf("budget bypassed target-only sampler: ids=%v err=%v", ids, err)
	}
}

// fak-test:runtime medium est=2s lane=default
func TestInKernelStreamOfferedVerdictOrdering(t *testing.T) {
	const raw = `before <tool_call>{"name":"Bash","arguments":{}}</tool_call> after`
	for _, names := range [][]string{nil, {"Read"}, {"Bash"}} {
		const outputID = 259
		var doc map[string]any
		if err := json.Unmarshal([]byte(buildByteVocab()), &doc); err != nil {
			t.Fatal(err)
		}
		doc["added_tokens"] = append(doc["added_tokens"].([]any), map[string]any{"id": outputID, "content": raw, "special": true})
		encoded, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		tok, err := tokenizer.ParseJSON(encoded)
		if err != nil {
			t.Fatal(err)
		}
		m := model.NewSynthetic(tinyConcurrencyConfig())
		p := NewInKernelPlanner(m, tok, "offered-stream", false, nil, false)
		p.quant = false
		tools := make([]ToolDef, len(names))
		for i, name := range names {
			tools[i] = ToolDef{Type: "function", Function: ToolDefFunction{Name: name}}
		}
		var streamed strings.Builder
		comp, err := p.CompleteStream(context.Background(), func(piece string) error { streamed.WriteString(piece); return nil }, []Message{{Role: RoleUser, Content: "probe"}}, tools,
			WithPerTokenStream(true), WithMaxTokens(1), WithLogitBias(map[int]float64{outputID: 100}))
		if err != nil {
			t.Fatal(err)
		}
		if streamed.String() != comp.Message.Content {
			t.Fatalf("names=%v streamed=%q final=%q", names, streamed.String(), comp.Message.Content)
		}
		if len(names) == 0 || names[0] != "Bash" {
			if comp.Message.Content != raw || comp.ToolCallsDropped || len(comp.Message.ToolCalls) != 0 {
				t.Fatalf("unoffered native result: %+v", comp)
			}
		} else if len(comp.Message.ToolCalls) != 1 {
			t.Fatalf("offered call not lifted: %+v", comp)
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelDeferredToolVerdictSinkFailure(t *testing.T) {
	sentinel := errors.New("sink closed")
	calls := 0
	p := newInKernelStreamProjector(func(string) error {
		calls++
		if calls > 1 {
			return sentinel
		}
		return nil
	}, nil)
	p.spans.deferTools = true
	const raw = `before <tool_call>{"name":"Bash"}</tool_call> after`
	for _, piece := range []string{"before ", "<tool_", `call>{"name":"Bash"}</tool_call> after`} {
		if err := p.feed(piece); err != nil {
			t.Fatal(err)
		}
	}
	if calls != 1 {
		t.Fatalf("tail emitted before verdict: %d", calls)
	}
	if err := p.finish(raw, true); !errors.Is(err, sentinel) {
		t.Fatalf("finish error=%v", err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelDeferredVerdictWhitespaceBoundaries(t *testing.T) {
	for _, raw := range []string{
		`before <tool_call>{"name":"Bash","arguments":{}}</tool_call>`,
		`  before <tool_call>{"name":"Bash","arguments":{}}</tool_call> after  `,
		`before <tool_call>{"name":"Bash","arguments":{}}</tool_call> after`,
		`{"name":"Bash","arguments":{}}`,
		"```json\n{\"name\":\"Bash\",\"arguments\":{}}\n```",
		"ordinary text with trailing spaces  ",
		"\u00a0before <tool_call>{\"name\":\"Bash\",\"arguments\":{}}</tool_call>",
		"before\u2003<tool_call>{\"name\":\"Bash\",\"arguments\":{}}</tool_call>",
		"before café \u00a0",
		"before\xc2",
	} {
		for _, offered := range []OfferedTools{{}, offeredTestNames("Read"), offeredTestNames("Bash")} {
			var out strings.Builder
			p := newInKernelStreamProjector(func(piece string) error { out.WriteString(piece); return nil }, nil)
			p.spans.deferTools = true
			for _, b := range []byte(raw) {
				if err := p.feed(string(b)); err != nil {
					t.Fatal(err)
				}
			}
			final := LiftTextToolCalls(Message{Content: raw}, offered).Content
			if err := p.finish(final, true); err != nil {
				t.Fatalf("raw=%q finish=%v", raw, err)
			}
			if out.String() != final {
				t.Fatalf("raw=%q got=%q want=%q", raw, out.String(), final)
			}
		}
	}
	var out strings.Builder
	p := newInKernelStreamProjector(func(piece string) error { out.WriteString(piece); return nil }, nil)
	p.spans.deferTools = true
	p.spans.cap = 8
	raw := "before" + strings.Repeat(" \t", 20) + "after"
	for _, b := range []byte(raw) {
		if err := p.feed(string(b)); err != nil {
			t.Fatal(err)
		}
	}
	if !p.spans.deferred || p.spans.trailingSpace.Len() > 8 {
		t.Fatal("whitespace hold exceeded cap")
	}
	if err := p.finish(raw, true); err != nil {
		t.Fatal(err)
	}
	if out.String() != raw {
		t.Fatal("overflow deferral lost content")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInKernelNoToolLateReasoningKeepsLegacyStatus(t *testing.T) {
	for _, answer := range []string{"answer", "```go\nfmt.Println(1)\n```", "[1,2]", `{"value":1}`} {
		var out strings.Builder
		p := newInKernelStreamProjector(func(piece string) error { out.WriteString(piece); return nil }, nil)
		p.spans.deferTools = true
		if err := p.feed("prose<think>reasoning</think>" + answer); err != nil {
			t.Fatal(err)
		}
		if hasTextToolCandidate(answer) {
			t.Fatalf("ordinary answer marked as tool: %q", answer)
		}
		if err := p.finish(answer, hasTextToolCandidate(answer)); err != nil {
			t.Fatalf("new error on existing no-tool late reasoning: %v", err)
		}
		// The historical prefix mismatch remains deferred, without losing newly
		// held ordinary code or nameless JSON after the reasoning span.
		if out.String() != "prose"+answer {
			t.Fatalf("legacy stream=%q", out.String())
		}
	}
}

// fak-test:runtime medium est=2s lane=default
func TestInKernelLegacyReasoningMultipleClosesAndSinkFailure(t *testing.T) {
	for _, answer := range []string{"answer", "[1,2]", "```go\nx()\n```"} {
		var out strings.Builder
		p := newInKernelStreamProjector(func(piece string) error { out.WriteString(piece); return nil }, nil)
		p.spans.deferTools = true
		raw := "prose<think>first</think>middle<think>second</think>" + answer
		for _, b := range []byte(raw) {
			if err := p.feed(string(b)); err != nil {
				t.Fatal(err)
			}
		}
		final := inKernelDecodeToCompletion(t, raw, "stop", nil).Message.Content
		if final != answer {
			t.Fatalf("actual Complete final=%q want=%q", final, answer)
		}
		if err := p.finish(final, false); err != nil {
			t.Fatal(err)
		}
		if out.String() != "prosemiddle"+answer {
			t.Fatalf("duplicated or lost suffix: %q", out.String())
		}
	}
	sentinel := errors.New("legacy suffix sink closed")
	writes := 0
	p := newInKernelStreamProjector(func(string) error {
		writes++
		if writes > 1 {
			return sentinel
		}
		return nil
	}, nil)
	p.spans.deferTools = true
	if err := p.feed("prose<think>reasoning</think>[1,2]"); err != nil {
		t.Fatal(err)
	}
	if err := p.finish("[1,2]", false); !errors.Is(err, sentinel) {
		t.Fatalf("sink error=%v", err)
	}
	if writes != 2 {
		t.Fatalf("writes=%d", writes)
	}
}
