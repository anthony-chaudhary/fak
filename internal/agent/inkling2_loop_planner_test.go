package agent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func TestInkling2LoopPlannerContextBoundary(t *testing.T) {
	p := inkling2StreamPlanner(t, &inkling2StreamGen{ids: []int{'a', 'b'}})
	msgs := []Message{{Role: RoleUser, Content: "hello"}}
	turn, err := p.Complete(t.Context(), msgs, nil)
	if err != nil {
		t.Fatal(err)
	}
	p.contextTokens = turn.Usage.PromptTokens
	_, err = p.Complete(t.Context(), msgs, nil)
	var overflow *InKernelContextLengthError
	if !errors.As(err, &overflow) || overflow.PromptTokens != p.contextTokens || overflow.MaxContext != p.contextTokens || overflow.MaxNewTokens != Inkling2LoopDefaultMaxTokens {
		t.Fatalf("full prompt: got %#v, err %v", overflow, err)
	}
	p.contextTokens++
	turn, err = p.Complete(t.Context(), msgs, nil)
	if err != nil || turn.Usage.CompletionTokens != 1 {
		t.Fatalf("one token free: turn=%+v err=%v", turn, err)
	}
}

type inkling2FailingGenerator struct{ err error }

func (g inkling2FailingGenerator) Generate(context.Context, []int, int, map[int]bool, func(int) bool) (Inkling2LoopGeneration, error) {
	return Inkling2LoopGeneration{}, g.err
}

func TestInkling2LoopPlannerGenerationFailures(t *testing.T) {
	for _, tc := range []struct {
		name      string
		cause     error
		local     bool
		retryable bool
	}{
		{"unsupported checkpoint", errors.New("checkpoint unavailable"), true, false},
		{"allocation failure", &compute.DeviceAllocError{Bytes: 4096, Class: compute.MemoryScratchpad}, true, true},
		{"existing classification", NewLocalForwardError("prefill", errors.New("device busy"), true), true, true},
		{"device stage", &model.BackendForwardOperationError{}, false, false},
		{"canceled", context.Canceled, false, false},
		{"deadline", context.DeadlineExceeded, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := inkling2StreamPlanner(t, inkling2FailingGenerator{tc.cause})
			_, err := p.Complete(t.Context(), []Message{{Role: RoleUser, Content: "hello"}}, nil)
			if !errors.Is(err, tc.cause) || IsLocalForwardError(err) != tc.local || LocalForwardErrorRetryable(err) != tc.retryable {
				t.Fatalf("unexpected error classification: %v", err)
			}
		})
	}
}

// Byte-level fixture: "Ã" and "©" are the two UTF-8 bytes of "é", so a stop
// string can straddle a token boundary inside one rune.
const (
	inkling2PlannerTokA = iota
	inkling2PlannerTokB
	inkling2PlannerTokLead
	inkling2PlannerTokCont
	inkling2PlannerTokSpecial
)

const inkling2PlannerTokInvalid = 99

func inkling2PlannerTestTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := []string{"a", "b", "Ã", "©", "<|content_text|>"}
	tok, err := tokenizer.FromGGML(vocab, []string{"a b"}, nil, "inkling2")
	if err != nil {
		t.Fatal(err)
	}
	got, err := tok.Decode([]int{inkling2PlannerTokLead, inkling2PlannerTokCont})
	if err != nil || got != "é" {
		t.Fatalf("fixture decode = %q, err %v; want the two bytes of é", got, err)
	}
	return tok
}

// inkling2PlannerLegacyScan is the original planner stop logic: decode the whole
// output on every token. It is the reference the scanner must match byte for byte.
func inkling2PlannerLegacyScan(decode func([]int) (string, error), ids []int, stops []string) (string, bool, error) {
	var out []int
	stopped := false
	for _, id := range ids {
		out = append(out, id)
		if len(stops) == 0 {
			continue
		}
		text, err := decode(out)
		if err != nil {
			continue
		}
		for _, s := range stops {
			if s != "" && strings.Contains(text, s) {
				stopped = true
				break
			}
		}
		if stopped {
			break
		}
	}
	text, err := decode(out)
	if err != nil {
		return "", false, fmt.Errorf("inkling2_loop planner: decode output: %w", err)
	}
	if stopped {
		for _, s := range stops {
			if i := strings.Index(text, s); s != "" && i >= 0 {
				text = text[:i]
			}
		}
	}
	return text, stopped, nil
}

// inkling2PlannerScan drives the scanner the way generation does: feed ids until
// push reports a stop, then finish.
func inkling2PlannerScan(decode func([]int) (string, error), ids []int, stops []string) (string, bool, error) {
	scan := newInkling2StopScanner(decode, stops)
	for _, id := range ids {
		if !scan.push(id) {
			break
		}
	}
	text, err := scan.finish()
	return text, scan.stopped, err
}

func TestInkling2PlannerStopScanLinear(t *testing.T) {
	tok := inkling2PlannerTestTokenizer(t)
	for _, n := range []int{250, 1000} {
		decoded := 0
		counting := func(ids []int) (string, error) {
			decoded += len(ids)
			return tok.Decode(ids)
		}
		scan := newInkling2StopScanner(counting, []string{"zz", "é"})
		for i := 0; i < n; i++ {
			if !scan.push(inkling2PlannerTokA) {
				t.Fatalf("n=%d: scan stopped at token %d without a stop in the output", n, i)
			}
		}
		if _, err := scan.finish(); err != nil {
			t.Fatal(err)
		}
		if decoded > 3*n {
			t.Fatalf("n=%d: decoded %d tokens in total; want O(n) (at most %d)", n, decoded, 3*n)
		}
	}
}

func TestInkling2PlannerStopScanMatchesLegacy(t *testing.T) {
	tok := inkling2PlannerTestTokenizer(t)
	decode := tok.Decode
	alphabet := []int{
		inkling2PlannerTokA, inkling2PlannerTokB, inkling2PlannerTokLead,
		inkling2PlannerTokCont, inkling2PlannerTokSpecial, inkling2PlannerTokInvalid,
	}
	stopSets := [][]string{
		{"é"},
		{"ab"},
		{"b©", "a"},
		{"", "Ã"},
		{"ba", "b"},
		{"<|content_text|>b"},
		{"ÃÃ"},
		{"zz"},
		nil,
	}
	const maxSequence = 5
	var sequences [][]int
	var grow func(prefix []int)
	grow = func(prefix []int) {
		sequences = append(sequences, slices.Clone(prefix))
		if len(prefix) == maxSequence {
			return
		}
		for _, id := range alphabet {
			grow(append(prefix, id))
		}
	}
	grow(nil)

	for _, stops := range stopSets {
		for _, ids := range sequences {
			wantText, wantStopped, wantErr := inkling2PlannerLegacyScan(decode, ids, stops)
			gotText, gotStopped, gotErr := inkling2PlannerScan(decode, ids, stops)
			if (wantErr == nil) != (gotErr == nil) || (wantErr != nil && wantErr.Error() != gotErr.Error()) {
				t.Fatalf("ids %v stops %q: err %v, want %v", ids, stops, gotErr, wantErr)
			}
			if gotText != wantText || gotStopped != wantStopped {
				t.Fatalf("ids %v stops %q: text %q stopped %v; want %q stopped %v", ids, stops, gotText, gotStopped, wantText, wantStopped)
			}
		}
	}
}

func TestInkling2PlannerStopStraddlesTokenBoundary(t *testing.T) {
	tok := inkling2PlannerTestTokenizer(t)
	cases := []struct {
		name  string
		ids   []int
		stops []string
		want  string
	}{
		{"rune split across two tokens", []int{inkling2PlannerTokA, inkling2PlannerTokLead, inkling2PlannerTokCont, inkling2PlannerTokB}, []string{"é"}, "a"},
		{"stop ends on the newest token", []int{inkling2PlannerTokA, inkling2PlannerTokB, inkling2PlannerTokB}, []string{"ab"}, ""},
		{"no stop present", []int{inkling2PlannerTokA, inkling2PlannerTokB}, []string{"zz"}, "ab"},
		{"ordered truncation", []int{inkling2PlannerTokA, inkling2PlannerTokB, inkling2PlannerTokB}, []string{"b", "ab"}, "a"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, _, err := inkling2PlannerScan(tok.Decode, c.ids, c.stops)
			if err != nil || got != c.want {
				t.Fatalf("text %q, err %v; want %q", got, err, c.want)
			}
		})
	}
}

// inkling2FakeLogitGen implements both Inkling2LoopGenerator (greedy) and
// Inkling2LoopLogitGenerator (logit sampling) for deterministic sampling tests.
type inkling2FakeLogitGen struct {
	steps       [][]float32
	greedyCalls int
	logitCalls  int
}

func (g *inkling2FakeLogitGen) Generate(_ context.Context, _ []int, maxNew int, stops map[int]bool, emit func(int) bool) (Inkling2LoopGeneration, error) {
	g.greedyCalls++
	var res Inkling2LoopGeneration
	for _, row := range g.steps {
		if res.Generated >= maxNew {
			return res, nil
		}
		best := 0
		for i, v := range row {
			if v > row[best] {
				best = i
			}
		}
		if stops[best] {
			res.Stopped = true
			return res, nil
		}
		res.Generated++
		if !emit(best) {
			res.Stopped = true
			return res, nil
		}
	}
	res.Stopped = true
	return res, nil
}

func (g *inkling2FakeLogitGen) GenerateLogits(_ context.Context, _ []int, maxNew int, stops map[int]bool, sample func([]float32) (int, error), emit func(int) bool) (Inkling2LoopGeneration, error) {
	g.logitCalls++
	var res Inkling2LoopGeneration
	for _, row := range g.steps {
		if res.Generated >= maxNew {
			return res, nil
		}
		id, err := sample(row)
		if err != nil {
			return res, err
		}
		if stops[id] {
			res.Stopped = true
			return res, nil
		}
		res.Generated++
		if !emit(id) {
			res.Stopped = true
			return res, nil
		}
	}
	res.Stopped = true
	return res, nil
}

func ink2Ptr[T any](v T) *T { return &v }

// ink2LogitRow builds a vocab-sized logit row with -100 everywhere except the
// explicitly supplied token IDs.
func ink2LogitRow(entries map[int]float32) []float32 {
	row := make([]float32, 210)
	for i := range row {
		row[i] = -100
	}
	for id, val := range entries {
		row[id] = val
	}
	return row
}

func TestInkling2LoopPlannerSampling(t *testing.T) {
	msgs := []Message{{Role: RoleUser, Content: "hello"}}

	// 12 identical steps where 'A' is slightly ahead of 'B', 'C', 'D'.
	steps := make([][]float32, 12)
	for i := range steps {
		steps[i] = ink2LogitRow(map[int]float32{'A': 2.0, 'B': 1.9, 'C': 1.8, 'D': 1.7})
	}

	t.Run("greedy fast path when temperature omitted or <= 0", func(t *testing.T) {
		gen := &inkling2FakeLogitGen{steps: steps}
		p := inkling2StreamPlanner(t, gen)
		c1, err := p.Complete(t.Context(), msgs, nil)
		if err != nil {
			t.Fatal(err)
		}
		c2, err := p.Complete(t.Context(), msgs, nil, WithTemperature(ink2Ptr(0.0)))
		if err != nil {
			t.Fatal(err)
		}
		if gen.greedyCalls != 2 || gen.logitCalls != 0 {
			t.Fatalf("calls greedy=%d logit=%d, want 2/0", gen.greedyCalls, gen.logitCalls)
		}
		want := strings.Repeat("A", 12)
		if c1.Message.Content != want || c2.Message.Content != want {
			t.Fatalf("greedy content = %q / %q, want %q", c1.Message.Content, c2.Message.Content, want)
		}
	})

	t.Run("seeded sampling is deterministic and non-greedy", func(t *testing.T) {
		gen := &inkling2FakeLogitGen{steps: steps}
		p := inkling2StreamPlanner(t, gen)
		c1, err := p.Complete(t.Context(), msgs, nil, WithTemperature(ink2Ptr(1.5)), WithSeed(ink2Ptr(int64(42))))
		if err != nil {
			t.Fatal(err)
		}
		c2, err := p.Complete(t.Context(), msgs, nil, WithTemperature(ink2Ptr(1.5)), WithSeed(ink2Ptr(int64(42))))
		if err != nil {
			t.Fatal(err)
		}
		c3, err := p.Complete(t.Context(), msgs, nil, WithTemperature(ink2Ptr(1.5)), WithSeed(ink2Ptr(int64(99))))
		if err != nil {
			t.Fatal(err)
		}
		if gen.logitCalls != 3 || gen.greedyCalls != 0 {
			t.Fatalf("calls greedy=%d logit=%d, want 0/3", gen.greedyCalls, gen.logitCalls)
		}
		if c1.Message.Content != c2.Message.Content {
			t.Fatalf("same seed diverged: %q vs %q", c1.Message.Content, c2.Message.Content)
		}
		if c1.Message.Content == strings.Repeat("A", 12) {
			t.Fatalf("expected non-greedy tokens at temp=1.5, got all argmax %q", c1.Message.Content)
		}
		if c1.Message.Content == c3.Message.Content {
			t.Fatalf("different seeds produced identical output %q", c1.Message.Content)
		}
	})

	t.Run("top_k restricts candidate set", func(t *testing.T) {
		gen := &inkling2FakeLogitGen{steps: steps}
		p := inkling2StreamPlanner(t, gen)
		// top_k=1 must always pick argmax (A).
		c1, err := p.Complete(t.Context(), msgs, nil, WithTemperature(ink2Ptr(2.0)), WithTopK(ink2Ptr(1)), WithSeed(ink2Ptr(int64(7))))
		if err != nil {
			t.Fatal(err)
		}
		if c1.Message.Content != strings.Repeat("A", 12) {
			t.Fatalf("top_k=1 content = %q, want all A", c1.Message.Content)
		}
		// top_k=2 must only ever emit A or B (never C or D).
		c2, err := p.Complete(t.Context(), msgs, nil, WithTemperature(ink2Ptr(2.0)), WithTopK(ink2Ptr(2)), WithSeed(ink2Ptr(int64(7))))
		if err != nil {
			t.Fatal(err)
		}
		if strings.ContainsAny(c2.Message.Content, "CD") {
			t.Fatalf("top_k=2 emitted excluded token in %q", c2.Message.Content)
		}
	})

	t.Run("top_p restricts nucleus", func(t *testing.T) {
		// Dominant token A has ~99% mass; top_p=0.5 keeps only A.
		sharp := make([][]float32, 8)
		for i := range sharp {
			sharp[i] = ink2LogitRow(map[int]float32{'A': 6.0, 'B': 0.0, 'C': 0.0, 'D': 0.0})
		}
		gen := &inkling2FakeLogitGen{steps: sharp}
		p := inkling2StreamPlanner(t, gen)
		c, err := p.Complete(t.Context(), msgs, nil, WithTemperature(ink2Ptr(1.0)), WithTopP(ink2Ptr(0.5)), WithSeed(ink2Ptr(int64(13))))
		if err != nil {
			t.Fatal(err)
		}
		if c.Message.Content != strings.Repeat("A", 8) {
			t.Fatalf("top_p=0.5 content = %q, want all A", c.Message.Content)
		}
	})

	t.Run("non-finite logits return ErrNonFiniteLogits", func(t *testing.T) {
		gen := &inkling2FakeLogitGen{steps: [][]float32{ink2LogitRow(map[int]float32{'A': float32(math.NaN()), 'B': 1.0})}}
		p := inkling2StreamPlanner(t, gen)
		_, err := p.Complete(t.Context(), msgs, nil, WithTemperature(ink2Ptr(0.8)), WithSeed(ink2Ptr(int64(1))))
		if !errors.Is(err, ErrNonFiniteLogits) {
			t.Fatalf("err = %v, want ErrNonFiniteLogits", err)
		}
	})

	t.Run("CompleteStream wires sampling and streams effort=none prefilled text", func(t *testing.T) {
		gen := &inkling2FakeLogitGen{steps: steps[:6]}
		p := inkling2StreamPlanner(t, gen)
		var chunks []string
		comp, err := p.CompleteStream(t.Context(), func(delta string) error {
			chunks = append(chunks, delta)
			return nil
		}, msgs, nil, WithTemperature(ink2Ptr(1.5)), WithSeed(ink2Ptr(int64(42))), WithReasoningEffort("none"))
		if err != nil {
			t.Fatal(err)
		}
		if gen.logitCalls != 1 {
			t.Fatalf("logitCalls = %d, want 1", gen.logitCalls)
		}
		if got := strings.Join(chunks, ""); got != comp.Message.Content {
			t.Fatalf("streamed %q != final %q", got, comp.Message.Content)
		}
	})

	t.Run("max_tokens exhausted inside hidden reasoning reports length and empty content", func(t *testing.T) {
		// Emit <|message_model|><|content_thinking|>AAAA with max_tokens=6 (no end_message).
		// In inkling2StreamTokenizer: 201=<|message_model|>, 205=<|content_thinking|>, 'A'=65.
		gen := &inkling2StreamGen{ids: []int{201, 205, 'A', 'A', 'A', 'A'}}
		p := inkling2StreamPlanner(t, gen)
		comp, err := p.Complete(t.Context(), msgs, nil, WithMaxTokens(6))
		if err != nil {
			t.Fatal(err)
		}
		if comp.FinishReason != "length" || comp.Message.Content != "" || comp.Message.ReasoningContent != "AAAA" {
			t.Fatalf("comp = %+v, want length/empty content/reasoning=AAAA", comp)
		}
	})
}

// fak-test:runtime fast est=1s lane=default
func TestInkling2LoopPlannerSamplerDefaultsPenaltiesAndCycleBreaker(t *testing.T) {
	msgs := []Message{{Role: RoleUser, Content: "hello"}}

	steps := make([][]float32, 16)
	for i := range steps {
		steps[i] = ink2LogitRow(map[int]float32{'A': 2.0, 'B': 1.9, 'C': 1.8, 'D': 1.7})
	}

	t.Run("omitted client temperature defaults to non-greedy GenerateLogits temp=1.0", func(t *testing.T) {
		gen := &inkling2FakeLogitGen{steps: steps}
		p := inkling2StreamPlanner(t, gen)
		comp, err := p.Complete(t.Context(), msgs, nil, WithClientTemperature(nil), WithSeed(ink2Ptr(int64(42))))
		if err != nil {
			t.Fatal(err)
		}
		if gen.logitCalls != 1 || gen.greedyCalls != 0 {
			t.Fatalf("calls greedy=%d logit=%d, want 0/1", gen.greedyCalls, gen.logitCalls)
		}
		if comp.Message.Content == strings.Repeat("A", len(steps)) {
			t.Fatalf("expected non-greedy output at default temp=1.0, got all argmax %q", comp.Message.Content)
		}
	})

	t.Run("temp=0 with presence_penalty routes to GenerateLogits and applies penalty", func(t *testing.T) {
		gen := &inkling2FakeLogitGen{steps: steps[:4]}
		p := inkling2StreamPlanner(t, gen)
		comp, err := p.Complete(t.Context(), msgs, nil,
			WithTemperature(ink2Ptr(0.0)),
			WithPresencePenalty(ink2Ptr(1.5)),
		)
		if err != nil {
			t.Fatal(err)
		}
		if gen.logitCalls != 1 || gen.greedyCalls != 0 {
			t.Fatalf("calls greedy=%d logit=%d, want 0/1", gen.greedyCalls, gen.logitCalls)
		}
		// Step 0 picks A (2.0), step 1 penalizes A to 0.5 so picks B (1.9), then C (1.8), then D (1.7).
		if comp.Message.Content != "ABCD" {
			t.Fatalf("temp=0 + presence_penalty content = %q, want %q", comp.Message.Content, "ABCD")
		}
	})

	t.Run("temp=0 with frequency_penalty or logit_bias routes to GenerateLogits", func(t *testing.T) {
		gen := &inkling2FakeLogitGen{steps: steps[:3]}
		p := inkling2StreamPlanner(t, gen)
		c1, err := p.Complete(t.Context(), msgs, nil,
			WithTemperature(ink2Ptr(0.0)),
			WithFrequencyPenalty(ink2Ptr(1.0)),
		)
		if err != nil {
			t.Fatal(err)
		}
		if c1.Message.Content != "ABC" {
			t.Fatalf("temp=0 + frequency_penalty content = %q, want %q", c1.Message.Content, "ABC")
		}
		c2, err := p.Complete(t.Context(), msgs, nil,
			WithTemperature(ink2Ptr(0.0)),
			WithLogitBias(map[int]float64{'A': -10.0}),
		)
		if err != nil {
			t.Fatal(err)
		}
		if c2.Message.Content != "BBB" {
			t.Fatalf("temp=0 + logit_bias content = %q, want %q", c2.Message.Content, "BBB")
		}
		if gen.logitCalls != 2 || gen.greedyCalls != 0 {
			t.Fatalf("calls greedy=%d logit=%d, want 0/2", gen.greedyCalls, gen.logitCalls)
		}
	})

	t.Run("synthetic 20-token repeating cycle halts within <= 4 repetitions", func(t *testing.T) {
		const period = 20
		cycle := make([]int, period)
		for i := range cycle {
			cycle[i] = 'A' + i
		}
		var repeated []int
		for rep := 0; rep < 20; rep++ {
			repeated = append(repeated, cycle...)
		}
		gen := &inkling2StreamGen{ids: repeated}
		p := inkling2StreamPlanner(t, gen)
		comp, err := p.Complete(t.Context(), msgs, nil, WithMaxTokens(len(repeated)))
		if err != nil {
			t.Fatal(err)
		}
		if comp.FinishReason != "stop" {
			t.Fatalf("FinishReason = %q, want %q", comp.FinishReason, "stop")
		}
		if comp.Usage.CompletionTokens > period*inkling2CycleMinRepeats {
			t.Fatalf("generated %d tokens, want <= %d (<= %d repetitions of period %d)",
				comp.Usage.CompletionTokens, period*inkling2CycleMinRepeats, inkling2CycleMinRepeats, period)
		}
		if comp.Usage.CompletionTokens != period*inkling2CycleMinRepeats {
			t.Fatalf("generated %d tokens, want exact break at %d", comp.Usage.CompletionTokens, period*inkling2CycleMinRepeats)
		}
	})
}
