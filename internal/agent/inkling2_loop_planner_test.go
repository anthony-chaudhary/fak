package agent

import (
	"context"
	"errors"
	"fmt"
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
