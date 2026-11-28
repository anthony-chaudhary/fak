package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolgrammar"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// FusionToolCallSpec describes a single grammar-constrained tool call in a fused turn.
type FusionToolCallSpec struct {
	Index   int
	ID      string
	Tool    string
	Grammar *toolgrammar.CompiledGrammar
	Payload string
	Meta    map[string]string
}

// FusionTurnReceipt summarizes the deterministic outcome of a fused
// toolgrammar.ForcedDraft + vdso.StreamingPrefetcher turn.
type FusionTurnReceipt struct {
	SequentialSteps    int                    `json:"sequential_steps"`
	FusedSteps         int                    `json:"fused_steps"`
	StepReductionRatio float64                `json:"step_reduction_ratio"`
	PrefetchedHits     int                    `json:"prefetched_hits"`
	RefusedWrites      int                    `json:"refused_writes"`
	ForcedTokens       int                    `json:"forced_tokens"`
	ForcedSpans        int                    `json:"forced_spans"`
	PrefetchReceipts   []vdso.PrefetchReceipt `json:"prefetch_receipts,omitempty"`
}

// KernelFusionController fuses zero-compute toolgrammar.ForcedDraft spans with
// vdso.StreamingPrefetcher default-deny Tier-2 read-only prefetching.
type KernelFusionController struct {
	vocab      toolgrammar.VocabMatcher
	vdso       *vdso.VDSO
	prefetcher *vdso.StreamingPrefetcher
}

// NewKernelFusionController constructs a KernelFusionController backed by vocab,
// vDSO cache v (or vdso.Default when nil), and prefetch executor exec.
func NewKernelFusionController(
	vocab toolgrammar.VocabMatcher,
	v *vdso.VDSO,
	exec vdso.PrefetchExecutor,
) (*KernelFusionController, error) {
	if vocab == nil {
		return nil, fmt.Errorf("agent: nil vocab matcher")
	}
	if v == nil {
		v = vdso.Default
	}
	return &KernelFusionController{
		vocab:      vocab,
		vdso:       v,
		prefetcher: vdso.NewStreamingPrefetcher(v, exec),
	}, nil
}

// RunTurn executes a sequence of grammar-constrained tool calls through the fused
// ForcedDraft + StreamingPrefetcher pipeline and returns a deterministic receipt.
func (c *KernelFusionController) RunTurn(ctx context.Context, specs []FusionToolCallSpec) (FusionTurnReceipt, error) {
	if c == nil {
		return FusionTurnReceipt{}, fmt.Errorf("agent: nil KernelFusionController")
	}

	var receipt FusionTurnReceipt
	for _, spec := range specs {
		if err := c.runToolCall(ctx, spec, &receipt); err != nil {
			return receipt, err
		}
	}

	if receipt.FusedSteps > 0 {
		receipt.StepReductionRatio = float64(receipt.SequentialSteps) / float64(receipt.FusedSteps)
	}
	receipt.PrefetchReceipts = c.prefetcher.Receipts()
	return receipt, nil
}

func (c *KernelFusionController) runToolCall(
	ctx context.Context,
	spec FusionToolCallSpec,
	receipt *FusionTurnReceipt,
) error {
	seqSteps, err := c.countSequentialTokens(spec.Payload)
	if err != nil {
		return fmt.Errorf("agent: tool %q sequential tokenize: %w", spec.Tool, err)
	}
	receipt.SequentialSteps += seqSteps

	state, err := toolgrammar.NewJumpForwardState(spec.Grammar)
	if err != nil {
		return fmt.Errorf("agent: tool %q init jump-forward: %w", spec.Tool, err)
	}

	// Seed tool identity and metadata on the stream slot before argument fragments arrive.
	c.prefetcher.Feed(ctx, vdso.StreamToolDelta{
		Index: spec.Index,
		ID:    spec.ID,
		Tool:  spec.Tool,
		Meta:  spec.Meta,
	})

	argsExtractor := newArgsFragmentExtractor(spec.Grammar != nil && spec.Grammar.IncludeToolTag)
	rem := spec.Payload
	for !state.IsComplete() {
		forcedToks, forcedTxt := toolgrammar.ForcedDraft(state, c.vocab)
		if len(forcedToks) > 0 && strings.HasPrefix(rem, forcedTxt) {
			if err := state.AdvanceText(forcedTxt); err != nil {
				return fmt.Errorf("agent: tool %q advance forced %q: %w", spec.Tool, forcedTxt, err)
			}
			receipt.ForcedTokens += len(forcedToks)
			receipt.ForcedSpans++
			receipt.FusedSteps++
			rem = rem[len(forcedTxt):]
			if frag := argsExtractor.feed(forcedTxt); frag != "" {
				c.prefetcher.Feed(ctx, vdso.StreamToolDelta{
					Index:        spec.Index,
					ArgsFragment: frag,
				})
			}
			continue
		}

		if len(rem) == 0 {
			return fmt.Errorf("agent: tool %q incomplete grammar state with empty remainder", spec.Tool)
		}
		_, n, ok := c.vocab.MatchPrefix(rem)
		if !ok || n <= 0 || n > len(rem) {
			return fmt.Errorf("agent: tool %q vocab match failed on %q", spec.Tool, rem)
		}
		tokTxt := rem[:n]
		if err := state.AdvanceText(tokTxt); err != nil {
			return fmt.Errorf("agent: tool %q advance token %q: %w", spec.Tool, tokTxt, err)
		}
		receipt.FusedSteps++
		rem = rem[n:]
		if frag := argsExtractor.feed(tokTxt); frag != "" {
			c.prefetcher.Feed(ctx, vdso.StreamToolDelta{
				Index:        spec.Index,
				ArgsFragment: frag,
			})
		}
	}

	c.prefetcher.Feed(ctx, vdso.StreamToolDelta{
		Index:        spec.Index,
		FinishReason: "tool_calls",
	})

	prefRec, ok := c.prefetcher.Receipt(spec.Index)
	if !ok {
		return nil
	}
	if prefRec.Reason.Refused() {
		receipt.RefusedWrites++
		return nil
	}

	probe := &abi.ToolCall{
		Tool: spec.Tool,
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: prefRec.Args,
			Len:    int64(len(prefRec.Args)),
		},
		Meta: spec.Meta,
	}
	if _, hit := c.vdso.Lookup(ctx, probe); hit {
		receipt.PrefetchedHits++
	}
	return nil
}

func (c *KernelFusionController) countSequentialTokens(payload string) (int, error) {
	steps := 0
	rem := payload
	for len(rem) > 0 {
		_, n, ok := c.vocab.MatchPrefix(rem)
		if !ok || n <= 0 || n > len(rem) {
			return 0, fmt.Errorf("vocab failed to match prefix of %q", rem)
		}
		rem = rem[n:]
		steps++
	}
	return steps, nil
}

// argsFragmentExtractor forwards only the inner "arguments" JSON object bytes to
// vdso.StreamingPrefetcher when the grammar wraps calls in {"name":...,"arguments":...}.
type argsFragmentExtractor struct {
	includeToolTag bool
	inArgs         bool
	done           bool
	depth          int
	inString       bool
	escaped        bool
	pending        strings.Builder
}

func newArgsFragmentExtractor(includeToolTag bool) *argsFragmentExtractor {
	return &argsFragmentExtractor{includeToolTag: includeToolTag}
}

func (e *argsFragmentExtractor) feed(chunk string) string {
	if !e.includeToolTag {
		return chunk
	}
	if e.done || chunk == "" {
		return ""
	}
	if !e.inArgs {
		e.pending.WriteString(chunk)
		buf := e.pending.String()
		const marker = `"arguments":`
		idx := strings.Index(buf, marker)
		if idx < 0 {
			return ""
		}
		e.inArgs = true
		chunk = buf[idx+len(marker):]
		e.pending.Reset()
	}

	var out strings.Builder
	for i := 0; i < len(chunk); i++ {
		ch := chunk[i]
		if e.depth == 0 && (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
			continue
		}
		out.WriteByte(ch)
		if e.inString {
			if e.escaped {
				e.escaped = false
				continue
			}
			if ch == '\\' {
				e.escaped = true
				continue
			}
			if ch == '"' {
				e.inString = false
			}
			continue
		}
		switch ch {
		case '"':
			e.inString = true
		case '{':
			e.depth++
		case '}':
			e.depth--
			if e.depth == 0 {
				e.done = true
				return out.String()
			}
		}
	}
	return out.String()
}
