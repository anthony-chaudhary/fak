package main

import (
	"context"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func inkling2ServeForcedDraftTokenizer(t *testing.T) *tokenizer.Tokenizer {
	t.Helper()
	vocab := map[string]int{
		"<unk>": 0,
		"\n":    1,
		"\t":    2,
	}
	nextID := len(vocab)
	for b := 0x20; b <= 0x7e; b++ {
		ch := string(byte(b))
		if b == ' ' {
			ch = "Ġ"
		}
		vocab[ch] = nextID
		nextID++
	}
	for _, piece := range []string{
		"name",
		"arguments",
		"read_file",
		"path",
		`{"name":"`,
		`","arguments":{"path":"`,
		`"}}`,
		"go",
		"mod",
	} {
		if _, exists := vocab[piece]; !exists {
			vocab[piece] = nextID
			nextID++
		}
	}
	specials := []string{
		"<|end_message|>",
		agent.Inkling2EndOfTurnMarker,
		"<|message_user|>",
		"<|message_system|>",
		"<|message_model|>",
		"<|content_text|>",
		"<|content_thinking|>",
		"<|content_invoke_tool_json|>",
		"<|content_xml|>",
	}
	added := make([]map[string]any, 0, len(specials))
	for _, s := range specials {
		vocab[s] = nextID
		added = append(added, map[string]any{"id": nextID, "content": s, "special": true})
		nextID++
	}
	doc := map[string]any{
		"model": map[string]any{
			"type":   "BPE",
			"vocab":  vocab,
			"merges": []string{},
		},
		"decoder":      map[string]any{"type": "ByteLevel"},
		"added_tokens": added,
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.ParseJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// inkling2ForcedCountingSession models a KV session backed by a target text
// payload and counts exact Prefill batches and serial Step calls regardless of
// multi-token chunking by ForcedDraft.
type inkling2ForcedCountingSession struct {
	tok         *tokenizer.Tokenizer
	targetText  string
	stopID      int
	vocabSize   int
	emittedText string
	prefills    [][]int
	steps       []int
	lastNext    int
}

func (s *inkling2ForcedCountingSession) predictNext() int {
	if len(s.emittedText) >= len(s.targetText) {
		return s.stopID
	}
	rest := s.targetText[len(s.emittedText):]
	ids, err := s.tok.Encode(rest[:1])
	if err != nil || len(ids) == 0 {
		return s.stopID
	}
	return ids[0]
}

func (s *inkling2ForcedCountingSession) appendTokens(tokens []int) {
	if len(tokens) == 0 {
		return
	}
	if txt, err := s.tok.Decode(tokens); err == nil {
		s.emittedText += txt
	}
}

func (s *inkling2ForcedCountingSession) Step(token int) (int, error) {
	s.steps = append(s.steps, token)
	s.appendTokens([]int{token})
	s.lastNext = s.predictNext()
	return s.lastNext, nil
}

func (s *inkling2ForcedCountingSession) LastLogits() ([]float32, error) {
	logits := make([]float32, s.vocabSize)
	if s.lastNext >= 0 && s.lastNext < s.vocabSize {
		logits[s.lastNext] = 10
	}
	return logits, nil
}

func (s *inkling2ForcedCountingSession) Close() error { return nil }

// inkling2PromptAwareSession separates prompt tokens from fused forced-draft
// suffix tokens in Prefill accounting by tracking initialPromptLen.
type inkling2PromptAwareSession struct {
	*inkling2ForcedCountingSession
	initialPromptLen int
	prefilledPrompt  int
}

func (s *inkling2PromptAwareSession) Prefill(_ context.Context, tokens []int) (int, error) {
	cp := append([]int(nil), tokens...)
	s.prefills = append(s.prefills, cp)
	if s.prefilledPrompt < s.initialPromptLen {
		remPrompt := s.initialPromptLen - s.prefilledPrompt
		if len(cp) <= remPrompt {
			s.prefilledPrompt += len(cp)
		} else {
			s.prefilledPrompt = s.initialPromptLen
			s.appendTokens(cp[remPrompt:])
		}
	} else {
		s.appendTokens(cp)
	}
	s.lastNext = s.predictNext()
	return s.lastNext, nil
}

// fak-test:runtime fast est=50ms lane=default
func TestServeInkling2ForcedDraftReducesStepsAndPreservesParity(t *testing.T) {
	tok := inkling2ServeForcedDraftTokenizer(t)
	stopID, _ := tok.SpecialID(agent.Inkling2EndOfTurnMarker)
	tools := []agent.ToolDef{{
		Type: "function",
		Function: agent.ToolDefFunction{
			Name:       "read_file",
			Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}`),
		},
	}}
	choice := json.RawMessage(`{"type":"function","function":{"name":"read_file"}}`)
	const targetText = `{"name":"read_file","arguments":{"path":"go.mod"}}`

	// Measure prompt token length with a baseline step-by-step planner (no GenerateDraft).
	var baseSess *inkling2PromptAwareSession
	baseEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		baseSess = &inkling2PromptAwareSession{
			inkling2ForcedCountingSession: &inkling2ForcedCountingSession{
				tok:        tok,
				targetText: targetText,
				stopID:     stopID,
				vocabSize:  tok.Vocab(),
			},
			initialPromptLen: 1 << 30,
		}
		return baseSess, nil
	})
	defer baseEng.Close()

	type plainGenOnly struct{ agent.Inkling2LoopGenerator }
	plainPlanner, err := newServeInkling2LoopPlanner(plainGenOnly{baseEng}, tok, "inkling2-mini", 4096)
	if err != nil {
		t.Fatal(err)
	}

	msgs := []agent.Message{{Role: agent.RoleUser, Content: "read go.mod"}}
	baseComp, err := plainPlanner.Complete(context.Background(), msgs, tools, agent.WithToolChoice(choice), agent.WithReasoningEffort(agent.EffortTierNone), agent.WithMaxTokens(128))
	if err != nil {
		t.Fatal(err)
	}
	if len(baseSess.steps) == 0 {
		t.Fatal("baseline step-by-step decode unexpectedly ran zero Step calls")
	}
	promptTokens := baseComp.Usage.PromptTokens

	// Run 2: ForcedDraft wired (both greedy temp=0 and sampled temp=0.7).
	for _, temp := range []float64{0, 0.7} {
		var draftSess *inkling2PromptAwareSession
		draftEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
			draftSess = &inkling2PromptAwareSession{
				inkling2ForcedCountingSession: &inkling2ForcedCountingSession{
					tok:        tok,
					targetText: targetText,
					stopID:     stopID,
					vocabSize:  tok.Vocab(),
				},
				initialPromptLen: promptTokens,
			}
			return draftSess, nil
		})
		draftPlanner, err := newServeInkling2LoopPlanner(draftEng, tok, "inkling2-mini", 4096)
		if err != nil {
			t.Fatal(err)
		}
		tVal := temp
		draftComp, err := draftPlanner.Complete(context.Background(), msgs, tools, agent.WithToolChoice(choice), agent.WithReasoningEffort(agent.EffortTierNone), agent.WithTemperature(&tVal), agent.WithMaxTokens(128))
		draftEng.Close()
		if err != nil {
			t.Fatalf("temp=%g draft Complete failed: %v", temp, err)
		}

		if len(draftComp.Message.ToolCalls) != 1 || len(baseComp.Message.ToolCalls) != 1 {
			t.Fatalf("temp=%g missing tool call: draft=%+v base=%+v", temp, draftComp.Message.ToolCalls, baseComp.Message.ToolCalls)
		}
		if draftComp.Message.ToolCalls[0].Function != baseComp.Message.ToolCalls[0].Function {
			t.Fatalf("temp=%g tool call mismatch: got %+v, want %+v", temp, draftComp.Message.ToolCalls[0].Function, baseComp.Message.ToolCalls[0].Function)
		}
		if draftComp.Usage.CompletionTokens > baseComp.Usage.CompletionTokens {
			t.Fatalf("temp=%g completion tokens=%d exceeded baseline %d", temp, draftComp.Usage.CompletionTokens, baseComp.Usage.CompletionTokens)
		}
		baseForwardCalls := len(baseSess.prefills) + len(baseSess.steps)
		draftForwardCalls := len(draftSess.prefills) + len(draftSess.steps)
		if len(draftSess.steps) >= len(baseSess.steps) {
			t.Fatalf("temp=%g steps not reduced: draft=%d baseline=%d", temp, len(draftSess.steps), len(baseSess.steps))
		}
		if draftForwardCalls >= baseForwardCalls {
			t.Fatalf("temp=%g forward calls not reduced: draft=%d baseline=%d", temp, draftForwardCalls, baseForwardCalls)
		}
		t.Logf("temp=%g step reduction: baseline_steps=%d draft_steps=%d (total_forward: %d -> %d, batched_prefills=%d)",
			temp, len(baseSess.steps), len(draftSess.steps), baseForwardCalls, draftForwardCalls, len(draftSess.prefills))
	}

	// Run 3: Batch engine routes GenerateDraft cleanly through scalarGenerateDraft.
	var batchSess *inkling2PromptAwareSession
	scalarEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		batchSess = &inkling2PromptAwareSession{
			inkling2ForcedCountingSession: &inkling2ForcedCountingSession{
				tok:        tok,
				targetText: targetText,
				stopID:     stopID,
				vocabSize:  tok.Vocab(),
			},
			initialPromptLen: promptTokens,
		}
		return batchSess, nil
	})
	defer scalarEng.Close()
	batchEng := newInkling2LoopBatchEngineWith(scalarEng.Generate, nil, testInkling2Admission(inkling2RealClock{}), 4096)
	batchEng.scalarGenerateLogits = scalarEng.GenerateLogits
	batchEng.scalarGenerateDraft = scalarEng.GenerateDraft
	go batchEng.run()
	defer close(batchEng.queue)

	batchPlanner, err := newServeInkling2LoopPlanner(batchEng, tok, "inkling2-mini", 4096)
	if err != nil {
		t.Fatal(err)
	}
	batchComp, err := batchPlanner.Complete(context.Background(), msgs, tools, agent.WithToolChoice(choice), agent.WithReasoningEffort(agent.EffortTierNone), agent.WithMaxTokens(128))
	if err != nil {
		t.Fatal(err)
	}
	if len(batchComp.Message.ToolCalls) != 1 || batchComp.Message.ToolCalls[0].Function != baseComp.Message.ToolCalls[0].Function {
		t.Fatalf("batch ForcedDraft mismatch: got %+v, want %+v", batchComp.Message.ToolCalls, baseComp.Message.ToolCalls)
	}
	if len(batchSess.steps) >= len(baseSess.steps) {
		t.Fatalf("batch ForcedDraft did not reduce steps: got %d, baseline %d", len(batchSess.steps), len(baseSess.steps))
	}

	// Run 4: max_tokens truncation mid-initial-forced span stops cleanly without panics or extra tokens.
	var truncSess *inkling2PromptAwareSession
	truncEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		truncSess = &inkling2PromptAwareSession{
			inkling2ForcedCountingSession: &inkling2ForcedCountingSession{
				tok:        tok,
				targetText: targetText,
				stopID:     stopID,
				vocabSize:  tok.Vocab(),
			},
			initialPromptLen: promptTokens,
		}
		return truncSess, nil
	})
	defer truncEng.Close()
	truncPlanner, err := newServeInkling2LoopPlanner(truncEng, tok, "inkling2-mini", 4096)
	if err != nil {
		t.Fatal(err)
	}
	truncComp, err := truncPlanner.Complete(context.Background(), msgs, tools, agent.WithToolChoice(choice), agent.WithReasoningEffort(agent.EffortTierNone), agent.WithMaxTokens(2))
	if err != nil {
		t.Fatal(err)
	}
	if truncComp.Usage.CompletionTokens != 2 || len(truncSess.steps) != 0 {
		t.Fatalf("max_tokens=2 forced truncation: completion=%d steps=%d, want 2 completion and 0 steps", truncComp.Usage.CompletionTokens, len(truncSess.steps))
	}
	if !strings.Contains(truncComp.FinishReason, "length") && !truncComp.ToolCallsDropped {
		t.Fatalf("expected length/oversize finish on forced truncation, got %+v", truncComp)
	}
	if slices.Contains(truncSess.prefills[0], stopID) {
		t.Fatalf("unexpected stop token in prefill: %v", truncSess.prefills[0])
	}
}

type inkling2ForcedVerifyMockSession struct {
	*inkling2PromptAwareSession
	forcedSpanCommits [][]int
}

func (s *inkling2ForcedVerifyMockSession) CommitForcedSpan(_ context.Context, prevToken int, forced []int) (int, bool, error) {
	span := make([]int, 1+len(forced))
	span[0] = prevToken
	copy(span[1:], forced)
	s.forcedSpanCommits = append(s.forcedSpanCommits, span)
	s.appendTokens(span)
	s.lastNext = s.predictNext()
	return s.lastNext, true, nil
}

// fak-test:runtime fast est=50ms lane=default
func TestServeInkling2ForcedDraftUsesForcedVerifySession(t *testing.T) {
	tok := inkling2ServeForcedDraftTokenizer(t)
	stopID, _ := tok.SpecialID(agent.Inkling2EndOfTurnMarker)
	tools := []agent.ToolDef{{
		Type: "function",
		Function: agent.ToolDefFunction{
			Name:       "read_file",
			Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}`),
		},
	}}
	choice := json.RawMessage(`{"type":"function","function":{"name":"read_file"}}`)
	const targetText = `{"name":"read_file","arguments":{"path":"go.mod"}}`

	var baseSess *inkling2PromptAwareSession
	baseEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		baseSess = &inkling2PromptAwareSession{
			inkling2ForcedCountingSession: &inkling2ForcedCountingSession{
				tok:        tok,
				targetText: targetText,
				stopID:     stopID,
				vocabSize:  tok.Vocab(),
			},
			initialPromptLen: 1 << 30,
		}
		return baseSess, nil
	})
	defer baseEng.Close()
	type plainGenOnly struct{ agent.Inkling2LoopGenerator }
	plainPlanner, err := newServeInkling2LoopPlanner(plainGenOnly{baseEng}, tok, "inkling2-mini", 4096)
	if err != nil {
		t.Fatal(err)
	}
	msgs := []agent.Message{{Role: agent.RoleUser, Content: "read go.mod"}}
	baseComp, err := plainPlanner.Complete(context.Background(), msgs, tools, agent.WithToolChoice(choice), agent.WithReasoningEffort(agent.EffortTierNone), agent.WithMaxTokens(128))
	if err != nil {
		t.Fatal(err)
	}

	var fvSess *inkling2ForcedVerifyMockSession
	fvEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		fvSess = &inkling2ForcedVerifyMockSession{
			inkling2PromptAwareSession: &inkling2PromptAwareSession{
				inkling2ForcedCountingSession: &inkling2ForcedCountingSession{
					tok:        tok,
					targetText: targetText,
					stopID:     stopID,
					vocabSize:  tok.Vocab(),
				},
				initialPromptLen: baseComp.Usage.PromptTokens,
			},
		}
		return fvSess, nil
	})
	defer fvEng.Close()
	fvPlanner, err := newServeInkling2LoopPlanner(fvEng, tok, "inkling2-mini", 4096)
	if err != nil {
		t.Fatal(err)
	}
	fvComp, err := fvPlanner.Complete(context.Background(), msgs, tools, agent.WithToolChoice(choice), agent.WithReasoningEffort(agent.EffortTierNone), agent.WithMaxTokens(128))
	if err != nil {
		t.Fatal(err)
	}
	if len(fvComp.Message.ToolCalls) != 1 || fvComp.Message.ToolCalls[0].Function != baseComp.Message.ToolCalls[0].Function {
		t.Fatalf("ForcedVerify tool call mismatch: got %+v, want %+v", fvComp.Message.ToolCalls, baseComp.Message.ToolCalls)
	}
	if len(fvSess.forcedSpanCommits) == 0 {
		t.Fatal("expected serve() to call CommitForcedSpan on Inkling2LoopForcedVerifySession")
	}
	if len(fvSess.prefills) != 1 {
		t.Fatalf("expected only initial Prefill (1 call) when CommitForcedSpan handles mid-decode forced spans, got %d prefills", len(fvSess.prefills))
	}
}

type inkling2MaskedLogitBiasSession struct {
	*inkling2ForcedVerifyMockSession
	poisonID int
	validID  int
}

func (s *inkling2MaskedLogitBiasSession) LastLogits() ([]float32, error) {
	logits, err := s.inkling2ForcedVerifyMockSession.LastLogits()
	if err != nil {
		return nil, err
	}
	if s.lastNext == s.validID && s.poisonID >= 0 && s.poisonID < len(logits) {
		// Force an invalid grammar token above the valid token unless MaskLogits masks it.
		logits[s.poisonID] = 100
		logits[s.validID] = 50
	}
	return logits, nil
}

// fak-test:runtime fast est=50ms lane=default
func TestServeInkling2ForcedDraftAppliesMaskLogitsAtBranchingPositions(t *testing.T) {
	tok := inkling2ServeForcedDraftTokenizer(t)
	stopID, _ := tok.SpecialID(agent.Inkling2EndOfTurnMarker)
	tools := []agent.ToolDef{{
		Type: "function",
		Function: agent.ToolDefFunction{
			Name:       "read_file",
			Parameters: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"count":{"type":"integer"}},"required":["path","count"],"additionalProperties":false}`),
		},
	}}
	choice := json.RawMessage(`{"type":"function","function":{"name":"read_file"}}`)
	const targetText = `{"name":"read_file","arguments":{"path":"go.mod","count":7}}`

	digitIDs, err := tok.Encode("7")
	if err != nil || len(digitIDs) != 1 {
		t.Fatalf("encode 7: ids=%v err=%v", digitIDs, err)
	}
	quoteIDs, err := tok.Encode(`"`)
	if err != nil || len(quoteIDs) != 1 {
		t.Fatalf("encode quote: ids=%v err=%v", quoteIDs, err)
	}

	msgs := []agent.Message{{Role: agent.RoleUser, Content: "read go.mod"}}
	baseEng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		return &inkling2PromptAwareSession{
			inkling2ForcedCountingSession: &inkling2ForcedCountingSession{
				tok:        tok,
				targetText: targetText,
				stopID:     stopID,
				vocabSize:  tok.Vocab(),
			},
			initialPromptLen: 1 << 30,
		}, nil
	})
	type plainGenOnly struct{ agent.Inkling2LoopGenerator }
	basePlanner, err := newServeInkling2LoopPlanner(plainGenOnly{baseEng}, tok, "inkling2-mini", 4096)
	if err != nil {
		baseEng.Close()
		t.Fatal(err)
	}
	baseComp, err := basePlanner.Complete(context.Background(), msgs, tools, agent.WithToolChoice(choice), agent.WithReasoningEffort(agent.EffortTierNone), agent.WithMaxTokens(128))
	baseEng.Close()
	if err != nil {
		t.Fatal(err)
	}
	promptTokens := baseComp.Usage.PromptTokens

	for _, temp := range []float64{0, 0.7} {
		var sess *inkling2MaskedLogitBiasSession
		eng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
			sess = &inkling2MaskedLogitBiasSession{
				inkling2ForcedVerifyMockSession: &inkling2ForcedVerifyMockSession{
					inkling2PromptAwareSession: &inkling2PromptAwareSession{
						inkling2ForcedCountingSession: &inkling2ForcedCountingSession{
							tok:        tok,
							targetText: targetText,
							stopID:     stopID,
							vocabSize:  tok.Vocab(),
						},
						initialPromptLen: promptTokens,
					},
				},
				poisonID: quoteIDs[0],
				validID:  digitIDs[0],
			}
			return sess, nil
		})
		planner, err := newServeInkling2LoopPlanner(eng, tok, "inkling2-mini", 4096)
		if err != nil {
			eng.Close()
			t.Fatal(err)
		}
		tVal := temp
		comp, err := planner.Complete(context.Background(), msgs, tools, agent.WithToolChoice(choice), agent.WithReasoningEffort(agent.EffortTierNone), agent.WithTemperature(&tVal), agent.WithMaxTokens(128))
		eng.Close()
		if err != nil {
			t.Fatalf("temp=%g Complete: %v", temp, err)
		}
		if len(comp.Message.ToolCalls) != 1 || comp.Message.ToolCalls[0].Function.Name != "read_file" {
			t.Fatalf("temp=%g missing read_file tool call: %+v", temp, comp.Message.ToolCalls)
		}
		if comp.Message.ToolCalls[0].Function.Arguments != `{"path":"go.mod","count":7}` {
			t.Fatalf("temp=%g args = %q, want %q", temp, comp.Message.ToolCalls[0].Function.Arguments, `{"path":"go.mod","count":7}`)
		}
		if len(sess.forcedSpanCommits) == 0 {
			t.Fatalf("temp=%g expected CommitForcedSpan to be invoked", temp)
		}
	}
}
