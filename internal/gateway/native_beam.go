package gateway

import (
	"fmt"
	"math"
	"net/http"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/beam"
)

func chatBeamParams(req ChatRequest) *agent.NativeBeamSearchParams {
	if req.Fak == nil {
		return nil
	}
	return req.Fak.BeamSearch
}

func validateChatBeamSearch(w http.ResponseWriter, req ChatRequest) bool {
	params := chatBeamParams(req)
	if params == nil {
		return true
	}
	fail := func(reason string) bool {
		writeErrCode(w, http.StatusBadRequest, "native_beam_search_unsupported", reason)
		return false
	}
	if err := agent.ValidateNativeBeamSearchParams(params); err != nil {
		return fail(err.Error())
	}
	if req.Stream || req.StreamOptions != nil {
		return fail("beam search requires a nonstreaming request")
	}
	if req.MaxTokens < 1 || req.MaxTokens > beam.MaxSearchSteps {
		return fail(fmt.Sprintf("beam search requires explicit max_tokens in [1,%d]", beam.MaxSearchSteps))
	}
	if len(req.Tools) > 0 || len(req.ToolChoice) > 0 || len(req.ResponseFormat) > 0 || len(req.GuidedDecodeFields()) > 0 ||
		len(req.Stop) > 0 || len(req.LogitBias) > 0 || req.TopP != nil || req.TopK != nil || req.MinP != nil ||
		req.Seed != nil || req.FrequencyPenalty != nil || req.PresencePenalty != nil || req.SpeculativeNMax != nil ||
		len(req.ChatTemplateKwargs) > 0 || len(req.ThinkingConfig) > 0 ||
		(req.ReasoningEffort != "" && req.ReasoningEffort != agent.EffortTierNone) ||
		(req.Temperature != nil && *req.Temperature != 0) ||
		req.FakDecodeTrace || req.Fak.NativeInferenceReceipt || req.Fak.NativeDecodeTokenIDs || req.Fak.NativeDecodeTop5 || req.Fak.MultimodalMode != "" {
		return fail("beam search does not support tools, sampling modifiers, constraints, multimodal input or ordinary token-trace receipts")
	}
	for _, m := range req.Messages {
		if (m.Role != agent.RoleSystem && m.Role != agent.RoleUser && m.Role != agent.RoleAssistant) ||
			m.ContentWasStructured || len(m.VisionImages) > 0 || m.VisionPrompt != "" || len(m.ToolCalls) > 0 ||
			m.FunctionCall != nil || m.ToolCallID != "" || m.Thinking != "" || m.ThinkingSignature != "" ||
			len(m.RedactedThinking) > 0 || m.ReasoningContent != "" {
			return fail("beam search requires plain text non-tool message history")
		}
	}
	return true
}

func (s *Server) chatBeamSearchSupported(reqModel string) bool {
	p := s.planner
	if dual, ok := p.(*DualPlanner); ok {
		if !dual.RoutesLocal(reqModel) {
			return false
		}
		p = dual.Local()
	}
	capable, ok := p.(agent.NativeBeamSearchPlanner)
	return ok && capable.NativeBeamSearchSupported()
}

// NativeBeamSearchWireReceipt deliberately contains NO model content or token
// IDs. The internal agent receipt is trusted, raw search state; copying it onto
// the public wire would bypass the normal winner's adjudication/redaction.
type NativeBeamSearchWireReceipt struct {
	Schema           string                         `json:"schema"`
	Width            int                            `json:"width"`
	LengthPenalty    float64                        `json:"length_penalty"`
	Sequences        []NativeBeamSearchWireSequence `json:"sequences"`
	ForwardCalls     int                            `json:"forward_calls"`
	ReplayedTokens   int                            `json:"replayed_tokens"`
	ReplayTokenBound int                            `json:"replay_token_bound"`
	SessionOwnership string                         `json:"session_ownership"`
}

type NativeBeamSearchWireSequence struct {
	Rank              int     `json:"rank"`
	GeneratedTokens   int     `json:"generated_tokens"`
	CumulativeLogprob float64 `json:"cumulative_logprob"`
	Score             float64 `json:"score"`
	FinishReason      string  `json:"finish_reason"`
}

// nativeBeamWireReceipt copies only finite, bounded non-content fields and
// closed-vocabulary strings. Invalid internal metadata is omitted, never used
// as an alternate text publication channel. Winner admission is unchanged.
func nativeBeamWireReceipt(in *agent.NativeBeamSearchReceipt) *NativeBeamSearchWireReceipt {
	if in == nil || in.Schema != "fak.native-beam-search/1" ||
		in.SessionOwnership != "serial-fresh-owned-cpu-replay" ||
		in.Width < 1 || in.Width > beam.MaxSearchWidth ||
		len(in.Sequences) < 1 || len(in.Sequences) > in.Width ||
		math.IsNaN(in.LengthPenalty) || math.IsInf(in.LengthPenalty, 0) ||
		in.LengthPenalty < 0 || in.LengthPenalty > 2 ||
		in.ForwardCalls < 1 || in.ForwardCalls > 1+in.Width*(beam.MaxSearchSteps-1) ||
		in.ReplayedTokens < in.ForwardCalls || in.ReplayedTokens > in.ReplayTokenBound ||
		in.ReplayTokenBound < 1 || in.ReplayTokenBound > beam.MaxSearchReplayTokens {
		return nil
	}
	out := &NativeBeamSearchWireReceipt{
		Schema: "fak.native-beam-search-scores/1", Width: in.Width,
		LengthPenalty: in.LengthPenalty, ForwardCalls: in.ForwardCalls,
		ReplayedTokens: in.ReplayedTokens, ReplayTokenBound: in.ReplayTokenBound,
		SessionOwnership: "serial-fresh-owned-cpu-replay",
		Sequences:        make([]NativeBeamSearchWireSequence, len(in.Sequences)),
	}
	for i, row := range in.Sequences {
		if row.GeneratedTokens < 1 || row.GeneratedTokens > beam.MaxSearchSteps ||
			math.IsNaN(row.CumulativeLogprob) || math.IsInf(row.CumulativeLogprob, 0) || row.CumulativeLogprob > 0 ||
			math.IsNaN(row.Score) || math.IsInf(row.Score, 0) || row.Score > 0 ||
			(row.FinishReason != "stop" && row.FinishReason != "length") {
			return nil
		}
		out.Sequences[i] = NativeBeamSearchWireSequence{
			Rank: i + 1, GeneratedTokens: row.GeneratedTokens,
			CumulativeLogprob: row.CumulativeLogprob, Score: row.Score, FinishReason: row.FinishReason,
		}
	}
	return out
}
