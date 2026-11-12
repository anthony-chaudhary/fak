package agent_test

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// admitsNonThinkingKwargs is exercised end to end through the planner: the
// non-thinking DS4.1 profile admits an explicit enable_thinking=false kwarg
// (the shape cmd/fak/bench_h2h.go sends by default) and refuses every other
// contract, closing the typed-502 deviation recorded on the H200 DS4.1 runs.
// fak-test:runtime medium est=2s lane=default
func TestDeepSeekV41ChatProfileAdmitsNonThinkingKwarg(t *testing.T) {
	p := v41SyntheticPlanner(t, v41OracleSubsetTokenizer(t, map[int]string{
		v41UserID: "<｜User｜>", v41AssistantID: "<｜Assistant｜>", v41ThinkEndID: "</think>",
	}), "", nil)
	msgs := []agent.Message{{Role: agent.RoleUser, Content: "hi"}}

	cases := []struct {
		name    string
		kwargs  string
		wantErr bool
	}{
		{"absent kwarg admitted", ``, false},
		{"enable_thinking false admitted", `{"enable_thinking":false}`, false},
		{"whitespace tolerated", ` { "enable_thinking" : false } `, false},
		{"enable_thinking true refused", `{"enable_thinking":true}`, true},
		{"extra key refused", `{"enable_thinking":false,"top_k":1}`, true},
		{"unknown key refused", `{"preserve_thinking":true}`, true},
		{"non-object refused", `"nope"`, true},
		{"non-bool value refused", `{"enable_thinking":"false"}`, true},
		{"empty object refused", `{}`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var opts []agent.SampleOpt
			if tc.kwargs != "" {
				opts = append(opts, agent.WithChatTemplateKwargs(json.RawMessage(tc.kwargs)))
			}
			_, err := p.EncodePrompt(context.Background(), msgs, nil, opts...)
			if tc.wantErr && !errors.Is(err, agent.ErrDeepSeekV41ChatProfile) {
				t.Fatalf("want ErrDeepSeekV41ChatProfile, got %v", err)
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("want admission, got %v", err)
			}
		})
	}
}
