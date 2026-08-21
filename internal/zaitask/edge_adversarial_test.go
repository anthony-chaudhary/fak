package zaitask

import (
	"strings"
	"testing"
)

func TestZAIProductRoutingEdgeAndAdversarialInputs(t *testing.T) {
	cases := []struct {
		name, prompt, class, wantReason string
		wantSuitable                    bool
	}{
		{name: "empty", wantReason: "no task content"},
		{name: "blank", prompt: "  \n\t", wantReason: "no task content"},
		{name: "oversized", prompt: strings.Repeat("x", 1<<20), class: "bounded", wantReason: "task exceeds"},
		{name: "hostile class", prompt: "summarize", class: "bounded; frontier", wantReason: "unsupported task class"},
		{name: "case confused class", prompt: "summarize", class: "BOUNDED", wantReason: "unsupported task class"},
		{name: "frontier", prompt: "design architecture", class: "frontier", wantReason: "reserved"},
		{name: "bounded", prompt: "summarize this diff", class: "bounded", wantSuitable: true, wantReason: "bounded task"},
		{name: "default class", prompt: "summarize this diff", wantSuitable: true, wantReason: "bounded task"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Classify(tc.prompt, tc.class)
			if got.Suitable != tc.wantSuitable || !strings.Contains(got.Reason, tc.wantReason) {
				t.Fatalf("Classify=%+v want suitable=%v reason %q", got, tc.wantSuitable, tc.wantReason)
			}
		})
	}
}
