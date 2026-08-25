package trajectory

import (
	"reflect"
	"testing"
)

func TestDetectQwenMutationChurnReportsRepeatedUnwitnessedWrites(t *testing.T) {
	events := []QwenMutationEvent{
		{TranscriptID: "tx-b", Target: "b.go", Kind: QwenMutationWrite, AccountedTokens: 7, HypothesisID: "h1"},
		{TranscriptID: "tx-b", Target: "b.go", Kind: QwenMutationWrite, AccountedTokens: 11, HypothesisID: "h1"},
		{TranscriptID: "tx-a", Target: "a.go", Kind: QwenMutationWrite, AccountedTokens: 3, HypothesisID: "h2"},
		{TranscriptID: "tx-a", Target: "a.go", Kind: QwenMutationWrite, AccountedTokens: 5, HypothesisID: "h2"},
		{TranscriptID: "tx-a", Target: "a.go", Kind: QwenMutationWrite, AccountedTokens: 13, HypothesisID: "h2"},
	}

	got := DetectQwenMutationChurn(events)
	want := []QwenMutationChurn{
		{TranscriptID: "tx-b", Target: "b.go", Count: 2, AccountedTokens: 18, Intervention: QwenMutationObserveReproFirst},
		{TranscriptID: "tx-a", Target: "a.go", Count: 3, AccountedTokens: 21, Intervention: QwenMutationObserveReproFirst},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("DetectQwenMutationChurn() = %#v, want %#v", got, want)
	}
}

func TestDetectQwenMutationChurnPreservesLegitimateWrites(t *testing.T) {
	tests := []struct {
		name   string
		events []QwenMutationEvent
	}{
		{
			name: "changed hypothesis",
			events: []QwenMutationEvent{
				{TranscriptID: "tx", Target: "a.go", Kind: QwenMutationWrite, AccountedTokens: 2, HypothesisID: "before"},
				{TranscriptID: "tx", Target: "a.go", Kind: QwenMutationWrite, AccountedTokens: 3, HypothesisID: "after"},
			},
		},
		{
			name: "intervening witness",
			events: []QwenMutationEvent{
				{TranscriptID: "tx", Target: "a.go", Kind: QwenMutationWrite, AccountedTokens: 2, HypothesisID: "h"},
				{TranscriptID: "tx", Target: "a.go", Kind: QwenMutationWitness, AccountedTokens: 3, HypothesisID: "h"},
				{TranscriptID: "tx", Target: "a.go", Kind: QwenMutationWrite, AccountedTokens: 5, HypothesisID: "h"},
			},
		},
		{
			name: "multi-file edit",
			events: []QwenMutationEvent{
				{TranscriptID: "tx", Target: "a.go", Kind: QwenMutationWrite, AccountedTokens: 2, HypothesisID: "h"},
				{TranscriptID: "tx", Target: "b.go", Kind: QwenMutationWrite, AccountedTokens: 3, HypothesisID: "h"},
				{TranscriptID: "tx", Target: "a.go", Kind: QwenMutationWrite, AccountedTokens: 5, HypothesisID: "h"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DetectQwenMutationChurn(tt.events); len(got) != 0 {
				t.Fatalf("DetectQwenMutationChurn() = %#v, want no churn", got)
			}
		})
	}
}

func TestDetectQwenMutationChurnSeparatesTranscripts(t *testing.T) {
	events := []QwenMutationEvent{
		{TranscriptID: "tx-a", Target: "same.go", Kind: QwenMutationWrite, AccountedTokens: 2, HypothesisID: "h"},
		{TranscriptID: "tx-b", Target: "same.go", Kind: QwenMutationWrite, AccountedTokens: 3, HypothesisID: "h"},
		{TranscriptID: "tx-a", Target: "same.go", Kind: QwenMutationWrite, AccountedTokens: 5, HypothesisID: "h"},
	}

	if got := DetectQwenMutationChurn(events); len(got) != 0 {
		t.Fatalf("DetectQwenMutationChurn() = %#v, want no cross-transcript churn", got)
	}
}
