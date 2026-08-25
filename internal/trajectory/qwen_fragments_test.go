package trajectory

import (
	"errors"
	"reflect"
	"testing"
)

func TestCanonicalizeQwenFragmentsDeduplicatesMessagesAndRetainsProvenance(t *testing.T) {
	fragments := []QwenFragment{
		{SourceKind: QwenFragmentClaude, SourcePath: "z.jsonl", TranscriptID: "transcript-1", FragmentID: "part-2", Messages: []QwenFragmentMessage{{ID: "m2", Usage: QwenCanonicalUsage{InputTokens: 7, OutputTokens: 3}}}},
		{SourceKind: QwenFragmentClaude, SourcePath: "a.jsonl", TranscriptID: "transcript-1", FragmentID: "part-1", Messages: []QwenFragmentMessage{{ID: "m1", Usage: QwenCanonicalUsage{InputTokens: 5, CacheReadTokens: 2}}, {ID: "m2", Usage: QwenCanonicalUsage{InputTokens: 7, OutputTokens: 3}}}},
		{SourceKind: QwenFragmentClaude, SourcePath: "empty.jsonl", TranscriptID: "transcript-1", FragmentID: "part-3"},
	}

	got, err := CanonicalizeQwenFragments(fragments)
	if err != nil {
		t.Fatal(err)
	}
	if got.RawFragmentCount != 3 || got.CanonicalTranscriptCount != 1 {
		t.Fatalf("counts = raw %d canonical %d", got.RawFragmentCount, got.CanonicalTranscriptCount)
	}
	wantUsage := QwenCanonicalUsage{InputTokens: 12, CacheReadTokens: 2, OutputTokens: 3}
	if !reflect.DeepEqual(got.Transcripts[0].Usage, wantUsage) {
		t.Fatalf("usage = %+v, want %+v", got.Transcripts[0].Usage, wantUsage)
	}
	wantProvenance := []QwenFragmentProvenance{{SourcePath: "a.jsonl", FragmentID: "part-1"}, {SourcePath: "empty.jsonl", FragmentID: "part-3"}, {SourcePath: "z.jsonl", FragmentID: "part-2"}}
	if !reflect.DeepEqual(got.Transcripts[0].Provenance, wantProvenance) {
		t.Fatalf("provenance = %#v, want %#v", got.Transcripts[0].Provenance, wantProvenance)
	}
}

func TestCanonicalizeQwenFragmentsKeepsUnrelatedTranscriptsSeparateAndStable(t *testing.T) {
	fragments := []QwenFragment{
		{SourceKind: QwenFragmentCodex, SourcePath: "b", SessionID: "session-b", Messages: []QwenFragmentMessage{{ID: "same", Usage: QwenCanonicalUsage{OutputTokens: 2}}}},
		{SourceKind: QwenFragmentClaude, SourcePath: "a", TranscriptID: "transcript-a", Messages: []QwenFragmentMessage{{ID: "same", Usage: QwenCanonicalUsage{OutputTokens: 1}}}},
	}
	got, err := CanonicalizeQwenFragments(fragments)
	if err != nil {
		t.Fatal(err)
	}
	if got.CanonicalTranscriptCount != 2 {
		t.Fatalf("canonical count = %d", got.CanonicalTranscriptCount)
	}
	if got.Transcripts[0].SourceKind != QwenFragmentClaude || got.Transcripts[0].Identity != "transcript-a" || got.Transcripts[1].Identity != "session-b" {
		t.Fatalf("unstable transcripts: %#v", got.Transcripts)
	}
}

func TestCanonicalizeQwenFragmentsRefusesMissingOrAmbiguousIdentity(t *testing.T) {
	tests := []struct {
		name string
		in   QwenFragment
		code QwenFragmentRefusalCode
	}{
		{name: "missing transcript", in: QwenFragment{SourceKind: QwenFragmentClaude}, code: QwenFragmentIdentityMissing},
		{name: "ambiguous claude identity", in: QwenFragment{SourceKind: QwenFragmentClaude, TranscriptID: "t", SessionID: "s"}, code: QwenFragmentIdentityAmbiguous},
		{name: "missing message id", in: QwenFragment{SourceKind: QwenFragmentCodex, SessionID: "s", Messages: []QwenFragmentMessage{{}}}, code: QwenFragmentIdentityMissing},
		{name: "conflicting duplicate usage", in: QwenFragment{SourceKind: QwenFragmentCodex, SessionID: "s", Messages: []QwenFragmentMessage{{ID: "m", Usage: QwenCanonicalUsage{InputTokens: 1}}, {ID: "m", Usage: QwenCanonicalUsage{InputTokens: 2}}}}, code: QwenFragmentIdentityAmbiguous},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := CanonicalizeQwenFragments([]QwenFragment{tt.in})
			var refusal *QwenFragmentRefusal
			if !errors.As(err, &refusal) || refusal.Code != tt.code {
				t.Fatalf("error = %#v, want refusal %q", err, tt.code)
			}
		})
	}
}
