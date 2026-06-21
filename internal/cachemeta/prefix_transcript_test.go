package cachemeta

import "testing"

// prefix_transcript_test.go — witnesses for the §A3 transcript front-end seam.

func TestPartKindRoleMapping(t *testing.T) {
	cases := []struct {
		role string
		want SegmentKind
	}{
		{"system", SegStable},
		{"tool_schema", SegToolSchema},
		{"tools", SegToolSchema},
		{"tool_result", SegToolResult},
		{"user", SegMessage},
		{"assistant", SegMessage},
		{"anything-else", SegMessage},
	}
	for _, c := range cases {
		if got := partKind(ConvPart{Role: c.role}); got != c.want {
			t.Fatalf("role %q -> %q, want %q", c.role, got, c.want)
		}
	}
}

func TestPartKindSealedAndVolatilePrecedence(t *testing.T) {
	if got := partKind(ConvPart{Role: "tool_result", Sealed: true}); got != SegSealed {
		t.Fatalf("sealed tool_result -> %q, want sealed", got)
	}
	if got := partKind(ConvPart{Role: "system", Sealed: true, Volatile: true}); got != SegSealed {
		t.Fatalf("sealed wins over volatile: got %q", got)
	}
	if got := partKind(ConvPart{Role: "system", Volatile: true}); got != SegVolatile {
		t.Fatalf("volatile system -> %q, want volatile", got)
	}
}

func TestEstTokensUsesExplicitElseEstimates(t *testing.T) {
	if got := estTokens(ConvPart{Tokens: 42, Content: []byte("ignored")}); got != 42 {
		t.Fatalf("explicit tokens: got %d, want 42", got)
	}
	if got := estTokens(ConvPart{Content: make([]byte, 40)}); got != 10 {
		t.Fatalf("estimate: 40 bytes -> %d, want 10 (~4 B/tok)", got)
	}
	if got := estTokens(ConvPart{Content: []byte("ab")}); got != 1 {
		t.Fatalf("estimate: tiny non-empty -> %d, want 1", got)
	}
	if got := estTokens(ConvPart{}); got != 0 {
		t.Fatalf("empty -> %d, want 0", got)
	}
}

func TestSealedTranscriptPartCapsCacheableRun(t *testing.T) {
	parts := []ConvPart{
		{Role: "system", Tokens: 100, Content: []byte("sys")},
		{Role: "tool_schema", Tokens: 200, Content: []byte("schema")},
		{Role: "tool_result", Tokens: 80, Content: []byte("SECRET"), Sealed: true},
		{Role: "user", Tokens: 20, Content: []byte("next")},
	}
	segs := SegmentsFromParts(parts)
	if segs[2].Kind != SegSealed {
		t.Fatalf("sealed result not mapped to SegSealed: %q", segs[2].Kind)
	}
	if got := frontCacheableRun(segs); got != 300 {
		t.Fatalf("cacheable front run %d, want 300 (sealed result caps it)", got)
	}
}

func TestTurnsFromConversationBuildsCumulativePrefixes(t *testing.T) {
	parts := []ConvPart{
		{Role: "system", Tokens: 100, Content: []byte("sys")},
		{Role: "tool_schema", Tokens: 200, Content: []byte("schema")},
		{Role: "user", Tokens: 10, Content: []byte("q1")},
		{Role: "assistant", Tokens: 5, Content: []byte("a1")},
		{Role: "user", Tokens: 12, Content: []byte("q2")},
		{Role: "assistant", Tokens: 6, Content: []byte("a2")},
	}
	turns := TurnsFromConversation(parts)
	if len(turns) != 2 {
		t.Fatalf("got %d turns, want 2 (one per assistant request)", len(turns))
	}
	if len(turns[0]) != 3 {
		t.Fatalf("turn 1 has %d segments, want 3 (sys, schema, q1)", len(turns[0]))
	}
	if len(turns[1]) != 5 {
		t.Fatalf("turn 2 has %d segments, want 5", len(turns[1]))
	}
	r := AnalyzeStability(turns)
	if r.BrokeAtTurn != -1 {
		t.Fatalf("clean cumulative conversation should not break: BrokeAtTurn=%d", r.BrokeAtTurn)
	}
	if r.CacheableTokens != 310 {
		t.Fatalf("cacheable across turns = %d, want 310 (turn1's sys+schema+q1)", r.CacheableTokens)
	}
}
