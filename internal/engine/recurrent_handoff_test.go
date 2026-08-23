package engine

import (
	"bytes"
	"testing"
)

func measuredPrefixClone() CacheCapability {
	return CacheCapability{Verdict: CachePrefixClone, Provenance: ProvenanceKernel, ColdPathCorrect: true}
}

func TestRecurrentCheckpointHandoffForksState(t *testing.T) {
	backend := []byte{1, 2, 3, 4}
	cp, err := NewRecurrentStateCheckpoint("qwen3.8", "shared-tools", 128, backend)
	if err != nil {
		t.Fatal(err)
	}
	backend[0] = 9

	req := RecurrentHandoffRequest{
		ModelID: "qwen3.8", PrefixKey: "shared-tools", PrefixTokens: 128,
		PrefixClone: measuredPrefixClone(), Checkpoint: cp,
	}
	first, refusal := AdmitRecurrentHandoff(req)
	if refusal.Refused() {
		t.Fatalf("unexpected refusal: %s", refusal)
	}
	second, refusal := AdmitRecurrentHandoff(req)
	if refusal.Refused() {
		t.Fatalf("unexpected refusal: %s", refusal)
	}

	firstState := first.State()
	firstState[1] = 8
	if got := first.State(); !bytes.Equal(got, []byte{1, 2, 3, 4}) {
		t.Fatalf("handoff exposed mutable state: %v", got)
	}
	if got := second.State(); !bytes.Equal(got, []byte{1, 2, 3, 4}) {
		t.Fatalf("forks shared mutable state: %v", got)
	}
	if first.ModelID() != "qwen3.8" || first.PrefixKey() != "shared-tools" || first.PrefixTokens() != 128 {
		t.Fatalf("handoff lost boundary identity: %#v", first)
	}
}

func TestRecurrentHandoffJointAdmissionFailsClosed(t *testing.T) {
	cp, err := NewRecurrentStateCheckpoint("qwen3.8", "shared-tools", 128, []byte{1})
	if err != nil {
		t.Fatal(err)
	}
	base := RecurrentHandoffRequest{
		ModelID: "qwen3.8", PrefixKey: "shared-tools", PrefixTokens: 128,
		PrefixClone: measuredPrefixClone(), Checkpoint: cp,
	}
	tests := []struct {
		name   string
		mutate func(*RecurrentHandoffRequest)
		want   RecurrentHandoffRefusal
	}{
		{"kv prefix clone refused", func(r *RecurrentHandoffRequest) { r.PrefixClone.ColdPathCorrect = false }, RecurrentHandoffPrefixRefused},
		{"model mismatch", func(r *RecurrentHandoffRequest) { r.ModelID = "other" }, RecurrentHandoffCheckpointMismatch},
		{"prefix mismatch", func(r *RecurrentHandoffRequest) { r.PrefixKey = "other" }, RecurrentHandoffCheckpointMismatch},
		{"boundary mismatch", func(r *RecurrentHandoffRequest) { r.PrefixTokens++ }, RecurrentHandoffCheckpointMismatch},
		{"malformed request", func(r *RecurrentHandoffRequest) { r.ModelID = "" }, RecurrentHandoffRequestMalformed},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := base
			tc.mutate(&req)
			handoff, got := AdmitRecurrentHandoff(req)
			if got != tc.want {
				t.Fatalf("refusal=%q want %q", got, tc.want)
			}
			if handoff.State() != nil {
				t.Fatal("refused joint admission exposed recurrent state")
			}
		})
	}
}

func TestNewRecurrentStateCheckpointRejectsIncompleteBoundary(t *testing.T) {
	for _, tc := range []struct {
		model, key string
		tokens     uint64
		state      []byte
	}{
		{"", "p", 1, []byte{1}}, {"m", "", 1, []byte{1}}, {"m", "p", 0, []byte{1}}, {"m", "p", 1, nil},
	} {
		if _, err := NewRecurrentStateCheckpoint(tc.model, tc.key, tc.tokens, tc.state); err == nil {
			t.Fatalf("accepted incomplete checkpoint: %#v", tc)
		}
	}
}
