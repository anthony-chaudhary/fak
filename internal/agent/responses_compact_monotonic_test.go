package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func monotonicCompactFixture(middle ...Message) []Message {
	messages := []Message{{Role: RoleSystem, Content: "Pinned rules."}, {Role: RoleUser, Content: "Initial task."}}
	messages = append(messages, middle...)
	return append(messages, Message{Role: RoleAssistant, Content: "Recent answer."}, Message{Role: RoleUser, Content: "Continue."})
}

func checkMonotonicCompactAccounting(t *testing.T, before, after []Message, outcome CompactOutcome) {
	t.Helper()
	original := EstimateResponsesMessagesTokens(before)
	current := EstimateResponsesMessagesTokens(after)
	if current > original {
		t.Fatalf("compaction expanded estimated tokens: %d -> %d", original, current)
	}
	if outcome.OriginalTokens != original || outcome.CompactedTokens != current || outcome.ShedTokens != original-current {
		t.Fatalf("accounting=%+v, actual=%d -> %d", outcome, original, current)
	}
}

// fak-test:runtime fast est=1s
func TestResponsesCompactionRejectsNonSavingStubs(t *testing.T) {
	// Find an exact estimated-token tie with a valid custom CAS reference.
	equal := ""
	for n := 1; n < 512; n++ {
		body := strings.Repeat("x", n)
		stub := Message{Role: RoleAssistant, Content: fmt.Sprintf("[fak:compacted id=r size=%d turns=1 hint=\"fak_context_restore\"]", len(body))}
		if EstimateResponsesMessageTokens(Message{Role: RoleAssistant, Content: body}) == EstimateResponsesMessageTokens(stub) {
			equal = body
			break
		}
	}
	if equal == "" {
		t.Fatal("fixture did not find an estimated-token tie")
	}
	for _, tc := range []struct {
		name, text, ref string
	}{
		{"empty", "", "r"},
		{"short", "ok", "r"},
		{"unicode_short", "é", "r"},
		{"equal_estimate", equal, "r"},
		{"oversized_custom_ref", strings.Repeat("ordinary ", 40), strings.Repeat("a", 4096)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := monotonicCompactFixture(Message{Role: RoleAssistant, Content: tc.text})
			original := append([]Message(nil), before...)
			after, outcome, err := CompactResponsesMessages(before, 1, func([]byte) string { return tc.ref })
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(after, original) || !reflect.DeepEqual(before, original) {
				t.Fatal("a non-saving candidate changed history")
			}
			checkMonotonicCompactAccounting(t, before, after, outcome)
			if outcome.ShedTurns != 0 || outcome.RestoreStubsCreated != 0 {
				t.Fatalf("non-saving candidate reported as compacted: %+v", outcome)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestResponsesCompactionRetainsPriorDecisions(t *testing.T) {
	for _, stub := range []string{
		"[fak:compacted id=" + strings.Repeat("a", 4096) + ` size=10000 turns=1 hint="fak_context_restore"]`,
		`{"_paged":true,"ref":"prior","size":10000,"turns":1,"padding":"` + strings.Repeat("x", 4096) + `"}`,
	} {
		if _, _, _, ok := ParseResponsesCompactStub(stub); !ok {
			t.Fatal("fixture is not a recognized prior decision")
		}
		large := strings.Repeat("new compressible middle context ", 200)
		before := monotonicCompactFixture(Message{Role: RoleAssistant, Content: stub}, Message{Role: RoleAssistant, Content: large})
		var saved []string
		after, outcome, err := CompactResponsesMessages(before, 1, func(body []byte) string {
			saved = append(saved, string(body))
			digest := sha256.Sum256(body)
			return hex.EncodeToString(digest[:])
		})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(after[2], before[2]) {
			t.Fatal("prior compacted decision was rewritten")
		}
		if len(saved) != 1 || saved[0] != large {
			t.Fatalf("saved %d bodies, want only new large prose", len(saved))
		}
		if outcome.ShedTurns != 1 || outcome.RestoreStubsCreated != 1 || after[3].Content == large {
			t.Fatalf("new profitable content did not compact: %+v", outcome)
		}
		checkMonotonicCompactAccounting(t, before, after, outcome)
	}
}

// fak-test:runtime fast est=1s
func TestResponsesCompactionMonotonicAcrossTurns(t *testing.T) {
	messages := monotonicCompactFixture(
		Message{Role: RoleAssistant, Content: "ok"},
		Message{Role: RoleAssistant, Content: strings.Repeat("First long historical explanation. ", 100)},
	)
	decisions := map[int]Message{}
	cas := map[string]string{}
	store := func(body []byte) string {
		if _, _, _, ok := ParseResponsesCompactStub(string(body)); ok {
			t.Error("attempted to store a previous restore stub")
		}
		digest := sha256.Sum256(body)
		ref := hex.EncodeToString(digest[:])
		cas[ref] = string(body)
		return ref
	}
	for turn := 0; turn < 80; turn++ {
		before := append([]Message(nil), messages...)
		after, outcome, err := CompactResponsesMessages(messages, 1, store)
		if err != nil {
			t.Fatal(err)
		}
		checkMonotonicCompactAccounting(t, before, after, outcome)
		for i, old := range decisions {
			if !reflect.DeepEqual(after[i], old) {
				t.Fatalf("turn %d rewrote prior compacted message %d", turn, i)
			}
		}
		for i, message := range after {
			if ref, _, _, ok := ParseResponsesCompactStub(message.Content); ok {
				if _, found := cas[ref]; !found {
					t.Fatalf("turn %d: missing original payload for %q", turn, ref)
				}
				decisions[i] = message
			}
		}
		if !reflect.DeepEqual(after[:2], before[:2]) || !reflect.DeepEqual(after[len(after)-2:], before[len(before)-2:]) || !reflect.DeepEqual(messages, before) {
			t.Fatal("prefix, recent suffix, or caller input changed")
		}
		messages = append(after,
			Message{Role: RoleAssistant, Content: strings.Repeat(fmt.Sprintf("New explanation %d. ", turn), 100)},
			Message{Role: RoleUser, Content: "Continue."})
	}
	if len(decisions) < 70 {
		t.Fatalf("only %d profitable decisions, fixture failed to exercise progression", len(decisions))
	}
}
