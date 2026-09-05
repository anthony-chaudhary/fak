package agent

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
)

func TestIsGoalContinuationMessage(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    bool
	}{
		{
			name:    "empty",
			content: "",
			want:    false,
		},
		{
			name:    "plain query",
			content: "Can you help me refactor the database connector?",
			want:    false,
		},
		{
			name:    "casual mid-sentence mention",
			content: "Our goal is to finish the sprint by Friday.",
			want:    false,
		},
		{
			name:    "compact goal marker",
			content: "[fak:goal] migrate token store to new rotation runbook",
			want:    true,
		},
		{
			name:    "bracket goal continuation marker",
			content: "[goal_continuation] please proceed with phase 2",
			want:    true,
		},
		{
			name:    "codex internal context tag",
			content: "<codex_internal_context source=\"goal\">\nGoal: stabilize prefix\nTurn: 2\n</codex_internal_context>",
			want:    true,
		},
		{
			name:    "goal continuation tag",
			content: "<goal_continuation>\nRefactor logging\n</goal_continuation>",
			want:    true,
		},
		{
			name:    "goal tag pair",
			content: "<goal>Complete the test suite</goal>",
			want:    true,
		},
		{
			name:    "header goal colon",
			content: "Goal: Implement authentication middleware\nTurn: 1",
			want:    true,
		},
		{
			name:    "markdown header goal",
			content: "# Goal Continuation\nKeep running tests until green",
			want:    true,
		},
		{
			name:    "bold header goal",
			content: "**Goal:** Deploy service to staging\n**Budget:** 1000",
			want:    true,
		},
		{
			name:    "json goal continuation",
			content: `{"type": "goal_continuation", "goal": "migrate database", "turn": 3}`,
			want:    true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := IsGoalContinuationMessage(tc.content)
			if got != tc.want {
				t.Errorf("IsGoalContinuationMessage(%q) = %v, want %v", tc.content, got, tc.want)
			}
		})
	}
}

func TestStripVolatileContinuationFields(t *testing.T) {
	t.Run("text with volatile fields at tail", func(t *testing.T) {
		input := "Goal: Refactor database client\nTurn: 3\nBudget: 2000 tokens remaining\nProgress: 60% done"
		stable, volatile := StripVolatileContinuationFields(input)

		if !strings.Contains(stable, "Goal: Refactor database client") {
			t.Errorf("expected stable to contain goal, got %q", stable)
		}
		if strings.Contains(stable, "Turn:") || strings.Contains(stable, "Budget:") || strings.Contains(stable, "Progress:") {
			t.Errorf("stable should not contain volatile fields: %q", stable)
		}
		if !strings.Contains(volatile, "Turn: 3") || !strings.Contains(volatile, "Budget: 2000") || !strings.Contains(volatile, "Progress: 60%") {
			t.Errorf("volatile delta missing fields: %q", volatile)
		}
	})

	t.Run("text with volatile fields at head", func(t *testing.T) {
		input := "Turn: 3\nBudget: 2000\nProgress: 60%\nGoal: Refactor database client"
		stable, volatile := StripVolatileContinuationFields(input)

		if !strings.Contains(stable, "Goal: Refactor database client") {
			t.Errorf("expected stable to contain goal, got %q", stable)
		}
		if strings.Contains(stable, "Turn:") {
			t.Errorf("stable should not contain volatile fields: %q", stable)
		}
		if !strings.Contains(volatile, "Turn: 3") {
			t.Errorf("volatile delta missing fields: %q", volatile)
		}
	})

	t.Run("multi-line bullet points under progress", func(t *testing.T) {
		input := "Goal: Build cache\nProgress:\n- Implemented L1\n- Working on L2\nBudget: $50"
		stable, volatile := StripVolatileContinuationFields(input)

		if stable != "Goal: Build cache" {
			t.Errorf("unexpected stable: %q", stable)
		}
		if !strings.Contains(volatile, "Implemented L1") || !strings.Contains(volatile, "Working on L2") {
			t.Errorf("volatile missing progress bullets: %q", volatile)
		}
	})

	t.Run("xml wrapped codex context", func(t *testing.T) {
		input := "<codex_internal_context source=\"goal\">\nGoal: Implement auth\n<turn>2</turn>\n<budget>1000 tokens</budget>\n<progress>40%</progress>\n</codex_internal_context>"
		stable, volatile := StripVolatileContinuationFields(input)

		if !strings.Contains(stable, "<codex_internal_context source=\"goal\">") || !strings.Contains(stable, "Goal: Implement auth") || !strings.Contains(stable, "</codex_internal_context>") {
			t.Errorf("stable missing xml structure: %q", stable)
		}
		if strings.Contains(stable, "<turn>") || strings.Contains(stable, "<budget>") {
			t.Errorf("stable contains volatile tags: %q", stable)
		}
		if !strings.Contains(volatile, "<turn>2</turn>") || !strings.Contains(volatile, "<budget>1000 tokens</budget>") {
			t.Errorf("volatile missing tags: %q", volatile)
		}
	})

	t.Run("json payload", func(t *testing.T) {
		input := `{"goal":"Implement auth","turn":2,"budget":1000,"progress":"40%"}`
		stable, volatile := StripVolatileContinuationFields(input)

		var sm, vm map[string]any
		if err := json.Unmarshal([]byte(stable), &sm); err != nil {
			t.Fatalf("stable is not valid json: %v", err)
		}
		if sm["goal"] != "Implement auth" || sm["turn"] != nil {
			t.Errorf("unexpected stable json: %v", sm)
		}

		if err := json.Unmarshal([]byte(volatile), &vm); err != nil {
			t.Fatalf("volatile is not valid json: %v", err)
		}
		if vm["turn"] != float64(2) || vm["budget"] != float64(1000) || vm["goal"] != nil {
			t.Errorf("unexpected volatile json: %v", vm)
		}
	})

	t.Run("no volatile fields", func(t *testing.T) {
		input := "Goal: Deploy to production\nEnsure all unit tests pass before merge."
		stable, volatile := StripVolatileContinuationFields(input)

		if stable != input {
			t.Errorf("stable should equal input, got %q", stable)
		}
		if volatile != "" {
			t.Errorf("volatile should be empty, got %q", volatile)
		}
	})
}

func TestStabilizeGoalContinuationMessages(t *testing.T) {
	t.Run("single message with volatile head is canonicalized", func(t *testing.T) {
		input := []Message{
			{
				Role:    "user",
				Content: "Turn: 1\nBudget: 100\nGoal: Build index",
			},
		}

		out, changed := StabilizeGoalContinuationMessages(input)
		if !changed {
			t.Fatalf("expected changed to be true")
		}
		if len(out) != 1 {
			t.Fatalf("expected 1 message, got %d", len(out))
		}

		// Goal definition should be leading, volatile delta at tail
		if !strings.HasPrefix(out[0].Content, "Goal: Build index") {
			t.Errorf("expected goal to lead in canonical output, got %q", out[0].Content)
		}
		if !strings.Contains(out[0].Content, "Turn: 1\nBudget: 100") {
			t.Errorf("expected volatile delta at tail, got %q", out[0].Content)
		}

		// Idempotence
		out2, changed2 := StabilizeGoalContinuationMessages(out)
		if changed2 {
			t.Errorf("second pass should not change already stabilized message, got changed=true: %q", out2[0].Content)
		}
	})

	t.Run("multi-turn prefix byte stability", func(t *testing.T) {
		// Turn 1
		turn1Msgs := []Message{
			{
				Role:    "user",
				Content: "Goal: Refactor auth service\nTurn: 1\nBudget: 1000",
			},
		}
		_, _ = StabilizeGoalContinuationMessages(turn1Msgs)

		// Turn 2: client resends message 0 with updated turn/budget, plus assistant and new user turn
		turn2Msgs := []Message{
			{
				Role:    "user",
				Content: "Goal: Refactor auth service\nTurn: 2\nBudget: 900",
			},
			{
				Role:    "assistant",
				Content: "Inspecting auth packages...",
			},
			{
				Role:    "user",
				Content: "Proceed with step 2",
			},
		}
		s2, changed2 := StabilizeGoalContinuationMessages(turn2Msgs)
		if !changed2 {
			t.Fatalf("expected turn 2 to be stabilized")
		}

		// Turn 3: client resends message 0 with Turn: 3, earlier turns, plus new turn
		turn3Msgs := []Message{
			{
				Role:    "user",
				Content: "Goal: Refactor auth service\nTurn: 3\nBudget: 800",
			},
			{
				Role:    "assistant",
				Content: "Inspecting auth packages...",
			},
			{
				Role:    "user",
				Content: s2[2].Content,
			},
			{
				Role:    "assistant",
				Content: "Refactored token.go",
			},
			{
				Role:    "user",
				Content: "Run the tests",
			},
		}
		s3, changed3 := StabilizeGoalContinuationMessages(turn3Msgs)
		if !changed3 {
			t.Fatalf("expected turn 3 to be stabilized")
		}

		// KEY INVARIANT: message 0 in turn 2 and turn 3 MUST BE BYTE-IDENTICAL!
		if s2[0].Content != s3[0].Content {
			t.Fatalf("prefix message 0 diverged across turns!\nTurn 2: %q\nTurn 3: %q", s2[0].Content, s3[0].Content)
		}
		if s2[0].Content != "Goal: Refactor auth service" {
			t.Errorf("expected static goal in message 0, got %q", s2[0].Content)
		}

		// The volatile progress delta for Turn 2 must be on Turn 2's tail message
		if !strings.Contains(s2[2].Content, "Turn: 2") || !strings.Contains(s2[2].Content, "Budget: 900") {
			t.Errorf("turn 2 tail message missing delta: %q", s2[2].Content)
		}

		// The volatile progress delta for Turn 3 must be on Turn 3's tail message
		if !strings.Contains(s3[4].Content, "Turn: 3") || !strings.Contains(s3[4].Content, "Budget: 800") {
			t.Errorf("turn 3 tail message missing delta: %q", s3[4].Content)
		}

		// Intermediate turn 2 message in turn 3 must match s2[2].Content (prefix stability!)
		if s3[2].Content != s2[2].Content {
			t.Fatalf("intermediate message 2 diverged between turn 2 and turn 3!\ns2[2]: %q\ns3[2]: %q", s2[2].Content, s3[2].Content)
		}
	})

	t.Run("copy on write", func(t *testing.T) {
		input := []Message{
			{Role: "user", Content: "Turn: 1\nGoal: Do task"},
		}
		orig := input[0].Content
		out, _ := StabilizeGoalContinuationMessages(input)

		if input[0].Content != orig {
			t.Errorf("input was mutated in place! orig: %q, now: %q", orig, input[0].Content)
		}
		if out[0].Content == orig {
			t.Errorf("output was not changed")
		}
	})
}

func TestFilterRedundantWorldState(t *testing.T) {
	ws1 := "<world_state>\nbranch: main\ncommit: a1b2c3d\nstatus: clean\n</world_state>"
	ws2Identical := "<world_state>\nbranch: main\ncommit: a1b2c3d\nstatus: clean\n</world_state>"
	ws3Modified := "<world_state>\nbranch: main\ncommit: e5f6g7h\nstatus: clean\n</world_state>"

	t.Run("single world state is preserved verbatim", func(t *testing.T) {
		msgs := []Message{
			{Role: "system", Content: ws1},
			{Role: "user", Content: "Run tests"},
		}
		out, count := FilterRedundantWorldState(msgs)
		if count != 0 {
			t.Errorf("expected count=0 for single world state, got %d", count)
		}
		if out[0].Content != ws1 {
			t.Errorf("first world state must be preserved verbatim, got %q", out[0].Content)
		}
	})

	t.Run("subsequent identical world state converted to compact unchanged diff", func(t *testing.T) {
		msgs := []Message{
			{Role: "system", Content: ws1},
			{Role: "user", Content: "Step 1"},
			{Role: "assistant", Content: "Done step 1"},
			{Role: "user", Content: "Next step\n" + ws2Identical},
		}

		out, count := FilterRedundantWorldState(msgs)
		if count != 1 {
			t.Fatalf("expected count=1, got %d", count)
		}
		if out[0].Content != ws1 {
			t.Errorf("first world state must be preserved verbatim, got %q", out[0].Content)
		}
		if !strings.Contains(out[3].Content, `<world_state_diff status="unchanged"/>`) {
			t.Errorf("expected compact unchanged diff in message 3, got %q", out[3].Content)
		}
		if strings.Contains(out[3].Content, ws2Identical) {
			t.Errorf("redundant full world state was not filtered: %q", out[3].Content)
		}

		// Idempotence
		_, count2 := FilterRedundantWorldState(out)
		if count2 != 0 {
			t.Errorf("second pass should filter 0, got %d", count2)
		}
	})

	t.Run("subsequent modified world state converted to compact diff", func(t *testing.T) {
		msgs := []Message{
			{Role: "system", Content: ws1},
			{Role: "user", Content: "Step 1"},
			{Role: "assistant", Content: "Done step 1"},
			{Role: "user", Content: ws3Modified},
		}

		out, count := FilterRedundantWorldState(msgs)
		if count != 1 {
			t.Fatalf("expected count=1, got %d", count)
		}
		if out[0].Content != ws1 {
			t.Errorf("first world state must be preserved verbatim, got %q", out[0].Content)
		}
		if !strings.Contains(out[3].Content, "<world_state_diff>") || !strings.Contains(out[3].Content, "commit: e5f6g7h") {
			t.Errorf("expected compact diff with changed line in message 3, got %q", out[3].Content)
		}
		if strings.Contains(out[3].Content, "status: clean") {
			t.Errorf("unchanged lines should not be in compact diff: %q", out[3].Content)
		}
	})

	t.Run("json world state deduplication", func(t *testing.T) {
		j1 := `{"type":"world_state","branch":"main","commit":"abc1234","dirty":false}`
		j2 := `{"type":"world_state","branch":"main","commit":"abc1234","dirty":false}`
		j3 := `{"type":"world_state","branch":"main","commit":"def5678","dirty":false}`

		msgs := []Message{
			{Role: "system", Content: j1},
			{Role: "user", Content: "Query 1"},
			{Role: "system", Content: j2},
			{Role: "user", Content: "Query 2"},
			{Role: "system", Content: j3},
		}

		out, count := FilterRedundantWorldState(msgs)
		if count != 2 {
			t.Fatalf("expected count=2, got %d", count)
		}
		if out[0].Content != j1 {
			t.Errorf("first json world state must be preserved verbatim: %q", out[0].Content)
		}
		if !strings.Contains(out[2].Content, `"status":"unchanged"`) {
			t.Errorf("expected unchanged diff in message 2, got %q", out[2].Content)
		}
		if !strings.Contains(out[4].Content, "def5678") || strings.Contains(out[4].Content, "abc1234") {
			t.Errorf("expected compact diff with changed commit in message 4, got %q", out[4].Content)
		}
	})
}

func TestStabilizePromptPrefix(t *testing.T) {
	ws := "<world_state>\nbranch: main\ncommit: 112233\n</world_state>"
	input := []Message{
		{
			Role:    "system",
			Content: ws,
		},
		{
			Role:    "user",
			Content: "<codex_internal_context source=\"goal\">\nGoal: Migrate telemetry\n<turn>1</turn>\n<budget>5000 tokens</budget>\n</codex_internal_context>",
		},
		{
			Role:    "assistant",
			Content: "Reading telemetry files...",
		},
		{
			Role:    "user",
			Content: ws + "\nProceed with turn 2",
		},
	}

	out, changed := StabilizePromptPrefix(input)
	if !changed {
		t.Fatalf("expected changed=true")
	}

	// 1. First world state preserved
	if out[0].Content != ws {
		t.Errorf("first world state changed: %q", out[0].Content)
	}

	// 2. Intermediate goal continuation message stripped of volatile fields
	if strings.Contains(out[1].Content, "<turn>") || strings.Contains(out[1].Content, "<budget>") {
		t.Errorf("message 1 still contains volatile tags: %q", out[1].Content)
	}
	if !strings.Contains(out[1].Content, "Goal: Migrate telemetry") {
		t.Errorf("message 1 missing static goal: %q", out[1].Content)
	}

	// 3. Second world state deduplicated to compact unchanged diff
	if !strings.Contains(out[3].Content, `<world_state_diff status="unchanged"/>`) {
		t.Errorf("message 3 missing world state diff: %q", out[3].Content)
	}

	// 4. Volatile fields from message 1 moved to the tail message
	if !strings.Contains(out[3].Content, "<turn>1</turn>") || !strings.Contains(out[3].Content, "<budget>5000 tokens</budget>") {
		t.Errorf("tail message missing volatile progress delta: %q", out[3].Content)
	}

	// 5. Idempotence test
	out2, changed2 := StabilizePromptPrefix(out)
	if changed2 {
		t.Errorf("StabilizePromptPrefix must be idempotent, got changed=true on second pass: %v", out2)
	}
}

func TestGoalContinuationByteStabilityAcrossTurns(t *testing.T) {
	// Turn 1 and Turn 2 continuation prompts with volatile budget and turn metadata
	turn1Raw := "Goal: Ship #10671 | Budget left: 50 | Turn: 1"
	turn2Raw := "Goal: Ship #10671 | Budget left: 45 | Turn: 2"

	t.Run("single turn prompt prefix byte stability", func(t *testing.T) {
		turn1 := []Message{
			{Role: "user", Content: turn1Raw},
		}
		turn2 := []Message{
			{Role: "user", Content: turn2Raw},
		}

		s1, changed1 := StabilizePromptPrefix(turn1)
		if !changed1 {
			t.Fatalf("expected turn 1 to be stabilized")
		}
		s2, changed2 := StabilizePromptPrefix(turn2)
		if !changed2 {
			t.Fatalf("expected turn 2 to be stabilized")
		}

		// Extract static prefix from Turn 1
		staticPrefix, _ := StripVolatileContinuationFields(turn1Raw)
		if staticPrefix == "" || !strings.HasPrefix(s1[0].Content, staticPrefix) {
			t.Fatalf("turn 1 does not start with static prefix %q: %q", staticPrefix, s1[0].Content)
		}

		prefix1 := staticPrefix
		prefix2 := s2[0].Content

		if len(prefix2) < len(prefix1) {
			t.Fatalf("turn 2 prefix length %d < prefix 1 length %d", len(prefix2), len(prefix1))
		}

		h1 := sha256.Sum256([]byte(prefix1))
		h2 := sha256.Sum256([]byte(prefix2[:len(prefix1)]))
		if h1 != h2 {
			t.Fatalf("byte prefix diverged across turns!\nsha256(prefix1) = %x\nsha256(prefix2[:len(prefix1)]) = %x\nprefix1: %q\nprefix2[:len(prefix1)]: %q",
				h1, h2, prefix1, prefix2[:len(prefix1)])
		}
	})

	t.Run("multi-turn conversation session prefix stability", func(t *testing.T) {
		// In a realistic multi-turn session, Turn 1 has initial prompt + response.
		// Turn 2 continues with resent message 0 containing updated volatile fields.
		turn1 := []Message{
			{Role: "user", Content: turn1Raw},
			{Role: "assistant", Content: "Inspecting issue #10671 and planning test cases..."},
		}

		turn2 := []Message{
			{Role: "user", Content: turn2Raw},
			{Role: "assistant", Content: "Inspecting issue #10671 and planning test cases..."},
			{Role: "user", Content: "Proceed with implementation of byte-stable continuation."},
			{Role: "assistant", Content: "Implemented prefix stabilizer, verifying tests."},
		}

		s1, changed1 := StabilizePromptPrefix(turn1)
		if !changed1 {
			t.Fatalf("expected turn 1 to be modified")
		}
		s2, changed2 := StabilizePromptPrefix(turn2)
		if !changed2 {
			t.Fatalf("expected turn 2 to be modified")
		}

		// In intermediate turns, message 0 has all volatile fields stripped
		if s1[0].Content != "Goal: Ship #10671" {
			t.Errorf("turn 1 message 0 expected 'Goal: Ship #10671', got %q", s1[0].Content)
		}
		if s2[0].Content != "Goal: Ship #10671" {
			t.Errorf("turn 2 message 0 expected 'Goal: Ship #10671', got %q", s2[0].Content)
		}

		prefix1 := s1[0].Content
		prefix2 := s2[0].Content

		h1 := sha256.Sum256([]byte(prefix1))
		h2 := sha256.Sum256([]byte(prefix2[:len(prefix1)]))
		if h1 != h2 {
			t.Fatalf("multi-turn message 0 SHA256 mismatch:\nh1: %x\nh2: %x", h1, h2)
		}

		// Also check that the entire Turn 2 prompt begins with Turn 1's static prefix
		var sb2 strings.Builder
		for _, m := range s2 {
			sb2.WriteString(m.Content)
			sb2.WriteString("\n")
		}
		p2 := sb2.String()
		hp1 := sha256.Sum256([]byte(prefix1))
		hp2 := sha256.Sum256([]byte(p2[:len(prefix1)]))
		if hp1 != hp2 {
			t.Fatalf("prefix diverged: %x != %x", hp1, hp2)
		}

		// Verify volatile deltas moved to the tail message of each turn
		if !strings.Contains(s1[len(s1)-1].Content, "Budget left: 50 | Turn: 1") {
			t.Errorf("turn 1 tail message missing volatile delta: %q", s1[len(s1)-1].Content)
		}
		if !strings.Contains(s2[len(s2)-1].Content, "Budget left: 45 | Turn: 2") {
			t.Errorf("turn 2 tail message missing volatile delta: %q", s2[len(s2)-1].Content)
		}
	})
}

func TestRedundantWorldStateDedupAcrossTurns(t *testing.T) {
	wsFull := `<world_state full="true">
branch: main
commit: 10671abc4d
tree_dirty: false
subsystems:
  gateway: healthy
  adjudicator: healthy
  engine: healthy
</world_state>`

	// Build 5 turns where world_state full is injected on every turn
	var turns [][]Message
	for turnIdx := 0; turnIdx < 5; turnIdx++ {
		var msgs []Message
		for prev := 0; prev < turnIdx; prev++ {
			msgs = append(msgs,
				Message{Role: "user", Content: fmt.Sprintf("Turn %d instruction\n%s", prev, wsFull)},
				Message{Role: "assistant", Content: fmt.Sprintf("Turn %d completed successfully.", prev)},
			)
		}
		msgs = append(msgs,
			Message{Role: "user", Content: fmt.Sprintf("Turn %d instruction\n%s", turnIdx, wsFull)},
			Message{Role: "assistant", Content: fmt.Sprintf("Turn %d completed successfully.", turnIdx)},
		)
		turns = append(turns, msgs)
	}

	serializeMsgs := func(msgs []Message) string {
		var sb strings.Builder
		for _, m := range msgs {
			sb.WriteString(m.Role)
			sb.WriteString(":")
			sb.WriteString(m.Content)
			sb.WriteString("\n---\n")
		}
		return sb.String()
	}

	var filteredTurns [][]Message
	for turnIdx := 0; turnIdx < 5; turnIdx++ {
		filtered, count := FilterRedundantWorldState(turns[turnIdx])
		filteredTurns = append(filteredTurns, filtered)

		if turnIdx == 0 {
			// Turn 0: first full injection must be preserved verbatim, count = 0
			if count != 0 {
				t.Fatalf("turn 0 expected count=0, got %d", count)
			}
			if !strings.Contains(filtered[0].Content, wsFull) {
				t.Fatalf("turn 0 did not preserve full world state verbatim: %q", filtered[0].Content)
			}
		} else {
			// Subsequent turns: must filter redundant world states, count == turnIdx
			if count != turnIdx {
				t.Fatalf("turn %d expected count=%d, got %d", turnIdx, turnIdx, count)
			}
			// Message 0 still contains the first injection verbatim
			if !strings.Contains(filtered[0].Content, wsFull) {
				t.Fatalf("turn %d message 0 must preserve first world state verbatim: %q", turnIdx, filtered[0].Content)
			}
			// Redundant injections in subsequent turns are filtered out to compact unchanged diff
			for k := 1; k <= turnIdx; k++ {
				injectedMsg := filtered[k*2]
				if strings.Contains(injectedMsg.Content, wsFull) {
					t.Fatalf("turn %d message %d still contains redundant full world state: %q", turnIdx, k*2, injectedMsg.Content)
				}
				if !strings.Contains(injectedMsg.Content, `<world_state_diff status="unchanged"/>`) {
					t.Fatalf("turn %d message %d missing unchanged diff marker: %q", turnIdx, k*2, injectedMsg.Content)
				}
			}
		}
	}

	// Verify prefix byte stability across turns:
	// For each subsequent turn, its prompt prefix up to the previous turn's length
	// must be bit-for-bit identical in SHA-256 hash.
	for turnIdx := 1; turnIdx < 5; turnIdx++ {
		prevSerialized := serializeMsgs(filteredTurns[turnIdx-1])
		currSerialized := serializeMsgs(filteredTurns[turnIdx])

		if len(currSerialized) < len(prevSerialized) {
			t.Fatalf("turn %d serialized length %d < turn %d length %d",
				turnIdx, len(currSerialized), turnIdx-1, len(prevSerialized))
		}

		hPrev := sha256.Sum256([]byte(prevSerialized))
		hCurrPrefix := sha256.Sum256([]byte(currSerialized[:len(prevSerialized)]))

		if hPrev != hCurrPrefix {
			t.Fatalf("prefix byte stability broken between turn %d and turn %d!\nprev: %x\ncurr prefix: %x",
				turnIdx-1, turnIdx, hPrev, hCurrPrefix)
		}
	}
}

func TestGoalContinuationWithCodexTags(t *testing.T) {
	t.Run("xml progress tags stripped from codex context", func(t *testing.T) {
		input := `<codex_internal_context source="goal">
Goal: Ship #10671 byte-stable prefix
<turn>2</turn>
<step>5</step>
<iteration>1</iteration>
<budget>4500 tokens remaining</budget>
<tokens_used>1200</tokens_used>
<progress>60% complete</progress>
<status>in_progress</status>
<cost>$0.04</cost>
<elapsed>45s</elapsed>
<timestamp>2026-09-03T10:15:00Z</timestamp>
</codex_internal_context>`

		if !IsGoalContinuationMessage(input) {
			t.Fatalf("expected IsGoalContinuationMessage to return true for codex tag")
		}

		stable, volatile := StripVolatileContinuationFields(input)

		// Stable prefix must retain the codex wrapper and goal
		if !strings.Contains(stable, `<codex_internal_context source="goal">`) {
			t.Errorf("stable prefix missing opening tag: %q", stable)
		}
		if !strings.Contains(stable, `</codex_internal_context>`) {
			t.Errorf("stable prefix missing closing tag: %q", stable)
		}
		if !strings.Contains(stable, "Goal: Ship #10671 byte-stable prefix") {
			t.Errorf("stable prefix missing goal text: %q", stable)
		}

		// Stable prefix must NOT contain volatile tags
		volatileTags := []string{"<turn>", "<step>", "<iteration>", "<budget>", "<tokens_used>", "<progress>", "<status>", "<cost>", "<elapsed>", "<timestamp>"}
		for _, tag := range volatileTags {
			if strings.Contains(stable, tag) {
				t.Errorf("stable prefix should not contain volatile tag %s: %q", tag, stable)
			}
		}

		// Volatile delta must contain the extracted tokens
		for _, tag := range volatileTags {
			if !strings.Contains(volatile, tag) {
				t.Errorf("volatile delta missing expected tag %s: %q", tag, volatile)
			}
		}
	})

	t.Run("line based and pipe volatile tokens inside codex context", func(t *testing.T) {
		input := `<codex_internal_context source="goal">
Goal: Ship #10671 byte-stable prefix | Turn: 3 | Budget left: 3500 | Progress: 75%
</codex_internal_context>`

		stable, volatile := StripVolatileContinuationFields(input)
		if !strings.Contains(stable, "Goal: Ship #10671 byte-stable prefix") {
			t.Errorf("stable prefix missing goal: %q", stable)
		}
		if strings.Contains(stable, "Turn: 3") || strings.Contains(stable, "Budget left:") {
			t.Errorf("stable prefix contains volatile fields: %q", stable)
		}
		if !strings.Contains(volatile, "Turn: 3") || !strings.Contains(volatile, "Budget left: 3500") {
			t.Errorf("volatile delta missing fields: %q", volatile)
		}
	})

	t.Run("multi-turn byte stability with codex context tags", func(t *testing.T) {
		turn1Content := `<codex_internal_context source="goal">
Goal: Ship #10671
<turn>1</turn>
<budget>5000 tokens</budget>
<progress>10%</progress>
</codex_internal_context>`

		turn2Content := `<codex_internal_context source="goal">
Goal: Ship #10671
<turn>2</turn>
<budget>4500 tokens</budget>
<progress>25%</progress>
</codex_internal_context>`

		turn3Content := `<codex_internal_context source="goal">
Goal: Ship #10671
<turn>3</turn>
<budget>4000 tokens</budget>
<progress>50%</progress>
</codex_internal_context>`

		turn1 := []Message{
			{Role: "user", Content: turn1Content},
			{Role: "assistant", Content: "Turn 1 response."},
		}
		turn2 := []Message{
			{Role: "user", Content: turn2Content},
			{Role: "assistant", Content: "Turn 1 response."},
			{Role: "user", Content: "Turn 2 instruction."},
			{Role: "assistant", Content: "Turn 2 response."},
		}
		turn3 := []Message{
			{Role: "user", Content: turn3Content},
			{Role: "assistant", Content: "Turn 1 response."},
			{Role: "user", Content: "Turn 2 instruction."},
			{Role: "assistant", Content: "Turn 2 response."},
			{Role: "user", Content: "Turn 3 instruction."},
			{Role: "assistant", Content: "Turn 3 response."},
		}

		s1, _ := StabilizePromptPrefix(turn1)
		s2, _ := StabilizePromptPrefix(turn2)
		s3, _ := StabilizePromptPrefix(turn3)

		// Message 0 in Turn 1, 2, 3 must be bit-for-bit identical
		h1 := sha256.Sum256([]byte(s1[0].Content))
		h2 := sha256.Sum256([]byte(s2[0].Content))
		h3 := sha256.Sum256([]byte(s3[0].Content))

		if h1 != h2 {
			t.Fatalf("turn 1 and turn 2 message 0 diverged:\ns1[0]: %q\ns2[0]: %q", s1[0].Content, s2[0].Content)
		}
		if h2 != h3 {
			t.Fatalf("turn 2 and turn 3 message 0 diverged:\ns2[0]: %q\ns3[0]: %q", s2[0].Content, s3[0].Content)
		}

		expectedCleanGoal := `<codex_internal_context source="goal">
Goal: Ship #10671
</codex_internal_context>`
		if s1[0].Content != expectedCleanGoal {
			t.Errorf("unexpected canonical goal:\ngot:  %q\nwant: %q", s1[0].Content, expectedCleanGoal)
		}

		// Tail messages must carry the respective turn's progress delta
		if !strings.Contains(s2[len(s2)-1].Content, "<turn>2</turn>") {
			t.Errorf("turn 2 tail message missing progress delta: %q", s2[len(s2)-1].Content)
		}
		if !strings.Contains(s3[len(s3)-1].Content, "<turn>3</turn>") {
			t.Errorf("turn 3 tail message missing progress delta: %q", s3[len(s3)-1].Content)
		}
	})

	t.Run("escaped quote codex context wrapper", func(t *testing.T) {
		input := `<codex_internal_context source=\"goal\">
Goal: Ship #10671
<turn>1</turn>
<step>1</step>
</codex_internal_context>`

		if !IsGoalContinuationMessage(input) {
			t.Fatalf("expected IsGoalContinuationMessage to return true for escaped codex tag")
		}

		stable, volatile := StripVolatileContinuationFields(input)
		if !strings.Contains(stable, "Goal: Ship #10671") {
			t.Errorf("stable missing goal: %q", stable)
		}
		if strings.Contains(stable, "<turn>") || strings.Contains(stable, "<step>") {
			t.Errorf("stable contains volatile tags: %q", stable)
		}
		if !strings.Contains(volatile, "<turn>1</turn>") || !strings.Contains(volatile, "<step>1</step>") {
			t.Errorf("volatile delta missing tags: %q", volatile)
		}
	})
}

func TestGoalPrefix(t *testing.T) {
	t.Run("GoalContinuationByteStabilityAcrossTurns", TestGoalContinuationByteStabilityAcrossTurns)
	t.Run("RedundantWorldStateDedupAcrossTurns", TestRedundantWorldStateDedupAcrossTurns)
	t.Run("GoalContinuationWithCodexTags", TestGoalContinuationWithCodexTags)
}

func makeStabilizerBenchMessages(size int) []Message {
	if size <= 0 {
		return nil
	}
	msgs := make([]Message, 0, size)
	msgs = append(msgs, Message{
		Role: "user",
		Content: `<codex_internal_context source="goal">
Goal: Implement authentication and authorization middleware
<turn>1</turn>
<step>1</step>
<budget>5000 tokens remaining</budget>
<tokens_used>120</tokens_used>
<progress>10% complete</progress>
<status>in_progress</status>
<cost>$0.01</cost>
<elapsed>5s</elapsed>
<timestamp>2026-09-04T12:00:00Z</timestamp>
</codex_internal_context>`,
	})

	wsFull := `<world_state full="true">
branch: main
commit: 10671abc4d
tree_dirty: false
subsystems:
  gateway: healthy
  adjudicator: healthy
  engine: healthy
</world_state>`

	for i := 1; i < size; i++ {
		turnNum := (i / 2) + 1
		if i%2 == 1 {
			msgs = append(msgs, Message{
				Role:    "assistant",
				Content: fmt.Sprintf("Turn %d completed step %d of the implementation.", turnNum, i),
			})
		} else {
			switch {
			case i%6 == 0:
				msgs = append(msgs, Message{
					Role:    "user",
					Content: fmt.Sprintf("Turn %d context update:\n%s", turnNum, wsFull),
				})
			case i%4 == 0:
				msgs = append(msgs, Message{
					Role: "user",
					Content: fmt.Sprintf(`<codex_internal_context source="goal">
Goal: Implement authentication and authorization middleware
<turn>%d</turn>
<step>%d</step>
<budget>%d tokens remaining</budget>
<progress>%d%% complete</progress>
</codex_internal_context>`, turnNum, i, 5000-i*40, (i*100)/size),
				})
			default:
				msgs = append(msgs, Message{
					Role:    "user",
					Content: fmt.Sprintf("Goal: Implement authentication and authorization middleware | Turn: %d | Budget left: %d | Progress: %d%%", turnNum, 5000-i*40, (i*100)/size),
				})
			}
		}
	}
	return msgs
}

func BenchmarkStabilizePromptPrefix(b *testing.B) {
	for _, size := range []int{1, 10, 100} {
		msgs := makeStabilizerBenchMessages(size)
		b.Run(fmt.Sprintf("history_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, _ = StabilizePromptPrefix(msgs)
			}
		})
	}
}
