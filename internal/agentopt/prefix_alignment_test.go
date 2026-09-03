package agentopt

import (
	"fmt"
	"testing"
)

func TestRadixPrefixCacheAlignment(t *testing.T) {
	stabilizer := NewPrefixStabilizer()

	// 1. Define static prompt blocks in different orders across consecutive turns.
	blockAlpha := "## Operational Constraints\nAlways follow security standards and fail-closed rules."
	blockBeta := "## Output Conventions\nReturn structured facts with inline evidence."
	blockGamma := "## Tooling Contract\nInvoke read-only operations when inspecting state."

	sysPromptTurn1 := fmt.Sprintf("%s\n\n%s\n\n%s", blockBeta, blockAlpha, blockGamma)
	sysPromptTurn2 := fmt.Sprintf("%s\n\n%s\n\n%s\n\n", blockGamma, blockBeta, blockAlpha)
	sysPromptTurn3 := fmt.Sprintf("  %s  \n\n%s\n\n%s", blockAlpha, blockGamma, blockBeta)

	// 2. Define tools in different orders with permuted property ordering and required fields.
	toolReadFile := ToolSchema{
		Name:        "read_file",
		Description: "Read content from repository",
		Properties: map[string]PropertySchema{
			"path":   {Type: TypeString},
			"offset": {Type: TypeInteger},
		},
		Required: []string{"path", "offset"},
	}
	toolReadFilePermuted := ToolSchema{
		Name:        "read_file",
		Description: "Read content from repository",
		Properties: map[string]PropertySchema{
			"offset": {Type: TypeInteger},
			"path":   {Type: TypeString},
		},
		Required: []string{"offset", "path"}, // permuted required slice
	}

	toolRunCommand := ToolSchema{
		Name:        "run_command",
		Description: "Execute shell command in workspace",
		Properties: map[string]PropertySchema{
			"command": {Type: TypeString},
			"timeout": {Type: TypeInteger},
		},
		Required: []string{"command"},
	}

	toolListDir := ToolSchema{
		Name:        "list_dir",
		Description: "List files in directory",
		Properties: map[string]PropertySchema{
			"dir": {Type: TypeString},
		},
		Required: []string{"dir"},
	}

	toolsTurn1 := []ToolSchema{toolRunCommand, toolReadFile, toolListDir}
	toolsTurn2 := []ToolSchema{toolListDir, toolReadFilePermuted, toolRunCommand}
	toolsTurn3 := []ToolSchema{toolReadFile, toolListDir, toolRunCommand}

	// 3. Stabilize prefix for each turn.
	aligned1 := stabilizer.StabilizePrefix(sysPromptTurn1, toolsTurn1)
	aligned2 := stabilizer.StabilizePrefix(sysPromptTurn2, toolsTurn2)
	aligned3 := stabilizer.StabilizePrefix(sysPromptTurn3, toolsTurn3)

	t.Run("BasePrefix100PercentIdentity", func(t *testing.T) {
		if aligned1.PrefixHash == "" {
			t.Fatal("expected non-empty prefix hash")
		}
		if aligned1.PrefixHash != aligned2.PrefixHash {
			t.Fatalf("consecutive turn 1 and turn 2 prefix hashes differ:\nTurn1: %s\nTurn2: %s",
				aligned1.PrefixHash, aligned2.PrefixHash)
		}
		if aligned2.PrefixHash != aligned3.PrefixHash {
			t.Fatalf("consecutive turn 2 and turn 3 prefix hashes differ:\nTurn2: %s\nTurn3: %s",
				aligned2.PrefixHash, aligned3.PrefixHash)
		}

		// Ensure 100% canonical byte stream equality
		if string(aligned1.CanonicalBytes) != string(aligned2.CanonicalBytes) {
			t.Fatal("canonical byte representation mismatch between turn 1 and turn 2")
		}
		if string(aligned2.CanonicalBytes) != string(aligned3.CanonicalBytes) {
			t.Fatal("canonical byte representation mismatch between turn 2 and turn 3")
		}

		t.Logf("100%% invariant base prefix hash: %s", aligned1.PrefixHash)
	})

	t.Run("ToolLexicographicalOrdering", func(t *testing.T) {
		expectedOrder := []string{"list_dir", "read_file", "run_command"}
		if len(aligned1.ToolSchemas) != len(expectedOrder) {
			t.Fatalf("expected %d tools, got %d", len(expectedOrder), len(aligned1.ToolSchemas))
		}
		for i, name := range expectedOrder {
			if aligned1.ToolSchemas[i].Name != name {
				t.Errorf("tool [%d] expected %s, got %s", i, name, aligned1.ToolSchemas[i].Name)
			}
			if aligned2.ToolSchemas[i].Name != name {
				t.Errorf("turn 2 tool [%d] expected %s, got %s", i, name, aligned2.ToolSchemas[i].Name)
			}
		}

		// Verify required fields are sorted
		readTool := aligned2.ToolSchemas[1]
		if len(readTool.Required) != 2 || readTool.Required[0] != "offset" || readTool.Required[1] != "path" {
			t.Errorf("expected sorted required fields [offset, path], got %v", readTool.Required)
		}
	})

	t.Run("PromptBlockLexicographicalOrdering", func(t *testing.T) {
		if len(aligned1.SystemBlocks) != 3 {
			t.Fatalf("expected 3 system blocks, got %d", len(aligned1.SystemBlocks))
		}
		if aligned1.SystemBlocks[0] != blockAlpha {
			t.Errorf("block 0 expected %q, got %q", blockAlpha, aligned1.SystemBlocks[0])
		}
		if aligned1.SystemBlocks[1] != blockBeta {
			t.Errorf("block 1 expected %q, got %q", blockBeta, aligned1.SystemBlocks[1])
		}
		if aligned1.SystemBlocks[2] != blockGamma {
			t.Errorf("block 2 expected %q, got %q", blockGamma, aligned1.SystemBlocks[2])
		}
	})

	t.Run("MultiTurnRadixPrefixChainIdentity", func(t *testing.T) {
		turn0 := TurnInteraction{
			TurnIndex: 0,
			Prompt:    "Inspect repository files",
			Output:    "Discovered 42 go source files",
			ToolCalls: []ToolCall{
				{ID: "call_0", Name: "list_dir", Args: map[string]any{"dir": "internal"}, ReadOnly: true},
			},
		}

		turn1 := TurnInteraction{
			TurnIndex: 1,
			Prompt:    "Run the test suite",
			Output:    "All tests passed in 0.25s",
			ToolCalls: []ToolCall{
				{ID: "call_1", Name: "run_command", Args: map[string]any{"command": "go test ./..."}, ReadOnly: false},
			},
		}

		turn2 := TurnInteraction{
			TurnIndex: 2,
			Prompt:    "Generate summary report",
			Output:    "Summary generated successfully",
		}

		// Turn 1 builds prefix chain over history [turn0]
		chainTurn1 := aligned1.BuildPrefixChain([]TurnInteraction{turn0})
		// Turn 2 builds prefix chain over history [turn0, turn1]
		chainTurn2 := aligned2.BuildPrefixChain([]TurnInteraction{turn0, turn1})
		// Turn 3 builds prefix chain over history [turn0, turn1, turn2]
		chainTurn3 := aligned3.BuildPrefixChain([]TurnInteraction{turn0, turn1, turn2})

		// H0 (base prefix) must be identical across all turns
		if chainTurn1[0] != chainTurn2[0] || chainTurn2[0] != chainTurn3[0] {
			t.Fatalf("H0 mismatch across turns: %s vs %s vs %s", chainTurn1[0], chainTurn2[0], chainTurn3[0])
		}

		// H1 (prefix through turn 0) must be identical between turn 1, turn 2, and turn 3
		if chainTurn1[1] != chainTurn2[1] {
			t.Fatalf("H1 mismatch across turns: %s vs %s", chainTurn1[1], chainTurn2[1])
		}
		if chainTurn2[1] != chainTurn3[1] {
			t.Fatalf("H1 mismatch across turn 2 and turn 3: %s vs %s", chainTurn2[1], chainTurn3[1])
		}

		// H2 (prefix through turn 1) must be identical between turn 2 and turn 3
		if chainTurn2[2] != chainTurn3[2] {
			t.Fatalf("H2 mismatch across turn 2 and turn 3: %s vs %s", chainTurn2[2], chainTurn3[2])
		}

		// Radix tree alignment:
		// When Turn 1 runs, it populates the tree with chainTurn1.
		tree := NewRadixPrefixTree()
		inserted1 := tree.Insert(chainTurn1)
		if inserted1 != 2 {
			t.Fatalf("expected 2 nodes inserted for turn 1, got %d", inserted1)
		}

		// When Turn 2 runs, the prefix chain up to turn 0 must match with 100% ratio.
		turn2Prefix := chainTurn2[:2]
		matchedLen := tree.MatchLength(turn2Prefix)
		if matchedLen != 2 {
			t.Fatalf("expected match length 2, got %d", matchedLen)
		}
		matchRatio := tree.MatchRatio(turn2Prefix)
		if matchRatio != 1.0 {
			t.Fatalf("expected 100%% prefix match ratio, got %.4f", matchRatio)
		}

		// Insert turn 2 full chain into tree
		inserted2 := tree.Insert(chainTurn2)
		if inserted2 != 1 {
			t.Fatalf("expected exactly 1 new node (H2) for turn 2, got %d", inserted2)
		}

		// When Turn 3 runs, prefix chain up to turn 1 must match 100%
		turn3Prefix := chainTurn3[:3]
		matchedLen3 := tree.MatchLength(turn3Prefix)
		if matchedLen3 != 3 {
			t.Fatalf("expected match length 3, got %d", matchedLen3)
		}
		matchRatio3 := tree.MatchRatio(turn3Prefix)
		if matchRatio3 != 1.0 {
			t.Fatalf("expected 100%% prefix match ratio for turn 3, got %.4f", matchRatio3)
		}

		t.Logf("Radix prefix tree successfully demonstrated 100%% prefix reuse across consecutive turns")
	})

	t.Run("StabilizeTurnHelper", func(t *testing.T) {
		turn0 := TurnInteraction{TurnIndex: 0, Input: "Hello"}
		turnRes1 := stabilizer.StabilizeTurn(sysPromptTurn1, toolsTurn1, []TurnInteraction{turn0})
		turnRes2 := stabilizer.StabilizeTurn(sysPromptTurn2, toolsTurn2, []TurnInteraction{turn0})

		if turnRes1.PrefixHash != turnRes2.PrefixHash {
			t.Fatalf("StabilizeTurn hashes diverge across consecutive turns: %s vs %s",
				turnRes1.PrefixHash, turnRes2.PrefixHash)
		}
	})

	t.Run("DefaultStabilizerFallback", func(t *testing.T) {
		var nilStabilizer *PrefixStabilizer
		p := nilStabilizer.StabilizePrefix("Test prompt", nil)
		if p.PrefixHash == "" {
			t.Fatal("expected non-empty prefix hash from nil stabilizer")
		}
	})
}
