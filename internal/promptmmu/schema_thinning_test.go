package promptmmu

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestMinifyToolSchema(t *testing.T) {
	input := `{
		"type": "function",
		"function": {
			"name": "super_verbose_tool",
			"title": "Super Verbose Tool Definition That Is Redundant",
			"description": "This is a super long and detailed description of a tool that explains what it does in extensive multi-sentence prose that will exceed eighty characters easily.",
			"parameters": {
				"type": "object",
				"title": "Parameters Object Title",
				"properties": {
					"filePath": {
						"type": "string",
						"title": "File Path Parameter",
						"description": "The absolute or relative file path to the destination file that will be read by the system.",
						"default": null,
						"examples": []
					}
				},
				"required": ["filePath"]
			}
		}
	}`

	minified, err := MinifyToolSchema([]byte(input))
	if err != nil {
		t.Fatalf("MinifyToolSchema failed: %v", err)
	}

	minStr := string(minified)

	// Verify "title" is stripped
	if strings.Contains(minStr, "title") {
		t.Errorf("expected title to be stripped, got: %s", minStr)
	}

	// Verify "examples" empty array is stripped
	if strings.Contains(minStr, "examples") {
		t.Errorf("expected empty examples to be stripped, got: %s", minStr)
	}

	// Verify null default is stripped
	if strings.Contains(minStr, "default") {
		t.Errorf("expected null default to be stripped, got: %s", minStr)
	}

	// Verify long descriptions are truncated (ending in "...")
	var parsed map[string]any
	if err := json.Unmarshal(minified, &parsed); err != nil {
		t.Fatalf("failed to parse minified JSON: %v", err)
	}

	fn := parsed["function"].(map[string]any)
	desc := fn["description"].(string)
	if len(desc) > 80 {
		t.Errorf("expected description <= 80 chars, got length %d: %s", len(desc), desc)
	}
	if !strings.HasSuffix(desc, "...") {
		t.Errorf("expected truncated description to end in '...', got: %s", desc)
	}
}

func TestThinLocalTools(t *testing.T) {
	toolsJSON := `[
		{
			"type": "function",
			"function": {
				"name": "Read",
				"description": "Read file contents.",
				"parameters": {"type": "object"}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "Edit",
				"description": "Edit file contents.",
				"parameters": {"type": "object"}
			}
		},
		{
			"type": "function",
			"function": {
				"name": "deploy_kubernetes_cluster",
				"description": "Deploy an entire k8s cluster across AWS and GCP regions.",
				"parameters": {"type": "object"}
			}
		}
	]`

	hot := []string{"Read", "Edit"}
	var faulted []string

	thinnedJSON, pruned, err := ThinLocalTools([]byte(toolsJSON), hot, faulted)
	if err != nil {
		t.Fatalf("ThinLocalTools failed: %v", err)
	}

	if len(pruned) != 1 || pruned[0] != "deploy_kubernetes_cluster" {
		t.Errorf("expected deploy_kubernetes_cluster to be pruned, got %v", pruned)
	}

	thinnedStr := string(thinnedJSON)
	if !strings.Contains(thinnedStr, "Read") {
		t.Errorf("expected Read to be kept, got: %s", thinnedStr)
	}
	if !strings.Contains(thinnedStr, "Edit") {
		t.Errorf("expected Edit to be kept, got: %s", thinnedStr)
	}
	if strings.Contains(thinnedStr, "deploy_kubernetes_cluster") {
		t.Errorf("did not expect deploy_kubernetes_cluster in thinned tools: %s", thinnedStr)
	}
	// Verify fak_tools_search was injected
	if !strings.Contains(thinnedStr, "fak_tools_search") {
		t.Errorf("expected fak_tools_search to be injected when cold tools exist: %s", thinnedStr)
	}
}

func TestFaultInColdTools(t *testing.T) {
	catalog := map[string]string{
		"Read":                      "Read file",
		"Edit":                      "Edit file",
		"deploy_kubernetes_cluster": "Deploy a k8s cluster to the cloud",
		"query_sql_database":        "Run a read-only SQL query against Postgres",
		"launch_docker_container":   "Run a containerized workload",
	}

	active := map[string]bool{
		"read": true,
		"edit": true,
	}

	// Search for "sql"
	faulted := FaultInColdTools("sql", catalog, active)
	if len(faulted) != 1 || faulted[0] != "query_sql_database" {
		t.Fatalf("expected query_sql_database to be faulted in, got: %v", faulted)
	}
	if !active["query_sql_database"] {
		t.Fatalf("query_sql_database should now be marked active")
	}

	// Second query with already active tool should not re-fault
	faultedAgain := FaultInColdTools("sql", catalog, active)
	if len(faultedAgain) != 0 {
		t.Fatalf("expected 0 tools faulted again, got: %v", faultedAgain)
	}
}

func TestFaultedToolTracker_TurnDecayAcceptanceCriteria(t *testing.T) {
	// Acceptance criteria from Issue #11153:
	// Cold tool faulted at turn 2 is pruned at turn 7 when MaxIdleTurns=5.
	tracker := NewFaultedToolTracker(5)
	if tracker.MaxIdleTurns != 5 {
		t.Fatalf("expected MaxIdleTurns=5, got %d", tracker.MaxIdleTurns)
	}

	tracker.RecordFault("deploy_k8s", 2)

	if !tracker.IsActive("deploy_k8s") {
		t.Fatalf("expected deploy_k8s to be active")
	}
	if !tracker.ActiveSet()["deploy_k8s"] {
		t.Fatalf("expected deploy_k8s in ActiveSet")
	}

	// Turns 3 through 6: idle turns < 5, should not prune
	for turn := 3; turn <= 6; turn++ {
		pruned := tracker.PruneColdTools(turn)
		if len(pruned) != 0 {
			t.Fatalf("expected 0 tools pruned at turn %d, got: %v", turn, pruned)
		}
		if !tracker.IsActive("deploy_k8s") {
			t.Fatalf("expected deploy_k8s to remain active at turn %d", turn)
		}
	}

	// Turn 7: idle turns = 7 - 2 = 5 >= 5, must be pruned
	pruned := tracker.PruneColdTools(7)
	expectedPruned := []string{"deploy_k8s"}
	if !reflect.DeepEqual(pruned, expectedPruned) {
		t.Fatalf("expected pruned=%v at turn 7, got %v", expectedPruned, pruned)
	}

	if tracker.IsActive("deploy_k8s") {
		t.Fatalf("expected deploy_k8s to be pruned and inactive at turn 7")
	}
	if tracker.ActiveSet()["deploy_k8s"] {
		t.Fatalf("expected deploy_k8s removed from ActiveSet at turn 7")
	}

	// Subsequent turn should prune nothing further
	prunedLater := tracker.PruneColdTools(8)
	if len(prunedLater) != 0 {
		t.Fatalf("expected 0 tools pruned at turn 8, got: %v", prunedLater)
	}
}

func TestFaultedToolTracker_InvocationResetsIdleDecay(t *testing.T) {
	// Acceptance criteria from Issue #11153:
	// Tool invocation resets idle decay (e.g. faulted at turn 2, invoked at turn 5, pruned at turn 10).
	tracker := NewFaultedToolTracker(5)

	tracker.RecordFault("query_db", 2)

	// Invoke at turn 5
	tracker.RecordInvocation("query_db", 5)

	// At turn 7: idle turns = 7 - max(2, 5) = 2 < 5
	pruned7 := tracker.PruneColdTools(7)
	if len(pruned7) != 0 {
		t.Fatalf("expected 0 tools pruned at turn 7, got: %v", pruned7)
	}

	// At turn 9: idle turns = 9 - 5 = 4 < 5
	pruned9 := tracker.PruneColdTools(9)
	if len(pruned9) != 0 {
		t.Fatalf("expected 0 tools pruned at turn 9, got: %v", pruned9)
	}
	if !tracker.IsActive("query_db") {
		t.Fatalf("expected query_db to remain active at turn 9")
	}

	// At turn 10: idle turns = 10 - 5 = 5 >= 5, must be pruned
	pruned10 := tracker.PruneColdTools(10)
	expectedPruned := []string{"query_db"}
	if !reflect.DeepEqual(pruned10, expectedPruned) {
		t.Fatalf("expected pruned=%v at turn 10, got %v", expectedPruned, pruned10)
	}
	if tracker.IsActive("query_db") {
		t.Fatalf("expected query_db to be pruned at turn 10")
	}
}

func TestFaultedToolTracker_DeterministicSortOrder(t *testing.T) {
	tracker := NewFaultedToolTracker(5)

	// Insert tools in non-alphabetical order
	tools := []string{"zebra_tool", "apple_tool", "mango_tool", "banana_tool"}
	for _, tool := range tools {
		tracker.RecordFault(tool, 1)
	}

	expectedSorted := []string{"apple_tool", "banana_tool", "mango_tool", "zebra_tool"}

	// Verify ActiveFaultedTools returns deterministic sorted order
	active := tracker.ActiveFaultedTools()
	if !reflect.DeepEqual(active, expectedSorted) {
		t.Fatalf("expected active tools=%v, got %v", expectedSorted, active)
	}

	// At turn 6, all tools should be pruned together in sorted order
	pruned := tracker.PruneColdTools(6)
	if !reflect.DeepEqual(pruned, expectedSorted) {
		t.Fatalf("expected pruned tools=%v, got %v", expectedSorted, pruned)
	}

	// Verify tracker is now empty
	if len(tracker.ActiveFaultedTools()) != 0 {
		t.Fatalf("expected 0 active tools after pruning all, got: %v", tracker.ActiveFaultedTools())
	}
}

func TestFaultInColdToolsWithTracker(t *testing.T) {
	catalog := map[string]string{
		"Read":                      "Read file",
		"Edit":                      "Edit file",
		"deploy_kubernetes_cluster": "Deploy a k8s cluster to the cloud",
		"query_sql_database":        "Run a read-only SQL query against Postgres",
		"launch_docker_container":   "Run a containerized workload",
	}

	tracker := NewFaultedToolTracker(5)

	// Turn 1: fault in "sql"
	faulted1 := FaultInColdToolsWithTracker("sql", catalog, tracker, 1)
	expected1 := []string{"query_sql_database"}
	if !reflect.DeepEqual(faulted1, expected1) {
		t.Fatalf("expected faulted1=%v, got %v", expected1, faulted1)
	}
	if !tracker.IsActive("query_sql_database") {
		t.Fatalf("expected query_sql_database to be active in tracker")
	}

	// Turn 2: searching again for "sql" returns nil / 0 tools (no re-faulting)
	faultedAgain := FaultInColdToolsWithTracker("sql", catalog, tracker, 2)
	if len(faultedAgain) != 0 {
		t.Fatalf("expected 0 tools faulted again, got: %v", faultedAgain)
	}

	// Turn 2: fault in "docker"
	faulted2 := FaultInColdToolsWithTracker("docker", catalog, tracker, 2)
	expected2 := []string{"launch_docker_container"}
	if !reflect.DeepEqual(faulted2, expected2) {
		t.Fatalf("expected faulted2=%v, got %v", expected2, faulted2)
	}

	// Active tools should be sorted alphabetically
	active := tracker.ActiveFaultedTools()
	expectedActive := []string{"launch_docker_container", "query_sql_database"}
	if !reflect.DeepEqual(active, expectedActive) {
		t.Fatalf("expected active=%v, got %v", expectedActive, active)
	}

	// Turn 6: query_sql_database has been idle for 6 - 1 = 5 turns -> pruned
	// launch_docker_container has been idle for 6 - 2 = 4 turns -> kept
	pruned := tracker.PruneColdTools(6)
	expectedPruned := []string{"query_sql_database"}
	if !reflect.DeepEqual(pruned, expectedPruned) {
		t.Fatalf("expected pruned=%v, got %v", expectedPruned, pruned)
	}
	if tracker.IsActive("query_sql_database") {
		t.Fatalf("query_sql_database should be pruned")
	}
	if !tracker.IsActive("launch_docker_container") {
		t.Fatalf("launch_docker_container should still be active")
	}

	// Turn 6: re-faulting "sql" should now succeed because it was pruned
	reFaulted := FaultInColdToolsWithTracker("sql", catalog, tracker, 6)
	if !reflect.DeepEqual(reFaulted, expected1) {
		t.Fatalf("expected re-faulted=%v, got %v", expected1, reFaulted)
	}
	if !tracker.IsActive("query_sql_database") {
		t.Fatalf("query_sql_database should be active again")
	}
}

func TestFaultedToolTracker_DefaultMaxIdleTurnsAndNilSafety(t *testing.T) {
	t0 := NewFaultedToolTracker(0)
	if t0.MaxIdleTurns != 5 {
		t.Fatalf("expected default 5 for maxIdleTurns=0, got %d", t0.MaxIdleTurns)
	}

	tNeg := NewFaultedToolTracker(-3)
	if tNeg.MaxIdleTurns != 5 {
		t.Fatalf("expected default 5 for maxIdleTurns=-3, got %d", tNeg.MaxIdleTurns)
	}

	var nilTracker *FaultedToolTracker
	if nilTracker.IsActive("tool") {
		t.Fatalf("expected nil tracker IsActive to return false")
	}
	if len(nilTracker.ActiveFaultedTools()) != 0 {
		t.Fatalf("expected nil tracker ActiveFaultedTools to return empty slice")
	}
	if len(nilTracker.PruneColdTools(10)) != 0 {
		t.Fatalf("expected nil tracker PruneColdTools to return nil")
	}
	if len(nilTracker.ActiveSet()) != 0 {
		t.Fatalf("expected nil tracker ActiveSet to return empty map")
	}
	nilTracker.RecordFault("tool", 1)
	nilTracker.RecordInvocation("tool", 1)
	if faulted := FaultInColdToolsWithTracker("sql", map[string]string{"sql": "db"}, nil, 1); faulted != nil {
		t.Fatalf("expected nil for nil tracker FaultInColdToolsWithTracker, got %v", faulted)
	}
}

func TestFaultedToolTracker_CaseNormalization(t *testing.T) {
	tracker := NewFaultedToolTracker(5)
	tracker.RecordFault("  Query_SQL_Database  ", 2)

	if !tracker.IsActive("query_sql_database") {
		t.Fatalf("expected IsActive to match lowercase")
	}
	if !tracker.IsActive("QUERY_SQL_DATABASE") {
		t.Fatalf("expected IsActive to match uppercase")
	}

	// Invocation with mixed case and whitespace
	tracker.RecordInvocation(" Query_Sql_Database ", 5)

	// Turn 9: idle turns = 9 - 5 = 4 < 5 -> kept
	pruned := tracker.PruneColdTools(9)
	if len(pruned) != 0 {
		t.Fatalf("expected 0 pruned at turn 9, got %v", pruned)
	}

	// Turn 10: idle turns = 10 - 5 = 5 >= 5 -> pruned
	pruned = tracker.PruneColdTools(10)
	expectedPruned := []string{"Query_SQL_Database"}
	if !reflect.DeepEqual(pruned, expectedPruned) {
		t.Fatalf("expected pruned=%v, got %v", expectedPruned, pruned)
	}
}
