package promptmmu

import (
	"encoding/json"
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
