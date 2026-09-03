package agentopt

import (
	"strings"
	"sync"
	"testing"
)

func TestTieredModelRouter(t *testing.T) {
	router := NewTieredModelRouter(nil)
	classifier := router.Classifier()

	t.Run("RoutineFormatting_T0", func(t *testing.T) {
		formatTools := []string{
			"format", "fmt", "gofmt", "prettier", "black", "ruff_format", "format_code", "autofmt",
		}
		for _, tool := range formatTools {
			tier, reason := classifier.ClassifyToolCall(tool, nil, "")
			if tier != TierT0 {
				t.Fatalf("tool %s classified as %s, want %s", tool, tier, TierT0)
			}
			if !strings.Contains(reason, "routine formatting") {
				t.Fatalf("tool %s reason unexpected: %s", tool, reason)
			}

			targetTier, endpoint := router.RouteToolTask(tool, nil, "")
			if targetTier != TierT0 {
				t.Fatalf("tool %s routed to %s, want %s", tool, targetTier, TierT0)
			}
			if !strings.Contains(endpoint, "t0-fast") {
				t.Fatalf("tool %s endpoint %s, want t0 endpoint", tool, endpoint)
			}
		}

		// Formatting commands in bash/exec
		formatCommands := []string{
			"gofmt -w main.go",
			"prettier --write src/index.ts",
			"black ./src",
			"ruff format .",
			"npm run format",
			"cargo fmt",
		}
		for _, cmd := range formatCommands {
			args := map[string]any{"command": cmd}
			tier, reason := classifier.ClassifyToolCall("bash", args, "")
			if tier != TierT0 {
				t.Fatalf("command %q classified as %s, want %s (reason: %s)", cmd, tier, TierT0, reason)
			}

			targetTier, endpoint := router.RouteToolTask("bash", args, "")
			if targetTier != TierT0 {
				t.Fatalf("command %q routed to %s, want %s", cmd, targetTier, TierT0)
			}
			if endpoint != router.EndpointForTier(TierT0) {
				t.Fatalf("command %q endpoint %s, want %s", cmd, endpoint, router.EndpointForTier(TierT0))
			}
		}

		// Formatting prompt without tool
		tier, _ := classifier.ClassifyToolCall("", nil, "Please format code in main.go using gofmt")
		if tier != TierT0 {
			t.Fatalf("formatting prompt classified as %s, want %s", tier, TierT0)
		}
	})

	t.Run("RoutineLinting_T0", func(t *testing.T) {
		lintTools := []string{
			"lint", "linter", "eslint", "flake8", "golangci-lint", "ruff", "pylint", "clippy",
		}
		for _, tool := range lintTools {
			tier, reason := classifier.ClassifyToolCall(tool, nil, "")
			if tier != TierT0 {
				t.Fatalf("tool %s classified as %s, want %s", tool, tier, TierT0)
			}
			if !strings.Contains(reason, "routine linting") {
				t.Fatalf("tool %s reason unexpected: %s", tool, reason)
			}

			targetTier, endpoint := router.RouteToolTask(tool, nil, "")
			if targetTier != TierT0 {
				t.Fatalf("tool %s routed to %s, want %s", tool, targetTier, TierT0)
			}
			if endpoint != router.EndpointForTier(TierT0) {
				t.Fatalf("tool %s endpoint mismatch: got %s, want %s", tool, endpoint, router.EndpointForTier(TierT0))
			}
		}

		// Linting commands in bash
		lintCommands := []string{
			"golangci-lint run ./...",
			"ruff check .",
			"eslint src/**/*.ts",
			"npm run lint",
			"cargo clippy",
			"go vet ./...",
		}
		for _, cmd := range lintCommands {
			args := map[string]any{"command": cmd}
			tier, reason := classifier.ClassifyToolCall("bash", args, "")
			if tier != TierT0 {
				t.Fatalf("command %q classified as %s, want %s (reason: %s)", cmd, tier, TierT0, reason)
			}

			targetTier, _ := router.RouteToolTask("bash", args, "")
			if targetTier != TierT0 {
				t.Fatalf("command %q routed to %s, want %s", cmd, targetTier, TierT0)
			}
		}

		// Linting prompt
		tier, _ := classifier.ClassifyToolCall("", nil, "Run linter and check lint errors")
		if tier != TierT0 {
			t.Fatalf("lint prompt classified as %s, want %s", tier, TierT0)
		}
	})

	t.Run("SimpleReadGrep_T0", func(t *testing.T) {
		readGrepTools := []string{
			"Read", "read_file", "cat", "head", "tail", "Grep", "grep", "search_files", "Glob", "glob", "list_dir", "ls",
		}
		for _, tool := range readGrepTools {
			tier, reason := classifier.ClassifyToolCall(tool, map[string]any{"path": "file.go"}, "")
			if tier != TierT0 {
				t.Fatalf("tool %s classified as %s, want %s (reason: %s)", tool, tier, TierT0, reason)
			}

			targetTier, _ := router.RouteToolTask(tool, map[string]any{"path": "file.go"}, "")
			if targetTier != TierT0 {
				t.Fatalf("tool %s routed to %s, want %s", tool, targetTier, TierT0)
			}
		}

		// Read/grep commands
		readGrepCommands := []string{
			"cat README.md",
			"grep -rn 'TODO' .",
			"rg 'func New' internal/",
			"ls -la",
			"find . -name '*.go'",
			"head -n 20 main.go",
		}
		for _, cmd := range readGrepCommands {
			args := map[string]any{"command": cmd}
			tier, reason := classifier.ClassifyToolCall("bash", args, "")
			if tier != TierT0 {
				t.Fatalf("command %q classified as %s, want %s (reason: %s)", cmd, tier, TierT0, reason)
			}

			targetTier, _ := router.RouteToolTask("bash", args, "")
			if targetTier != TierT0 {
				t.Fatalf("command %q routed to %s, want %s", cmd, targetTier, TierT0)
			}
		}

		// Read prompt
		tier, _ := classifier.ClassifyToolCall("", nil, "read file internal/agentopt/doc.go")
		if tier != TierT0 {
			t.Fatalf("read prompt classified as %s, want %s", tier, TierT0)
		}
	})

	t.Run("TrivialJSONExtraction_T0", func(t *testing.T) {
		jsonTools := []string{
			"extract_json", "json_extract", "parse_json", "jq", "format_json",
		}
		for _, tool := range jsonTools {
			tier, reason := classifier.ClassifyToolCall(tool, nil, "")
			if tier != TierT0 {
				t.Fatalf("tool %s classified as %s, want %s (reason: %s)", tool, tier, TierT0, reason)
			}

			targetTier, _ := router.RouteToolTask(tool, nil, "")
			if targetTier != TierT0 {
				t.Fatalf("tool %s routed to %s, want %s", tool, targetTier, TierT0)
			}
		}

		// jq command
		args := map[string]any{"command": "jq '.dependencies' package.json"}
		tier, _ := classifier.ClassifyToolCall("bash", args, "")
		if tier != TierT0 {
			t.Fatalf("jq command classified as %s, want %s", tier, TierT0)
		}

		// JSON extraction prompt
		tier, _ = classifier.ClassifyToolCall("", nil, "extract json from the stdout output")
		if tier != TierT0 {
			t.Fatalf("json prompt classified as %s, want %s", tier, TierT0)
		}
	})

	t.Run("StandardTask_T1", func(t *testing.T) {
		// Single file edit
		tier, reason := classifier.ClassifyToolCall("Edit", map[string]any{"filePath": "foo.go"}, "Fix nil pointer check")
		if tier != TierT1 {
			t.Fatalf("single file edit classified as %s, want %s (reason: %s)", tier, TierT1, reason)
		}

		targetTier, endpoint := router.RouteToolTask("Edit", map[string]any{"filePath": "foo.go"}, "Fix nil pointer check")
		if targetTier != TierT1 {
			t.Fatalf("single file edit routed to %s, want %s", targetTier, TierT1)
		}
		if endpoint != router.EndpointForTier(TierT1) {
			t.Fatalf("endpoint mismatch: got %s, want %s", endpoint, router.EndpointForTier(TierT1))
		}

		// Write unit test
		tier, _ = classifier.ClassifyToolCall("Write", map[string]any{"filePath": "foo_test.go"}, "Add unit test for helper")
		if tier != TierT1 {
			t.Fatalf("write unit test classified as %s, want %s", tier, TierT1)
		}

		// Run tests command
		tier, _ = classifier.ClassifyToolCall("bash", map[string]any{"command": "go test -v ./internal/agentopt"}, "Run package tests")
		if tier != TierT1 {
			t.Fatalf("test execution classified as %s, want %s", tier, TierT1)
		}
	})

	t.Run("ComplexMultiFileEdits_T2", func(t *testing.T) {
		// Multiple target files in args
		args := map[string]any{
			"files": []string{"a.go", "b.go", "c.go"},
		}
		tier, reason := classifier.ClassifyToolCall("edit", args, "Update function signature")
		if tier != TierT2 {
			t.Fatalf("multi-file edit classified as %s, want %s (reason: %s)", tier, TierT2, reason)
		}
		if !strings.Contains(reason, "multi-file") {
			t.Fatalf("reason missing multi-file: %s", reason)
		}

		targetTier, endpoint := router.RouteToolTask("edit", args, "Update function signature")
		if targetTier != TierT2 {
			t.Fatalf("multi-file edit routed to %s, want %s", targetTier, TierT2)
		}
		if endpoint != router.EndpointForTier(TierT2) {
			t.Fatalf("endpoint mismatch: got %s, want %s", endpoint, router.EndpointForTier(TierT2))
		}

		// Prompt indicating multi-file refactoring across codebase
		tier, _ = classifier.ClassifyToolCall("Edit", map[string]any{"filePath": "a.go"}, "Perform multi-file refactoring across the codebase")
		if tier != TierT2 {
			t.Fatalf("multi-file prompt classified as %s, want %s", tier, TierT2)
		}

		// Prompt indicating repo-wide sweep
		tier, _ = classifier.ClassifyToolCall("", nil, "Sweep all packages and update error wrapping")
		if tier != TierT2 {
			t.Fatalf("sweep all packages classified as %s, want %s", tier, TierT2)
		}
	})

	t.Run("ArchitectureTasks_T2", func(t *testing.T) {
		architecturePrompts := []string{
			"Design the overall system architecture for the storage subsystem",
			"Audit concurrency invariants and lock ordering to eliminate race hazards",
			"Formalize the distributed consensus model and spec",
			"Perform security audit and update threat model",
			"Execute protocol migration across client and server boundaries",
			"Analyze frozen abi invariants before modifying types",
		}
		for _, prompt := range architecturePrompts {
			tier, reason := classifier.ClassifyToolCall("Edit", map[string]any{"filePath": "core.go"}, prompt)
			if tier != TierT2 {
				t.Fatalf("architecture prompt %q classified as %s, want %s (reason: %s)", prompt, tier, TierT2, reason)
			}

			targetTier, endpoint := router.RouteToolTask("Edit", map[string]any{"filePath": "core.go"}, prompt)
			if targetTier != TierT2 {
				t.Fatalf("architecture prompt %q routed to %s, want %s", prompt, targetTier, TierT2)
			}
			if endpoint != router.EndpointForTier(TierT2) {
				t.Fatalf("endpoint mismatch: got %s, want %s", endpoint, router.EndpointForTier(TierT2))
			}
		}
	})

	t.Run("CustomToolRegistration", func(t *testing.T) {
		c := NewTierClassifier()
		c.RegisterToolTier("custom_fast_lint", TierT0)
		c.RegisterToolTier("custom_deep_verifier", TierT2)

		tier, _ := c.ClassifyToolCall("custom_fast_lint", nil, "")
		if tier != TierT0 {
			t.Fatalf("custom T0 tool classified as %s, want %s", tier, TierT0)
		}

		tier, _ = c.ClassifyToolCall("custom_deep_verifier", nil, "")
		if tier != TierT2 {
			t.Fatalf("custom T2 tool classified as %s, want %s", tier, TierT2)
		}

		// Router method
		r := NewTieredModelRouterWithClassifier(c, nil)
		targetTier, _ := r.RouteToolTask("custom_deep_verifier", nil, "")
		if targetTier != TierT2 {
			t.Fatalf("custom T2 routed to %s, want %s", targetTier, TierT2)
		}
	})

	t.Run("EndpointOverrides", func(t *testing.T) {
		customEndpoints := map[ModelTier]string{
			TierT0: "http://10.0.0.1:9000/v1/fast",
			TierT1: "http://10.0.0.2:9000/v1/normal",
			TierT2: "http://10.0.0.3:9000/v1/frontier",
		}
		r := NewTieredModelRouter(customEndpoints)

		tier0, ep0 := r.RouteToolTask("gofmt", nil, "")
		if tier0 != TierT0 || ep0 != customEndpoints[TierT0] {
			t.Fatalf("T0 routed to (%s, %s), want (%s, %s)", tier0, ep0, TierT0, customEndpoints[TierT0])
		}

		tier1, ep1 := r.RouteToolTask("Edit", map[string]any{"filePath": "foo.go"}, "simple fix")
		if tier1 != TierT1 || ep1 != customEndpoints[TierT1] {
			t.Fatalf("T1 routed to (%s, %s), want (%s, %s)", tier1, ep1, TierT1, customEndpoints[TierT1])
		}

		tier2, ep2 := r.RouteToolTask("", nil, "System architecture and distributed consensus redesign")
		if tier2 != TierT2 || ep2 != customEndpoints[TierT2] {
			t.Fatalf("T2 routed to (%s, %s), want (%s, %s)", tier2, ep2, TierT2, customEndpoints[TierT2])
		}

		// Dynamic update
		r.SetEndpoint(TierT0, "http://10.0.0.99:9000/v1/new-fast")
		if got := r.EndpointForTier(TierT0); got != "http://10.0.0.99:9000/v1/new-fast" {
			t.Fatalf("updated endpoint got %s, want new-fast", got)
		}
	})

	t.Run("ToolCallRouting", func(t *testing.T) {
		r := NewTieredModelRouter(nil)
		callT0 := ToolCall{Name: "prettier", Args: map[string]any{"file": "app.css"}}
		tier, _ := r.RouteCall(callT0, "")
		if tier != TierT0 {
			t.Fatalf("callT0 routed to %s, want %s", tier, TierT0)
		}

		callT1 := ToolCall{Name: "Edit", Args: map[string]any{"filePath": "app.go"}}
		choice := r.RouteCallChoice(callT1, "fix error handling")
		if choice.TargetModelTier != TierT1 {
			t.Fatalf("callT1 choice tier %s, want %s", choice.TargetModelTier, TierT1)
		}
		if choice.TargetEndpoint != r.EndpointForTier(TierT1) {
			t.Fatalf("callT1 choice endpoint %s, want %s", choice.TargetEndpoint, r.EndpointForTier(TierT1))
		}
	})

	t.Run("ConcurrentRouting", func(t *testing.T) {
		r := NewTieredModelRouter(nil)
		var wg sync.WaitGroup
		for i := 0; i < 50; i++ {
			wg.Add(3)
			go func() {
				defer wg.Done()
				tier, _ := r.RouteToolTask("gofmt", nil, "")
				if tier != TierT0 {
					t.Errorf("concurrent T0 got %s", tier)
				}
			}()
			go func() {
				defer wg.Done()
				tier, _ := r.RouteToolTask("Edit", map[string]any{"filePath": "foo.go"}, "fix bug")
				if tier != TierT1 {
					t.Errorf("concurrent T1 got %s", tier)
				}
			}()
			go func() {
				defer wg.Done()
				tier, _ := r.RouteToolTask("", nil, "Design system architecture")
				if tier != TierT2 {
					t.Errorf("concurrent T2 got %s", tier)
				}
			}()
		}
		wg.Wait()
	})
}
