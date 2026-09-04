package toolcatalog

import (
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func commandAdapterRegistration(t *testing.T, helper, result string) Registration {
	t.Helper()
	adapter := map[string]any{
		"version": "fak.command-adapter/v1",
		"argv": []map[string]any{
			{"literal": "-test.run=TestCommandAdapterHelperProcess"},
			{"literal": "--"},
			{"field": "query", "flag": "--query"},
		},
		"stdin":  map[string]any{"field": "body"},
		"env":    map[string]string{"FAK_TOOL_TEST_ENV": "token"},
		"result": result,
	}
	adapterJSON, err := json.Marshal(adapter)
	if err != nil {
		t.Fatal(err)
	}
	skill := "---\nname: go_helper\ndescription: exercise a Go command\n---\n```fak-program\n" +
		`{"version":"fak.skill-program/v1","name":"go_helper","input_schema":{"type":"object","properties":{"query":{"type":"string"},"body":{"type":"string"},"token":{"type":"string"}},"required":["query","body","token"]},"executor":{"argv":[` + quoteJSON(helper) + `],"adapter":` + string(adapterJSON) + `}}` + "\n```"
	reg, err := CompileSkill([]byte(skill), "skills/go-helper/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	return reg
}

func quoteJSON(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestRunCommandMapsFieldsWithoutShellInterpolation(t *testing.T) {
	reg := commandAdapterRegistration(t, os.Args[0], "json")
	input := json.RawMessage(`{"query":"a b; echo never","body":"stdin bytes","token":"env value"}`)
	result, err := RunCommand(context.Background(), reg, input)
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(result.JSON, &got); err != nil {
		t.Fatal(err)
	}
	if got["query"] != "a b; echo never" || got["stdin"] != "stdin bytes" || got["env"] != "env value" {
		t.Fatalf("result = %s", result.JSON)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(os.Args[0]), "never")); err == nil {
		t.Fatal("field was interpreted as a shell command")
	}
}

func TestRunCommandRefusesMissingFieldAndNonzeroExit(t *testing.T) {
	reg := commandAdapterRegistration(t, os.Args[0], "json")
	if _, err := RunCommand(context.Background(), reg, json.RawMessage(`{"body":"x","token":"y"}`)); err == nil || !strings.Contains(err.Error(), "TOOL_COMMAND_FIELD_MISSING: query") {
		t.Fatalf("missing field error = %v", err)
	}
	input := json.RawMessage(`{"query":"exit-7","body":"x","token":"y"}`)
	result, err := RunCommand(context.Background(), reg, input)
	if err == nil || result.ExitCode != 7 || !strings.Contains(result.Stderr, "requested failure") {
		t.Fatalf("result=%#v error=%v", result, err)
	}
}

func TestRunCommandRefusesUndeclaredAdapterAndBadResultShape(t *testing.T) {
	reg, err := CompileSkill([]byte(skillFixture), ".claude/skills/repo-search/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RunCommand(context.Background(), reg, json.RawMessage(`{"query":"x"}`)); err == nil || !strings.Contains(err.Error(), "TOOL_COMMAND_ADAPTER_UNDECLARED") {
		t.Fatalf("undeclared adapter error = %v", err)
	}
	bad := commandAdapterRegistration(t, os.Args[0], "json")
	input := json.RawMessage(`{"query":"bad-json","body":"x","token":"y"}`)
	if _, err := RunCommand(context.Background(), bad, input); err == nil || !strings.Contains(err.Error(), "TOOL_COMMAND_RESULT_JSON") {
		t.Fatalf("bad result error = %v", err)
	}
}

func TestCommandAdapterHelperProcess(t *testing.T) {
	t.Helper()
	if runtime.GOOS == "windows" && !strings.Contains(strings.Join(os.Args, " "), "TestCommandAdapterHelperProcess") {
		return
	}
	query := ""
	for i, arg := range os.Args {
		if arg == "--query" && i+1 < len(os.Args) {
			query = os.Args[i+1]
		}
	}
	if query == "" {
		t.Skip("helper process")
		return
	}
	if query == "exit-7" {
		_, _ = os.Stderr.WriteString("requested failure")
		os.Exit(7)
	}
	if query == "bad-json" {
		_, _ = os.Stdout.WriteString("not json")
		os.Exit(0)
	}
	body, _ := io.ReadAll(os.Stdin)
	_ = json.NewEncoder(os.Stdout).Encode(map[string]string{"query": query, "stdin": string(body), "env": os.Getenv("FAK_TOOL_TEST_ENV")})
	os.Exit(0)
}
