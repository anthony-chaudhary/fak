package testroute

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestDecide(t *testing.T) {
	cases := []struct {
		name     string
		probe    Probe
		kind     Kind
		template []string
	}{
		{
			name:  "windows blocked native uses wsl wrapper",
			probe: Probe{GOOS: "windows", NativeTestAllowed: false, WSLPresent: true, CIReachable: true},
			kind:  KindWSL,
			template: []string{
				"powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "test.ps1", ArgsPlaceholder,
			},
		},
		{
			name:     "native allowed wins",
			probe:    Probe{GOOS: "linux", NativeTestAllowed: true, WSLPresent: false, CIReachable: true},
			kind:     KindNative,
			template: []string{"go", "test", ArgsPlaceholder},
		},
		{
			name:  "ci fallback",
			probe: Probe{GOOS: "windows", NativeTestAllowed: false, WSLPresent: false, CIReachable: true},
			kind:  KindCI,
			template: []string{
				"gh", "workflow", "run", "ci.yml", "--ref", "main", "-f", "go_test_args_json=" + CIArgsJSONPlaceholder,
			},
		},
		{
			name:     "unavailable",
			probe:    Probe{GOOS: "windows", NativeTestAllowed: false, WSLPresent: false, CIReachable: false},
			kind:     KindUnavailable,
			template: nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			route := Decide(tc.probe)
			if route.Kind != tc.kind {
				t.Fatalf("kind = %q, want %q", route.Kind, tc.kind)
			}
			if !reflect.DeepEqual(route.CommandTemplate, tc.template) {
				t.Fatalf("template = %v, want %v", route.CommandTemplate, tc.template)
			}
			if route.Reason == "" {
				t.Fatal("reason must be populated")
			}
		})
	}
}

func TestCommandExpandsPlaceholder(t *testing.T) {
	template := []string{"powershell", "-File", "test.ps1", ArgsPlaceholder}
	got := Command(template, []string{"-short", "./..."})
	want := []string{"powershell", "-File", "test.ps1", "-short", "./..."}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("command = %v, want %v", got, want)
	}
}

func TestCIOnlyRouteMatchesDispatchContract(t *testing.T) {
	route := Decide(Probe{GOOS: "windows", CIReachable: true})
	got := Command(route.CommandTemplate, []string{"-run", "TestName with spaces", "./internal/testroute"})
	want := []string{
		"gh", "workflow", "run", "ci.yml", "--ref", "main", "-f",
		`go_test_args_json=["-run","TestName with spaces","./internal/testroute"]`,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("command = %#v, want %#v", got, want)
	}

	workflowPath := os.Getenv("TESTROUTE_CI_WORKFLOW")
	if workflowPath == "" {
		workflowPath = "../../.github/workflows/ci.yml"
	}
	workflowBytes, err := os.ReadFile(workflowPath)
	if err != nil {
		t.Fatal(err)
	}
	workflow := string(workflowBytes)
	for _, required := range []string{
		"workflow_dispatch:",
		"go_test_args_json:",
		"testroute-ci-only:",
		"runs-on: ubuntu-24.04",
		`GO_TEST_ARGS_JSON: ${{ inputs.go_test_args_json }}`,
		`go test "${go_test_args[@]}"`,
	} {
		if !strings.Contains(workflow, required) {
			t.Errorf("ci.yml lacks executable CI-only route contract %q", required)
		}
	}
}
