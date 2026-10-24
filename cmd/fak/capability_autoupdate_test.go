package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

const capabilityAutoUpgradeTestCommit = "0123456789abcdef0123456789abcdef01234567"

// fak-test:runtime fast est=10ms lane=default
func TestCapabilityAutoUpgradeHILHelpStaysLocal(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		args []string
		help bool
	}{
		{name: "long help", args: []string{"--help"}, help: true},
		{name: "short help", args: []string{"-h"}, help: true},
		{name: "single dash help", args: []string{"-help"}, help: true},
		{name: "double dash short help", args: []string{"--h"}, help: true},
		{name: "help after boolean", args: []string{"--probe", "--help"}, help: true},
		{name: "help after value", args: []string{"--lan-host", "test-host", "-h"}, help: true},
		{name: "help after explicit false", args: []string{"--probe=false", "--help"}, help: true},
		{name: "micro doses"},
		{name: "probe", args: []string{"--probe"}},
		{name: "inventory", args: []string{"--inventory"}},
		{name: "comparison audit", args: []string{"--audit-comparison"}},
		{name: "help is string value", args: []string{"--candidate", "--help"}},
		{name: "help is inline value", args: []string{"--candidate=-h"}},
		{name: "help follows terminator", args: []string{"--", "--help"}},
		{name: "help follows positional", args: []string{"operand", "--help"}},
		{name: "help follows invalid flag", args: []string{"--unknown", "--help"}},
		{name: "help follows invalid value", args: []string{"--candidate-val=invalid", "--help"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			argv := append([]string{"fak", "hil"}, tc.args...)
			if got := needsCapabilityAutoUpgrade(argv); got == tc.help {
				t.Fatalf("needsCapabilityAutoUpgrade(%q) = %v, want %v", argv, got, !tc.help)
			}
			if !tc.help {
				return
			}
			deps := capabilityAutoUpgradeDeps{
				getenv: func(string) string {
					t.Fatal("help must not consult updater configuration")
					return ""
				},
				resolveCheckout: func() (string, string, error) {
					t.Fatal("help must not resolve the checkout")
					return "", "", nil
				},
				process: func(string, []string, io.Reader, io.Writer, io.Writer, []string) error {
					t.Fatal("help must not start the updater")
					return nil
				},
			}
			var stdout, stderr bytes.Buffer
			if code, handled := runCapabilityAutoUpgrade(nil, &stdout, &stderr, argv, deps); code != 0 || handled {
				t.Fatalf("early gate = (%d, %v), want (0, false)", code, handled)
			}
			if stdout.Len() != 0 || stderr.Len() != 0 {
				t.Fatalf("early gate emitted output: stdout=%q stderr=%q", stdout.String(), stderr.String())
			}
			if code := runHIL(&stdout, &stderr, tc.args); code != 0 || stdout.Len() != 0 || !strings.Contains(stderr.String(), "Usage of hil:") {
				t.Fatalf("help = %d, stdout=%q stderr=%q", code, stdout.String(), stderr.String())
			}
		})
	}
}

func TestCapabilityAutoUpgradeEligibilityAndEscapeStayCheap(t *testing.T) {
	tests := []struct {
		name string
		argv []string
		env  string
	}{
		{name: "unrelated verb", argv: []string{"fak", "serve"}},
		{name: "validate without strix", argv: []string{"fak", "validate", "--mine", "cmd/fak"}},
		{name: "explicit off", argv: []string{"fak", "hil"}, env: "off"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			deps := capabilityAutoUpgradeDeps{
				getenv: func(key string) string {
					if key == capabilityAutoUpgradeEnv {
						return tc.env
					}
					return ""
				},
				resolveCheckout: func() (string, string, error) {
					t.Fatal("checkout resolution must not run")
					return "", "", nil
				},
			}
			if code, handled := runCapabilityAutoUpgrade(nil, io.Discard, io.Discard, tc.argv, deps); code != 0 || handled {
				t.Fatalf("result = (%d, %v), want (0, false)", code, handled)
			}
		})
	}
}

func TestCapabilityAutoUpgradeNoRepoSkipsProcesses(t *testing.T) {
	deps := capabilityAutoUpgradeTestDeps()
	deps.resolveCheckout = func() (string, string, error) { return "", "", nil }
	deps.process = func(string, []string, io.Reader, io.Writer, io.Writer, []string) error {
		t.Fatal("no-repo fast path must not start a process")
		return nil
	}
	if code, handled := runCapabilityAutoUpgrade(nil, io.Discard, io.Discard, []string{"fak", "hil"}, deps); code != 0 || handled {
		t.Fatalf("result = (%d, %v), want (0, false)", code, handled)
	}
}

func TestCapabilityAutoUpgradeCurrentChecksOnceAndDispatches(t *testing.T) {
	const executable = `C:\tools\fak.exe`
	deps := capabilityAutoUpgradeTestDeps()
	deps.resolveCheckout = func() (string, string, error) { return `C:\src\fak`, executable, nil }
	calls := 0
	deps.process = func(gotExecutable string, args []string, _ io.Reader, stdout, _ io.Writer, _ []string) error {
		calls++
		if gotExecutable != executable {
			t.Fatalf("executable = %q, want %q", gotExecutable, executable)
		}
		wantArgs := []string{"self-update", "--check", "--json", "--root", `C:\src\fak`, "--target", executable}
		if !reflect.DeepEqual(args, wantArgs) {
			t.Fatalf("args = %#v, want %#v", args, wantArgs)
		}
		return writeCapabilityReceipt(stdout, capabilityReceipt("current", executable, "", 0))
	}
	if code, handled := runCapabilityAutoUpgrade(nil, io.Discard, io.Discard, []string{"fak", "validate", "--strix"}, deps); code != 0 || handled {
		t.Fatalf("result = (%d, %v), want (0, false)", code, handled)
	}
	if calls != 1 {
		t.Fatalf("process calls = %d, want 1", calls)
	}
}

func TestCapabilityAutoUpgradeStaleUpdatesAndReexecsExactInvocation(t *testing.T) {
	const (
		root       = `C:\src\fak`
		executable = `C:\tools\fak.exe`
	)
	argv := []string{"fak-from-path", "validate", "--strix", "--mine", "cmd/fak"}
	stdin := strings.NewReader("original stdin")
	var stdout, stderr bytes.Buffer
	deps := capabilityAutoUpgradeTestDeps()
	deps.resolveCheckout = func() (string, string, error) { return root, executable, nil }
	deps.environ = func() []string { return []string{"PATH=test", capabilityAutoUpgradeReexecEnv + "=stale"} }
	deps.getpid = func() int { return 4242 }
	calls := 0
	deps.process = func(gotExecutable string, args []string, gotStdin io.Reader, gotStdout, gotStderr io.Writer, env []string) error {
		calls++
		if gotExecutable != executable {
			t.Fatalf("call %d executable = %q", calls, gotExecutable)
		}
		switch calls {
		case 1:
			want := []string{"self-update", "--check", "--json", "--root", root, "--target", executable}
			if !reflect.DeepEqual(args, want) || gotStdin != nil {
				t.Fatalf("check invocation = %#v stdin=%v", args, gotStdin)
			}
			return writeCapabilityReceipt(gotStdout, capabilityReceipt("stale", executable, capabilityAutoUpgradeTestCommit, 0))
		case 2:
			want := []string{"self-update", "--json", "--root", root, "--target", executable}
			if !reflect.DeepEqual(args, want) || gotStdin != nil || gotStderr != &stderr {
				t.Fatalf("update invocation did not preserve transaction streams: args=%#v", args)
			}
			return writeCapabilityReceipt(gotStdout, capabilityReceipt("updated", executable, capabilityAutoUpgradeTestCommit, 1))
		case 3:
			if !reflect.DeepEqual(args, argv[1:]) {
				t.Fatalf("re-exec args = %#v, want %#v", args, argv[1:])
			}
			if gotStdin != stdin || gotStdout != &stdout || gotStderr != &stderr {
				t.Fatal("re-exec did not preserve stdin/stdout/stderr")
			}
			wantMarker := capabilityAutoUpgradeReexecEnv + "=" + capabilityAutoUpgradeTestCommit + ":4242"
			if !capabilityContainsString(env, wantMarker) {
				t.Fatalf("re-exec env lacks %q: %#v", wantMarker, env)
			}
			if capabilityContainsString(env, capabilityAutoUpgradeReexecEnv+"=stale") {
				t.Fatalf("re-exec env retained inherited marker: %#v", env)
			}
			return nil
		default:
			t.Fatalf("unexpected process call %d", calls)
			return nil
		}
	}
	if code, handled := runCapabilityAutoUpgrade(stdin, &stdout, &stderr, argv, deps); code != 0 || !handled {
		t.Fatalf("result = (%d, %v), want (0, true); stderr=%s", code, handled, stderr.String())
	}
	if calls != 3 {
		t.Fatalf("process calls = %d, want 3", calls)
	}
}

func TestCapabilityAutoUpgradeFailsClosedBeforeDispatch(t *testing.T) {
	const executable = `C:\tools\fak.exe`
	tests := []struct {
		name    string
		process capabilityAutoUpgradeProcess
	}{
		{
			name: "invalid check receipt",
			process: func(_ string, _ []string, _ io.Reader, stdout, _ io.Writer, _ []string) error {
				_, _ = io.WriteString(stdout, "not-json")
				return nil
			},
		},
		{
			name: "update failure",
			process: func(_ string, args []string, _ io.Reader, stdout, _ io.Writer, _ []string) error {
				if capabilityContainsString(args, "--check") {
					return writeCapabilityReceipt(stdout, capabilityReceipt("stale", executable, capabilityAutoUpgradeTestCommit, 0))
				}
				return errors.New("transaction failed")
			},
		},
		{
			name: "unverified update receipt",
			process: func(_ string, args []string, _ io.Reader, stdout, _ io.Writer, _ []string) error {
				if capabilityContainsString(args, "--check") {
					return writeCapabilityReceipt(stdout, capabilityReceipt("stale", executable, capabilityAutoUpgradeTestCommit, 0))
				}
				return writeCapabilityReceipt(stdout, capabilityReceipt("current", executable, capabilityAutoUpgradeTestCommit, 0))
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			deps := capabilityAutoUpgradeTestDeps()
			deps.resolveCheckout = func() (string, string, error) { return `C:\src\fak`, executable, nil }
			deps.process = tc.process
			var stderr bytes.Buffer
			if code, handled := runCapabilityAutoUpgrade(nil, io.Discard, &stderr, []string{"fak", "hil"}, deps); code != 1 || !handled {
				t.Fatalf("result = (%d, %v), want (1, true)", code, handled)
			}
			if stderr.Len() == 0 {
				t.Fatal("failure must explain the closed gate")
			}
		})
	}
}

func TestCapabilityAutoUpgradeReexecMarkerIsOneGeneration(t *testing.T) {
	marker := capabilityAutoUpgradeTestCommit + ":77"
	markerPresent := true
	deps := capabilityAutoUpgradeTestDeps()
	deps.getenv = func(key string) string {
		if key == capabilityAutoUpgradeReexecEnv && markerPresent {
			return marker
		}
		return ""
	}
	deps.parentPID = func() int { return 77 }
	deps.unsetenv = func(key string) error {
		if key != capabilityAutoUpgradeReexecEnv {
			t.Fatalf("unset %q", key)
		}
		markerPresent = false
		return nil
	}
	deps.resolveCheckout = func() (string, string, error) {
		t.Fatal("accepted child marker must bypass a second check")
		return "", "", nil
	}
	if code, handled := runCapabilityAutoUpgrade(nil, io.Discard, io.Discard, []string{"fak", "hil"}, deps); code != 0 || handled {
		t.Fatalf("result = (%d, %v), want (0, false)", code, handled)
	}
	if markerPresent {
		t.Fatal("accepted marker was not consumed")
	}

	deps = capabilityAutoUpgradeTestDeps()
	deps.getenv = func(key string) string {
		if key == capabilityAutoUpgradeReexecEnv {
			return marker
		}
		return ""
	}
	deps.parentPID = func() int { return 78 }
	if code, handled := runCapabilityAutoUpgrade(nil, io.Discard, io.Discard, []string{"fak", "hil"}, deps); code != 1 || !handled {
		t.Fatalf("inherited marker result = (%d, %v), want (1, true)", code, handled)
	}
}

func TestCapabilityAutoUpgradePathAttestationIsOSAware(t *testing.T) {
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" {
		t.Skip("case-insensitive filesystems treat distinct casing as the same file")
	}
	dir := t.TempDir()
	lower := filepath.Join(dir, "fak")
	upper := filepath.Join(dir, "FAK")
	if err := os.WriteFile(lower, []byte("lower"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(upper, []byte("upper"), 0o600); err != nil {
		t.Fatal(err)
	}
	receipt := capabilityReceipt("current", upper, "", 0)
	if capabilityReceiptAttestsPrimary(receipt, lower) {
		t.Fatal("case-distinct Unix executable paths were treated as the same target")
	}
}

func capabilityAutoUpgradeTestDeps() capabilityAutoUpgradeDeps {
	return capabilityAutoUpgradeDeps{
		getenv:   func(string) string { return "" },
		unsetenv: func(string) error { return nil },
		environ:  func() []string { return []string{"PATH=test"} },
		getpid:   func() int { return 1 },
		parentPID: func() int {
			return 1
		},
		resolveCheckout: func() (string, string, error) { return "", "", nil },
		process: func(string, []string, io.Reader, io.Writer, io.Writer, []string) error {
			return nil
		},
	}
}

func capabilityReceipt(status, executable, revision string, changed int) selfUpdateReceipt {
	receipt := selfUpdateReceipt{
		Schema:        selfUpdateReceiptSchema,
		SchemaVersion: 1,
		Status:        status,
		Targets:       []selfUpdateReceiptTarget{{Role: "primary", Path: executable}},
		Changed:       changed,
	}
	if revision != "" {
		receipt.NewRevision = &revision
	}
	return receipt
}

func writeCapabilityReceipt(w io.Writer, receipt selfUpdateReceipt) error {
	return json.NewEncoder(w).Encode(receipt)
}

func capabilityContainsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
