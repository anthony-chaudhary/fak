package main

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

func main() {
	root, err := repoRoot()
	must(err)
	goCmd := filepath.Join(root, "cmd", "fak")
	demo := filepath.Join(root, "examples", "security-fleet-governance")
	floor := filepath.Join(demo, "central-floor.json")
	narrow := filepath.Join(demo, "identity-narrowing.json")
	tmp, err := os.MkdirTemp("", "fak-security-fleet-governance-")
	must(err)
	defer os.RemoveAll(tmp)
	reload := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusNoContent) }))
	defer reload.Close()
	active := filepath.Join(tmp, "active-policy.json")
	must(copyFile(floor, active))

	run(root, "go", "run", goCmd, "policy", "--check", floor)
	run(root, "go", "run", goCmd, "attest", "--policy", floor, "--quiet")
	want(root, "ALLOW", "go", "run", goCmd, "preflight", "--policy", floor, "--tool", "read_corp_kb", "--args", `{"uri":"corp://security/identity/runbook"}`)
	want(root, "DENY", "go", "run", goCmd, "preflight", "--policy", floor, "--tool", "read_corp_kb", "--args", `{"uri":"https://unapproved.example/runbook"}`)
	want(root, "DENY", "go", "run", goCmd, "preflight", "--policy", floor, "--tool", "rotate_credentials", "--args", `{}`)

	preview := run(root, "go", "run", goCmd, "policy", "land-rule", "--policy", active, "--candidate", narrow)
	contains(preview, `"allow_glob": "corp://security/identity/**"`)
	landed := run(root, "go", "run", goCmd, "policy", "land-rule", "--policy", active, "--candidate", narrow, "--land", "--reload-url", reload.URL)
	contains(landed, "landed")
	want(root, "ALLOW", "go", "run", goCmd, "preflight", "--policy", active, "--tool", "read_corp_kb", "--args", `{"uri":"corp://security/identity/runbook"}`)
	want(root, "DENY", "go", "run", goCmd, "preflight", "--policy", active, "--tool", "read_corp_kb", "--args", `{"uri":"corp://security/payments/runbook"}`)

	rolled := run(root, "go", "run", goCmd, "policy", "land-rule", "--policy", active, "--rollback", "--reload-url", reload.URL)
	contains(rolled, "restored")
	want(root, "ALLOW", "go", "run", goCmd, "preflight", "--policy", active, "--tool", "read_corp_kb", "--args", `{"uri":"corp://security/payments/runbook"}`)
	fmt.Println("PASS security-fleet-governance: floor, attestation, canonical references, narrowing, read-back, and rollback")
}

func repoRoot() (string, error) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "", fmt.Errorf("locate demo source")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", "..")), nil
}

func run(dir, name string, args ...string) string {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "$ %s %s\n%s", name, strings.Join(args, " "), out.String())
		must(err)
	}
	return out.String()
}

func want(dir, verdict, name string, args ...string) {
	out := run(dir, name, args...)
	contains(out, verdict)
}

func contains(out, needle string) {
	if !strings.Contains(out, needle) {
		must(fmt.Errorf("output does not contain %q:\n%s", needle, out))
	}
}

func copyFile(src, dst string) error {
	b, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, b, 0o600)
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "FAIL security-fleet-governance:", err)
		os.Exit(1)
	}
}
