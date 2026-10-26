package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/hostapproval"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

type cliFixture struct {
	exe, installer, policy, context, codex, sentinel string
}

// fak-test:runtime medium est=20s lane=default
func TestNativeCLIPrepareVerifyAndPassiveStatus(t *testing.T) {
	f := newCLIFixture(t)
	prepareArgs := []string{
		"prepare", "--installer", f.installer, "--policy", f.policy,
		"--context", f.context, "--codex-config", f.codex,
		"--server", "fak-test", "--task", "task-14030",
		"--session", "session-14030", "--ttl", "2m",
	}
	raw, prepareStderr, code := runCLI(t, f.exe, prepareArgs, nil)
	if code != 0 {
		t.Fatalf("prepare code=%d stderr=%q", code, prepareStderr)
	}
	request, err := hostapproval.Decode(raw)
	if err != nil {
		t.Fatalf("prepare output is not a strict request: %v\n%s", err, raw)
	}
	digest, err := hostapproval.Digest(request)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(prepareStderr, "request_sha256="+digest) || !strings.Contains(prepareStderr, "nonce_status=untrusted_pending_broker") {
		t.Fatalf("prepare receipt = %q", prepareStderr)
	}

	verified, stderr, code := runCLI(t, f.exe, []string{
		"verify", "--session", "session-14030", "--request-sha256", digest,
	}, raw)
	if code != 0 {
		t.Fatalf("verify code=%d stderr=%q", code, stderr)
	}
	var result verificationResult
	if err := json.Unmarshal(verified, &result); err != nil {
		t.Fatalf("verify output: %v", err)
	}
	if result.Authorized || !result.BrokerRequired || result.BrokerAvailable || result.RequestSHA256 != digest {
		t.Fatalf("verify authority result = %+v", result)
	}

	statusRaw, stderr, code := runCLI(t, f.exe, []string{"status"}, nil)
	if code != 0 {
		t.Fatalf("status code=%d stderr=%q", code, stderr)
	}
	var status statusResult
	if err := json.Unmarshal(statusRaw, &status); err != nil {
		t.Fatalf("status output: %v", err)
	}
	if status.Authorized || !status.BrokerRequired || status.BrokerAvailable {
		t.Fatalf("status authority result = %+v", status)
	}
	if runtime.GOOS == "windows" && !status.WebAuthnAPIAvailable {
		t.Fatalf("Windows WebAuthn API not observed: %+v", status)
	}

	t.Run("request mutations fail closed", func(t *testing.T) {
		var unknown map[string]any
		if err := json.Unmarshal(raw, &unknown); err != nil {
			t.Fatal(err)
		}
		unknown["origin"] = "caller"
		unknownRaw, err := json.Marshal(unknown)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, code := runCLI(t, f.exe, []string{"verify", "--session", "session-14030", "--request-sha256", digest}, unknownRaw); code == 0 {
			t.Fatal("caller-supplied origin verified")
		}

		mutated := request
		mutated.SessionID = "other-session"
		mutatedRaw, err := json.Marshal(mutated)
		if err != nil {
			t.Fatal(err)
		}
		mutatedDigest, err := hostapproval.Digest(mutated)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, code := runCLI(t, f.exe, []string{"verify", "--session", "session-14030", "--request-sha256", mutatedDigest}, mutatedRaw); code == 0 {
			t.Fatal("request rebound to another session verified")
		}
		if _, _, code := runCLI(t, f.exe, []string{"verify", "--session", "session-14030", "--request-sha256", strings.Repeat("0", 64)}, raw); code == 0 {
			t.Fatal("wrong request digest verified")
		}
	})

	for _, forbidden := range []string{"--origin", "--principal", "--approve", "--approval"} {
		t.Run("forbidden flag "+forbidden, func(t *testing.T) {
			args := append(append([]string(nil), prepareArgs...), forbidden, "caller")
			if _, _, code := runCLI(t, f.exe, args, nil); code == 0 {
				t.Fatalf("%s accepted", forbidden)
			}
		})
	}
	for name, input := range map[string][]byte{
		"malformed JSON": []byte(`{"schema":`),
		"duplicate JSON": duplicateSchema(raw),
	} {
		t.Run(name, func(t *testing.T) {
			if _, _, code := runCLI(t, f.exe, []string{"verify", "--session", "session-14030", "--request-sha256", digest}, input); code == 0 {
				t.Fatal("ambiguous request verified")
			}
		})
	}
	if _, _, code := runCLI(t, f.exe, []string{
		"verify", "--session", "session-14030", "--session", "session-14030", "--request-sha256", digest,
	}, raw); code == 0 {
		t.Fatal("duplicate --session accepted")
	}
	if got, err := os.ReadFile(f.sentinel); err != nil || string(got) != "untouched" {
		t.Fatalf("CLI mutated execution sentinel: %q %v", got, err)
	}
}

func newCLIFixture(t *testing.T) cliFixture {
	t.Helper()
	dir := t.TempDir()
	exe := filepath.Join(dir, "fak-host-approval"+exeSuffix())
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	build := exec.CommandContext(ctx, "go", "build", "-buildvcs=false", "-o", exe, ".")
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build native CLI: %v\n%s", err, output)
	}
	exeBytes, err := os.ReadFile(exe)
	if err != nil {
		t.Fatal(err)
	}
	installer := filepath.Join(dir, "fak-test"+exeSuffix())
	adapter := filepath.Join(dir, "adapter-test"+exeSuffix())
	for _, path := range []string{installer, adapter} {
		if err := os.WriteFile(path, exeBytes, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	policy := filepath.Join(dir, "policy.json")
	if err := os.WriteFile(policy, []byte(`{"version":"fak-policy/v1","posture":"default_open"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	adapterSum := sha256.Sum256(exeBytes)
	context := filepath.Join(dir, "remote-maintenance.json")
	cfg := remotemaint.Config{Schema: remotemaint.ConfigSchema, Profiles: []remotemaint.Profile{{
		ProfileID: "profile-a", Target: "node-a", HostKey: testHostKey(1),
		Operation: remotemaint.OperationSystemdProfileInstallReload, PayloadSHA256: testDigest(2), Holder: "holder-a",
		Paths: []string{"/etc/fak/profile.conf"}, PathPrefixes: []remotemaint.PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "maintenance-stage."}},
		Units: []string{"fak.service"}, LeaseNodes: []string{"node-a"}, ExpiresAt: time.Now().UTC().Add(10 * time.Minute),
		Authority: remotemaint.Command{Executable: adapter, SHA256: hex.EncodeToString(adapterSum[:]), Args: []string{"--native-adapter"}},
	}}}
	contextRaw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(context, contextRaw, 0o600); err != nil {
		t.Fatal(err)
	}
	codex := filepath.Join(dir, "config.toml")
	if err := os.WriteFile(codex, []byte("model = \"test\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	sentinel := filepath.Join(dir, "must-not-execute")
	if err := os.WriteFile(sentinel, []byte("untouched"), 0o600); err != nil {
		t.Fatal(err)
	}
	return cliFixture{exe: exe, installer: installer, policy: policy, context: context, codex: codex, sentinel: sentinel}
}

func runCLI(t *testing.T, exe string, args []string, stdin []byte) ([]byte, string, int) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, exe, args...)
	cmd.Stdin = bytes.NewReader(stdin)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if err == nil {
		return stdout.Bytes(), stderr.String(), 0
	}
	if exitErr, ok := err.(*exec.ExitError); ok {
		return stdout.Bytes(), stderr.String(), exitErr.ExitCode()
	}
	t.Fatalf("run CLI: %v", err)
	return nil, "", -1
}

func exeSuffix() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}

func testDigest(b byte) string {
	return strings.Repeat(hex.EncodeToString([]byte{b}), sha256.Size)
}

func testHostKey(b byte) string {
	raw := make([]byte, sha256.Size)
	for i := range raw {
		raw[i] = b
	}
	return "SHA256:" + base64.RawStdEncoding.EncodeToString(raw)
}

func duplicateSchema(raw []byte) []byte {
	trimmed := bytes.TrimSpace(raw)
	out := append([]byte(nil), trimmed[:len(trimmed)-1]...)
	return append(out, []byte(`,"schema":"fak.host-approval.request.v1"}`)...)
}
