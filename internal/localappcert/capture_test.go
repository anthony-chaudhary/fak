package localappcert

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const captureHelperEnvironment = "GO_WANT_LOCALAPPCERT_CAPTURE_HELPER"

func TestCaptureHelperProcess(t *testing.T) {
	if os.Getenv(captureHelperEnvironment) != "1" {
		return
	}
	separator := -1
	for index, arg := range os.Args {
		if arg == "--" {
			separator = index
			break
		}
	}
	if separator < 0 || separator+1 >= len(os.Args) {
		os.Exit(90)
	}
	args := os.Args[separator+1:]
	switch args[0] {
	case "success":
		fmt.Fprint(os.Stdout, args[1])
		fmt.Fprint(os.Stderr, args[2])
		os.Exit(0)
	case "failure":
		fmt.Fprint(os.Stdout, args[1])
		os.Exit(23)
	case "timeout":
		fmt.Fprint(os.Stdout, args[1])
		time.Sleep(30 * time.Second)
		os.Exit(0)
	default:
		os.Exit(91)
	}
}

func TestValidateCaptureSpecRefusesIncompleteDuplicateAndUnknownBeforeRun(t *testing.T) {
	t.Setenv(captureHelperEnvironment, "1")
	base := completeCaptureSpec(t, "success", "stdout", "stderr")
	tests := []struct {
		name   string
		mutate func(*CaptureSpec)
		want   string
	}{
		{
			name: "missing",
			mutate: func(spec *CaptureSpec) {
				spec.Scenarios = spec.Scenarios[:len(spec.Scenarios)-1]
			},
			want: "missing capture scenarios",
		},
		{
			name: "duplicate",
			mutate: func(spec *CaptureSpec) {
				spec.Scenarios = append(spec.Scenarios, spec.Scenarios[0])
			},
			want: "duplicate capture scenarios",
		},
		{
			name: "unknown",
			mutate: func(spec *CaptureSpec) {
				spec.Scenarios[0].Name = "not-a-required-scenario"
			},
			want: "unknown capture scenarios",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := cloneCaptureSpec(base)
			test.mutate(&spec)
			evidenceDir := filepath.Join(t.TempDir(), "must-not-exist")
			rows, err := Capture(context.Background(), spec, CaptureOptions{EvidenceDir: evidenceDir})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("Capture() error = %v, want containing %q", err, test.want)
			}
			if rows != nil {
				t.Fatalf("Capture() rows = %#v, want nil preflight refusal", rows)
			}
			if _, statErr := os.Stat(evidenceDir); !os.IsNotExist(statErr) {
				t.Fatalf("evidence directory was created before completeness refusal: %v", statErr)
			}
		})
	}
}

func TestParseCaptureSpecRejectsDuplicateScenarioKey(t *testing.T) {
	entry := `{"command":["command"],"timeout":"1s"}`
	var pairs []string
	for _, name := range RequiredScenarios {
		pairs = append(pairs, fmt.Sprintf("%q:%s", name, entry))
	}
	pairs = append(pairs, fmt.Sprintf("%q:%s", RequiredScenarios[0], entry))
	raw := fmt.Sprintf(`{"schema":%q,"runtime_revision":"runtime-r1","artifact":"artifact@sha256:abc","scenarios":{%s}}`, CaptureSchema, strings.Join(pairs, ","))
	if _, err := ParseCaptureSpec([]byte(raw)); err == nil || !strings.Contains(err.Error(), "duplicate capture scenario") {
		t.Fatalf("ParseCaptureSpec() error = %v, want duplicate refusal", err)
	}
}

func TestParseCaptureSpecLoadsCompleteCommandMap(t *testing.T) {
	spec, err := ParseCaptureSpec(validCaptureJSON(t))
	if err != nil {
		t.Fatal(err)
	}
	if spec.Schema != CaptureSchema || spec.RuntimeRevision != "runtime-r1" || spec.Artifact != "artifact@sha256:abc" {
		t.Fatalf("capture metadata = %#v", spec)
	}
	if len(spec.Scenarios) != len(RequiredScenarios) {
		t.Fatalf("got %d commands, want %d", len(spec.Scenarios), len(RequiredScenarios))
	}
	if err := ValidateCaptureSpec(spec, CaptureOptions{EvidenceDir: "evidence"}); err != nil {
		t.Fatal(err)
	}
}

func TestCaptureWritesDeterministicEvidenceAndHashes(t *testing.T) {
	t.Setenv(captureHelperEnvironment, "1")
	spec := completeCaptureSpec(t, "success", "stdout-bytes", "stderr-bytes")
	evidenceDir := t.TempDir()
	rows, err := Capture(context.Background(), spec, CaptureOptions{EvidenceDir: evidenceDir})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != len(RequiredScenarios) {
		t.Fatalf("got %d rows, want %d", len(rows), len(RequiredScenarios))
	}
	wantOutput := []byte("stdout-bytesstderr-bytes")
	wantDigest := sha256.Sum256(wantOutput)
	for index, row := range rows {
		if row.Name != RequiredScenarios[index] || row.Status != StatusPass {
			t.Fatalf("row %d identity/status = %q/%q", index, row.Name, row.Status)
		}
		wantEvidence := filepath.Join(evidenceDir, fmt.Sprintf("%02d-%s.log", index+1, row.Name))
		if row.Evidence != wantEvidence {
			t.Fatalf("row %d evidence = %q, want %q", index, row.Evidence, wantEvidence)
		}
		gotOutput, readErr := os.ReadFile(row.Evidence)
		if readErr != nil {
			t.Fatal(readErr)
		}
		if string(gotOutput) != string(wantOutput) {
			t.Fatalf("row %d output = %q, want %q", index, gotOutput, wantOutput)
		}
		if row.EvidenceSHA256 != hex.EncodeToString(wantDigest[:]) || row.EvidenceBytes != int64(len(wantOutput)) {
			t.Fatalf("row %d evidence identity = %q/%d", index, row.EvidenceSHA256, row.EvidenceBytes)
		}
		if row.Receipt == nil || row.Receipt.Engine != "fak-native" || row.Receipt.Fallback != "none" || row.Receipt.Artifact != spec.Artifact || row.Receipt.RuntimeRevision != spec.RuntimeRevision {
			t.Fatalf("row %d receipt = %#v", index, row.Receipt)
		}
	}
	matrix := validMatrix()
	matrix.Envelopes[0].RuntimeRevision = spec.RuntimeRevision
	matrix.Envelopes[0].Scenarios = rows
	if err := Validate(matrix); err != nil {
		t.Fatalf("captured scenario rows do not satisfy the matrix validator: %v", err)
	}
}

func TestCaptureCommandFailureIsTypedFailWithEvidence(t *testing.T) {
	t.Setenv(captureHelperEnvironment, "1")
	spec := completeCaptureSpec(t, "success", "ok", "")
	failingName := RequiredScenarios[4]
	setCaptureCommand(&spec, failingName, helperCommand("failure", "failure-evidence"), "2s")
	rows, err := Capture(context.Background(), spec, CaptureOptions{EvidenceDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	row := captureRow(t, rows, failingName)
	if row.Status != StatusFail || row.Reason != ReasonCommandExitNonzero {
		t.Fatalf("failed row status/reason = %q/%q", row.Status, row.Reason)
	}
	got, err := os.ReadFile(row.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "failure-evidence" || row.EvidenceBytes != int64(len(got)) || row.EvidenceSHA256 == "" {
		t.Fatalf("failed row evidence = %q, bytes=%d, hash=%q", got, row.EvidenceBytes, row.EvidenceSHA256)
	}
	if last := rows[len(rows)-1]; last.Status != StatusPass {
		t.Fatalf("capture stopped after a command failure; final row = %#v", last)
	}
}

func TestCaptureTimeoutIsTypedFailWithEvidence(t *testing.T) {
	t.Setenv(captureHelperEnvironment, "1")
	spec := completeCaptureSpec(t, "success", "ok", "")
	timedOutName := RequiredScenarios[2]
	// Starting a fresh copy of the Go test binary can take hundreds of
	// milliseconds on a loaded or antivirus-scanned host. Leave ample startup
	// time so this witness proves output was captured before the running command
	// timed out, rather than racing process creation itself.
	setCaptureCommand(&spec, timedOutName, helperCommand("timeout", "before-timeout"), "1s")
	started := time.Now()
	rows, err := Capture(context.Background(), spec, CaptureOptions{EvidenceDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("per-scenario timeout took %v", elapsed)
	}
	row := captureRow(t, rows, timedOutName)
	if row.Status != StatusFail || row.Reason != ReasonCommandTimeout {
		t.Fatalf("timed-out row status/reason = %q/%q", row.Status, row.Reason)
	}
	got, err := os.ReadFile(row.Evidence)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "before-timeout" || row.EvidenceBytes != int64(len(got)) || row.EvidenceSHA256 == "" {
		t.Fatalf("timed-out row evidence = %q, bytes=%d, hash=%q", got, row.EvidenceBytes, row.EvidenceSHA256)
	}
}

func TestCaptureCannotSpoofFallback(t *testing.T) {
	raw := validCaptureJSON(t)
	var document map[string]any
	if err := json.Unmarshal(raw, &document); err != nil {
		t.Fatal(err)
	}
	document["fallback"] = "llama.cpp"
	spoofed, err := json.Marshal(document)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ParseCaptureSpec(spoofed); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("ParseCaptureSpec() spoof error = %v", err)
	}

	t.Setenv(captureHelperEnvironment, "1")
	spec := completeCaptureSpec(t, "success", "ok", "")
	rows, err := Capture(context.Background(), spec, CaptureOptions{EvidenceDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if row.Receipt == nil || row.Receipt.Engine != "fak-native" || row.Receipt.Fallback != "none" {
			t.Fatalf("capture minted spoofable receipt: %#v", row.Receipt)
		}
	}
}

func completeCaptureSpec(t *testing.T, behavior, stdout, stderr string) CaptureSpec {
	t.Helper()
	spec := CaptureSpec{Schema: CaptureSchema, RuntimeRevision: "runtime-r1", Artifact: "artifact@sha256:abc"}
	for _, name := range RequiredScenarios {
		spec.Scenarios = append(spec.Scenarios, CaptureScenarioSpec{
			Name: name, Command: helperCommand(behavior, stdout, stderr), Timeout: "2s",
		})
	}
	return spec
}

func helperCommand(args ...string) []string {
	return append([]string{os.Args[0], "-test.run=TestCaptureHelperProcess", "--"}, args...)
}

func cloneCaptureSpec(spec CaptureSpec) CaptureSpec {
	clone := spec
	clone.Scenarios = append([]CaptureScenarioSpec(nil), spec.Scenarios...)
	for index := range clone.Scenarios {
		clone.Scenarios[index].Command = append([]string(nil), clone.Scenarios[index].Command...)
	}
	return clone
}

func setCaptureCommand(spec *CaptureSpec, name string, command []string, timeout string) {
	for index := range spec.Scenarios {
		if spec.Scenarios[index].Name == name {
			spec.Scenarios[index].Command = command
			spec.Scenarios[index].Timeout = timeout
			return
		}
	}
}

func captureRow(t *testing.T, rows []Scenario, name string) Scenario {
	t.Helper()
	for _, row := range rows {
		if row.Name == name {
			return row
		}
	}
	t.Fatalf("capture has no row %q", name)
	return Scenario{}
}

func validCaptureJSON(t *testing.T) []byte {
	t.Helper()
	scenarios := make(map[string]any, len(RequiredScenarios))
	for _, name := range RequiredScenarios {
		scenarios[name] = map[string]any{"command": []string{"command"}, "timeout": "1s"}
	}
	b, err := json.Marshal(map[string]any{
		"schema": CaptureSchema, "runtime_revision": "runtime-r1", "artifact": "artifact@sha256:abc", "scenarios": scenarios,
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}
