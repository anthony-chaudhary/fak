package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVCacheFixturePutInspectCapturedProof(t *testing.T) {
	dir := t.TempDir()
	const payload = `{"answer":"fixture-secret"}`

	stdout, stderr, code := runVCacheForTest("put", "--dir", dir, "--payload", payload)
	if code != 2 || stdout != "" || stderr != "fak vcache put: refused: --fixture-mode must be offline or test\n" {
		t.Fatalf("ungated put = code %d stdout=%q stderr=%q", code, stdout, stderr)
	}

	stdout, stderr, code = runVCacheForTest(
		"put", "--dir", dir, "--payload", payload, "--fixture-mode", "test",
		"--tool", "fixture.lookup", "--args", `{"id":"alpha"}`, "--witness", "fixture-etag-1", "--json",
	)
	if code != 0 || stderr != "" {
		t.Fatalf("gated put = code %d stdout=%q stderr=%q", code, stdout, stderr)
	}
	var put vcacheFixturePutReport
	if err := json.Unmarshal([]byte(stdout), &put); err != nil {
		t.Fatalf("decode put: %v\n%s", err, stdout)
	}
	if !put.Stored || put.Schema != vcacheFixturePutSchema || len(put.Digest) != 64 {
		t.Fatalf("put report = %+v", put)
	}
	if put.Metadata.Tool != "fixture.lookup" || put.Metadata.Producer != "vdso" || put.Metadata.Plane != "tool_result" || put.Metadata.Eligibility != "read_only+idempotent" {
		t.Fatalf("put metadata did not come from eligible vDSO fill: %+v", put.Metadata)
	}

	stdout, stderr, code = runVCacheForTest("inspect", "--dir", dir, "--digest", put.Digest, "--json")
	if code != 0 || stderr != "" {
		t.Fatalf("default inspect = code %d stdout=%q stderr=%q", code, stdout, stderr)
	}
	if strings.Contains(stdout, "fixture-secret") || strings.Contains(stdout, "payload_base64") || strings.Contains(stdout, "\"payload\"") {
		t.Fatalf("default inspect exposed payload: %s", stdout)
	}
	var inspected vcacheInspectReport
	if err := json.Unmarshal([]byte(stdout), &inspected); err != nil {
		t.Fatalf("decode inspect: %v\n%s", err, stdout)
	}
	if inspected.Digest != put.Digest || !inspected.PayloadHidden || inspected.Payload != nil || len(inspected.PayloadBase64) != 0 {
		t.Fatalf("default inspect = %+v", inspected)
	}
	if inspected.Metadata != put.Metadata {
		t.Fatalf("inspect metadata = %+v, want %+v", inspected.Metadata, put.Metadata)
	}

	stdout, stderr, code = runVCacheForTest("inspect", "--dir", dir, "--digest", put.Digest, "--show-payload", "--json")
	if code != 2 || stdout != "" || stderr != "fak vcache inspect: refused: --show-payload requires --fixture-mode test\n" {
		t.Fatalf("ungated payload inspect = code %d stdout=%q stderr=%q", code, stdout, stderr)
	}

	stdout, stderr, code = runVCacheForTest(
		"inspect", "--dir", dir, "--digest", "sha256:"+put.Digest,
		"--show-payload", "--fixture-mode", "test", "--json",
	)
	if code != 0 || stderr != "" {
		t.Fatalf("gated payload inspect = code %d stdout=%q stderr=%q", code, stdout, stderr)
	}
	if err := json.Unmarshal([]byte(stdout), &inspected); err != nil {
		t.Fatalf("decode payload inspect: %v\n%s", err, stdout)
	}
	if inspected.PayloadHidden || inspected.Payload == nil || *inspected.Payload != payload {
		t.Fatalf("gated payload inspect = %+v", inspected)
	}
}

func TestVCacheFixturePutAllowsExplicitOfflineMode(t *testing.T) {
	dir := t.TempDir()
	stdout, stderr, code := runVCacheForTest(
		"put", "--dir", dir, "--payload", "offline fixture", "--fixture-mode", "offline", "--json",
	)
	if code != 0 || stderr != "" {
		t.Fatalf("offline put = code %d stdout=%q stderr=%q", code, stdout, stderr)
	}
	var put vcacheFixturePutReport
	if err := json.Unmarshal([]byte(stdout), &put); err != nil || put.Metadata.FixtureMode != "offline" {
		t.Fatalf("offline put report = %+v err=%v", put, err)
	}
}

func TestVCacheInspectFailsClosedOnDigestPathAndTampering(t *testing.T) {
	dir := t.TempDir()
	_, stderr, code := runVCacheForTest("inspect", "--dir", dir, "--digest", "../../payload", "--json")
	if code != 2 || stderr != "fak vcache inspect: --digest must be 64 lowercase hexadecimal SHA-256 characters\n" {
		t.Fatalf("path-shaped digest = code %d stderr=%q", code, stderr)
	}
	_, stderr, code = runVCacheForTest("inspect", "--dir", ".", "--digest", strings.Repeat("0", 64), "--json")
	if code != 2 || stderr != "fak vcache inspect: --dir must be an absolute clean path\n" {
		t.Fatalf("relative dir = code %d stderr=%q", code, stderr)
	}

	stdout, stderr, code := runVCacheForTest(
		"put", "--dir", dir, "--payload", "untampered", "--fixture-mode", "test", "--json",
	)
	if code != 0 || stderr != "" {
		t.Fatalf("seed = code %d stdout=%q stderr=%q", code, stdout, stderr)
	}
	var put vcacheFixturePutReport
	if err := json.Unmarshal([]byte(stdout), &put); err != nil {
		t.Fatal(err)
	}
	ledger := filepath.Join(dir, vcacheFixtureEventsFile)
	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	var event vcacheFixtureEvent
	if err := json.Unmarshal(bytes.TrimSpace(raw), &event); err != nil {
		t.Fatal(err)
	}
	event.Payload = []byte("tampered")
	raw, err = json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(ledger, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	stdout, stderr, code = runVCacheForTest("inspect", "--dir", dir, "--digest", put.Digest, "--json")
	if code != 1 || stdout != "" || !strings.Contains(stderr, "refused fixture ledger: line 1: payload digest mismatch") {
		t.Fatalf("tampered inspect = code %d stdout=%q stderr=%q", code, stdout, stderr)
	}
}

func TestVCacheUsageIncludesGuardedInspection(t *testing.T) {
	var out bytes.Buffer
	vcacheUsage(&out)
	for _, want := range []string{
		"fak vcache inspect --dir DIR --digest SHA256",
		"--show-payload --fixture-mode test",
		"fak vcache put --dir DIR (--payload TEXT|--payload-file FILE)",
		"--fixture-mode offline|test",
	} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("usage missing %q", want)
		}
	}
}

func runVCacheForTest(argv ...string) (stdout, stderr string, code int) {
	var out, errout bytes.Buffer
	code = runVCache(&out, &errout, argv)
	return out.String(), errout.String(), code
}
