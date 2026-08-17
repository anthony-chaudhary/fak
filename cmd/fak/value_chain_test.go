package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestValueChainSupportSpine(t *testing.T) {
	root := findRepoRoot("../..")
	var out, errOut bytes.Buffer
	code := runValueChain(&out, &errOut, []string{"audit", "--manifest", filepath.Join(root, "examples", "value-chain", "support-manifest.json"), "--observations", filepath.Join(root, "examples", "value-chain", "support-observations.json")})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"arm=baseline", "$/ticket_resolved=2.000000", "arm=shared", "sessions=2", "$/ticket_resolved=0.600000", "stage=gpu kind=hardware status=ABSENT", "design=paired"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("missing %q in\n%s", want, out.String())
		}
	}
}
func TestValueChainAgenticPacketAdapter(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "manifest.json")
	observations := filepath.Join(dir, "observations.json")
	packet := filepath.Join(dir, "packet.json")
	mustWrite := func(path, body string) {
		t.Helper()
		if err := os.WriteFile(path, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	mustWrite(manifest, `{"schema":"fak-value-chain/1","name":"latest-harness","stages":[{"id":"benchmark","kind":"agentic-benchmark"}],"arms":[{"id":"raw","default":true},{"id":"fak"}],"outcomes":[{"id":"safe_success","unit":"case"}]}`)
	mustWrite(observations, `{"schema":"fak-value-chain/1","observations":[]}`)
	mustWrite(packet, `{"schema":"fak.agentic-benchmark-result-packet.v1","status":"PASS_RESULT","result_claim_allowed":true,"value_chain":[{"role":"raw","trace_id":"r","pair_id":"case-1","turns":5,"outcomes":{"safe_success":1},"provenance":"official-grader"},{"role":"fak","trace_id":"f","pair_id":"case-1","turns":3,"cost_usd":0.3,"outcomes":{"safe_success":1},"provenance":"official-grader+bill"}]}`)
	var out, errOut bytes.Buffer
	if code := runValueChain(&out, &errOut, []string{"audit", "--manifest", manifest, "--observations", observations, "--agentic-packet", packet}); code != 0 {
		t.Fatalf("code=%d err=%s", code, errOut.String())
	}
	for _, want := range []string{"arm=raw", "$/turn=UNKNOWN", "arm=fak", "$/safe_success=0.300000", "design=paired"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("missing %q in %s", want, out.String())
		}
	}
}
