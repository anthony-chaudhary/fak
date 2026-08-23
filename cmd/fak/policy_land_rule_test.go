package main

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestPolicyLandRuleDryRunAndRefusals(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "policy.json")
	c := filepath.Join(dir, "candidate.json")
	base := `{"version":"fak-policy/v1","allow":["Bash"]}`
	os.WriteFile(p, []byte(base), 0600)
	os.WriteFile(c, []byte(`{"version":"fak-policy/v1","arg_rules":[{"tool":"Bash","arg":"command","deny_regex":"sponge","reason":"POLICY_BLOCK"}]}`), 0600)
	var out bytes.Buffer
	if err := runPolicyLandRule(p, c, "", false, false, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"deny_regex": "sponge"`) {
		t.Fatalf("dry run missing rule: %s", out.String())
	}
	got, _ := os.ReadFile(p)
	if string(got) != base {
		t.Fatal("dry run mutated policy")
	}
	os.WriteFile(c, []byte(`{"arg_rules":[{"tool":"Bash","arg":"command","deny_regex":"x"}],"self_modify":true}`), 0600)
	if err := runPolicyLandRule(p, c, "", false, false, &out); err == nil || !strings.Contains(err.Error(), "SELF_MODIFY") {
		t.Fatalf("self modify err=%v", err)
	}
	os.WriteFile(c, []byte(`{"arg_rules":[],"allow":["Bash"]}`), 0600)
	if err := runPolicyLandRule(p, c, "", false, false, &out); err == nil || !strings.Contains(err.Error(), "only") {
		t.Fatalf("non-ArgRules err=%v", err)
	}
}

func TestPolicyLandRuleRollbackRoundTrip(t *testing.T) {
	reloads := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { reloads++; w.WriteHeader(204) }))
	defer srv.Close()
	dir := t.TempDir()
	p := filepath.Join(dir, "policy.json")
	c := filepath.Join(dir, "candidate.json")
	base := []byte(`{"version":"fak-policy/v1","allow":["Bash"]}`)
	os.WriteFile(p, base, 0600)
	os.WriteFile(c, []byte(`{"arg_rules":[{"tool":"Bash","arg":"command","deny_regex":"sponge","reason":"POLICY_BLOCK"}]}`), 0600)
	var out bytes.Buffer
	if err := runPolicyLandRule(p, c, srv.URL, true, false, &out); err != nil {
		t.Fatal(err)
	}
	landed, _ := os.ReadFile(p)
	if bytes.Equal(landed, base) {
		t.Fatal("land did not change policy")
	}
	if err := runPolicyLandRule(p, "", srv.URL, false, true, &out); err != nil {
		t.Fatal(err)
	}
	restored, _ := os.ReadFile(p)
	if !bytes.Equal(restored, base) {
		t.Fatalf("rollback mismatch: %s", restored)
	}
	if reloads != 2 {
		t.Fatalf("reloads=%d want 2", reloads)
	}
}

func TestPolicyLandRuleReloadTimeoutRestoresExactPreimage(t *testing.T) {
	accepted := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		close(accepted)
		<-r.Context().Done()
	}))
	defer srv.Close()

	previousClient := landRuleHTTPClient
	landRuleHTTPClient = &http.Client{Timeout: 50 * time.Millisecond}
	t.Cleanup(func() { landRuleHTTPClient = previousClient })

	dir := t.TempDir()
	policyPath := filepath.Join(dir, "policy.json")
	candidatePath := filepath.Join(dir, "candidate.json")
	preimage := []byte("{\n  \"version\": \"fak-policy/v1\",\n  \"allow\": [\"Bash\"]\n}\n")
	if err := os.WriteFile(policyPath, preimage, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(candidatePath, []byte(`{"arg_rules":[{"tool":"Bash","arg":"command","deny_regex":"sponge","reason":"POLICY_BLOCK"}]}`), 0600); err != nil {
		t.Fatal(err)
	}

	started := time.Now()
	err := runPolicyLandRule(policyPath, candidatePath, srv.URL, true, false, io.Discard)
	if elapsed := time.Since(started); elapsed >= time.Second {
		t.Fatalf("land returned after %s, want a sub-second timeout", elapsed)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("land error = %v, want context deadline exceeded", err)
	}
	lowerErr := strings.ToLower(err.Error())
	if !strings.Contains(lowerErr, "reload policy") || !strings.Contains(lowerErr, "timeout") {
		t.Fatalf("land error = %q, want reload operation and timeout", err)
	}
	select {
	case <-accepted:
	default:
		t.Fatal("reload server never accepted the request")
	}
	restored, readErr := os.ReadFile(policyPath)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if !bytes.Equal(restored, preimage) {
		t.Fatalf("rollback mismatch:\n got %q\nwant %q", restored, preimage)
	}
}
