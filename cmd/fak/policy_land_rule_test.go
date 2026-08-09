package main

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
