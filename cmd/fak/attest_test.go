package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// restorePolicy resets the global adjudicator + IFC policy an `applyPolicy` call
// mutates, so one attest run cannot leak its floor into a sibling test.
func restorePolicy(t *testing.T) {
	t.Cleanup(func() {
		adjudicator.Default.SetPolicy(adjudicator.DefaultPolicy())
		ifc.ConfigureDefaultPolicy(ifc.Policy{})
	})
}

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	return p
}

func TestDeriveProbes(t *testing.T) {
	manifest := `{
		"allow": ["read_record"],
		"allow_prefix": ["search_", "get_"],
		"deny": { "delete_all": "POLICY_BLOCK", "exfiltrate": "SECRET_EXFIL" }
	}`
	probes, err := deriveProbes([]byte(manifest))
	if err != nil {
		t.Fatalf("deriveProbes: %v", err)
	}
	// 2 deny (sorted) + 1 allow + 2 allow_prefix + 1 default_deny = 6.
	if len(probes) != 6 {
		t.Fatalf("probe count = %d, want 6: %+v", len(probes), probes)
	}
	// deny rules are sorted and carry their cited reason.
	if probes[0].Tool != "delete_all" || probes[0].Expect != "deny" || probes[0].ExpectReason != "POLICY_BLOCK" {
		t.Fatalf("first probe = %+v, want delete_all/deny/POLICY_BLOCK", probes[0])
	}
	if probes[1].Tool != "exfiltrate" || probes[1].ExpectReason != "SECRET_EXFIL" {
		t.Fatalf("second probe = %+v, want exfiltrate/deny/SECRET_EXFIL", probes[1])
	}
	// allow + synthesized allow_prefix probes must expect ALLOW.
	var last probe = probes[len(probes)-1]
	if last.Origin != "default_deny" || last.Expect != "deny" || last.ExpectReason != "DEFAULT_DENY" {
		t.Fatalf("default-deny probe = %+v, want origin=default_deny/deny/DEFAULT_DENY", last)
	}
	for _, p := range probes {
		if p.Origin == "allow" || p.Origin == "allow_prefix" {
			if p.Expect != "allow" {
				t.Fatalf("allow probe %q expect = %q, want allow", p.Tool, p.Expect)
			}
		}
	}
}

func TestValidateProbe(t *testing.T) {
	cases := []struct {
		name    string
		p       probe
		wantErr bool
	}{
		{"allow-ok", probe{Tool: "x", Expect: "allow"}, false},
		{"deny-with-reason", probe{Tool: "x", Expect: "deny", ExpectReason: "POLICY_BLOCK"}, false},
		{"deny-no-reason", probe{Tool: "x", Expect: "deny"}, false},
		{"bad-expect", probe{Tool: "x", Expect: "maybe"}, true},
		{"unknown-reason", probe{Tool: "x", Expect: "deny", ExpectReason: "MADE_UP"}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateProbe(tc.p)
			if tc.wantErr && err == nil {
				t.Fatalf("want error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("want nil, got %v", err)
			}
		})
	}
}

func TestRunAttestDerivedProvesFloor(t *testing.T) {
	restorePolicy(t)
	policy := writeTemp(t, "floor.json", `{
		"allow": ["read_record"],
		"allow_prefix": ["search_"],
		"deny": { "delete_all": "POLICY_BLOCK", "exfiltrate": "SECRET_EXFIL" }
	}`)
	var out, errb bytes.Buffer
	code := runAttest(&out, &errb, []string{"--policy", policy})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "5 probe(s): 5 passed, 0 failed") || !strings.Contains(s, "PROVEN") {
		t.Fatalf("expected 5/5 PROVEN, got:\n%s", s)
	}
	// A declared deny must appear with the verdict it actually returned.
	if !strings.Contains(s, "delete_all") || !strings.Contains(s, "POLICY_BLOCK") {
		t.Fatalf("missing delete_all deny row:\n%s", s)
	}
}

func TestRunAttestDriftFails(t *testing.T) {
	restorePolicy(t)
	policy := writeTemp(t, "floor.json", `{"allow": ["read_record"], "deny": {"delete_all": "POLICY_BLOCK"}}`)
	// A probe set that expects the floor to ALLOW a tool it must DENY: the floor
	// is honest, so the probe drifts and the attestation must FAIL (exit 1).
	probes := writeTemp(t, "probes.json", `[{"tool":"delete_all","args":"{}","expect":"allow"}]`)
	var out, errb bytes.Buffer
	code := runAttest(&out, &errb, []string{"--policy", policy, "--probes", probes})
	if code != 1 {
		t.Fatalf("exit=%d, want 1 (drift must fail); stderr=%s out=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "NOT proven") {
		t.Fatalf("expected NOT proven, got:\n%s", out.String())
	}
}

func TestRunAttestJSONShape(t *testing.T) {
	restorePolicy(t)
	policy := writeTemp(t, "floor.json", `{"allow": ["read_record"], "deny": {"delete_all": "POLICY_BLOCK"}}`)
	var out, errb bytes.Buffer
	code := runAttest(&out, &errb, []string{"--policy", policy, "--json", "--quiet"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var got attestation
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v\nraw=%s", err, out.String())
	}
	if got.Schema != "fak-attestation/v1" {
		t.Fatalf("schema = %q", got.Schema)
	}
	if got.Policy.SHA256 == "" || len(got.Policy.SHA256) != 64 {
		t.Fatalf("policy sha256 = %q, want 64 hex chars", got.Policy.SHA256)
	}
	if !got.Summary.Pass || got.Summary.Probes == 0 {
		t.Fatalf("summary = %+v, want pass over >=1 probe", got.Summary)
	}
	// The sha256 must match a fresh hash of the file bytes.
	raw, _ := os.ReadFile(policy)
	if want := sha256Hex(raw); got.Policy.SHA256 != want {
		t.Fatalf("sha256 = %s, want %s", got.Policy.SHA256, want)
	}
}

func TestRunAttestRequiresPolicy(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runAttest(&out, &errb, []string{}); code != 2 {
		t.Fatalf("no --policy exit=%d, want 2", code)
	}
}

func TestRunAttestProbesFileRejectsUnknownReason(t *testing.T) {
	restorePolicy(t)
	policy := writeTemp(t, "floor.json", `{"allow": ["read_record"]}`)
	probes := writeTemp(t, "probes.json", `[{"tool":"x","expect":"deny","expect_reason":"INVENTED"}]`)
	var out, errb bytes.Buffer
	if code := runAttest(&out, &errb, []string{"--policy", policy, "--probes", probes}); code != 2 {
		t.Fatalf("unknown reason exit=%d, want 2 (fail-loud); stderr=%s", code, errb.String())
	}
}
