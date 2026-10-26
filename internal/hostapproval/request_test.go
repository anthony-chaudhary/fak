package hostapproval

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

type fixture struct {
	opts                       Options
	installer, policy, context string
	codex, adapter, sentinel   string
	adapterBytes               []byte
}

// fak-test:runtime fast est=3s lane=short
func TestPrepareVerifyNonAuthorizingRequest(t *testing.T) {
	f := newFixture(t)
	r, err := Prepare(f.opts)
	if err != nil {
		t.Fatal(err)
	}
	if r.Authorization != "none" || r.BrokerStatus != "required" {
		t.Fatalf("request authority shape = authorization %q broker %q", r.Authorization, r.BrokerStatus)
	}
	nonce, err := hex.DecodeString(r.Nonce)
	if err != nil || len(nonce) != sha256.Size || hex.EncodeToString(nonce) != r.Nonce {
		t.Fatalf("request nonce is not canonical lowercase hex for a 256-bit nonce: bytes=%d error=%v", len(nonce), err)
	}
	if err := Verify(r, f.opts.SessionID); err != nil {
		t.Fatalf("Verify: %v", err)
	}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := Decode(raw)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if d1, err := Digest(r); err != nil {
		t.Fatal(err)
	} else if d2, err := Digest(decoded); err != nil || d1 != d2 {
		t.Fatalf("digest mismatch %q %q %v", d1, d2, err)
	}
	if got := r.Plan[0].Args; strings.Join(got, " ") != "codex mcp install --source "+r.Installer.Path+" --policy "+r.Policy.Path+" --config "+r.CodexConfig.Path+" --server fak-test" {
		t.Fatalf("installer plan = %q", got)
	}
	if got := r.Plan[1].Args; strings.Join(got, " ") != "guard allow --remote-maintenance "+r.Context.Path+" --ttl 120s" {
		t.Fatalf("maintenance plan = %q", got)
	}
	second, err := Prepare(f.opts)
	if err != nil {
		t.Fatal(err)
	}
	if second.Nonce == r.Nonce {
		t.Fatal("request nonce was reused")
	}
	assertSentinel(t, f)

	mutations := map[string]func(*Request){
		"session":           func(x *Request) { x.SessionID = "other-session" },
		"context identity":  func(x *Request) { x.RemoteConfig.Profiles[0].ProfileID = "other-profile" },
		"target":            func(x *Request) { x.RemoteConfig.Profiles[0].Target = "other-target" },
		"host key":          func(x *Request) { x.RemoteConfig.Profiles[0].HostKey = hostKey(9) },
		"operation":         func(x *Request) { x.RemoteConfig.Profiles[0].Operation = "other-operation" },
		"scope":             func(x *Request) { x.RemoteConfig.Profiles[0].Paths = []string{"/etc/other"} },
		"installer binding": func(x *Request) { x.Installer.SHA256 = digestByte(8) },
		"policy binding":    func(x *Request) { x.Policy.SHA256 = digestByte(8) },
		"config binding":    func(x *Request) { x.CodexConfig.SHA256 = digestByte(8) },
		"adapter binding":   func(x *Request) { x.Adapters[0].SHA256 = digestByte(8) },
		"expiry":            func(x *Request) { x.ExpiresAt = x.PreparedAt.Add(-time.Second) },
		"authorization":     func(x *Request) { x.Authorization = "approved" },
		"plan":              func(x *Request) { x.Plan[0].Args = append(x.Plan[0].Args, "--skip-verify") },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			x := cloneRequest(t, r)
			mutate(&x)
			if err := Verify(x, f.opts.SessionID); err == nil {
				t.Fatal("mutated request verified")
			}
			assertSentinel(t, f)
		})
	}

	t.Run("wrong verifier session", func(t *testing.T) {
		if err := Verify(r, "wrong-session"); err == nil {
			t.Fatal("wrong verifier session accepted")
		}
	})
	t.Run("unknown json field", func(t *testing.T) {
		bad := append(raw[:len(raw)-1], []byte(`,"approved":true}`)...)
		if _, err := Decode(bad); err == nil {
			t.Fatal("unknown field accepted")
		}
	})
	t.Run("trailing json", func(t *testing.T) {
		if _, err := Decode(append(raw, []byte(` {}`)...)); err == nil {
			t.Fatal("trailing value accepted")
		}
	})

	fileMutations := []struct {
		name, path string
		body       []byte
	}{
		{"installer file", f.installer, []byte("changed")},
		{"policy file", f.policy, []byte(`{"version":"bad"}`)},
		{"context file", f.context, []byte(`{}`)},
		{"codex config file", f.codex, []byte("changed")},
		{"adapter executable", f.adapter, []byte("changed")},
	}
	for _, tc := range fileMutations {
		t.Run(tc.name, func(t *testing.T) {
			original, err := os.ReadFile(tc.path)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(tc.path, tc.body, 0o600); err != nil {
				t.Fatal(err)
			}
			if err := Verify(r, f.opts.SessionID); err == nil {
				t.Fatal("changed file verified")
			}
			if err := os.WriteFile(tc.path, original, 0o700); err != nil {
				t.Fatal(err)
			}
			assertSentinel(t, f)
		})
	}
	if err := Verify(r, f.opts.SessionID); err != nil {
		t.Fatalf("Verify after restored mutations: %v", err)
	}
	assertSentinel(t, f)
}

func newFixture(t *testing.T) fixture {
	t.Helper()
	dir := t.TempDir()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	selfBytes, err := os.ReadFile(self)
	if err != nil {
		t.Fatal(err)
	}
	installer := filepath.Join(dir, "fak-test"+filepath.Ext(self))
	adapter := filepath.Join(dir, "adapter-test"+filepath.Ext(self))
	if err := os.WriteFile(installer, selfBytes, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(adapter, selfBytes, 0o700); err != nil {
		t.Fatal(err)
	}
	policyPath := filepath.Join(dir, "policy.json")
	if err := os.WriteFile(policyPath, []byte(`{"version":"fak-policy/v1","posture":"default_open"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	adapterSum := sha256.Sum256(selfBytes)
	now := time.Now().UTC()
	cfg := remotemaint.Config{Schema: remotemaint.ConfigSchema, Profiles: []remotemaint.Profile{{
		ProfileID: "profile-a", Target: "node-a", HostKey: hostKey(1),
		Operation: remotemaint.OperationSystemdProfileInstallReload, PayloadSHA256: digestByte(2), Holder: "holder-a",
		Paths: []string{"/etc/fak/profile.conf"}, PathPrefixes: []remotemaint.PathPrefix{{Directory: "/var/tmp", FilenamePrefix: "maintenance-stage."}},
		Units: []string{"fak.service"}, LeaseNodes: []string{"node-a"}, ExpiresAt: now.Add(10 * time.Minute),
		Authority: remotemaint.Command{Executable: adapter, SHA256: hex.EncodeToString(adapterSum[:]), Args: []string{"--native-adapter"}},
	}}}
	contextPath := filepath.Join(dir, "remote-maintenance.json")
	raw, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(contextPath, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	codex := filepath.Join(dir, "config.toml")
	if err := os.WriteFile(codex, []byte("model = \"test\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	sentinel := filepath.Join(dir, "must-not-change")
	if err := os.WriteFile(sentinel, []byte("untouched"), 0o600); err != nil {
		t.Fatal(err)
	}
	return fixture{
		opts:      Options{Installer: installer, Policy: policyPath, Context: contextPath, CodexConfig: codex, Server: "fak-test", TaskID: "task-14030", SessionID: "session-14030", TTL: 2 * time.Minute},
		installer: installer, policy: policyPath, context: contextPath, codex: codex, adapter: adapter, sentinel: sentinel, adapterBytes: selfBytes,
	}
}

func cloneRequest(t *testing.T, in Request) Request {
	t.Helper()
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out Request
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	return out
}

func assertSentinel(t *testing.T, f fixture) {
	t.Helper()
	raw, err := os.ReadFile(f.sentinel)
	if err != nil || string(raw) != "untouched" {
		t.Fatalf("sentinel mutated: %q %v", raw, err)
	}
}

func digestByte(b byte) string { return strings.Repeat(hex.EncodeToString([]byte{b}), sha256.Size) }
func hostKey(b byte) string {
	raw := make([]byte, sha256.Size)
	for i := range raw {
		raw[i] = b
	}
	return "SHA256:" + base64.RawStdEncoding.EncodeToString(raw)
}
