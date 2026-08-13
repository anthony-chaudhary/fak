package portability

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeObj(t *testing.T, home, kind, name, body string) {
	t.Helper()
	p := filepath.Join(home, "managed", plural(kind), name+".json")
	if err := os.MkdirAll(filepath.Dir(p), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0600); err != nil {
		t.Fatal(err)
	}
}
func fixed(home string) Store {
	s := New(home)
	s.Now = func() time.Time { return time.Date(2026, 8, 13, 12, 0, 0, 0, time.UTC) }
	return s
}
func TestPersonalContinuitySpine(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	writeObj(t, a, "skill", "review", `{"behavior":"review-concisely"}`)
	writeObj(t, a, "workflow", "triage", `{"behavior":"triage-before-fix"}`)
	writeObj(t, a, "policy", "safe", `{"behavior":"deny-destructive"}`)
	writeObj(t, a, "adapter", "future", `{"behavior":"not-active"}`)
	sa, sb := fixed(a), fixed(b)
	pkgPath := filepath.Join(t.TempDir(), "p.json")
	p, er, err := sa.Export(pkgPath, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Objects) != 4 || er.Status != "committed" {
		t.Fatalf("package=%+v receipt=%+v", p, er)
	}
	if p.Objects[0].Active {
		t.Fatal("unknown object unexpectedly active")
	}
	ar, err := sb.Apply(pkgPath, true, 0)
	if err != nil {
		t.Fatal(err)
	}
	sr, err := sb.Switch(p.ID, true)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sb.Readback()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got["skill:review"] != "review-concisely" || got["workflow:triage"] != "triage-before-fix" || got["policy:safe"] != "deny-destructive" {
		t.Fatalf("behavior=%v", got)
	}
	rr, err := sb.Rollback(sr.ID, true)
	if err != nil {
		t.Fatal(err)
	}
	active, _ := sb.Active()
	if active != "" {
		t.Fatalf("rollback active=%q receipt=%+v apply=%+v", active, rr, ar)
	}
}
func TestDryRunIdempotenceInterruptionAndInvalidSafety(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	for _, k := range []string{"skill", "workflow", "policy"} {
		writeObj(t, a, k, "one", `{"behavior":"ok"}`)
	}
	sa, sb := fixed(a), fixed(b)
	path := filepath.Join(t.TempDir(), "p.json")
	p, _, err := sa.Export(path, nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("dry-run export mutated")
	}
	p, _, err = sa.Export(path, nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = sb.Apply(path, true, 1); err == nil {
		t.Fatal("expected interruption")
	}
	if active, _ := sb.Active(); active != "" {
		t.Fatal("interruption changed active")
	}
	if _, err = sb.Apply(path, true, 0); err != nil {
		t.Fatal(err)
	}
	r2, err := sb.Apply(path, true, 0)
	if err != nil || r2.Detail != "already applied" {
		t.Fatalf("idempotence: %+v %v", r2, err)
	}
	sr, err := sb.Switch(p.ID, true)
	if err != nil {
		t.Fatal(err)
	}
	bad := filepath.Join(t.TempDir(), "bad.json")
	os.WriteFile(bad, []byte(`{"schema":"fak.portability/v1","digest":"wrong"}`), 0600)
	if _, err = sb.Apply(bad, true, 0); err == nil {
		t.Fatal("invalid package accepted")
	}
	if active, _ := sb.Active(); active != p.ID {
		t.Fatalf("invalid apply changed %q", active)
	}
	if _, err = sb.Switch(p.ID, false); err != nil {
		t.Fatal(err)
	}
	if active, _ := sb.Active(); active != p.ID {
		t.Fatal("dry-run switch mutated")
	}
	_ = sr
}
func TestSecretPrivateAndHistoryFixturesFailClosed(t *testing.T) {
	cases := map[string]string{"credential": `{"api_token":"ghp_abcdefgh"}`, "private-host": `{"endpoint":"db.corp.internal"}`, "absolute-path": `{"path":"C:\\Users\\alice\\secret"}`, "history": `{"conversation_history":["undeclared"]}`}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			home := t.TempDir()
			writeObj(t, home, "skill", name, body)
			_, _, err := fixed(home).Export(filepath.Join(t.TempDir(), "p"), nil, true)
			if err == nil {
				t.Fatal("unsafe fixture exported")
			}
		})
	}
}
func TestTamperedPackageSecretFailsClosed(t *testing.T) {
	p := Package{Schema: Schema, ID: "x", Objects: []Object{{ID: "skill:x", Kind: "skill", Name: "x", Payload: json.RawMessage(`{"token":"sk-abcdefgh"}`)}}}
	sum := shaForTest(p.Objects[0].Payload)
	p.Objects[0].Digest = sum
	p.Digest = packageDigest(p)
	b, _ := json.Marshal(p)
	path := filepath.Join(t.TempDir(), "p")
	os.WriteFile(path, b, 0600)
	_, err := ReadPackage(path)
	if err == nil || !strings.Contains(err.Error(), "unsafe") {
		t.Fatalf("err=%v", err)
	}
}
func shaForTest(b []byte) string {
	var v any
	json.Unmarshal(b, &v)
	canon, _ := json.Marshal(v)
	h := sha256.Sum256(canon)
	return "sha256:" + hex.EncodeToString(h[:])
}

func plural(s string) string {
	if s == "policy" {
		return "policies"
	}
	return s + "s"
}
