package portability

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPublicEgressAdversarialCorpus(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("token=ghp_1234567890abcdef"))
	payload := json.RawMessage(`{
      "title":{"sensitivity":"public","value":"portable workflow"},
      "org":{"sensitivity":"organization","value":"roadmap"},
      "token":"ghp_1234567890abcdef",
      "private_url":"ssh://git@example.invalid/private/repo.git",
      "hostname":"builder.internal",
      "path":"C:\\Users\\operator\\private.txt",
      "history":{"messages":["Call operator@example.invalid"]},
      "encoded":"` + encoded + `",
      "credential_ref":"env:PORTABILITY_TOKEN",
      "credential":{"sensitivity":"credential-reference","value":"actual-secret"},
      "nested":{"dependencies":[{"url":"https://example.invalid/public"},{"password":"nested-secret"}]},
      "unknown":"untyped"
    }`)
	p, err := PreviewEgress(ChannelPublic, payload)
	if err != nil {
		t.Fatal(err)
	}
	if p.Allowed {
		t.Fatal("public plan containing forbidden material must deny")
	}
	if !bytes.Contains(p.Payload, []byte("portable workflow")) {
		t.Fatal("allowed public material did not survive")
	}
	for _, forbidden := range []string{"ghp_1234567890abcdef", "actual-secret", "nested-secret", encoded, "operator@example.invalid", "builder.internal", "private/repo.git", "Users"} {
		if bytes.Contains(p.Payload, []byte(forbidden)) {
			t.Fatalf("payload leaked forbidden fixture %q", forbidden)
		}
		b, _ := json.Marshal(p.Decisions)
		if bytes.Contains(b, []byte(forbidden)) {
			t.Fatalf("explanation leaked fixture %q", forbidden)
		}
	}
	got := map[string]EgressAction{}
	for _, d := range p.Decisions {
		got[d.Path] = d.Action
	}
	checks := map[string]EgressAction{"$/title": ActionInclude, "$/org": ActionRedact, "$/token": ActionDeny, "$/private_url": ActionRedact, "$/hostname": ActionRedact, "$/path": ActionRedact, "$/history/messages/0": ActionRedact, "$/encoded": ActionDeny, "$/credential_ref": ActionReference, "$/credential": ActionDeny, "$/nested/dependencies/1/password": ActionDeny, "$/unknown": ActionDeny}
	for path, want := range checks {
		if got[path] != want {
			t.Errorf("%s action=%s want %s", path, got[path], want)
		}
	}
}

func TestOrganizationUnknownDefaultsDenyAndAdapterExtends(t *testing.T) {
	adapter := SensitivityAdapterFunc(func(path string, v any) (Sensitivity, string, bool) {
		if strings.HasSuffix(path, "/team_note") {
			return SensitivityOrganization, "team-field", true
		}
		return "", "", false
	})
	p, err := PreviewEgress(ChannelOrganization, json.RawMessage(`{"public":{"sensitivity":"public","value":"ok"},"team_note":"inside","novel":42}`), adapter)
	if err != nil {
		t.Fatal(err)
	}
	if p.Allowed {
		t.Fatal("organization unknown sensitivity must deny")
	}
	actions := map[string]EgressAction{}
	for _, d := range p.Decisions {
		actions[d.Path] = d.Action
	}
	if actions["$/team_note"] != ActionInclude || actions["$/novel"] != ActionDeny {
		t.Fatalf("unexpected actions: %#v", actions)
	}
}

func TestRedactionDeterministicAndCredentialReferencesOnly(t *testing.T) {
	in := json.RawMessage(`{"home":"/srv/private/file","credential_ref":"vault:team/service","password_ref":"literal-value"}`)
	a, _ := PreviewEgress(ChannelPublic, in)
	b, _ := PreviewEgress(ChannelPublic, in)
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	if !bytes.Equal(ja, jb) {
		t.Fatal("egress plan is not deterministic")
	}
	if !bytes.Contains(a.Payload, []byte("vault:team/service")) {
		t.Fatal("credential reference was not preserved")
	}
	if bytes.Contains(a.Payload, []byte("literal-value")) {
		t.Fatal("credential value escaped as a reference")
	}
}

func TestExportEgressDenialPrecedesIdentityAndWrite(t *testing.T) {
	home := t.TempDir()
	dir := filepath.Join(home, "managed", "skills")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bad.json"), []byte(`{"name":{"sensitivity":"public","value":"safe"},"auth":{"sensitivity":"forbidden","value":"sensitive fixture"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(home, "exports", "bad.json")
	pkg, _, previews, err := New(home).ExportEgress(out, nil, ChannelPublic, true)
	if err == nil {
		t.Fatal("unsafe export succeeded")
	}
	if pkg.ID != "" || pkg.Digest != "" {
		t.Fatalf("denied bytes received identity: %#v", pkg)
	}
	if len(previews) != 1 || previews[0].Allowed {
		t.Fatalf("missing denied preview: %#v", previews)
	}
	if _, statErr := os.Stat(out); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("denied export wrote source-boundary bytes: %v", statErr)
	}
}
