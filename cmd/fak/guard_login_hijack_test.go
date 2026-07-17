package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

func TestGuardLoginHijackWatchWarnsAtCredentialRewrite(t *testing.T) {
	dir := t.TempDir()
	cred := filepath.Join(dir, ".credentials.json")
	if err := os.WriteFile(cred, []byte(`{"claudeAiOauth":{"accessToken":"old"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	regPath := filepath.Join(t.TempDir(), "registry.json")
	reg := accounts.Registry{Homes: []accounts.Home{{Name: "july4-netra", Dir: dir, Identity: accounts.Identity{Email: "july4@example.test", AccountUUID: "uuid-4"}}}}
	if err := accounts.SaveRegistry(regPath, reg); err != nil {
		t.Fatal(err)
	}
	probe := func(token string) (accounts.ProbedIdentity, error) {
		return accounts.ProbedIdentity{Email: "july17@example.test", AccountUUID: "uuid-17"}, nil
	}
	var warn bytes.Buffer
	stop := guardWatchLoginHijack(dir, regPath, probe, &warn, 5*time.Millisecond)
	if err := os.WriteFile(cred, []byte(`{"claudeAiOauth":{"accessToken":"new-token-longer"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(filepath.Join(dir, ".fak-login-hijack.jsonl")); err == nil {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	stop()
	got := warn.String()
	for _, want := range []string{"july4-netra", "july4@example.test", "uuid-4", "july17@example.test", "uuid-17", "fak accounts enroll-current --name"} {
		if !strings.Contains(got, want) {
			t.Fatalf("warning missing %q: %q", want, got)
		}
	}
}

func TestGuardLoginHijackWatchStaysQuietOnSameIdentity(t *testing.T) {
	dir := t.TempDir()
	cred := filepath.Join(dir, ".credentials.json")
	if err := os.WriteFile(cred, []byte(`{"claudeAiOauth":{"accessToken":"old"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	regPath := filepath.Join(t.TempDir(), "registry.json")
	identity := accounts.Identity{Email: "july4@example.test", AccountUUID: "uuid-4"}
	if err := accounts.SaveRegistry(regPath, accounts.Registry{Homes: []accounts.Home{{Name: "july4-netra", Dir: dir, Identity: identity}}}); err != nil {
		t.Fatal(err)
	}
	probe := func(string) (accounts.ProbedIdentity, error) {
		return accounts.ProbedIdentity{Email: identity.Email, AccountUUID: identity.AccountUUID}, nil
	}
	var warn bytes.Buffer
	stop := guardWatchLoginHijack(dir, regPath, probe, &warn, 5*time.Millisecond)
	if err := os.WriteFile(cred, []byte(`{"claudeAiOauth":{"accessToken":"same-identity-new-token"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	time.Sleep(40 * time.Millisecond)
	stop()
	if got := warn.String(); got != "" {
		t.Fatalf("same-identity re-login warned: %q", got)
	}
}
