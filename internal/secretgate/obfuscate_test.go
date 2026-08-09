package secretgate

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

const fixtureSecret = "sk-proj-AbCdEf0123456789XYZ_secret"

func TestObfuscationRoundTripDeepWalkAndExecutionBoundary(t *testing.T) {
	keyPath := filepath.Join(t.TempDir(), "install", "secretgate.key")
	o, err := loadObfuscator(keyPath, bytes.NewReader(bytes.Repeat([]byte{0x42}, keyBytes)))
	if err != nil {
		t.Fatal(err)
	}
	outbound := o.Substitute("Authorization: Bearer "+fixtureSecret, "github token")
	if strings.Contains(outbound, fixtureSecret) {
		t.Fatal("raw secret escaped to provider-visible payload")
	}
	if !strings.Contains(outbound, "$$GITHUBTOKEN_") {
		t.Fatalf("missing labeled placeholder: %q", outbound)
	}
	placeholder := strings.TrimPrefix(outbound, "Authorization: Bearer ")
	transcript := map[string]any{"headers": []any{map[string]any{"authorization": placeholder}}, "same": placeholder}
	before := cloneMap(transcript)
	var executed map[string]any
	if err := o.ExecuteRestored(transcript, func(args map[string]any) error { executed = args; return nil }); err != nil {
		t.Fatal(err)
	}
	got := executed["headers"].([]any)[0].(map[string]any)["authorization"]
	if got != fixtureSecret {
		t.Fatalf("tool received %q, want exact secret", got)
	}
	if !reflect.DeepEqual(transcript, before) {
		t.Fatalf("transcript mutated: got %#v want %#v", transcript, before)
	}
}

func TestObfuscationKeyIsInstallScopedAndPrivate(t *testing.T) {
	mk := func(fill byte) *Obfuscator {
		o, err := loadObfuscator(filepath.Join(t.TempDir(), "key"), bytes.NewReader(bytes.Repeat([]byte{fill}, keyBytes)))
		if err != nil {
			t.Fatal(err)
		}
		return o
	}
	a, b := mk(1), mk(2)
	keyPath := filepath.Join(t.TempDir(), "persisted", "key")
	if _, err := loadObfuscator(keyPath, bytes.NewReader(bytes.Repeat([]byte{4}, keyBytes))); err != nil {
		t.Fatal(err)
	}
	if info, err := os.Stat(keyPath); err != nil {
		t.Fatal(err)
	} else if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Fatalf("key permissions = %o, want 600", info.Mode().Perm())
	}
	pa, pb := a.Substitute(fixtureSecret, ""), b.Substitute(fixtureSecret, "")
	if pa == pb {
		t.Fatalf("two installs minted same placeholder %q", pa)
	}
	// Public strings contain neither key material nor the raw secret.
	for _, public := range []string{pa, pb, a.Warning(), b.Warning()} {
		if strings.Contains(public, fixtureSecret) || strings.Contains(public, strings.Repeat("\x01", keyBytes)) {
			t.Fatalf("private material leaked in %q", public)
		}
	}
}

func TestObfuscationCaseVariantsHaveIndependentBases(t *testing.T) {
	o, _ := loadObfuscator(filepath.Join(t.TempDir(), "key"), bytes.NewReader(bytes.Repeat([]byte{3}, keyBytes)))
	lower := o.placeholder("abcdefgh12345678", "")
	upper := o.placeholder("ABCDEFGH12345678", "")
	if strings.Split(lower, ":")[0] == strings.Split(upper, ":")[0] {
		t.Fatalf("case variants share base: %q %q", lower, upper)
	}
}

func TestObfuscationDefaultInert(t *testing.T) {
	t.Setenv("FAK_SECRETGATE", "")
	o, err := LoadObfuscator(filepath.Join(t.TempDir(), "key"))
	if err != nil {
		t.Fatal(err)
	}
	if o != nil {
		t.Fatal("obfuscation enabled without opt-in")
	}
	original := map[string]any{"token": fixtureSecret}
	if got := (*Obfuscator)(nil).RestoreArguments(original); !reflect.DeepEqual(got, original) {
		t.Fatalf("default changed bytes: %#v", got)
	}
}

func TestObfuscationEphemeralKeyWarnsAndNeverFailsOpen(t *testing.T) {
	parent := t.TempDir()
	blocker := filepath.Join(parent, "file")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	o, err := loadObfuscator(filepath.Join(blocker, "key"), bytes.NewReader(bytes.Repeat([]byte{4}, keyBytes)))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(o.Warning(), "ephemeral") {
		t.Fatalf("warning=%q", o.Warning())
	}
	if strings.Contains(o.Warning(), blocker) {
		t.Fatal("warning leaked key path")
	}
	if got := o.Substitute(fixtureSecret, ""); got == fixtureSecret {
		t.Fatal("persistence failure emitted raw secret")
	}
}

func TestObfuscationShortMatchFloor(t *testing.T) {
	o := &Obfuscator{key: bytes.Repeat([]byte{5}, keyBytes), restore: map[string]string{}}
	// Pin the public floor independently of canon's current minimum lengths.
	if got := o.substituteMatch("secret", ""); got != "secret" {
		t.Fatalf("short word obfuscated: %q", got)
	}
}

func TestExecuteRestoredPropagatesToolError(t *testing.T) {
	o := &Obfuscator{key: bytes.Repeat([]byte{6}, keyBytes), restore: map[string]string{}}
	want := errors.New("tool failed")
	if got := o.ExecuteRestored(map[string]any{}, func(map[string]any) error { return want }); !errors.Is(got, want) {
		t.Fatalf("got %v", got)
	}
}

func cloneMap(in map[string]any) map[string]any {
	out := make(map[string]any, len(in))
	for k, v := range in {
		switch x := v.(type) {
		case []any:
			y := make([]any, len(x))
			for i, e := range x {
				if m, ok := e.(map[string]any); ok {
					y[i] = cloneMap(m)
				} else {
					y[i] = e
				}
			}
			out[k] = y
		default:
			out[k] = v
		}
	}
	return out
}
