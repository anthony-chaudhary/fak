package secretload

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// openFast opens an EncryptedFile with a low PBKDF2 work factor so the round-trip tests
// stay fast; the at-rest format is identical (the iteration count is stored per file).
func openFast(t *testing.T, path string, pass []byte) *EncryptedFile {
	t.Helper()
	e, err := OpenEncryptedFile(path, pass)
	if err != nil {
		t.Fatalf("OpenEncryptedFile: %v", err)
	}
	e.iterations = 4096
	return e
}

func TestEncryptedFileRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.enc")
	pass := []byte("correct horse battery staple")

	w := openFast(t, path, pass)
	w.Set("OPENAI_API_KEY", "sk-secret-value-123456")
	w.Set("DB_PASSWORD", "p@ss w/ spaces & symbols ✓")
	if err := w.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	r := openFast(t, path, pass)
	if v, ok := r.Lookup("OPENAI_API_KEY"); !ok || v != "sk-secret-value-123456" {
		t.Fatalf("reopen OPENAI_API_KEY = (%q,%v)", v, ok)
	}
	if v, ok := r.Lookup("DB_PASSWORD"); !ok || v != "p@ss w/ spaces & symbols ✓" {
		t.Fatalf("reopen DB_PASSWORD = (%q,%v)", v, ok)
	}
	if _, ok := r.Lookup("ABSENT"); ok {
		t.Error("absent key reported present after reopen")
	}
	if r.Name() != "encrypted-file" {
		t.Errorf("Name = %q", r.Name())
	}
}

func TestEncryptedFileWrongPassphrase(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.enc")
	w := openFast(t, path, []byte("right-pass"))
	w.Set("K", "the-secret-value")
	if err := w.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}
	_, err := OpenEncryptedFile(path, []byte("wrong-pass"))
	if !errors.Is(err, ErrWrongPassphrase) {
		t.Fatalf("wrong passphrase err = %v, want ErrWrongPassphrase", err)
	}
}

func TestEncryptedFileTamperDetected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.enc")
	pass := []byte("pass")
	w := openFast(t, path, pass)
	w.Set("K", "the-secret-value")
	if err := w.Save(); err != nil {
		t.Fatalf("Save: %v", err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var env encEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		t.Fatal(err)
	}
	ct, _ := base64.StdEncoding.DecodeString(env.Ciphertext)
	if len(ct) == 0 {
		t.Fatal("empty ciphertext")
	}
	ct[0] ^= 0xff // flip a byte -> GCM auth must fail
	env.Ciphertext = base64.StdEncoding.EncodeToString(ct)
	out, _ := json.Marshal(env)
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := OpenEncryptedFile(path, pass); !errors.Is(err, ErrWrongPassphrase) {
		t.Fatalf("tamper err = %v, want ErrWrongPassphrase", err)
	}
}

func TestEncryptedFileMissingIsEmptyWritable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does-not-exist.enc")
	e, err := OpenEncryptedFile(path, []byte("pass"))
	if err != nil {
		t.Fatalf("missing file must open clean, got %v", err)
	}
	if _, ok := e.Lookup("K"); ok {
		t.Error("empty store reported a value")
	}
	// And it is writable from empty.
	e.iterations = 4096
	e.Set("K", "v-value")
	if err := e.Save(); err != nil {
		t.Fatalf("Save from empty: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("Save did not create the file: %v", err)
	}
}

func TestEncryptedFileDeleteOnEmptySet(t *testing.T) {
	path := filepath.Join(t.TempDir(), "store.enc")
	e := openFast(t, path, []byte("p"))
	e.Set("K", "v-value")
	e.Set("K", "") // delete
	if _, ok := e.Lookup("K"); ok {
		t.Error("empty Set should delete the key")
	}
}

func TestPassphraseFromEnv(t *testing.T) {
	if _, ok := PassphraseFromEnv(); ok {
		t.Skip("FAK_SECRETS_PASSPHRASE set in env; skipping unset-case assertion")
	}
	t.Setenv("FAK_SECRETS_PASSPHRASE", "hunter2")
	if p, ok := PassphraseFromEnv(); !ok || string(p) != "hunter2" {
		t.Fatalf("PassphraseFromEnv = (%q,%v)", p, ok)
	}
	t.Setenv("FAK_SECRETS_PASSPHRASE", "")
	if _, ok := PassphraseFromEnv(); ok {
		t.Error("empty passphrase env must report absent")
	}
}
