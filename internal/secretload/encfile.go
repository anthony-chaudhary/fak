package secretload

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/pbkdf2"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sync"
)

// EncryptedFile is a SecretSource backed by a local file encrypted at rest with
// AES-256-GCM. The 32-byte key is stretched from a caller-supplied passphrase with
// PBKDF2-HMAC-SHA256 over a per-file random salt; a fresh random nonce is drawn on every
// Save. GCM authenticates the ciphertext, so a wrong passphrase or any tampering fails the
// open/decrypt LOUDLY (ErrWrongPassphrase) rather than returning garbage.
//
// At-rest model (documented on purpose): the file is a small JSON envelope
//
//	{"version":1,"kdf":"pbkdf2-sha256","iterations":N,"salt":b64,"nonce":b64,"ciphertext":b64}
//
// whose plaintext is the JSON object of key→value secrets. The passphrase is the caller's
// to source — commonly the FAK_SECRETS_PASSPHRASE env var (see PassphraseFromEnv) or, as a
// documented follow-on, an OS keyring. Mechanism, not policy: this backend
// encrypts/decrypts; it never decides where the passphrase lives.
//
// Zero external dependencies — every primitive is standard library (crypto/aes,
// crypto/cipher, and crypto/pbkdf2, the last added to the stdlib in Go 1.24).
type EncryptedFile struct {
	path       string
	passphrase []byte
	iterations int

	mu      sync.RWMutex
	secrets map[string]string
}

const (
	encFileVersion   = 1
	pbkdf2Iterations = 240_000 // PBKDF2-HMAC-SHA256 work factor (OWASP-floor class)
	encKeyLen        = 32      // AES-256
	encSaltLen       = 16
)

// encEnvelope is the on-disk JSON form of an EncryptedFile.
type encEnvelope struct {
	Version    int    `json:"version"`
	KDF        string `json:"kdf"`
	Iterations int    `json:"iterations"`
	Salt       string `json:"salt"`       // base64
	Nonce      string `json:"nonce"`      // base64
	Ciphertext string `json:"ciphertext"` // base64
}

// ErrWrongPassphrase is returned by OpenEncryptedFile when the file exists but the
// passphrase does not decrypt it (GCM authentication failed) — distinct from a missing
// file, which opens an empty, writable store with no error.
var ErrWrongPassphrase = errors.New("secretload: wrong passphrase or corrupt encrypted store")

// OpenEncryptedFile opens (or, when path does not exist, initializes an empty, writable)
// encrypted secret store under passphrase. A present-but-undecryptable file returns
// ErrWrongPassphrase. The caller owns the passphrase lifecycle.
func OpenEncryptedFile(path string, passphrase []byte) (*EncryptedFile, error) {
	e := &EncryptedFile{
		path:       path,
		passphrase: append([]byte(nil), passphrase...),
		iterations: pbkdf2Iterations,
		secrets:    map[string]string{},
	}
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return e, nil // first run: empty store; Save creates the file
	}
	if err != nil {
		return nil, fmt.Errorf("secretload: read %s: %w", path, err)
	}
	if len(raw) == 0 {
		return e, nil
	}
	if err := e.decryptInto(raw); err != nil {
		return nil, err
	}
	return e, nil
}

func (e *EncryptedFile) decryptInto(raw []byte) error {
	var env encEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return fmt.Errorf("secretload: parse envelope: %w", err)
	}
	if env.Version != encFileVersion {
		return fmt.Errorf("secretload: unsupported envelope version %d", env.Version)
	}
	salt, err := base64.StdEncoding.DecodeString(env.Salt)
	if err != nil {
		return fmt.Errorf("secretload: decode salt: %w", err)
	}
	nonce, err := base64.StdEncoding.DecodeString(env.Nonce)
	if err != nil {
		return fmt.Errorf("secretload: decode nonce: %w", err)
	}
	ct, err := base64.StdEncoding.DecodeString(env.Ciphertext)
	if err != nil {
		return fmt.Errorf("secretload: decode ciphertext: %w", err)
	}
	iters := env.Iterations
	if iters <= 0 {
		iters = pbkdf2Iterations
	}
	gcm, err := e.gcm(salt, iters)
	if err != nil {
		return err
	}
	if len(nonce) != gcm.NonceSize() {
		return ErrWrongPassphrase
	}
	pt, err := gcm.Open(nil, nonce, ct, nil)
	if err != nil {
		return ErrWrongPassphrase
	}
	m := map[string]string{}
	if err := json.Unmarshal(pt, &m); err != nil {
		return fmt.Errorf("secretload: parse plaintext: %w", err)
	}
	e.mu.Lock()
	e.secrets = m
	e.iterations = iters
	e.mu.Unlock()
	return nil
}

func (e *EncryptedFile) gcm(salt []byte, iters int) (cipher.AEAD, error) {
	key, err := pbkdf2.Key(sha256.New, string(e.passphrase), salt, iters, encKeyLen)
	if err != nil {
		return nil, fmt.Errorf("secretload: derive key: %w", err)
	}
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, fmt.Errorf("secretload: aes: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("secretload: gcm: %w", err)
	}
	return gcm, nil
}

// Name identifies the source for diagnostics.
func (e *EncryptedFile) Name() string { return "encrypted-file" }

// Lookup returns the stored value for key, reporting a hit only for a non-empty value.
func (e *EncryptedFile) Lookup(key string) (string, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()
	v, ok := e.secrets[key]
	if !ok || v == "" {
		return "", false
	}
	return v, true
}

// Set stores key=value in memory; call Save to persist it encrypted. An empty value
// deletes the key.
func (e *EncryptedFile) Set(key, value string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if value == "" {
		delete(e.secrets, key)
		return
	}
	e.secrets[key] = value
}

// Save encrypts the current secret set and writes it to the file atomically (write a temp
// sibling, then rename). A fresh random salt and nonce are drawn on every Save, so two
// saves of identical content produce different ciphertext.
func (e *EncryptedFile) Save() error {
	e.mu.RLock()
	pt, err := json.Marshal(e.secrets)
	iters := e.iterations
	e.mu.RUnlock()
	if err != nil {
		return fmt.Errorf("secretload: marshal: %w", err)
	}
	if iters <= 0 {
		iters = pbkdf2Iterations
	}
	salt := make([]byte, encSaltLen)
	if _, err := rand.Read(salt); err != nil {
		return fmt.Errorf("secretload: salt: %w", err)
	}
	gcm, err := e.gcm(salt, iters)
	if err != nil {
		return err
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return fmt.Errorf("secretload: nonce: %w", err)
	}
	ct := gcm.Seal(nil, nonce, pt, nil)
	out, err := json.MarshalIndent(encEnvelope{
		Version:    encFileVersion,
		KDF:        "pbkdf2-sha256",
		Iterations: iters,
		Salt:       base64.StdEncoding.EncodeToString(salt),
		Nonce:      base64.StdEncoding.EncodeToString(nonce),
		Ciphertext: base64.StdEncoding.EncodeToString(ct),
	}, "", "  ")
	if err != nil {
		return fmt.Errorf("secretload: marshal envelope: %w", err)
	}
	tmp := e.path + ".tmp"
	if err := os.WriteFile(tmp, out, 0o600); err != nil {
		return fmt.Errorf("secretload: write: %w", err)
	}
	if err := os.Rename(tmp, e.path); err != nil {
		return fmt.Errorf("secretload: rename: %w", err)
	}
	return nil
}

// PassphraseFromEnv reads the encrypted-store passphrase from the FAK_SECRETS_PASSPHRASE
// environment variable. It returns (nil, false) when unset or empty, so a caller can decide
// whether a missing passphrase is fatal (the encrypted source is simply absent) or
// required.
func PassphraseFromEnv() ([]byte, bool) {
	v, ok := os.LookupEnv("FAK_SECRETS_PASSPHRASE")
	if !ok || v == "" {
		return nil, false
	}
	return []byte(v), true
}
