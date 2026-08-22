package harnessartifact

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var safeName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)

func GenerateKey() (ed25519.PublicKey, ed25519.PrivateKey, error) {
	return ed25519.GenerateKey(rand.Reader)
}
func EncodePrivateKey(k ed25519.PrivateKey) string { return base64.StdEncoding.EncodeToString(k) }
func EncodePublicKey(k ed25519.PublicKey) string   { return base64.StdEncoding.EncodeToString(k) }
func DecodePrivateKey(s string) (ed25519.PrivateKey, error) {
	b, e := base64.StdEncoding.DecodeString(strings.TrimSpace(s))
	if e != nil {
		return nil, e
	}
	if len(b) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("private key has %d bytes, want %d", len(b), ed25519.PrivateKeySize)
	}
	return ed25519.PrivateKey(b), nil
}
func DecodePublicKey(s string) (ed25519.PublicKey, error) {
	b, e := base64.StdEncoding.DecodeString(strings.TrimSpace(s))
	if e != nil {
		return nil, e
	}
	if len(b) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("public key has %d bytes, want %d", len(b), ed25519.PublicKeySize)
	}
	return ed25519.PublicKey(b), nil
}
func Fingerprint(k ed25519.PublicKey) string {
	s := sha256.Sum256(k)
	return "sha256:" + hex.EncodeToString(s[:])
}
func artifactID(payload Payload, publicKey string) (string, error) {
	raw, e := json.Marshal(struct {
		Payload   Payload `json:"payload"`
		PublicKey string  `json:"public_key"`
	}{payload, publicKey})
	if e != nil {
		return "", e
	}
	s := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(s[:]), nil
}
func signBytes(id string, payload Payload, pub string) ([]byte, error) {
	return json.Marshal(struct {
		ID        string  `json:"id"`
		Payload   Payload `json:"payload"`
		PublicKey string  `json:"public_key"`
	}{id, payload, pub})
}
func digestHex(id string) (string, error) {
	if !strings.HasPrefix(id, "sha256:") {
		return "", fmt.Errorf("artifact reference must pin sha256 digest")
	}
	h := strings.TrimPrefix(id, "sha256:")
	if len(h) != 64 {
		return "", fmt.Errorf("invalid sha256 digest")
	}
	if _, e := hex.DecodeString(h); e != nil {
		return "", fmt.Errorf("invalid sha256 digest")
	}
	return h, nil
}
func blobPath(root, id string) (string, error) {
	h, e := digestHex(id)
	if e != nil {
		return "", e
	}
	return filepath.Join(root, "blobs", "sha256", h+".json"), nil
}
func channelPath(root, name string) (string, error) {
	if !safeName.MatchString(name) {
		return "", fmt.Errorf("invalid channel %q", name)
	}
	return filepath.Join(root, "channels", name+".json"), nil
}
func writeJSONAtomic(path string, v any, perm os.FileMode) error {
	raw, e := json.MarshalIndent(v, "", "  ")
	if e != nil {
		return e
	}
	raw = append(raw, '\n')
	if e = os.MkdirAll(filepath.Dir(path), 0o755); e != nil {
		return e
	}
	tmp := path + ".tmp"
	if e = os.WriteFile(tmp, raw, perm); e != nil {
		return e
	}
	return os.Rename(tmp, path)
}
func readJSON(path string, v any) error {
	raw, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if e = d.Decode(v); e != nil {
		return e
	}
	if d.Decode(&struct{}{}) != io.EOF {
		return fmt.Errorf("trailing JSON in %s", path)
	}
	return nil
}

func unique(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, value := range in {
		if !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	return out
}

func contains(in []string, value string) bool {
	for _, item := range in {
		if item == value {
			return true
		}
	}
	return false
}
