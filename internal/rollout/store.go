package rollout

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const storeFormatVersion = 1

// PointerName identifies a rollout role used when admitting new sessions.
type PointerName string

const (
	PointerStable        PointerName = "stable"
	PointerCanary        PointerName = "candidate"
	PointerLastKnownGood PointerName = "last-known-good"
)

// Store keeps immutable generation artifacts and manifests below one local root.
// PointerName only replaces small pointer files; it never rewrites generation data.
type Store struct {
	root string
	mu   sync.RWMutex
}

type generationManifest struct {
	Version    int        `json:"version"`
	Generation Generation `json:"generation"`
}

type pointerManifest struct {
	Version    int        `json:"version"`
	Generation Generation `json:"generation"`
}

// NewStore opens a local generation store. Directories are created lazily.
func NewStore(root string) *Store {
	return &Store{root: root}
}

// Install persists artifact as an immutable generation. The supplied digest
// must be sha256:<hex> for the exact artifact bytes.
func (s *Store) Install(g Generation, artifact []byte) error {
	if err := g.validate("stored"); err != nil {
		return err
	}
	digestHex, err := parseDigest(g.Digest)
	if err != nil {
		return err
	}
	actual := sha256.Sum256(artifact)
	if !bytes.Equal(actual[:], mustDecodeHex(digestHex)) {
		return fmt.Errorf("generation %q digest mismatch: manifest has %s, artifact has sha256:%x", g.ID, g.Digest, actual)
	}
	if s == nil || s.root == "" {
		return errors.New("generation store requires a root")
	}

	if err := os.MkdirAll(filepath.Join(s.root, "objects"), 0o755); err != nil {
		return fmt.Errorf("create object store: %w", err)
	}
	if err := os.MkdirAll(filepath.Join(s.root, "generations"), 0o755); err != nil {
		return fmt.Errorf("create manifest store: %w", err)
	}

	objectPath := s.objectPath(digestHex)
	if err := writeImmutable(objectPath, artifact); err != nil {
		return fmt.Errorf("store artifact: %w", err)
	}
	if err := verifyFileDigest(objectPath, digestHex); err != nil {
		return fmt.Errorf("verify stored artifact: %w", err)
	}

	manifestBytes, err := json.Marshal(generationManifest{Version: storeFormatVersion, Generation: g})
	if err != nil {
		return fmt.Errorf("encode generation manifest: %w", err)
	}
	manifestPath := s.manifestPath(g.ID)
	if err := writeImmutable(manifestPath, append(manifestBytes, '\n')); err != nil {
		return fmt.Errorf("store generation manifest: %w", err)
	}

	stored, err := s.Load(g.ID)
	if err != nil {
		return err
	}
	if stored != g {
		return fmt.Errorf("generation %q already identifies digest %s", g.ID, stored.Digest)
	}
	return nil
}

// Load reads a generation manifest and verifies its artifact before returning it.
func (s *Store) Load(id string) (Generation, error) {
	if s == nil || s.root == "" {
		return Generation{}, errors.New("generation store requires a root")
	}
	if id == "" {
		return Generation{}, errors.New("generation ID cannot be empty")
	}

	data, err := os.ReadFile(s.manifestPath(id))
	if err != nil {
		return Generation{}, fmt.Errorf("read generation %q manifest: %w", id, err)
	}
	var manifest generationManifest
	if err := decodeExactJSON(data, &manifest); err != nil {
		return Generation{}, fmt.Errorf("decode generation %q manifest: %w", id, err)
	}
	if manifest.Version != storeFormatVersion {
		return Generation{}, fmt.Errorf("generation %q manifest version %d is unsupported", id, manifest.Version)
	}
	if err := manifest.Generation.validate("stored"); err != nil {
		return Generation{}, err
	}
	if manifest.Generation.ID != id {
		return Generation{}, fmt.Errorf("generation manifest identity mismatch: requested %q, found %q", id, manifest.Generation.ID)
	}
	digestHex, err := parseDigest(manifest.Generation.Digest)
	if err != nil {
		return Generation{}, err
	}
	if err := verifyFileDigest(s.objectPath(digestHex), digestHex); err != nil {
		return Generation{}, fmt.Errorf("verify generation %q artifact: %w", id, err)
	}
	return manifest.Generation, nil
}

// Artifact returns verified immutable bytes for g.
func (s *Store) Artifact(g Generation) ([]byte, error) {
	stored, err := s.Load(g.ID)
	if err != nil {
		return nil, err
	}
	if stored != g {
		return nil, fmt.Errorf("generation %q digest mismatch: requested %s, stored %s", g.ID, g.Digest, stored.Digest)
	}
	digestHex, _ := parseDigest(g.Digest)
	return os.ReadFile(s.objectPath(digestHex))
}

// Activate atomically changes a named pointer after verifying the generation.
func (s *Store) Activate(name PointerName, g Generation) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := validatePointerName(name); err != nil {
		return err
	}
	stored, err := s.Load(g.ID)
	if err != nil {
		return err
	}
	if stored != g {
		return fmt.Errorf("activation %q generation mismatch: requested %s, stored %s", name, g.Digest, stored.Digest)
	}
	if err := os.MkdirAll(filepath.Join(s.root, "activations"), 0o755); err != nil {
		return fmt.Errorf("create activation store: %w", err)
	}
	data, err := json.Marshal(pointerManifest{Version: storeFormatVersion, Generation: g})
	if err != nil {
		return fmt.Errorf("encode activation %q: %w", name, err)
	}
	if err := atomicWrite(filepath.Join(s.root, "activations", string(name)+".json"), append(data, '\n')); err != nil {
		return fmt.Errorf("activate %q: %w", name, err)
	}
	return nil
}

// Active resolves a named pointer and verifies the referenced generation.
func (s *Store) Active(name PointerName) (Generation, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := validatePointerName(name); err != nil {
		return Generation{}, err
	}
	data, err := readPointerNameFile(filepath.Join(s.root, "activations", string(name)+".json"))
	if err != nil {
		return Generation{}, fmt.Errorf("read activation %q: %w", name, err)
	}
	var pointer pointerManifest
	if err := decodeExactJSON(data, &pointer); err != nil {
		return Generation{}, fmt.Errorf("decode activation %q: %w", name, err)
	}
	if pointer.Version != storeFormatVersion {
		return Generation{}, fmt.Errorf("activation %q version %d is unsupported", name, pointer.Version)
	}
	stored, err := s.Load(pointer.Generation.ID)
	if err != nil {
		return Generation{}, err
	}
	if stored != pointer.Generation {
		return Generation{}, fmt.Errorf("activation %q generation mismatch", name)
	}
	return stored, nil
}

func (s *Store) objectPath(digestHex string) string {
	return filepath.Join(s.root, "objects", digestHex)
}

func (s *Store) manifestPath(id string) string {
	sum := sha256.Sum256([]byte(id))
	return filepath.Join(s.root, "generations", hex.EncodeToString(sum[:])+".json")
}

func parseDigest(digest string) (string, error) {
	const prefix = "sha256:"
	if !strings.HasPrefix(digest, prefix) {
		return "", fmt.Errorf("digest %q must use sha256:<hex>", digest)
	}
	hexDigest := strings.TrimPrefix(digest, prefix)
	decoded, err := hex.DecodeString(hexDigest)
	if err != nil || len(decoded) != sha256.Size {
		return "", fmt.Errorf("digest %q must contain exactly %d SHA-256 bytes", digest, sha256.Size)
	}
	return strings.ToLower(hexDigest), nil
}

func mustDecodeHex(value string) []byte {
	decoded, _ := hex.DecodeString(value)
	return decoded
}

func verifyFileDigest(path, wantHex string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil {
		return err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != wantHex {
		return fmt.Errorf("artifact corruption: digest is sha256:%s, want sha256:%s", got, wantHex)
	}
	return nil
}

func writeImmutable(path string, data []byte) error {
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, ".install-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	if err := temp.Chmod(0o444); err != nil {
		temp.Close()
		return err
	}
	if _, err := temp.Write(data); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}

	if err := os.Link(tempPath, path); err == nil {
		return nil
	} else if !errors.Is(err, os.ErrExist) {
		return err
	}
	existing, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if !bytes.Equal(existing, data) {
		return errors.New("immutable file already exists with different contents")
	}
	return nil
}
func atomicWrite(path string, data []byte) error {
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, ".activate-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()
	defer os.Remove(tempPath)
	if _, err := temp.Write(data); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Sync(); err != nil {
		temp.Close()
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	var lastErr error
	for attempt := 0; attempt < 200; attempt++ {
		if err := os.Rename(tempPath, path); err == nil {
			return nil
		} else {
			lastErr = err
		}
		// Windows may briefly deny replacement while another process has the
		// old pointer open. Retrying preserves the complete old pointer until
		// one atomic replacement succeeds.
		time.Sleep(time.Millisecond)
	}
	return lastErr
}

func readPointerNameFile(path string) ([]byte, error) {
	var lastErr error
	for attempt := 0; attempt < 50; attempt++ {
		data, err := os.ReadFile(path)
		if err == nil {
			return data, nil
		}
		lastErr = err
		// Windows replacement can briefly deny a concurrent open even though
		// readers can only observe the complete old or complete new pointer.
		time.Sleep(time.Millisecond)
	}
	return nil, lastErr
}
func decodeExactJSON(data []byte, target any) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("unexpected trailing JSON value")
		}
		return err
	}
	return nil
}

func validatePointerName(name PointerName) error {
	switch name {
	case PointerStable, PointerCanary, PointerLastKnownGood:
		return nil
	default:
		return fmt.Errorf("unknown activation %q", name)
	}
}
