package deployment

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// DesiredContext is the locked, normalized input to a deployment derivation.
type DesiredContext struct {
	Objects        []ManagedObject `json:"objects"`
	LockedInputs   []LockedInput   `json:"locked_inputs"`
	AdapterVersion string          `json:"adapter_version"`
	SchemaVersion  string          `json:"schema_version"`
	Policy         string          `json:"policy"`
	Target         Target          `json:"target"`
	AllowedEffects []string        `json:"allowed_effects,omitempty"`
}

// ManagedObject declares a content-addressed entity required by the runtime environment.
type ManagedObject struct {
	Type    string `json:"type"`
	ID      string `json:"id"`
	Content []byte `json:"content"`
}

// LockedInput pins an external artifact dependency by its content digest.
type LockedInput struct {
	Name   string `json:"name"`
	Digest string `json:"digest"`
}

// Target names the capability envelope covered by a derivation and realization.
type Target struct {
	OS           string   `json:"os"`
	Architecture string   `json:"architecture"`
	Capabilities []string `json:"capabilities,omitempty"`
}

// DerivationID identifies declared inputs. It makes no claim about output bytes or behavior.
func DerivationID(desired DesiredContext) (string, error) {
	normalizeDesired(&desired)
	raw, err := json.Marshal(desired)
	if err != nil {
		return "", err
	}
	return digest(raw), nil
}

// Realization is one byte-identical output closure for a derivation and target.
type Realization struct {
	ID           string            `json:"id"`
	DerivationID string            `json:"derivation_id"`
	Target       Target            `json:"target"`
	Files        map[string][]byte `json:"files"`
}

// NewRealization computes output identity independently from derivation identity.
func NewRealization(derivationID string, target Target, files map[string][]byte) (Realization, error) {
	if derivationID == "" {
		return Realization{}, errors.New("derivation ID is required")
	}
	if err := validateFiles(files); err != nil {
		return Realization{}, err
	}
	target.Capabilities = sortedUnique(target.Capabilities)
	id, err := realizationID(files)
	if err != nil {
		return Realization{}, err
	}
	return Realization{ID: id, DerivationID: derivationID, Target: target, Files: cloneFiles(files)}, nil
}

// Compatible reports whether a trusted substitute exactly covers the declared target envelope.
func (r Realization) Compatible(derivationID string, target Target) bool {
	target.Capabilities = sortedUnique(target.Capabilities)
	r.Target.Capabilities = sortedUnique(r.Target.Capabilities)
	return r.DerivationID == derivationID && r.Target.OS == target.OS && r.Target.Architecture == target.Architecture && equalStrings(r.Target.Capabilities, target.Capabilities)
}

// Store is a local content-addressed realization store on disk.
type Store struct{ Root string }

// Materialize verifies and atomically installs a realization without activating it.
func (s Store) Materialize(r Realization) (string, error) {
	if s.Root == "" {
		return "", errors.New("store root is required")
	}
	verified, err := NewRealization(r.DerivationID, r.Target, r.Files)
	if err != nil {
		return "", err
	}
	if r.ID != verified.ID {
		return "", fmt.Errorf("realization ID mismatch: declared %s computed %s", r.ID, verified.ID)
	}
	final := filepath.Join(s.Root, r.ID)
	if _, err := os.Stat(final); err == nil {
		return final, nil
	} else if !errors.Is(err, fs.ErrNotExist) {
		return "", err
	}
	if err := os.MkdirAll(s.Root, 0o755); err != nil {
		return "", err
	}
	tmp, err := os.MkdirTemp(s.Root, ".realization-")
	if err != nil {
		return "", err
	}
	committed := false
	defer func() {
		if !committed {
			_ = os.RemoveAll(tmp)
		}
	}()
	for name, data := range r.Files {
		path := filepath.Join(tmp, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return "", err
		}
		if err := os.WriteFile(path, data, 0o444); err != nil {
			return "", err
		}
	}
	manifest, err := json.Marshal(struct {
		ID           string `json:"id"`
		DerivationID string `json:"derivation_id"`
		Target       Target `json:"target"`
	}{r.ID, r.DerivationID, r.Target})
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(tmp, "realization.json"), manifest, 0o444); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, final); err != nil {
		if _, statErr := os.Stat(final); statErr == nil {
			return final, nil
		}
		return "", err
	}
	committed = true
	return final, nil
}

// Generation is an immutable activation record. Sequence is monotonically increasing per name.
type Generation struct {
	Name         string   `json:"name"`
	Sequence     uint64   `json:"sequence"`
	Realizations []string `json:"realizations"`
	ConfigDigest string   `json:"config_digest"`
	CreatedAt    string   `json:"created_at"`
}

// Activator manages named generation records and small atomic active-pointer files.
type Activator struct{ Root string }

// Activate creates an immutable generation and atomically selects it.
func (a Activator) Activate(name string, realizationIDs []string, config []byte) (Generation, error) {
	if err := validName(name); err != nil {
		return Generation{}, err
	}
	if len(realizationIDs) == 0 {
		return Generation{}, errors.New("at least one realization is required")
	}
	ids := sortedUnique(realizationIDs)
	dir := filepath.Join(a.Root, "generations", name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Generation{}, err
	}
	seq, err := nextSequence(dir)
	if err != nil {
		return Generation{}, err
	}
	g := Generation{Name: name, Sequence: seq, Realizations: ids, ConfigDigest: digest(config), CreatedAt: time.Now().UTC().Format(time.RFC3339Nano)}
	raw, err := json.Marshal(g)
	if err != nil {
		return Generation{}, err
	}
	generationPath := filepath.Join(dir, fmt.Sprintf("%020d.json", seq))
	if err := writeExclusive(generationPath, raw); err != nil {
		return Generation{}, err
	}
	if err := a.switchActive(name, generationPath); err != nil {
		return Generation{}, err
	}
	return g, nil
}

// Active returns the selected immutable generation.
func (a Activator) Active(name string) (Generation, error) {
	if err := validName(name); err != nil {
		return Generation{}, err
	}
	raw, err := os.ReadFile(filepath.Join(a.Root, "active", name))
	if err != nil {
		return Generation{}, err
	}
	generationPath := filepath.Join(a.Root, filepath.FromSlash(strings.TrimSpace(string(raw))))
	var g Generation
	if raw, err = os.ReadFile(generationPath); err != nil {
		return Generation{}, err
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		return Generation{}, err
	}
	return g, nil
}

// Rollback atomically selects the generation immediately preceding the active one.
func (a Activator) Rollback(name string) (Generation, error) {
	active, err := a.Active(name)
	if err != nil {
		return Generation{}, err
	}
	if active.Sequence <= 1 {
		return Generation{}, errors.New("no previous generation")
	}
	path := filepath.Join(a.Root, "generations", name, fmt.Sprintf("%020d.json", active.Sequence-1))
	raw, err := os.ReadFile(path)
	if err != nil {
		return Generation{}, fmt.Errorf("previous generation unavailable: %w", err)
	}
	var previous Generation
	if err := json.Unmarshal(raw, &previous); err != nil {
		return Generation{}, err
	}
	if err := a.switchActive(name, path); err != nil {
		return Generation{}, err
	}
	return previous, nil
}

func (a Activator) switchActive(name, generationPath string) error {
	activeDir := filepath.Join(a.Root, "active")
	if err := os.MkdirAll(activeDir, 0o755); err != nil {
		return err
	}
	rel, err := filepath.Rel(a.Root, generationPath)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(activeDir, ".active-")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }()
	if _, err := tmp.WriteString(filepath.ToSlash(rel) + "\n"); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, filepath.Join(activeDir, name))
}

func normalizeDesired(d *DesiredContext) {
	sort.Slice(d.Objects, func(i, j int) bool {
		if d.Objects[i].Type != d.Objects[j].Type {
			return d.Objects[i].Type < d.Objects[j].Type
		}
		if d.Objects[i].ID != d.Objects[j].ID {
			return d.Objects[i].ID < d.Objects[j].ID
		}
		return string(d.Objects[i].Content) < string(d.Objects[j].Content)
	})
	sort.Slice(d.LockedInputs, func(i, j int) bool {
		if d.LockedInputs[i].Name != d.LockedInputs[j].Name {
			return d.LockedInputs[i].Name < d.LockedInputs[j].Name
		}
		return d.LockedInputs[i].Digest < d.LockedInputs[j].Digest
	})
	d.Target.Capabilities = sortedUnique(d.Target.Capabilities)
	d.AllowedEffects = sortedUnique(d.AllowedEffects)
}

func realizationID(files map[string][]byte) (string, error) {
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	sort.Strings(names)
	h := sha256.New()
	for _, name := range names {
		fmt.Fprintf(h, "%d:%s:%d:", len(name), name, len(files[name]))
		_, _ = h.Write(files[name])
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func validateFiles(files map[string][]byte) error {
	if len(files) == 0 {
		return errors.New("realization must contain at least one file")
	}
	for name := range files {
		clean := filepath.Clean(filepath.FromSlash(name))
		if name == "" || filepath.IsAbs(clean) || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) || clean == "realization.json" {
			return fmt.Errorf("unsafe realization path %q", name)
		}
	}
	return nil
}

func nextSequence(dir string) (uint64, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, err
	}
	var max uint64
	for _, entry := range entries {
		var n uint64
		if _, err := fmt.Sscanf(entry.Name(), "%020d.json", &n); err == nil && n > max {
			max = n
		}
	}
	return max + 1, nil
}

func writeExclusive(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o444)
	if err != nil {
		return err
	}
	if _, err = f.Write(data); err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func validName(name string) error {
	if name == "" || name == "." || name == ".." || strings.ContainsAny(name, `/\\`) {
		return fmt.Errorf("invalid generation name %q", name)
	}
	return nil
}

func digest(raw []byte) string { sum := sha256.Sum256(raw); return hex.EncodeToString(sum[:]) }
func sortedUnique(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	n := 0
	for _, v := range out {
		if n == 0 || out[n-1] != v {
			out[n] = v
			n++
		}
	}
	return out[:n]
}
func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
func cloneFiles(in map[string][]byte) map[string][]byte {
	out := make(map[string][]byte, len(in))
	for k, v := range in {
		out[k] = append([]byte(nil), v...)
	}
	return out
}
