package session

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// descriptorFileMagic is the self-describing schema-header magic on the durable
// session ledger — the forward-compat header for the live-deployment upgrade
// seam (#3424, mirroring #3395's L3 cache-record magic+version header). A ledger
// whose Version carries this magic but a version this fak build does not support
// is an INCOMPATIBLE SCHEMA JUMP across the version boundary, not corruption: the
// records are intact but unreadable by this binary, so the runtime refuses to
// start loudly (IncompatibleSchemaError) rather than quarantining them — because
// quarantine-and-start-empty would silently drop every live durable session
// across the upgrade seam, the exact loss the drain/cutover path exists to
// prevent. A version with no recognizable magic is still treated as genuine
// corruption and quarantined.
const descriptorFileMagic = "fak.session-descriptors."

const descriptorFileVersion = descriptorFileMagic + "v1"

const (
	childRegistrationSchema       = "fak-child-registration/1"
	descriptorStoreDefault        = "<UserConfigDir>/fak/session-registry.json"
	childRegistrationStoreDefault = "<UserConfigDir>/fak/child-registrations.jsonl"
)

// ledgerSchemaIncompatibleReason is the stable, greppable named reason surfaced
// when the durable session ledger's schema is incompatible with this build. It
// is the "refuses with a named reason rather than partially migrating" contract
// from #3424's acceptance criteria.
const ledgerSchemaIncompatibleReason = "LEDGER_SCHEMA_INCOMPATIBLE"

// IncompatibleSchemaError reports that the durable session ledger carries a
// well-formed schema header (descriptorFileMagic) at a version this fak build
// does not support — an incompatible schema jump across a live-deployment
// upgrade seam (#3424). Unlike CorruptDescriptorFileError it is NOT quarantined:
// the records are intact but unreadable by this version, so the runtime refuses
// to start against them rather than partially migrating or silently dropping
// live sessions. Recover by rolling back to a fak build that supports
// FileVersion, or by running a forward migration for Supported.
type IncompatibleSchemaError struct {
	FileVersion string
	Supported   string
}

func (e *IncompatibleSchemaError) Error() string {
	return fmt.Sprintf("%s: durable session ledger schema %q is incompatible with this fak build (supports %q); refusing to start rather than dropping live sessions — roll back to a build that supports %q, or run a forward migration",
		ledgerSchemaIncompatibleReason, e.FileVersion, e.Supported, e.FileVersion)
}

// IsIncompatibleSchema reports whether err is a refuse-to-start incompatible
// schema jump on the durable session ledger. It is deliberately distinct from
// IsCorruptDescriptorFile: an incompatible jump must NOT be recovered by
// quarantine (which would drop live sessions), it must halt the upgrade loudly.
func IsIncompatibleSchema(err error) bool {
	var target *IncompatibleSchemaError
	return errors.As(err, &target)
}

// SchemaCollisionError reports that the descriptor store was pointed at a
// child-lineage ledger. It is not corruption: callers must leave the file in
// place and refuse startup.
type SchemaCollisionError struct {
	FoundSchema              string
	ExpectedSchema           string
	DescriptorDefault        string
	ChildRegistrationDefault string
}

func (e *SchemaCollisionError) Error() string {
	return fmt.Sprintf("session registry schema collision: the session registry path (FAK_SESSION_REGISTRY/--session-registry) names schema %q, but the descriptor store requires schema %q; defaults are descriptor %s and child-registration %s; refusing without changing the lineage ledger",
		e.FoundSchema, e.ExpectedSchema, e.DescriptorDefault, e.ChildRegistrationDefault)
}

func (*SchemaCollisionError) schemaCollision() {}

// IsSchemaCollision reports whether err is a cross-store schema collision.
func IsSchemaCollision(err error) bool {
	var target *SchemaCollisionError
	return errors.As(err, &target)
}

// fileStoreFaultHook is a test-only crash-boundary seam. Production leaves it nil.
// Tests install it only in isolated helper processes, so concurrent callers never
// share mutable hook state.
var fileStoreFaultHook func(stage string)

func fileStoreBoundary(stage string) {
	if fileStoreFaultHook != nil {
		fileStoreFaultHook(stage)
	}
}

// FileStore persists Descriptor rows into one JSON file. It is the production
// DescriptorStore for the live session registry: Put/Delete rewrite the small
// descriptor index, while List reads the current file back. The file is an index
// of drive state only, not a transcript.
type FileStore struct {
	mu   sync.Mutex
	path string
}

// NewFileStore returns a DescriptorStore backed by path.
func NewFileStore(path string) *FileStore {
	return &FileStore{path: path}
}

// CorruptDescriptorFileError reports malformed or unsupported descriptor-index
// content. Callers may recover by quarantining the index: descriptors are a
// rebuildable projection of live session state, not the session transcript.
// Cause carries the normalized, privacy-safe corruption class so recovery
// observability never has to echo descriptor contents (#4658).
type CorruptDescriptorFileError struct {
	Cause RecoveryCause
	Err   error
}

func (e *CorruptDescriptorFileError) Error() string { return e.Err.Error() }
func (e *CorruptDescriptorFileError) Unwrap() error { return e.Err }

// IsCorruptDescriptorFile reports whether err means the descriptor index was
// readable but its contents could not be trusted.
func IsCorruptDescriptorFile(err error) bool {
	var target *CorruptDescriptorFileError
	return errors.As(err, &target)
}

func corruptDescriptorFileError(cause RecoveryCause, err error) error {
	return &CorruptDescriptorFileError{Cause: cause, Err: err}
}

type descriptorFile struct {
	Version     string       `json:"version"`
	Descriptors []Descriptor `json:"descriptors"`
}

// Put writes one descriptor keyed by ID, replacing any prior row for that ID.
// The cross-process lock orders writers: for the same ID, the last Put that
// acquires the lock wins, regardless of the descriptor's embedded Rev value.
func (s *FileStore) Put(d Descriptor) error {
	if d.ID == "" {
		return errBlankDescriptorID
	}
	if s == nil || s.path == "" {
		return registryError("descriptor file path must be non-empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	unlock, err := lockDescriptorFile(s.path)
	if err != nil {
		return err
	}
	defer unlock()
	if err := cleanupDescriptorTemps(s.path); err != nil {
		return err
	}
	byID, err := s.loadLocked()
	if err != nil {
		return err
	}
	byID[d.ID] = d
	return s.saveLocked(byID)
}

// Delete removes id from the file. Deleting a missing id is a no-op.
func (s *FileStore) Delete(id string) error {
	if s == nil || s.path == "" {
		return registryError("descriptor file path must be non-empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	unlock, err := lockDescriptorFile(s.path)
	if err != nil {
		return err
	}
	defer unlock()
	if err := cleanupDescriptorTemps(s.path); err != nil {
		return err
	}
	byID, err := s.loadLocked()
	if err != nil {
		return err
	}
	if _, ok := byID[id]; !ok {
		return nil
	}
	delete(byID, id)
	return s.saveLocked(byID)
}

// List returns every descriptor currently persisted in the file.
func (s *FileStore) List() ([]Descriptor, error) {
	if s == nil || s.path == "" {
		return nil, registryError("descriptor file path must be non-empty")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	byID, err := s.loadLocked()
	if err != nil {
		return nil, err
	}
	out := make([]Descriptor, 0, len(byID))
	for _, d := range byID {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

func (s *FileStore) loadLocked() (map[string]Descriptor, error) {
	if s == nil || s.path == "" {
		return nil, registryError("descriptor file path must be non-empty")
	}
	b, err := readDescriptorFile(s.path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]Descriptor{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read session descriptor file: %w", err)
	}
	if len(bytes.TrimSpace(b)) == 0 {
		return map[string]Descriptor{}, nil
	}
	if schema := lineageSchema(b); schema != "" {
		return nil, &SchemaCollisionError{
			FoundSchema: schema, ExpectedSchema: descriptorFileVersion,
			DescriptorDefault: descriptorStoreDefault, ChildRegistrationDefault: childRegistrationStoreDefault,
		}
	}
	var doc descriptorFile
	if err := json.Unmarshal(b, &doc); err != nil {
		return nil, corruptDescriptorFileError(RecoveryCauseDecode, fmt.Errorf("decode session descriptor file: %w", err))
	}
	if doc.Version != descriptorFileVersion {
		// Forward-compat header check (#3424): a well-formed schema magic at an
		// unsupported version is an incompatible upgrade-seam jump — refuse
		// loudly instead of quarantining, so no live session is silently
		// dropped. A version with no recognizable magic is genuine corruption.
		if strings.HasPrefix(doc.Version, descriptorFileMagic) {
			return nil, &IncompatibleSchemaError{FileVersion: doc.Version, Supported: descriptorFileVersion}
		}
		return nil, corruptDescriptorFileError(RecoveryCauseVersion, fmt.Errorf("unsupported session descriptor file version %q", doc.Version))
	}
	byID := make(map[string]Descriptor, len(doc.Descriptors))
	for _, d := range doc.Descriptors {
		if d.ID == "" {
			return nil, corruptDescriptorFileError(RecoveryCauseBlankID, errBlankDescriptorID)
		}
		byID[d.ID] = d
	}
	return byID, nil
}

func lineageSchema(data []byte) string {
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var header struct {
			Schema string `json:"schema"`
		}
		if json.Unmarshal(line, &header) != nil {
			return ""
		}
		if header.Schema == childRegistrationSchema || strings.HasPrefix(header.Schema, "fak-child-registration/") || strings.HasPrefix(header.Schema, "fak.sessionjournal.") {
			return header.Schema
		}
		return ""
	}
	return ""
}

func (s *FileStore) saveLocked(byID map[string]Descriptor) error {
	if s == nil || s.path == "" {
		return registryError("descriptor file path must be non-empty")
	}
	descs := make([]Descriptor, 0, len(byID))
	for _, d := range byID {
		descs = append(descs, d)
	}
	sort.Slice(descs, func(i, j int) bool { return descs[i].ID < descs[j].ID })
	doc := descriptorFile{Version: descriptorFileVersion, Descriptors: descs}

	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create session descriptor dir: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".session-descriptors-*.tmp")
	if err != nil {
		return fmt.Errorf("create session descriptor temp file: %w", err)
	}
	tmpName := tmp.Name()
	committed := false
	defer func() {
		if !committed {
			_ = os.Remove(tmpName)
		}
	}()

	enc := json.NewEncoder(tmp)
	enc.SetIndent("", "  ")
	if err := enc.Encode(doc); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("encode session descriptor file: %w", err)
	}
	fileStoreBoundary("encode")
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("flush session descriptor file: %w", err)
	}
	fileStoreBoundary("flush")
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close session descriptor file: %w", err)
	}
	fileStoreBoundary("close")
	if err := replaceFile(tmpName, s.path); err != nil {
		return err
	}
	committed = true
	return nil
}

// DriverName returns the driver identifier "file".
func (s *FileStore) DriverName() string { return "file" }

func init() {
	RegisterStoreDriver("file", &FileStore{})
}
