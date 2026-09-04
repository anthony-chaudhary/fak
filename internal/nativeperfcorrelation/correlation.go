// Package nativeperfcorrelation provides a bounded, scrubbed index that joins
// native-performance evidence without exporting high-cardinality identifiers as
// metric labels.
//
// Invariant: all indexed records scrub raw client identifiers into one-way sha256
// digests and store bounded relative locators so public metrics never leak trace data.
package nativeperfcorrelation

import (
	"crypto/sha256"
	"encoding/base32"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"path"
	"regexp"
	"slices"
	"strings"
	"sync"
)

const (
	// Schema defines the canonical JSON schema identifier for native performance correlation records.
	Schema = "fak/native-performance-correlation/v1"

	// NativeEngine identifies the required in-tree fak-native model execution runtime.
	NativeEngine = "fak-native"

	// DefaultMaxArtifact sets the default upper bound (64 MiB) for streaming artifact digest verification.
	DefaultMaxArtifact = int64(64 << 20)
)

var (
	// ErrNotFound indicates that the requested opaque correlation key is not present in the index.
	ErrNotFound = errors.New("native performance correlation not found")

	// ErrCollision indicates that distinct execution records produced an identical correlation key.
	ErrCollision = errors.New("native performance correlation key collision")

	// ErrArtifactMissing indicates that an expected receipt, trace, or profile artifact file is absent from the filesystem.
	ErrArtifactMissing = errors.New("native performance correlation artifact missing")

	// ErrDigestMismatch indicates that the computed artifact SHA-256 does not match the recorded manifest digest.
	ErrDigestMismatch = errors.New("native performance correlation artifact digest mismatch")

	// ErrArtifactTooLarge indicates that an artifact stream exceeded the allocated byte safety threshold.
	ErrArtifactTooLarge = errors.New("native performance correlation artifact exceeds size limit")

	hexDigestRE  = regexp.MustCompile(`^[0-9a-f]{64}$`)
	commitRE     = regexp.MustCompile(`^[0-9a-f]{7,64}$`)
	moduleRevRE  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]*@r[1-9][0-9]*\+g[0-9a-f]{7,64}$`)
	identityRE   = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/+:-]{0,127}$`)
	locatorSegRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
)

// Input is the unpersisted join material. Request, run, receipt, trace, and
// profile IDs are fingerprinted before a Record enters the index.
type Input struct {
	RequestID   string
	RunID       string
	CommitSHA   string
	ReceiptID   string
	TraceID     string
	ProfileID   string
	Engine      EngineIdentity
	ModuleAtRev string
	Artifacts   []Artifact
}

// EngineIdentity names the model execution path. Name must be "fak-native";
// Backend, Model, and Quantization are scrubbed bounded identities, not secrets.
type EngineIdentity struct {
	Name         string `json:"name"`
	Backend      string `json:"backend"`
	Model        string `json:"model"`
	Quantization string `json:"quantization,omitempty"`
}

// Artifact identifies one scrubbed artifact and its expected content digest.
type Artifact struct {
	Kind    ArtifactKind `json:"kind"`
	Locator string       `json:"locator"`
	SHA256  string       `json:"sha256"`
}

// ArtifactKind enumerates the closed set of verifiable evidence categories tracked by the index.
type ArtifactKind string

const (
	// ArtifactReceipt designates execution receipt artifacts containing admission and pricing telemetry.
	ArtifactReceipt ArtifactKind = "receipt"

	// ArtifactTrace designates session trace artifacts capturing tool call execution sequences.
	ArtifactTrace ArtifactKind = "trace"

	// ArtifactProfile designates performance profiling artifacts containing runtime memory and CPU samples.
	ArtifactProfile ArtifactKind = "profile"
)

// Record is safe to serialize into a public evidence index. It intentionally
// contains fingerprints instead of the supplied high-cardinality IDs.
type Record struct {
	Schema             string         `json:"schema"`
	Key                string         `json:"key"`
	RequestFingerprint string         `json:"request_fingerprint"`
	RunFingerprint     string         `json:"run_fingerprint"`
	CommitSHA          string         `json:"commit_sha"`
	ReceiptFingerprint string         `json:"receipt_fingerprint"`
	TraceFingerprint   string         `json:"trace_fingerprint"`
	ProfileFingerprint string         `json:"profile_fingerprint"`
	Engine             EngineIdentity `json:"engine"`
	ModuleAtRev        string         `json:"module_at_rev"`
	Artifacts          []Artifact     `json:"artifacts"`
}

// Exemplar returns the only high-cardinality metric attachment defined by this
// package. Callers should put Key in an exemplar or artifact link, never a
// Prometheus label set.
type Exemplar struct {
	CorrelationKey string `json:"correlation_key"`
}

// Contract: Index bounds in-memory retention strictly to capacity, evicting the oldest
// entry on overflow while maintaining exact idempotency for identical inputs.
//
// Invariant: Records never retain raw high-cardinality IDs; all request, run, receipt,
// trace, and profile identifiers are cryptographically fingerprinted using SHA-256 before admission.
//
// Index provides a concurrency-safe, bounded in-memory cache retaining at most Capacity records.
// When full, inserting a new record evicts the oldest insertion. Re-inserting an identical record is idempotent.
type Index struct {
	mu       sync.RWMutex
	capacity int
	keyFn    func(Record) string
	records  map[string]Record
	order    []string
}

// Option configures optional parameters during Index initialization, such as custom key derivation.
type Option func(*Index)

// WithKeyFunc exists for deterministic collision testing and specialized
// storage adapters. Production callers should use the default content key.
func WithKeyFunc(fn func(Record) string) Option {
	return func(index *Index) {
		if fn != nil {
			index.keyFn = fn
		}
	}
}

// Precondition: NewIndex requires capacity to be strictly positive (capacity > 0)
// to prevent zero-capacity indexes or unbounded memory growth.
//
// NewIndex constructs a thread-safe correlation index bounded by the specified positive capacity.
func NewIndex(capacity int, options ...Option) (*Index, error) {
	if capacity <= 0 {
		return nil, errors.New("native performance correlation capacity must be positive")
	}
	index := &Index{
		capacity: capacity,
		keyFn:    contentKey,
		records:  make(map[string]Record, capacity),
	}
	for _, option := range options {
		option(index)
	}
	return index, nil
}

// Postcondition: Add returns either an admitted immutable clone of the Record with a
// deterministic npc1_ correlation key, or an explicit validation error without mutating state.
//
// Add scrubs and admits execution input into the bounded index, evicting the oldest record if capacity is reached.
func (i *Index) Add(input Input) (Record, error) {
	record, err := scrub(input)
	if err != nil {
		return Record{}, err
	}
	record.Key = i.keyFn(record)
	if err := validateKey(record.Key); err != nil {
		return Record{}, err
	}

	i.mu.Lock()
	defer i.mu.Unlock()
	if existing, ok := i.records[record.Key]; ok {
		if equalRecord(existing, record) {
			return cloneRecord(existing), nil
		}
		return Record{}, fmt.Errorf("%w: %s", ErrCollision, record.Key)
	}
	if len(i.order) == i.capacity {
		delete(i.records, i.order[0])
		i.order = i.order[1:]
	}
	i.records[record.Key] = cloneRecord(record)
	i.order = append(i.order, record.Key)
	return cloneRecord(record), nil
}

// Lookup retrieves a defensively cloned Record by its opaque correlation key, or returns ErrNotFound.
func (i *Index) Lookup(key string) (Record, error) {
	i.mu.RLock()
	defer i.mu.RUnlock()
	record, ok := i.records[key]
	if !ok {
		return Record{}, fmt.Errorf("%w: %s", ErrNotFound, key)
	}
	return cloneRecord(record), nil
}

// Snapshot returns records in deterministic oldest-to-newest insertion order.
func (i *Index) Snapshot() []Record {
	i.mu.RLock()
	defer i.mu.RUnlock()
	out := make([]Record, 0, len(i.order))
	for _, key := range i.order {
		out = append(out, cloneRecord(i.records[key]))
	}
	return out
}

// Exemplar generates an OpenTelemetry-compatible metric exemplar referencing the indexed correlation key.
func (i *Index) Exemplar(key string) (Exemplar, error) {
	if _, err := i.Lookup(key); err != nil {
		return Exemplar{}, err
	}
	return Exemplar{CorrelationKey: key}, nil
}

// Guard: VerifyArtifact inspects artifact content streams bounded by maxBytes and enforces
// constant-time hex SHA-256 verification against the pre-indexed manifest digest.
//
// VerifyArtifact streams one indexed artifact from fsys, bounded by maxBytes,
// and verifies that it is the exact content named by the record.
func (i *Index) VerifyArtifact(key string, kind ArtifactKind, fsys fs.FS, maxBytes int64) error {
	record, err := i.Lookup(key)
	if err != nil {
		return err
	}
	artifact, ok := artifactByKind(record.Artifacts, kind)
	if !ok {
		return fmt.Errorf("%w: %s", ErrArtifactMissing, kind)
	}
	file, err := fsys.Open(artifact.Locator)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("%w: %s", ErrArtifactMissing, kind)
		}
		return fmt.Errorf("open %s artifact: %w", kind, err)
	}
	defer file.Close()
	if maxBytes <= 0 {
		maxBytes = DefaultMaxArtifact
	}
	hash := sha256.New()
	read, err := io.Copy(hash, io.LimitReader(file, maxBytes+1))
	if err != nil {
		return fmt.Errorf("hash %s artifact: %w", kind, err)
	}
	if read > maxBytes {
		return fmt.Errorf("%w: %s", ErrArtifactTooLarge, kind)
	}
	if got := hex.EncodeToString(hash.Sum(nil)); got != artifact.SHA256 {
		return fmt.Errorf("%w: %s", ErrDigestMismatch, kind)
	}
	return nil
}

// Fail-closed: Any unscrubbed engine backend, traversal path, non-relative locator,
// or missing required artifact kind immediately halts admission and returns an explicit error.
func scrub(input Input) (Record, error) {
	for name, value := range map[string]string{
		"request ID": input.RequestID,
		"run ID":     input.RunID,
		"receipt ID": input.ReceiptID,
		"trace ID":   input.TraceID,
		"profile ID": input.ProfileID,
	} {
		if strings.TrimSpace(value) == "" {
			return Record{}, fmt.Errorf("native performance correlation %s is required", name)
		}
	}
	commit := strings.ToLower(strings.TrimSpace(input.CommitSHA))
	if !commitRE.MatchString(commit) {
		return Record{}, errors.New("native performance correlation commit must be a 7-64 character lowercase hex SHA")
	}
	if input.Engine.Name != NativeEngine {
		return Record{}, fmt.Errorf("native performance correlation engine must be %q", NativeEngine)
	}
	if err := validateEngine(input.Engine); err != nil {
		return Record{}, err
	}
	if !moduleRevRE.MatchString(input.ModuleAtRev) {
		return Record{}, errors.New("native performance correlation module_at_rev must match module/path@rN+g<sha>")
	}
	artifacts, err := validateArtifacts(input.Artifacts)
	if err != nil {
		return Record{}, err
	}
	return Record{
		Schema:             Schema,
		RequestFingerprint: fingerprint(input.RequestID),
		RunFingerprint:     fingerprint(input.RunID),
		CommitSHA:          commit,
		ReceiptFingerprint: fingerprint(input.ReceiptID),
		TraceFingerprint:   fingerprint(input.TraceID),
		ProfileFingerprint: fingerprint(input.ProfileID),
		Engine:             input.Engine,
		ModuleAtRev:        input.ModuleAtRev,
		Artifacts:          artifacts,
	}, nil
}

func validateEngine(engine EngineIdentity) error {
	for name, value := range map[string]string{
		"backend": engine.Backend,
		"model":   engine.Model,
	} {
		if !identityRE.MatchString(value) {
			return fmt.Errorf("native performance correlation engine %s is not a scrubbed bounded identity", name)
		}
	}
	if engine.Quantization != "" && !identityRE.MatchString(engine.Quantization) {
		return errors.New("native performance correlation engine quantization is not a scrubbed bounded identity")
	}
	for name, value := range map[string]string{
		"backend":      engine.Backend,
		"model":        engine.Model,
		"quantization": engine.Quantization,
	} {
		if strings.Contains(value, "://") || strings.ContainsAny(value, "@?#\\") {
			return fmt.Errorf("native performance correlation engine %s contains endpoint or credential syntax", name)
		}
	}
	return nil
}

func validateArtifacts(in []Artifact) ([]Artifact, error) {
	if len(in) != 3 {
		return nil, errors.New("native performance correlation requires receipt, trace, and profile artifacts")
	}
	out := slices.Clone(in)
	slices.SortFunc(out, func(a, b Artifact) int { return strings.Compare(string(a.Kind), string(b.Kind)) })
	seen := make(map[ArtifactKind]bool, 3)
	for index := range out {
		artifact := &out[index]
		if artifact.Kind != ArtifactReceipt && artifact.Kind != ArtifactTrace && artifact.Kind != ArtifactProfile {
			return nil, fmt.Errorf("native performance correlation unknown artifact kind %q", artifact.Kind)
		}
		if seen[artifact.Kind] {
			return nil, fmt.Errorf("native performance correlation duplicate %s artifact", artifact.Kind)
		}
		seen[artifact.Kind] = true
		if err := validateLocator(artifact.Locator); err != nil {
			return nil, fmt.Errorf("native performance correlation %s locator: %w", artifact.Kind, err)
		}
		artifact.SHA256 = strings.ToLower(artifact.SHA256)
		if !hexDigestRE.MatchString(artifact.SHA256) {
			return nil, fmt.Errorf("native performance correlation %s digest must be lowercase SHA-256", artifact.Kind)
		}
	}
	return out, nil
}

func validateLocator(locator string) error {
	if locator == "" || len(locator) > 512 || !fs.ValidPath(locator) || path.IsAbs(locator) {
		return errors.New("must be a bounded relative fs.ValidPath")
	}
	if !strings.HasPrefix(locator, "artifacts/") {
		return errors.New("must be rooted under artifacts/")
	}
	for _, segment := range strings.Split(locator, "/") {
		if !locatorSegRE.MatchString(segment) {
			return errors.New("contains an unsanitized path segment")
		}
	}
	return nil
}

func validateKey(key string) error {
	if !strings.HasPrefix(key, "npc1_") || len(key) < len("npc1_")+16 || len(key) > 80 {
		return errors.New("native performance correlation key must be a bounded npc1_ opaque key")
	}
	for _, r := range key[len("npc1_"):] {
		if (r < 'a' || r > 'z') && (r < '2' || r > '7') {
			return errors.New("native performance correlation key contains invalid characters")
		}
	}
	return nil
}

func fingerprint(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func contentKey(record Record) string {
	record.Key = ""
	encoded, _ := json.Marshal(record)
	sum := sha256.Sum256(encoded)
	return "npc1_" + strings.ToLower(base32.StdEncoding.WithPadding(base32.NoPadding).EncodeToString(sum[:20]))
}

func artifactByKind(artifacts []Artifact, kind ArtifactKind) (Artifact, bool) {
	for _, artifact := range artifacts {
		if artifact.Kind == kind {
			return artifact, true
		}
	}
	return Artifact{}, false
}

func cloneRecord(record Record) Record {
	record.Artifacts = slices.Clone(record.Artifacts)
	return record
}

func equalRecord(a, b Record) bool {
	left, _ := json.Marshal(a)
	right, _ := json.Marshal(b)
	return string(left) == string(right)
}
