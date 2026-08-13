package portability

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// Support describes how faithfully an adapter implements its managed kind.
type Support string

const (
	SupportFull      Support = "full"
	SupportPartial   Support = "partial"
	SupportLocalOnly Support = "local-only"
	SupportInactive  Support = "inactive"
)

type Capability string

const (
	CapDiscover      Capability = "discover"
	CapRead          Capability = "read"
	CapWrite         Capability = "write"
	CapValidate      Capability = "validate"
	CapPreview       Capability = "preview"
	CapApply         Capability = "apply"
	CapRollback      Capability = "rollback"
	CapMigrate       Capability = "migrate"
	CapDiff          Capability = "diff"
	CapDependencies  Capability = "dependencies"
	CapIdentity      Capability = "identity"
	CapExport        Capability = "export"
	CapCompatibility Capability = "compatibility"
	CapDegradation   Capability = "degradation"
)

type AdapterInfo struct {
	Kind          string       `json:"kind"`
	Version       string       `json:"version"`
	Support       Support      `json:"support"`
	Sensitivity   string       `json:"sensitivity"`
	Compatibility string       `json:"compatibility"`
	Degradation   string       `json:"degradation,omitempty"`
	Capabilities  []Capability `json:"capabilities"`
}
type Record struct {
	Kind    string          `json:"kind"`
	Name    string          `json:"name"`
	Version string          `json:"version"`
	Active  bool            `json:"active"`
	Data    json.RawMessage `json:"data"`
	Unknown json.RawMessage `json:"unknown,omitempty"`
}
type Change struct {
	ID     string  `json:"id"`
	Before *Record `json:"before,omitempty"`
	After  *Record `json:"after,omitempty"`
}
type Plan struct {
	Kind     string   `json:"kind"`
	Changes  []Change `json:"changes"`
	Warnings []string `json:"warnings,omitempty"`
}
type AdapterReceipt struct {
	PlanID  string   `json:"plan_id"`
	Applied []Change `json:"applied"`
}
type Discovery struct {
	Records []Record `json:"records"`
}
type State interface {
	Discover(context.Context, string) ([]Record, error)
	Read(context.Context, string, string) (Record, error)
	Write(context.Context, string, Record) error
	Delete(context.Context, string, string) error
}

type Adapter interface {
	Info() AdapterInfo
	Discover(context.Context, State, string) (Discovery, error)
	Read(context.Context, State, string, string) (Record, error)
	Validate(context.Context, Record) error
	Preview(context.Context, State, string, []Record) (Plan, error)
	Apply(context.Context, State, string, Plan) (AdapterReceipt, error)
	Rollback(context.Context, State, string, AdapterReceipt) error
	Migrate(context.Context, Record, string) (Record, error)
	Diff(context.Context, Record, Record) ([]string, error)
	Dependencies(context.Context, Record) ([]string, error)
	Identity(context.Context, Record) (string, error)
	Export(context.Context, Record) ([]byte, error)
}

type Error struct {
	Code      string `json:"code"`
	Kind      string `json:"kind,omitempty"`
	Operation string `json:"operation"`
	Field     string `json:"field,omitempty"`
	Message   string `json:"message"`
}

func (e *Error) Error() string { return e.Code + ": " + e.Operation + ": " + e.Message }
func errx(code, kind, op, field, msg string) error {
	return &Error{Code: code, Kind: kind, Operation: op, Field: field, Message: msg}
}

// Registry is the additive registration seam. Register is deliberately the only
// mutation operation and rejects duplicate or malformed declarations.
type AdapterRegistry struct {
	mu       sync.RWMutex
	adapters map[string]Adapter
}

func NewAdapterRegistry() *AdapterRegistry { return &AdapterRegistry{adapters: map[string]Adapter{}} }
func (r *AdapterRegistry) Register(a Adapter) error {
	if a == nil {
		return errx("INVALID_ADAPTER", "", "register", "", "nil adapter")
	}
	i := a.Info()
	if i.Kind == "" || i.Version == "" {
		return errx("INVALID_ADAPTER", i.Kind, "register", "kind", "kind and version are required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.adapters[i.Kind]; ok {
		return errx("DUPLICATE_ADAPTER", i.Kind, "register", "kind", "adapter already registered")
	}
	r.adapters[i.Kind] = a
	return nil
}
func (r *AdapterRegistry) Adapter(kind string) Adapter {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if a := r.adapters[kind]; a != nil {
		return a
	}
	return opaqueAdapter{kind: kind}
}
func (r *AdapterRegistry) Matrix() []AdapterInfo {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]AdapterInfo, 0, len(r.adapters))
	for _, a := range r.adapters {
		out = append(out, a.Info())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Kind < out[j].Kind })
	return out
}
func (r *AdapterRegistry) RequireCoverage(kinds []string) error {
	sort.Strings(kinds)
	for _, k := range kinds {
		r.mu.RLock()
		_, ok := r.adapters[k]
		r.mu.RUnlock()
		if !ok {
			return errx("MISSING_ADAPTER", k, "coverage", "kind", "managed kind has no adapter/status")
		}
	}
	return nil
}

func canonical(v any) ([]byte, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var x any
	if err = json.Unmarshal(b, &x); err != nil {
		return nil, err
	}
	return json.Marshal(x)
}
func validateRecord(kind string, r Record) error {
	if r.Kind != kind {
		return errx("KIND_MISMATCH", kind, "validate", "kind", "record kind does not match adapter")
	}
	if strings.TrimSpace(r.Name) == "" {
		return errx("MALFORMED_RECORD", kind, "validate", "name", "name is required")
	}
	if !json.Valid(r.Data) {
		return errx("MALFORMED_RECORD", kind, "validate", "data", "data must be valid JSON")
	}
	return nil
}
func identity(r Record) (string, error) {
	c, err := canonical(struct {
		Kind, Name, Version string
		Data                json.RawMessage
		Unknown             json.RawMessage
	}{r.Kind, r.Name, r.Version, r.Data, r.Unknown})
	if err != nil {
		return "", err
	}
	h := sha256.Sum256(c)
	return "sha256:" + hex.EncodeToString(h[:]), nil
}

// MemoryState is a bounded host-supplied state seam used by conformance tests
// and embedders; adapters receive no filesystem path or credential capability.
type MemoryState struct {
	mu        sync.Mutex
	scopes    map[string]map[string]Record
	FailAfter int
	writes    int
}

func NewMemoryState() *MemoryState { return &MemoryState{scopes: map[string]map[string]Record{}} }
func (s *MemoryState) Discover(_ context.Context, scope string) ([]Record, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var o []Record
	for _, r := range s.scopes[scope] {
		o = append(o, r)
	}
	sort.Slice(o, func(i, j int) bool { return o[i].Name < o[j].Name })
	return o, nil
}
func (s *MemoryState) Read(_ context.Context, scope, name string) (Record, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r, ok := s.scopes[scope][name]
	if !ok {
		return Record{}, errors.New("not found")
	}
	return r, nil
}
func (s *MemoryState) Write(_ context.Context, scope string, r Record) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.writes++
	if s.FailAfter > 0 && s.writes > s.FailAfter {
		return errors.New("injected interruption")
	}
	if s.scopes[scope] == nil {
		s.scopes[scope] = map[string]Record{}
	}
	s.scopes[scope][r.Name] = r
	return nil
}
func (s *MemoryState) Delete(_ context.Context, scope, name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.scopes[scope], name)
	return nil
}

func stablePlanID(p Plan) string {
	c, _ := canonical(p)
	h := sha256.Sum256(c)
	return hex.EncodeToString(h[:16])
}
func equalRecord(a, b Record) bool {
	x, _ := canonical(a)
	y, _ := canonical(b)
	return bytes.Equal(x, y)
}
func secretLike(b []byte) bool {
	s := strings.ToLower(string(b))
	return strings.Contains(s, "secret") || strings.Contains(s, "token") || strings.Contains(s, "password") || strings.Contains(s, "credential")
}
func ensureNoSecret(kind, op string, r Record) error {
	if secretLike(r.Data) || secretLike(r.Unknown) {
		return errx("SENSITIVE_DATA", kind, op, "data", "credential-like material is not portable")
	}
	return nil
}
func unsupported(kind, op string) error {
	return errx("UNSUPPORTED", kind, op, "", "operation is unavailable for inactive adapter data")
}
func fmtPath(s string) string { return fmt.Sprintf("/%s", strings.ReplaceAll(s, "~", "~0")) }
