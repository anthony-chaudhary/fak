// Package sourceleasecontrol defines a finite, source-owned local lease release.
// Definitions and Authority implementations come from protected host Go code;
// request fields, recorded holder strings and filesystem pins confer no authority.
package sourceleasecontrol

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"time"
)

const (
	Tool                     = "fak_source_lease_control"
	OperationRelease         = "release"
	LockEffectExclusiveMutex = "exclusive-create-delete-mutex"
	ReceiptSchema            = "fak.source-lease-control.v1"
	OutcomeReleased          = "released"
	OutcomeRefused           = "refused"
	OutcomeUnknown           = "unknown_effect"
)

var (
	ErrInvalid       = errors.New("invalid source lease control")
	ErrAuthority     = errors.New("source lease authority unavailable")
	ErrChanged       = errors.New("source lease scope changed")
	ErrConsumed      = errors.New("source lease operation consumed")
	ErrUnknownEffect = errors.New("source lease effect unknown")
)

type Request struct {
	ProfileID  string `json:"profile_id"`
	PlanDigest string `json:"plan_digest"`
}

// Definition names an existing append store and one absent exclusive mutex leaf.
// Footprint includes the mutex creation/deletion and every existing regular file
// the provider may change. No directory writes, patterns or command are admitted.
type Definition struct {
	ProfileID     string    `json:"profile_id"`
	Operation     string    `json:"operation"`
	WorkspaceRoot string    `json:"workspace_root"`
	StorePath     string    `json:"store_path"`
	LockPath      string    `json:"lock_path"`
	LockEffect    string    `json:"lock_effect"`
	LeaseKey      string    `json:"lease_key"`
	Generation    uint64    `json:"generation"`
	RowSHA256     string    `json:"row_sha256"`
	Footprint     []string  `json:"footprint"`
	ExpiresAt     time.Time `json:"expires_at"`
}

// Authority is retained in host memory, never deserialized from a tool call.
// Validate checks preliminary source pins, not fresh permission to execute.
// Execute MUST open the canonical provider transaction, verify file identities,
// and recheck authenticated actor authority, exact row/generation, owner/session
// and child state, and preservation facts while holding that transaction, at the
// last pre-effect boundary. CAS alone is not authorization. Public runtime pins
// cannot replace those provider-specific checks or make them atomic.
// The mutex MUST be acquired by canonical exclusive creation, retain its own
// file/nonce identity, and remove only that owned mutex. A race is refusal;
// observing an absent mutex during Prepare never reserves it.
// Execute MUST durably consume the operation identity and retain an unknown
// outcome barrier inside its transaction. Prepared supplies only per-instance
// single use: another Prepare is not a replay fence. A host facade must also
// retain per-plan consumption across RPCs; neither may retry an unknown effect.
type Authority interface {
	Resolve(context.Context, string) (Definition, error)
	Validate(context.Context, Definition) error
	Execute(context.Context, Definition) (Receipt, error)
}

// Receipt is closed control metadata. It carries no arbitrary observations.
type Receipt struct {
	Schema     string    `json:"schema"`
	ProfileID  string    `json:"profile_id"`
	PlanDigest string    `json:"plan_digest"`
	Generation uint64    `json:"generation"`
	RowSHA256  string    `json:"row_sha256"`
	Outcome    string    `json:"outcome"`
	Verified   bool      `json:"verified"`
	StartedAt  time.Time `json:"started_at"`
	EndedAt    time.Time `json:"ended_at"`
}

type Prepared struct {
	mu         sync.Mutex
	authority  Authority
	definition Definition
	digest     string
	pins       map[string]os.FileInfo
	consumed   bool
}

func ParseRequest(raw []byte) (Request, error) {
	var r Request
	if len(raw) == 0 || len(raw) > 1024 {
		return r, ErrInvalid
	}
	d := json.NewDecoder(strings.NewReader(string(raw)))
	first, err := d.Token()
	if err != nil || first != json.Delim('{') {
		return r, ErrInvalid
	}
	seen := map[string]bool{}
	for d.More() {
		key, err := d.Token()
		if err != nil {
			return r, ErrInvalid
		}
		name, ok := key.(string)
		if !ok || seen[name] {
			return r, ErrInvalid
		}
		seen[name] = true
		switch name {
		case "profile_id":
			err = d.Decode(&r.ProfileID)
		case "plan_digest":
			err = d.Decode(&r.PlanDigest)
		default:
			return r, ErrInvalid
		}
		if err != nil {
			return r, ErrInvalid
		}
	}
	if last, err := d.Token(); err != nil || last != json.Delim('}') {
		return r, ErrInvalid
	}
	if _, err := d.Token(); !errors.Is(err, io.EOF) {
		return r, ErrInvalid
	}
	if !validName(r.ProfileID) || !validDigest(r.PlanDigest) {
		return r, ErrInvalid
	}
	return r, nil
}

func Clone(d Definition) Definition {
	d.Footprint = append([]string(nil), d.Footprint...)
	return d
}

// PlanDigest validates the finite definition and hashes its canonical JSON.
func PlanDigest(d Definition) (string, error) {
	d = Clone(d)
	if _, err := pinDefinition(d, time.Now().UTC()); err != nil {
		return "", err
	}
	return definitionDigest(d), nil
}

// DefinitionDigest hashes canonical definition JSON without reading files or
// validating live scope. It is usable while the provider holds its mutex. A
// digest proves neither permission nor validity; Prepare still validates scope,
// and the provider must revalidate authority inside its held transaction.
func DefinitionDigest(d Definition) (string, error) {
	d = Clone(d)
	d.ExpiresAt = d.ExpiresAt.UTC()
	b, err := json.Marshal(d)
	if err != nil {
		return "", ErrInvalid
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:]), nil
}

func definitionDigest(d Definition) string {
	digest, _ := DefinitionDigest(d)
	return digest
}

func Prepare(ctx context.Context, a Authority, r Request) (*Prepared, error) {
	if ctx == nil || nilAuthority(a) {
		return nil, ErrAuthority
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !validName(r.ProfileID) || !validDigest(r.PlanDigest) {
		return nil, ErrInvalid
	}
	d, err := a.Resolve(ctx, r.ProfileID)
	if err != nil {
		return nil, ErrAuthority
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	d = Clone(d)
	pins, err := pinDefinition(d, time.Now().UTC())
	if err != nil {
		return nil, err
	}
	digest := definitionDigest(d)
	if d.ProfileID != r.ProfileID || digest != r.PlanDigest {
		return nil, ErrChanged
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := a.Validate(ctx, Clone(d)); err != nil {
		return nil, ErrAuthority
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &Prepared{authority: a, definition: d, digest: digest, pins: pins}, nil
}

func (p *Prepared) Definition() Definition {
	if p == nil {
		return Definition{}
	}
	return Clone(p.definition)
}

func (p *Prepared) Execute(ctx context.Context) (Receipt, error) {
	if p == nil || ctx == nil || p.authority == nil {
		return Receipt{}, ErrAuthority
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return Receipt{}, err
	}
	if p.consumed {
		return Receipt{}, ErrConsumed
	}
	p.consumed = true
	if err := p.revalidate(); err != nil {
		return Receipt{}, err
	}
	if err := ctx.Err(); err != nil {
		return Receipt{}, err
	}
	if err := p.authority.Validate(ctx, Clone(p.definition)); err != nil {
		return Receipt{}, ErrAuthority
	}
	if err := ctx.Err(); err != nil {
		return Receipt{}, err
	}
	if err := p.revalidate(); err != nil {
		return Receipt{}, err
	}
	if err := ctx.Err(); err != nil {
		return Receipt{}, err
	}
	started := time.Now().UTC()
	r, err := invokeEffect(ctx, p.authority, Clone(p.definition))
	if err != nil || ctx.Err() != nil || r.StartedAt.Before(started) || ValidateReceipt(p.definition, p.digest, r, time.Now().UTC()) != nil {
		return UnknownReceipt(p.definition, p.digest, started, time.Now().UTC()), ErrUnknownEffect
	}
	if r.Outcome == OutcomeUnknown {
		return r, ErrUnknownEffect
	}
	if r.Outcome == OutcomeRefused {
		return r, ErrAuthority
	}
	return r, nil
}

func invokeEffect(ctx context.Context, a Authority, d Definition) (receipt Receipt, err error) {
	defer func() {
		if recover() != nil {
			receipt, err = Receipt{}, ErrUnknownEffect
		}
	}()
	return a.Execute(ctx, d)
}

func nilAuthority(a Authority) bool {
	if a == nil {
		return true
	}
	v := reflect.ValueOf(a)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return v.IsNil()
	}
	return false
}

func (p *Prepared) revalidate() error {
	pins, err := pinDefinition(p.definition, time.Now().UTC())
	if err != nil {
		return err
	}
	for name, before := range p.pins {
		if after, ok := pins[name]; !ok || !os.SameFile(before, after) {
			return ErrChanged
		}
	}
	return nil
}

func UnknownReceipt(d Definition, digest string, started, ended time.Time) Receipt {
	if ended.Before(started) {
		ended = started
	}
	return Receipt{Schema: ReceiptSchema, ProfileID: d.ProfileID, PlanDigest: digest, Generation: d.Generation, RowSHA256: d.RowSHA256, Outcome: OutcomeUnknown, StartedAt: started.UTC(), EndedAt: ended.UTC()}
}

func ValidateReceipt(d Definition, digest string, r Receipt, now time.Time) error {
	if digest != definitionDigest(d) || r.Schema != ReceiptSchema || r.ProfileID != d.ProfileID || r.PlanDigest != digest || r.Generation != d.Generation || r.RowSHA256 != d.RowSHA256 || r.StartedAt.IsZero() || r.EndedAt.Before(r.StartedAt) || r.EndedAt.After(now) || r.StartedAt.After(d.ExpiresAt) {
		return ErrChanged
	}
	switch r.Outcome {
	case OutcomeReleased:
		if !r.Verified {
			return ErrInvalid
		}
	case OutcomeRefused, OutcomeUnknown:
		if r.Verified {
			return ErrInvalid
		}
	default:
		return ErrInvalid
	}
	return nil
}

func validName(s string) bool {
	if len(s) == 0 || len(s) > 128 {
		return false
	}
	for _, c := range s {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return true
}

func validDigest(s string) bool {
	if len(s) != 64 || strings.ToLower(s) != s {
		return false
	}
	b, err := hex.DecodeString(s)
	return err == nil && len(b) == 32
}

func canonical(path string) (os.FileInfo, error) {
	if len(path) == 0 || len(path) > 4096 || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, ErrInvalid
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil || resolved != path {
		return nil, ErrInvalid
	}
	info, err := os.Lstat(path)
	if err != nil || info.Mode()&os.ModeSymlink != 0 {
		return nil, ErrInvalid
	}
	return info, nil
}

func pinDefinition(d Definition, now time.Time) (map[string]os.FileInfo, error) {
	if !validName(d.ProfileID) || d.Operation != OperationRelease || d.LockEffect != LockEffectExclusiveMutex || !validName(d.LeaseKey) || d.Generation == 0 || !validDigest(d.RowSHA256) || d.ExpiresAt.IsZero() || !d.ExpiresAt.After(now) || d.ExpiresAt.After(now.Add(15*time.Minute)) || len(d.Footprint) < 2 || len(d.Footprint) > 32 {
		return nil, ErrInvalid
	}
	root, err := canonical(d.WorkspaceRoot)
	if err != nil || !root.IsDir() {
		return nil, ErrInvalid
	}
	pins := map[string]os.FileInfo{d.WorkspaceRoot: root}
	lockDeclared := false
	for i, name := range d.Footprint {
		if i > 0 && d.Footprint[i-1] >= name {
			return nil, ErrInvalid
		}
		rel, err := filepath.Rel(d.WorkspaceRoot, name)
		if err != nil || rel == "." || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
			return nil, ErrInvalid
		}
		if name == d.LockPath {
			if !filepath.IsAbs(name) || filepath.Clean(name) != name || len(name) > 4096 {
				return nil, ErrInvalid
			}
			if _, err := os.Lstat(name); !errors.Is(err, os.ErrNotExist) {
				return nil, ErrChanged
			}
			parent := filepath.Dir(name)
			info, err := canonical(parent)
			if err != nil || !info.IsDir() {
				return nil, ErrInvalid
			}
			pins[parent] = info
			lockDeclared = true
			continue
		}
		info, err := canonical(name)
		if err != nil || !info.Mode().IsRegular() {
			return nil, ErrInvalid
		}
		for _, existing := range pins {
			if os.SameFile(existing, info) {
				return nil, ErrInvalid
			}
		}
		pins[name] = info
	}
	if d.StorePath == d.LockPath || pins[d.StorePath] == nil || !pins[d.StorePath].Mode().IsRegular() || !lockDeclared {
		return nil, fmt.Errorf("%w: distinct store and absent exclusive mutex required", ErrInvalid)
	}
	return pins, nil
}
