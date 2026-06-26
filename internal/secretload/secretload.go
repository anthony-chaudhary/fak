package secretload

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/canon"
)

// SecretSource is one priority-ordered origin of secret material. A Loader queries its
// sources in order and the FIRST hit wins, so an earlier source overrides a later one.
//
// Backends are additive by construction: the OS-env, encrypted-file, and .env sources in
// this package — and any future external vault (HashiCorp Vault KV2, AWS Secrets Manager,
// Azure Key Vault) — are all just a SecretSource. Adding a vault is wiring a new source
// into the loader, never rewriting the loader.
type SecretSource interface {
	// Name identifies the source for diagnostics and for the startup checklist's
	// "resolved by" attribution. It carries no secret material.
	Name() string

	// Lookup returns the value bound to key in this source and whether this source
	// supplied it. A source that does not know key MUST return ("", false); it must
	// never report a zero value as a hit, or it would shadow a later source that does
	// know the key (an absent value is not the empty value).
	Lookup(key string) (value string, ok bool)
}

// Loader resolves secrets across a priority-ordered list of SecretSources, tracks the
// required-secret checklist, and masks known secret material for safe logging. The zero
// Loader is usable (no sources, nothing required); use New to seed it with sources. A
// Loader is safe for concurrent use.
type Loader struct {
	mu       sync.RWMutex
	sources  []SecretSource
	required []requirement
	// masked is the set of secret VALUES Redact scrubs. It is grown by MarkSecret and
	// automatically whenever a Require'd (declared-secret) key resolves, so a resolved
	// credential is masked in logs without the caller threading it anywhere.
	masked map[string]struct{}
}

// New returns a Loader seeded with sources in priority order: sources[0] is consulted
// first and wins ties.
func New(sources ...SecretSource) *Loader {
	return &Loader{sources: append([]SecretSource(nil), sources...)}
}

// Default returns a Loader with the always-safe base source — the process environment.
// Callers AddSource an EncryptedFile or DotEnv (lower priority) as policy dictates.
func Default() *Loader { return New(OSEnvSource()) }

// AddSource appends s as the LOWEST-priority source (consulted last). Use it to wire an
// opt-in backend (a .env file, an encrypted store, a vault) after the defaults. A nil
// source is ignored.
func (l *Loader) AddSource(s SecretSource) {
	if s == nil {
		return
	}
	l.mu.Lock()
	l.sources = append(l.sources, s)
	l.mu.Unlock()
}

// Sources returns the source names in priority order — a diagnostic; it never returns
// values.
func (l *Loader) Sources() []string {
	l.mu.RLock()
	defer l.mu.RUnlock()
	names := make([]string, 0, len(l.sources))
	for _, s := range l.sources {
		names = append(names, s.Name())
	}
	return names
}

// resolve walks the sources in priority order. It is a pure read with no masking side
// effect, so the checklist can decide masking explicitly.
func (l *Loader) resolve(key string) (value, source string, ok bool) {
	l.mu.RLock()
	srcs := l.sources
	l.mu.RUnlock()
	for _, s := range srcs {
		if v, found := s.Lookup(key); found {
			return v, s.Name(), true
		}
	}
	return "", "", false
}

// Lookup resolves key against the sources in priority order. If key was declared with
// Require, the resolved value is registered for log-masking (Redact) automatically.
func (l *Loader) Lookup(key string) (string, bool) {
	v, _, ok := l.LookupSource(key)
	return v, ok
}

// LookupSource is Lookup that also reports which source supplied the value.
func (l *Loader) LookupSource(key string) (value, source string, ok bool) {
	v, src, ok := l.resolve(key)
	if !ok {
		return "", "", false
	}
	l.mu.RLock()
	req := l.isRequiredLocked(key)
	l.mu.RUnlock()
	if req {
		l.MarkSecret(v)
	}
	return v, src, true
}

// Get returns the resolved value for key, or "" if no source supplies it — the terse form
// of Lookup for callers that do not need the present/absent distinction.
func (l *Loader) Get(key string) string {
	v, _ := l.Lookup(key)
	return v
}

// requirement is one declared-required secret: the key, a human description for the
// startup checklist, and an optional validator run against the resolved value.
type requirement struct {
	key      string
	desc     string
	validate func(string) error
}

// Require declares key as a required secret with a human-readable description and an
// optional validator (nil = presence is enough). Requirements drive the startup checklist
// (CheckRequired / StartupReport): a missing or invalid required secret is reported LOUDLY
// at startup instead of failing late inside a request. Declaring a key required also marks
// its resolved value for log-masking (Redact). A repeated Require for the same key
// replaces the earlier declaration.
func (l *Loader) Require(key, description string, validate func(string) error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for i := range l.required {
		if l.required[i].key == key {
			l.required[i] = requirement{key, description, validate}
			return
		}
	}
	l.required = append(l.required, requirement{key, description, validate})
}

// isRequiredLocked reports whether key was declared with Require. The caller holds l.mu.
func (l *Loader) isRequiredLocked(key string) bool {
	for i := range l.required {
		if l.required[i].key == key {
			return true
		}
	}
	return false
}

// Status is a required-secret's checklist verdict.
type Status string

const (
	StatusOK      Status = "ok"      // present and (if a validator was set) valid
	StatusMissing Status = "missing" // no source supplied it
	StatusInvalid Status = "invalid" // supplied but failed the validator
)

// CheckResult is one row of the required-secret checklist. It names the key, the source
// that supplied it, and any validator error — never the secret value.
type CheckResult struct {
	Key         string
	Description string
	Status      Status
	Source      string // the source that supplied the value (StatusOK / StatusInvalid)
	Err         error  // the validator error (StatusInvalid only)
}

// OK reports whether the row passed (present and valid).
func (r CheckResult) OK() bool { return r.Status == StatusOK }

// CheckRequired resolves every declared-required secret and returns one CheckResult per
// requirement, in declaration order. A resolved required secret is registered for masking
// as a side effect (it is, by declaration, a credential).
func (l *Loader) CheckRequired() []CheckResult {
	l.mu.RLock()
	reqs := append([]requirement(nil), l.required...)
	l.mu.RUnlock()

	out := make([]CheckResult, 0, len(reqs))
	for _, r := range reqs {
		v, src, ok := l.resolve(r.key)
		if !ok {
			out = append(out, CheckResult{Key: r.key, Description: r.desc, Status: StatusMissing})
			continue
		}
		l.MarkSecret(v)
		if r.validate != nil {
			if err := r.validate(v); err != nil {
				out = append(out, CheckResult{Key: r.key, Description: r.desc, Status: StatusInvalid, Source: src, Err: err})
				continue
			}
		}
		out = append(out, CheckResult{Key: r.key, Description: r.desc, Status: StatusOK, Source: src})
	}
	return out
}

// Missing returns the subset of CheckRequired that did not pass (missing or invalid) — the
// rows a startup gate must act on.
func (l *Loader) Missing() []CheckResult {
	var bad []CheckResult
	for _, r := range l.CheckRequired() {
		if !r.OK() {
			bad = append(bad, r)
		}
	}
	return bad
}

// StartupReport resolves the required-secret checklist and renders a human-readable,
// value-free report. ok is true iff every required secret is present and valid; report is
// a multi-line summary safe to print at startup — it names keys, sources, and validator
// errors, never secret values.
func (l *Loader) StartupReport() (ok bool, report string) {
	results := l.CheckRequired()
	ok = true
	var b strings.Builder
	for _, r := range results {
		switch r.Status {
		case StatusOK:
			fmt.Fprintf(&b, "  ok      %s (via %s)\n", r.Key, r.Source)
		case StatusMissing:
			ok = false
			fmt.Fprintf(&b, "  MISSING %s — %s\n", r.Key, r.Description)
		case StatusInvalid:
			ok = false
			fmt.Fprintf(&b, "  INVALID %s — %s: %v\n", r.Key, r.Description, r.Err)
		}
	}
	if len(results) == 0 {
		b.WriteString("  (no required secrets declared)\n")
	}
	return ok, b.String()
}

// minMaskLen is the shortest secret VALUE Redact will scrub by exact match. Masking a
// very short value (a 1–4 char flag like "on" or a port) would corrupt unrelated text and
// leak nothing, so exact-value masking is gated on a credential-plausible length. The
// canon.SecretPatterns shape pass is unaffected — it is already shape-anchored.
const minMaskLen = 5

// RedactPlaceholder is the token Redact substitutes for a masked secret.
const RedactPlaceholder = "[REDACTED]"

// MarkSecret registers value as secret material to be scrubbed by Redact. The empty string
// and values shorter than minMaskLen are ignored. Safe for concurrent use; idempotent.
func (l *Loader) MarkSecret(value string) {
	if len(value) < minMaskLen {
		return
	}
	l.mu.Lock()
	if l.masked == nil {
		l.masked = make(map[string]struct{})
	}
	l.masked[value] = struct{}{}
	l.mu.Unlock()
}

// Redact returns s with every known secret scrubbed: first each registered secret VALUE
// (longest first, so a value that contains a shorter one is fully covered), then every
// canon.SecretPatterns shape (the one detector normgate and recall share — never forked
// here). It is a log-safety helper, so it errs toward over-redaction: a shape match scrubs
// the flagged span even if that span was not a real credential. With nothing registered
// and no shape present, s is returned unchanged.
func (l *Loader) Redact(s string) string {
	if s == "" {
		return s
	}
	l.mu.RLock()
	vals := make([]string, 0, len(l.masked))
	for v := range l.masked {
		vals = append(vals, v)
	}
	l.mu.RUnlock()

	// Longest value first: masking "abc" before "abcdef" would leave "[REDACTED]def".
	sort.Slice(vals, func(i, j int) bool { return len(vals[i]) > len(vals[j]) })
	for _, v := range vals {
		s = strings.ReplaceAll(s, v, RedactPlaceholder)
	}
	return RedactShapes(s)
}

// RedactShapes masks only canon.SecretPatterns shapes in s — the loader-free form of
// Redact for a caller that has no Loader (e.g. a deep log line). It cannot mask a known
// value that does not also match a shape; use Loader.Redact when the loaded secret set is
// available.
func RedactShapes(s string) string {
	for _, re := range canon.SecretPatterns {
		s = re.ReplaceAllString(s, RedactPlaceholder)
	}
	return s
}
