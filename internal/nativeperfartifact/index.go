// Package nativeperfartifact provides a bounded, public-safe index from native
// performance correlation keys to benchmark artifacts.
package nativeperfartifact

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/nativeperfcorrelation"
)

const (
	Schema       = "fak-native-performance-artifacts/1"
	MaxArtifacts = 5
)

var moduleRevRE = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._/-]*@r[1-9][0-9]*\+g[0-9a-f]{7,64}$`)

var (
	ErrNotFound        = errors.New("native performance artifact not found")
	ErrExpired         = errors.New("native performance artifact expired")
	ErrBroken          = errors.New("native performance artifact is broken")
	ErrPrivate         = errors.New("native performance artifact locator is private")
	ErrUntrustedScheme = errors.New("native performance artifact locator uses an untrusted scheme")
	ErrRedacted        = errors.New("native performance artifact is redacted")
)

type Kind string

const (
	KindReceipt          Kind = "benchmark_receipt"
	KindMetalProfile     Kind = "metal_profile_bundle"
	KindCUDAProfile      Kind = "cuda_profile_bundle"
	KindKernelTrace      Kind = "kernel_trace"
	KindComparisonReport Kind = "comparison_report"
)

type State string

const (
	StateReady    State = "ready"
	StateBroken   State = "broken"
	StateRedacted State = "redacted"
)

type SeriesStatus string

const (
	SeriesProduced    SeriesStatus = "produced"
	SeriesUnavailable SeriesStatus = "unavailable"
)

type UnavailableReason string

const (
	ReasonNone            UnavailableReason = "none"
	ReasonIncomplete      UnavailableReason = "incomplete"
	ReasonStale           UnavailableReason = "stale"
	ReasonPrivate         UnavailableReason = "private"
	ReasonUntrustedScheme UnavailableReason = "untrusted_scheme"
)

// LiveObservation binds one completed Qwen3.8 Metal run to the scrubbed index
// record produced for it. Identity and revision are validation inputs, not
// metric labels, so an observation cannot expand Prometheus cardinality.
type LiveObservation struct {
	CompletedAt time.Time
	ModuleRev   string
	ModelFamily string
	Backend     string
	Record      Record
}

// LiveSeriesResult distinguishes an emitted bounded series from evidence that
// is honestly unavailable. Unavailable results carry only a closed reason and
// never echo source fields that could contain paths, locators, or digests.
type LiveSeriesResult struct {
	Status            SeriesStatus      `json:"status"`
	UnavailableReason UnavailableReason `json:"unavailable_reason"`
	Prometheus        string            `json:"prometheus,omitempty"`
}

// ProduceLiveSeries renders the existing fak_native_artifact_info contract for
// one fresh, complete Qwen3.8 Metal observation. The output has four fixed
// labels and at most MaxArtifacts samples; immutable revision and digest fields
// are validated but intentionally not exported as labels.
func ProduceLiveSeries(now time.Time, maxAge time.Duration, observation LiveObservation) LiveSeriesResult {
	unavailable := func(reason UnavailableReason) LiveSeriesResult {
		return LiveSeriesResult{Status: SeriesUnavailable, UnavailableReason: reason}
	}
	if now.IsZero() || maxAge <= 0 || observation.CompletedAt.IsZero() || now.Before(observation.CompletedAt) ||
		!moduleRevRE.MatchString(observation.ModuleRev) || observation.ModelFamily != "Qwen3.8" || observation.Backend != "metal" {
		return unavailable(ReasonIncomplete)
	}
	if now.Sub(observation.CompletedAt) > maxAge {
		return unavailable(ReasonStale)
	}

	record, err := scrub(observation.Record)
	if err != nil {
		switch {
		case errors.Is(err, ErrPrivate):
			return unavailable(ReasonPrivate)
		case errors.Is(err, ErrUntrustedScheme):
			return unavailable(ReasonUntrustedScheme)
		default:
			return unavailable(ReasonIncomplete)
		}
	}
	for _, artifact := range record.Artifacts {
		if !artifact.ExpiresAt.IsZero() && !now.Before(artifact.ExpiresAt) {
			return unavailable(ReasonStale)
		}
	}

	var b strings.Builder
	b.WriteString("# HELP fak_native_artifact_info Bounded public-safe native artifact evidence state.\n")
	b.WriteString("# TYPE fak_native_artifact_info gauge\n")
	for _, artifact := range record.Artifacts {
		fmt.Fprintf(&b, "fak_native_artifact_info{engine=\"fak-native\",correlation_key=%q,kind=%q,state=%q} 1\n",
			record.CorrelationKey, artifact.Kind, artifact.State)
	}
	return LiveSeriesResult{Status: SeriesProduced, UnavailableReason: ReasonNone, Prometheus: b.String()}
}

// Artifact is safe to serialize into a public index. Locator must be an HTTPS
// URL without credentials, query strings, fragments, or private path material.
type Artifact struct {
	Kind      Kind      `json:"kind"`
	Locator   string    `json:"locator,omitempty"`
	SHA256    string    `json:"sha256,omitempty"`
	ExpiresAt time.Time `json:"expires_at,omitempty"`
	State     State     `json:"state"`
}

// Record binds artifacts to one exact bounded key issued by nativeperfcorrelation.
type Record struct {
	Schema         string     `json:"schema"`
	CorrelationKey string     `json:"correlation_key"`
	Engine         string     `json:"engine"`
	Artifacts      []Artifact `json:"artifacts"`
}

type Index struct {
	mu       sync.RWMutex
	capacity int
	records  map[string]Record
	order    []string
}

func NewIndex(capacity int) (*Index, error) {
	if capacity <= 0 {
		return nil, errors.New("native performance artifact index capacity must be positive")
	}
	return &Index{capacity: capacity, records: make(map[string]Record, capacity)}, nil
}

func (i *Index) Add(record Record) error {
	record, err := scrub(record)
	if err != nil {
		return err
	}
	i.mu.Lock()
	defer i.mu.Unlock()
	if existing, exists := i.records[record.CorrelationKey]; exists {
		if !equalRecord(existing, record) {
			return errors.New("native performance artifact correlation key collision")
		}
		return nil
	}
	if len(i.order) == i.capacity {
		delete(i.records, i.order[0])
		i.order = i.order[1:]
	}
	i.order = append(i.order, record.CorrelationKey)
	i.records[record.CorrelationKey] = record
	return nil
}

// Resolve returns one exact artifact or an honest typed failure. It never
// redirects to another engine, correlation key, or artifact kind.
func (i *Index) Resolve(correlationKey string, kind Kind, now time.Time) (Artifact, error) {
	if err := validateCorrelationKey(correlationKey); err != nil {
		return Artifact{}, fmt.Errorf("%w: %v", ErrNotFound, err)
	}
	i.mu.RLock()
	record, ok := i.records[correlationKey]
	i.mu.RUnlock()
	if !ok {
		return Artifact{}, ErrNotFound
	}
	idx := sort.Search(len(record.Artifacts), func(n int) bool { return record.Artifacts[n].Kind >= kind })
	if idx == len(record.Artifacts) || record.Artifacts[idx].Kind != kind {
		return Artifact{}, ErrNotFound
	}
	artifact := record.Artifacts[idx]
	switch artifact.State {
	case StateReady:
		// Continue through expiry and locator validation below.
	case StateBroken:
		return Artifact{}, ErrBroken
	case StateRedacted:
		return Artifact{}, ErrRedacted
	}
	if !artifact.ExpiresAt.IsZero() && !now.Before(artifact.ExpiresAt) {
		return Artifact{}, ErrExpired
	}
	if err := validateLocator(artifact.Locator); err != nil {
		return Artifact{}, err
	}
	return artifact, nil
}

func (i *Index) Snapshot() []Record {
	i.mu.RLock()
	defer i.mu.RUnlock()
	out := make([]Record, 0, len(i.order))
	for _, key := range i.order {
		record := i.records[key]
		record.Artifacts = append([]Artifact(nil), record.Artifacts...)
		out = append(out, record)
	}
	return out
}

func scrub(record Record) (Record, error) {
	if err := validateCorrelationKey(record.CorrelationKey); err != nil {
		return Record{}, err
	}
	if record.Engine != nativeperfcorrelation.NativeEngine {
		return Record{}, errors.New("native performance artifact engine must be fak-native")
	}
	if len(record.Artifacts) == 0 || len(record.Artifacts) > MaxArtifacts {
		return Record{}, fmt.Errorf("native performance artifact count must be 1..%d", MaxArtifacts)
	}
	seen := make(map[Kind]struct{}, len(record.Artifacts))
	out := Record{Schema: Schema, CorrelationKey: record.CorrelationKey, Engine: record.Engine, Artifacts: append([]Artifact(nil), record.Artifacts...)}
	for n := range out.Artifacts {
		a := &out.Artifacts[n]
		if !validKind(a.Kind) {
			return Record{}, fmt.Errorf("unsupported native performance artifact kind %q", a.Kind)
		}
		if _, duplicate := seen[a.Kind]; duplicate {
			return Record{}, fmt.Errorf("duplicate native performance artifact kind %q", a.Kind)
		}
		seen[a.Kind] = struct{}{}
		if a.State == "" {
			a.State = StateReady
		}
		switch a.State {
		case StateReady:
			if len(a.SHA256) != 64 || !isLowerHex(a.SHA256) {
				return Record{}, errors.New("ready native performance artifact requires a lowercase sha256 digest")
			}
			if err := validateLocator(a.Locator); err != nil {
				return Record{}, err
			}
		case StateBroken:
			if a.Locator != "" || a.SHA256 != "" {
				return Record{}, errors.New("broken native performance artifact must not publish a locator or digest")
			}
		case StateRedacted:
			if a.Locator != "" || a.SHA256 != "" {
				return Record{}, errors.New("redacted native performance artifact must not publish a locator or digest")
			}
		default:
			return Record{}, fmt.Errorf("unsupported native performance artifact state %q", a.State)
		}
	}
	sort.Slice(out.Artifacts, func(a, b int) bool { return out.Artifacts[a].Kind < out.Artifacts[b].Kind })
	return out, nil
}

func validateCorrelationKey(key string) error {
	if len(key) != 37 || !strings.HasPrefix(key, "npc1_") {
		return errors.New("native performance correlation key must be a bounded npc1_ opaque key")
	}
	for _, r := range key[len("npc1_"):] {
		if !('0' <= r && r <= '9') && !('a' <= r && r <= 'f') {
			return errors.New("native performance correlation key contains invalid characters")
		}
	}
	return nil
}

func validateLocator(locator string) error {
	if locator == "" {
		return ErrBroken
	}
	u, err := url.Parse(locator)
	if err != nil || u.Scheme != "https" {
		return ErrUntrustedScheme
	}
	if u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return ErrPrivate
	}
	host := strings.ToLower(u.Hostname())
	if host == "" || host == "localhost" || strings.HasSuffix(host, ".local") || strings.HasSuffix(host, ".internal") {
		return ErrPrivate
	}
	if ip := net.ParseIP(host); ip != nil && (ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsUnspecified()) {
		return ErrPrivate
	}
	lowerPath := strings.ToLower(u.EscapedPath())
	for _, forbidden := range []string{"/private/", "/var/", "/tmp/", "/users/", "/home/", "credential", "token=", "raw-log"} {
		if strings.Contains(lowerPath, forbidden) {
			return ErrPrivate
		}
	}
	return nil
}

func validKind(kind Kind) bool {
	switch kind {
	case KindReceipt, KindMetalProfile, KindCUDAProfile, KindKernelTrace, KindComparisonReport:
		return true
	default:
		return false
	}
}

func isLowerHex(value string) bool {
	for _, r := range value {
		if !('0' <= r && r <= '9') && !('a' <= r && r <= 'f') {
			return false
		}
	}
	return true
}

func equalRecord(a, b Record) bool {
	if a.Schema != b.Schema || a.CorrelationKey != b.CorrelationKey || a.Engine != b.Engine || len(a.Artifacts) != len(b.Artifacts) {
		return false
	}
	for n := range a.Artifacts {
		if a.Artifacts[n] != b.Artifacts[n] {
			return false
		}
	}
	return true
}
