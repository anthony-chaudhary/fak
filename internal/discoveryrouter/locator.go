package discoveryrouter

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/url"
	"sort"
	"strings"
	"sync"
	"time"
)

// LocatorRecord is the transcript-free, signed discovery pointer for one logical session.
type LocatorRecord struct {
	LogicalID  string    `json:"logical_id"`
	Generation uint64    `json:"generation"`
	Epoch      uint64    `json:"epoch"`
	Endpoints  []string  `json:"endpoints"`
	ExpiresAt  time.Time `json:"expires_at"`
	Revoked    bool      `json:"revoked"`
	Signature  string    `json:"signature"`
}

type locatorPayload struct {
	LogicalID  string    `json:"logical_id"`
	Generation uint64    `json:"generation"`
	Epoch      uint64    `json:"epoch"`
	Endpoints  []string  `json:"endpoints"`
	ExpiresAt  time.Time `json:"expires_at"`
	Revoked    bool      `json:"revoked"`
}

func (r LocatorRecord) payload() ([]byte, error) {
	return json.Marshal(locatorPayload{r.LogicalID, r.Generation, r.Epoch, r.Endpoints, r.ExpiresAt.UTC(), r.Revoked})
}

func SignLocator(r *LocatorRecord, key ed25519.PrivateKey) error {
	if err := validateLocator(*r); err != nil {
		return err
	}
	payload, err := r.payload()
	if err != nil {
		return err
	}
	r.Signature = base64.RawURLEncoding.EncodeToString(ed25519.Sign(key, payload))
	return nil
}

func VerifyLocator(r LocatorRecord, key ed25519.PublicKey) error {
	if err := validateLocator(r); err != nil {
		return err
	}
	sig, err := base64.RawURLEncoding.DecodeString(r.Signature)
	if err != nil {
		return fmt.Errorf("locator signature: %w", err)
	}
	payload, err := r.payload()
	if err != nil {
		return err
	}
	if !ed25519.Verify(key, payload, sig) {
		return errors.New("locator signature is not trusted")
	}
	return nil
}

func validateLocator(r LocatorRecord) error {
	if strings.TrimSpace(r.LogicalID) == "" || r.Generation == 0 || r.Epoch == 0 || r.ExpiresAt.IsZero() {
		return errors.New("locator record is incomplete")
	}
	if len(r.Endpoints) == 0 && !r.Revoked {
		return errors.New("active locator has no endpoint")
	}
	for _, raw := range r.Endpoints {
		u, err := url.Parse(raw)
		if err != nil || u.Scheme != "https" || u.User != nil || u.Hostname() == "" {
			return fmt.Errorf("locator endpoint %q is not a public HTTPS URL", raw)
		}
		host := strings.ToLower(u.Hostname())
		ip := net.ParseIP(host)
		if host == "localhost" || strings.HasSuffix(host, ".local") || (ip != nil && (ip.IsPrivate() || ip.IsLoopback() || ip.IsLinkLocalUnicast())) {
			return fmt.Errorf("locator endpoint %q exposes a private host", raw)
		}
	}
	return nil
}

type LocatorBackend interface {
	Name() string
	Lookup(context.Context, string) ([]LocatorRecord, error)
	Put(context.Context, LocatorRecord) error
}

type ResolveCode string

const (
	Resolved    ResolveCode = "resolved"
	NotFound    ResolveCode = "not_found"
	Unreachable ResolveCode = "unreachable"
	SplitBrain  ResolveCode = "split_brain"
	TooOld      ResolveCode = "too_old"
	Revoked     ResolveCode = "revoked"
	Expired     ResolveCode = "expired"
)

type ResolveError struct {
	Code     ResolveCode
	Recovery string
}

func (e *ResolveError) Error() string { return string(e.Code) + ": " + e.Recovery }

type Resolver struct {
	Backends    []LocatorBackend
	TrustedKeys map[string]ed25519.PublicKey
	Now         func() time.Time
	observed    map[string]version
	mu          sync.Mutex
}
type version struct{ epoch, generation uint64 }

func versionOf(r LocatorRecord) version { return version{r.Epoch, r.Generation} }
func (v version) less(o version) bool {
	return v.epoch < o.epoch || (v.epoch == o.epoch && v.generation < o.generation)
}

func (r *Resolver) Resolve(ctx context.Context, logicalID string) (LocatorRecord, error) {
	var valid []LocatorRecord
	reachable := 0
	key := r.TrustedKeys[logicalID]
	for _, b := range r.Backends {
		records, err := b.Lookup(ctx, logicalID)
		if err != nil {
			continue
		}
		reachable++
		for _, record := range records {
			if record.LogicalID == logicalID && VerifyLocator(record, key) == nil {
				valid = append(valid, record)
			}
		}
	}
	if reachable == 0 {
		return LocatorRecord{}, &ResolveError{Unreachable, "retry another locator backend"}
	}
	if len(valid) == 0 {
		return LocatorRecord{}, &ResolveError{NotFound, "republish the logical session locator"}
	}
	sort.Slice(valid, func(i, j int) bool { return versionOf(valid[j]).less(versionOf(valid[i])) })
	best := valid[0]
	bestPayload, _ := best.payload()
	for _, candidate := range valid[1:] {
		if versionOf(candidate) != versionOf(best) {
			break
		}
		p, _ := candidate.payload()
		if string(p) != string(bestPayload) {
			return LocatorRecord{}, &ResolveError{SplitBrain, "quarantine conflicting replicas and republish a higher generation"}
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.observed == nil {
		r.observed = make(map[string]version)
	}
	if versionOf(best).less(r.observed[logicalID]) {
		return LocatorRecord{}, &ResolveError{TooOld, "query a current replica; do not attach"}
	}
	r.observed[logicalID] = versionOf(best)
	if best.Revoked {
		return LocatorRecord{}, &ResolveError{Revoked, "start a newly authorized logical session"}
	}
	now := time.Now()
	if r.Now != nil {
		now = r.Now()
	}
	if !now.Before(best.ExpiresAt) {
		return LocatorRecord{}, &ResolveError{Expired, "republish a fresh locator generation"}
	}
	return best, nil
}

// MemoryLocator is a deterministic backend for local registries and replicated locator fixtures.
type MemoryLocator struct {
	BackendName string
	Err         error
	mu          sync.RWMutex
	records     map[string][]LocatorRecord
}

func (m *MemoryLocator) Name() string { return m.BackendName }
func (m *MemoryLocator) Lookup(_ context.Context, id string) ([]LocatorRecord, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.Err != nil {
		return nil, m.Err
	}
	return append([]LocatorRecord(nil), m.records[id]...), nil
}
func (m *MemoryLocator) Put(_ context.Context, record LocatorRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.Err != nil {
		return m.Err
	}
	if m.records == nil {
		m.records = make(map[string][]LocatorRecord)
	}
	for _, old := range m.records[record.LogicalID] {
		if !versionOf(old).less(versionOf(record)) {
			return errors.New("locator publication is not monotonic")
		}
	}
	m.records[record.LogicalID] = append(m.records[record.LogicalID], record)
	return nil
}
