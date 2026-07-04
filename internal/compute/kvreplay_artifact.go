package compute

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

const KVReplayArtifactSchema = "fak.kvbm.replay/v1"

// KVReplayArtifact is a committed, model-free replay witness for the KVBM eviction policy.
// It keeps the validation shape close to the production contract: a fixed memory budget,
// replayed session touches, TTL-bounded pins, and payload bytes that must survive
// evict/restore without drift.
type KVReplayArtifact struct {
	Schema      string                  `json:"schema"`
	Name        string                  `json:"name"`
	BudgetBytes int64                   `json:"budget_bytes"`
	Events      []KVReplayArtifactEvent `json:"events"`
}

// KVReplayArtifactEvent records one prefix-span access in a replayed agent session.
type KVReplayArtifactEvent struct {
	Session      string `json:"session,omitempty"`
	SpanID       string `json:"span_id"`
	Tokens       int    `json:"tokens"`
	Bytes        int64  `json:"bytes,omitempty"`
	Payload      string `json:"payload"`
	PinTTLEvents int    `json:"pin_ttl_events,omitempty"`
}

// KVReplayArtifactReport compares LRU and cost-aware eviction on the same artifact.
type KVReplayArtifactReport struct {
	Name        string
	BudgetBytes int64
	LRU         KVReplayPolicyReport
	CostAware   KVReplayPolicyReport
}

// KVReplayPolicyReport is the per-policy validation evidence from one artifact replay.
type KVReplayPolicyReport struct {
	Policy              string
	HitTokens           int
	AccessTokens        int
	Evictions           int
	Restores            int
	PinnedSkips         int
	PinViolations       int
	BitDriftMismatches  int
	MaxResidentBytes    int64
	BlockedByActivePins int
}

// HitRate returns hitTokens/accessTokens as a convenience for callers that print reports.
func (r KVReplayPolicyReport) HitRate() float64 {
	if r.AccessTokens == 0 {
		return 0
	}
	return float64(r.HitTokens) / float64(r.AccessTokens)
}

func (r KVReplayArtifactReport) CostAwareAtLeastLRU() bool {
	return r.CostAware.HitTokens >= r.LRU.HitTokens
}

func (r KVReplayArtifactReport) PinViolations() int {
	return r.LRU.PinViolations + r.CostAware.PinViolations
}

func (r KVReplayArtifactReport) BitDriftMismatches() int {
	return r.LRU.BitDriftMismatches + r.CostAware.BitDriftMismatches
}

// ParseKVReplayArtifact parses and validates a replay artifact.
func ParseKVReplayArtifact(data []byte) (KVReplayArtifact, error) {
	var artifact KVReplayArtifact
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&artifact); err != nil {
		return KVReplayArtifact{}, err
	}
	if err := artifact.validate(); err != nil {
		return KVReplayArtifact{}, err
	}
	return artifact, nil
}

// ReplayKVArtifact runs the committed replay under both LRU and cost-aware policies.
func ReplayKVArtifact(artifact KVReplayArtifact) (KVReplayArtifactReport, error) {
	if err := artifact.validate(); err != nil {
		return KVReplayArtifactReport{}, err
	}
	lru := replayKVArtifactPolicy(artifact, KVEvictLRU)
	cost := replayKVArtifactPolicy(artifact, KVEvictCostAware)
	return KVReplayArtifactReport{
		Name:        artifact.Name,
		BudgetBytes: artifact.BudgetBytes,
		LRU:         lru,
		CostAware:   cost,
	}, nil
}

func (a KVReplayArtifact) validate() error {
	if a.Schema != KVReplayArtifactSchema {
		return fmt.Errorf("compute: replay artifact schema %q, want %q", a.Schema, KVReplayArtifactSchema)
	}
	if a.Name == "" {
		return errors.New("compute: replay artifact name is required")
	}
	if a.BudgetBytes < 0 {
		return errors.New("compute: replay artifact budget_bytes must be non-negative")
	}
	if len(a.Events) == 0 {
		return errors.New("compute: replay artifact must contain at least one event")
	}
	for i, ev := range a.Events {
		if ev.SpanID == "" {
			return fmt.Errorf("compute: replay event %d missing span_id", i)
		}
		if ev.Tokens <= 0 {
			return fmt.Errorf("compute: replay event %d has non-positive tokens %d", i, ev.Tokens)
		}
		if ev.Bytes < 0 {
			return fmt.Errorf("compute: replay event %d has negative bytes %d", i, ev.Bytes)
		}
		if ev.Payload == "" {
			return fmt.Errorf("compute: replay event %d missing payload bytes", i)
		}
		if ev.PinTTLEvents < 0 {
			return fmt.Errorf("compute: replay event %d has negative pin_ttl_events %d", i, ev.PinTTLEvents)
		}
	}
	return nil
}

type kvReplayResidentSpan struct {
	tokens     int
	bytes      int64
	hits       int
	lastUsed   uint64
	digest     string
	payload    string
	pinnedTill uint64
}

func replayKVArtifactPolicy(artifact KVReplayArtifact, policy KVEvictPolicy) KVReplayPolicyReport {
	report := KVReplayPolicyReport{Policy: kvReplayPolicyName(policy)}
	residentSpans := map[string]*kvReplayResidentSpan{}
	coldStore := map[string]kvReplayResidentSpan{}
	var residentBytes int64

	for i, ev := range artifact.Events {
		clock := uint64(i + 1)
		bytes := ev.Bytes
		if bytes == 0 {
			bytes = int64(ev.Tokens)
		}
		digest := kvReplayDigest(ev.Payload)
		report.AccessTokens += ev.Tokens

		if r, ok := residentSpans[ev.SpanID]; ok {
			if r.digest != digest || r.payload != ev.Payload {
				report.BitDriftMismatches++
			}
			r.hits++
			r.lastUsed = clock
			applyKVReplayPin(r, clock, ev.PinTTLEvents)
			report.HitTokens += ev.Tokens
			continue
		}

		if stored, ok := coldStore[ev.SpanID]; ok {
			report.Restores++
			if stored.digest != digest || stored.payload != ev.Payload || stored.tokens != ev.Tokens || stored.bytes != bytes {
				report.BitDriftMismatches++
			}
		}

		for artifact.BudgetBytes > 0 && residentBytes+bytes > artifact.BudgetBytes && len(residentSpans) > 0 {
			victimID, pinnedSkips := kvReplayVictim(residentSpans, policy, clock)
			report.PinnedSkips += pinnedSkips
			if victimID == "" {
				report.BlockedByActivePins++
				break
			}
			victim := residentSpans[victimID]
			if kvReplayPinActive(victim, clock) {
				report.PinViolations++
			}
			coldStore[victimID] = *victim
			residentBytes -= victim.bytes
			delete(residentSpans, victimID)
			report.Evictions++
		}

		residentSpans[ev.SpanID] = &kvReplayResidentSpan{
			tokens:   ev.Tokens,
			bytes:    bytes,
			lastUsed: clock,
			digest:   digest,
			payload:  ev.Payload,
		}
		applyKVReplayPin(residentSpans[ev.SpanID], clock, ev.PinTTLEvents)
		residentBytes += bytes
		if residentBytes > report.MaxResidentBytes {
			report.MaxResidentBytes = residentBytes
		}
	}
	return report
}

func kvReplayVictim(spans map[string]*kvReplayResidentSpan, policy KVEvictPolicy, clock uint64) (string, int) {
	victimID := ""
	var victimCost float64
	var victimLastUsed uint64
	pinnedSkips := 0
	for id, r := range spans {
		if kvReplayPinActive(r, clock) {
			pinnedSkips++
			continue
		}
		switch policy {
		case KVEvictLRU:
			if victimID == "" || r.lastUsed < victimLastUsed || (r.lastUsed == victimLastUsed && id < victimID) {
				victimID, victimLastUsed = id, r.lastUsed
			}
		case KVEvictCostAware:
			cost := KVEvictionCost(KVSpanStats{
				Tokens:   r.tokens,
				Bytes:    r.bytes,
				Hits:     r.hits,
				LastUsed: r.lastUsed,
			})
			switch {
			case victimID == "":
				victimID, victimCost, victimLastUsed = id, cost, r.lastUsed
			case cost < victimCost:
				victimID, victimCost, victimLastUsed = id, cost, r.lastUsed
			case cost == victimCost && (r.lastUsed < victimLastUsed || (r.lastUsed == victimLastUsed && id < victimID)):
				victimID, victimLastUsed = id, r.lastUsed
			}
		}
	}
	return victimID, pinnedSkips
}

func applyKVReplayPin(r *kvReplayResidentSpan, clock uint64, ttlEvents int) {
	if r == nil || ttlEvents <= 0 {
		return
	}
	until := clock + uint64(ttlEvents)
	if until > r.pinnedTill {
		r.pinnedTill = until
	}
}

func kvReplayPinActive(r *kvReplayResidentSpan, clock uint64) bool {
	return r != nil && r.pinnedTill >= clock
}

func kvReplayDigest(payload string) string {
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:])
}

func kvReplayPolicyName(policy KVEvictPolicy) string {
	switch policy {
	case KVEvictLRU:
		return "lru"
	case KVEvictCostAware:
		return "cost-aware"
	default:
		return "unknown"
	}
}
