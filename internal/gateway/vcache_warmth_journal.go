package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

const (
	vcacheWarmthDemotionJournalSchema = "fak.vcache.warmth-demotion.v1"
	vcacheWarmthDemotionJournalCap    = 1024
)

type vcacheWarmthDemotionRecord struct {
	Schema              string `json:"schema"`
	Seq                 uint64 `json:"seq"`
	TSUnixMillis        int64  `json:"ts_unix_millis"`
	Family              string `json:"family"`
	Action              string `json:"action"`
	Reason              string `json:"reason"`
	InputTokens         int64  `json:"input_tokens"`
	CacheReadTokens     int64  `json:"cache_read_tokens"`
	CacheCreationTokens int64  `json:"cache_creation_tokens"`
	PredictedWarm       bool   `json:"predicted_warm"`
	ActualWarm          bool   `json:"actual_warm"`
	StateBefore         string `json:"state_before"`
	StateAfter          string `json:"state_after"`
	DivergenceProbe     string `json:"divergence_probe"`
	PrevHash            string `json:"prev_hash"`
	Hash                string `json:"hash"`
}

type vcacheWarmthDemotionObservation struct {
	Family              string
	TSUnixMillis        int64
	InputTokens         int64
	CacheReadTokens     int64
	CacheCreationTokens int64
	StateBefore         cachemeta.EntryState
	StateAfter          cachemeta.EntryState
}

type vcacheWarmthDemotionJournal struct {
	mu       sync.Mutex
	records  []vcacheWarmthDemotionRecord
	seq      uint64
	lastHash string
}

func newVCacheWarmthDemotionJournal() *vcacheWarmthDemotionJournal {
	return &vcacheWarmthDemotionJournal{}
}

func (j *vcacheWarmthDemotionJournal) record(obs vcacheWarmthDemotionObservation) vcacheWarmthDemotionRecord {
	if j == nil || strings.TrimSpace(obs.Family) == "" {
		return vcacheWarmthDemotionRecord{}
	}
	j.mu.Lock()
	defer j.mu.Unlock()

	j.seq++
	r := vcacheWarmthDemotionRecord{
		Schema:              vcacheWarmthDemotionJournalSchema,
		Seq:                 j.seq,
		TSUnixMillis:        obs.TSUnixMillis,
		Family:              obs.Family,
		Action:              "mark_belief_cold",
		Reason:              string(vcachecal.FalseWarm),
		InputTokens:         obs.InputTokens,
		CacheReadTokens:     obs.CacheReadTokens,
		CacheCreationTokens: obs.CacheCreationTokens,
		PredictedWarm:       true,
		ActualWarm:          false,
		StateBefore:         string(obs.StateBefore),
		StateAfter:          string(obs.StateAfter),
		DivergenceProbe:     "not_wired",
		PrevHash:            j.lastHash,
	}
	r.Hash = hashVCacheWarmthDemotion(r.PrevHash, r)
	j.lastHash = r.Hash
	j.records = append(j.records, r)
	if len(j.records) > vcacheWarmthDemotionJournalCap {
		j.records = j.records[len(j.records)-vcacheWarmthDemotionJournalCap:]
	}
	return r
}

func (j *vcacheWarmthDemotionJournal) recordsCopy() []vcacheWarmthDemotionRecord {
	if j == nil {
		return nil
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	out := make([]vcacheWarmthDemotionRecord, len(j.records))
	copy(out, j.records)
	return out
}

func (j *vcacheWarmthDemotionJournal) count() uint64 {
	if j == nil {
		return 0
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.seq
}

func (m *gatewayMetrics) observeVCacheWarmthDemotion(family string, tsUnixMillis int64, turns []vcacheobserve.Turn) vcacheWarmthDemotionRecord {
	if m == nil || m.vcacheWarmth == nil || family == "" || len(turns) == 0 || !vcacheWindowHasCacheActivity(turns) {
		return vcacheWarmthDemotionRecord{}
	}
	obs, ok := vcacheWarmthDemotionForLatestTurn(family, tsUnixMillis, turns)
	if !ok {
		return vcacheWarmthDemotionRecord{}
	}
	return m.vcacheWarmth.record(obs)
}

func (m *gatewayMetrics) vcacheWarmthDemotionRecords() []vcacheWarmthDemotionRecord {
	if m == nil || m.vcacheWarmth == nil {
		return nil
	}
	return m.vcacheWarmth.recordsCopy()
}

func (m *gatewayMetrics) writeVCacheWarmthDemotionMetrics(b *strings.Builder) {
	if m == nil || m.vcacheWarmth == nil {
		return
	}
	n := m.vcacheWarmth.count()
	if n == 0 {
		return
	}
	writeCounter(b, "fak_vcache_warmth_demotions_total", "DECISION (M1 warmth belief): cumulative false-warm events where the live gateway marked the reconstructed provider-cache belief cold after a believed-warm turn returned cache_read=0. This is an alarm/demotion witness only; correctness and request content never depend on warmth.", int64(n))
}

func vcacheWarmthDemotionForLatestTurn(family string, tsUnixMillis int64, turns []vcacheobserve.Turn) (vcacheWarmthDemotionObservation, bool) {
	ft := make([]vcacheobserve.Turn, 0)
	for _, t := range turns {
		if t.Family == family {
			ft = append(ft, t)
		}
	}
	if len(ft) == 0 {
		return vcacheWarmthDemotionObservation{}, false
	}
	sort.SliceStable(ft, func(i, j int) bool { return ft[i].UnixMillis < ft[j].UnixMillis })
	if ft[len(ft)-1].UnixMillis != tsUnixMillis {
		return vcacheWarmthDemotionObservation{}, false
	}

	policy := vcachecal.DefaultBeliefPolicy()
	var belief vcachecal.Belief
	started := false
	for i, t := range ft {
		if started {
			belief = belief.Advance(policy, t.UnixMillis)
		}
		before := belief.Lifecycle.State
		var outcome vcachecal.PredictionOutcome
		belief, outcome = belief.Observe(policy, t.UnixMillis, t.CacheRead)
		if i == len(ft)-1 && outcome.Class() == vcachecal.FalseWarm {
			return vcacheWarmthDemotionObservation{
				Family:              family,
				TSUnixMillis:        t.UnixMillis,
				InputTokens:         t.InputTokens,
				CacheReadTokens:     t.CacheRead,
				CacheCreationTokens: t.CacheCreation,
				StateBefore:         before,
				StateAfter:          belief.Lifecycle.State,
			}, true
		}
		started = true
	}
	return vcacheWarmthDemotionObservation{}, false
}

func hashVCacheWarmthDemotion(prev string, r vcacheWarmthDemotionRecord) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\x1f%s\x1f%d\x1f%d\x1f%s\x1f%s\x1f%s\x1f%d\x1f%d\x1f%d\x1f%t\x1f%t\x1f%s\x1f%s\x1f%s",
		prev,
		r.Schema,
		r.Seq,
		r.TSUnixMillis,
		r.Family,
		r.Action,
		r.Reason,
		r.InputTokens,
		r.CacheReadTokens,
		r.CacheCreationTokens,
		r.PredictedWarm,
		r.ActualWarm,
		r.StateBefore,
		r.StateAfter,
		r.DivergenceProbe,
	)
	return hex.EncodeToString(h.Sum(nil))
}
