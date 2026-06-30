package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

const (
	vcacheGovernorDecisionJournalSchema = "fak.vcache.governor-decision.v1"
	vcacheGovernorJournalRecentCap      = 1024
)

type vcacheGovernorDecisionRecord struct {
	Schema              string `json:"schema"`
	Seq                 uint64 `json:"seq"`
	TSUnixMillis        int64  `json:"ts_unix_millis"`
	Family              string `json:"family"`
	Decision            string `json:"decision"`
	ArrivalRatePerSec   string `json:"arrival_rate_per_sec"`
	Turns               int    `json:"turns"`
	CacheReadTokens     int64  `json:"cache_read_tokens"`
	CacheCreationTokens int64  `json:"cache_creation_tokens"`
	SavedTokenEquiv     string `json:"saved_token_equiv"`
	Status              string `json:"status"`
	PrevHash            string `json:"prev_hash"`
	Hash                string `json:"hash"`
}

type vcacheGovernorDecisionJournal struct {
	mu       sync.Mutex
	records  []vcacheGovernorDecisionRecord
	seq      uint64
	lastHash string
}

func newVCacheGovernorDecisionJournal() *vcacheGovernorDecisionJournal {
	return &vcacheGovernorDecisionJournal{}
}

func (j *vcacheGovernorDecisionJournal) record(tsUnixMillis int64, fam vcacheobserve.Family) vcacheGovernorDecisionRecord {
	if j == nil || fam.Key == "" {
		return vcacheGovernorDecisionRecord{}
	}
	j.mu.Lock()
	defer j.mu.Unlock()

	j.seq++
	r := vcacheGovernorDecisionRecord{
		Schema:              vcacheGovernorDecisionJournalSchema,
		Seq:                 j.seq,
		TSUnixMillis:        tsUnixMillis,
		Family:              fam.Key,
		Decision:            string(fam.GovernorDecision),
		ArrivalRatePerSec:   strconv.FormatFloat(fam.ArrivalRatePerSec, 'g', -1, 64),
		Turns:               fam.Turns,
		CacheReadTokens:     fam.CacheReadTokens,
		CacheCreationTokens: fam.CacheCreationTokens,
		SavedTokenEquiv:     strconv.FormatFloat(fam.Economics.SavedTokenEquiv, 'g', -1, 64),
		Status:              string(fam.Economics.Status),
		PrevHash:            j.lastHash,
	}
	r.Hash = hashVCacheGovernorDecision(r.PrevHash, r)
	j.lastHash = r.Hash
	j.records = append(j.records, r)
	if len(j.records) > vcacheGovernorJournalRecentCap {
		j.records = j.records[len(j.records)-vcacheGovernorJournalRecentCap:]
	}
	return r
}

func (j *vcacheGovernorDecisionJournal) recordsCopy() []vcacheGovernorDecisionRecord {
	if j == nil {
		return nil
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	out := make([]vcacheGovernorDecisionRecord, len(j.records))
	copy(out, j.records)
	return out
}

func (j *vcacheGovernorDecisionJournal) latestDecisionCounts() map[vcachegov.GovernorDecision]int {
	counts := make(map[vcachegov.GovernorDecision]int, len(vcacheGovernorDecisionOrder))
	for _, decision := range vcacheGovernorDecisionOrder {
		counts[decision] = 0
	}
	if j == nil {
		return counts
	}
	latest := map[string]vcachegov.GovernorDecision{}
	for _, r := range j.recordsCopy() {
		latest[r.Family] = vcachegov.GovernorDecision(r.Decision)
	}
	for _, decision := range latest {
		if _, ok := counts[decision]; ok {
			counts[decision]++
		}
	}
	return counts
}

func (m *gatewayMetrics) observeVCacheGovernorDecision(family string, tsUnixMillis int64, turns []vcacheobserve.Turn) vcacheGovernorDecisionRecord {
	if m == nil || m.vcacheGovernor == nil || family == "" || len(turns) == 0 || !vcacheWindowHasCacheActivity(turns) {
		return vcacheGovernorDecisionRecord{}
	}
	rep := vcacheobserve.Observe(turns, vcacheobserve.DefaultMultipliers())
	for _, fam := range rep.Families {
		if fam.Key == family {
			return m.vcacheGovernor.record(tsUnixMillis, fam)
		}
	}
	return vcacheGovernorDecisionRecord{}
}

func (m *gatewayMetrics) vcacheGovernorDecisionRecords() []vcacheGovernorDecisionRecord {
	if m == nil || m.vcacheGovernor == nil {
		return nil
	}
	return m.vcacheGovernor.recordsCopy()
}

func hashVCacheGovernorDecision(prev string, r vcacheGovernorDecisionRecord) string {
	h := sha256.New()
	fmt.Fprintf(h, "%s\x1f%s\x1f%d\x1f%d\x1f%s\x1f%s\x1f%s\x1f%d\x1f%d\x1f%d\x1f%s\x1f%s",
		prev,
		r.Schema,
		r.Seq,
		r.TSUnixMillis,
		r.Family,
		r.Decision,
		r.ArrivalRatePerSec,
		r.Turns,
		r.CacheReadTokens,
		r.CacheCreationTokens,
		r.SavedTokenEquiv,
		r.Status,
	)
	return hex.EncodeToString(h.Sum(nil))
}
