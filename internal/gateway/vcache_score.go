package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcachescore"
)

// handleFakVCacheScore serves the same scorecard contract as `fak vcache score`.
// It is read-only: provider counters come from the gateway metrics roll-up, while
// pure-fak KV activation comes from the in-kernel cache observer.
func (s *Server) handleFakVCacheScore(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.vcacheScoreReport())
}

func (s *Server) vcacheScoreReport() vcachescore.Report {
	in := vcachescore.DefaultInput()
	var sum AdjudicationSummary
	if s != nil {
		sum = s.AdjudicationSummary()
		if sum.InputTokens > 0 || sum.CachedPromptTokens > 0 || sum.CacheCreationTokens > 0 {
			in.TelemetryRows = []vcachegov.TelemetryRow{{
				InputTokens:              float64(sum.InputTokens),
				CacheReadInputTokens:     float64(sum.CachedPromptTokens),
				CacheCreationInputTokens: float64(sum.CacheCreationTokens),
			}}
		}
	}
	if ctxEvidence, events, ok := contextPlaneEvidence(sum); ok {
		in.Context = ctxEvidence
		if in.AgenticActivation.ContextEvents == 0 {
			in.AgenticActivation.ContextEvents = events
		}
	}
	kv := cacheobs.Default.Snapshot()
	if kv.PromptTokens > 0 {
		if kv.ReusedTokens > 0 {
			in.AgenticActivation.KernelKVEvents = uint64ToInt(kv.FrozenTurns + kv.PartialTurns)
		}
		in.KernelKV = vcachescore.PlaneEvidenceInput{
			Available:          true,
			BaselineTokenEquiv: float64(kv.PromptTokens),
			SavedTokenEquiv:    float64(kv.ReusedTokens),
			CostTokenEquiv:     float64(kv.PromptTokens - kv.ReusedTokens),
			Reason:             "in-kernel KV-prefix reuse observed by gateway cacheobs",
		}
	}
	if ext, ok := s.externalEngineCacheEvidence(); ok {
		in.ExternalEngine = ext
	}
	return vcachescore.Score(in)
}

func contextPlaneEvidence(sum AdjudicationSummary) (vcachescore.PlaneEvidenceInput, int, bool) {
	if sum.CompactionFired == 0 && sum.CompactionShedTokens == 0 {
		return vcachescore.PlaneEvidenceInput{}, 0, false
	}
	ev := vcachescore.PlaneEvidenceInput{
		Available:       true,
		Provenance:      "WITNESSED",
		SavedTokenEquiv: float64(sum.CompactionShedTokens),
		Reason: fmt.Sprintf(
			"O(1) context/history compaction witnessed %d fired attempt(s), shed %d token(s), dropped %d turn(s)",
			sum.CompactionFired,
			sum.CompactionShedTokens,
			sum.CompactionDroppedTurns,
		),
	}
	if sum.CompactionBudget > 0 && sum.CompactionFired > 0 && sum.CompactionShedTokens > 0 {
		resident := float64(sum.CompactionBudget) * float64(sum.CompactionFired)
		ev.CostTokenEquiv = resident
		ev.BaselineTokenEquiv = resident + float64(sum.CompactionShedTokens)
	}
	return ev, uint64ToInt(sum.CompactionFired), true
}

func (s *Server) externalEngineCacheEvidence() (vcachescore.PlaneEvidenceInput, bool) {
	if s == nil || s.metrics == nil {
		return vcachescore.PlaneEvidenceInput{}, false
	}
	rows := s.metrics.servingEmitterRows()
	if len(rows) == 0 {
		return vcachescore.PlaneEvidenceInput{}, false
	}
	sum := 0.0
	count := 0
	engines := map[string]struct{}{}
	for _, row := range rows {
		if !row.PrefixCacheHitRate.Set {
			continue
		}
		sum += clamp01(row.PrefixCacheHitRate.Value)
		count++
		engine := strings.TrimSpace(row.Labels.Engine)
		if engine == "" {
			engine = "external"
		}
		engines[engine] = struct{}{}
	}
	if count == 0 {
		return vcachescore.PlaneEvidenceInput{}, false
	}
	return vcachescore.PlaneEvidenceInput{
		Available:  true,
		Provenance: "OBSERVED",
		HitRate:    sum / float64(count),
		Reason:     fmt.Sprintf("external serving prefix-cache hit rate observed from %d row(s): %s", count, joinedSet(engines)),
	}, true
}

func uint64ToInt(n uint64) int {
	maxInt := int(^uint(0) >> 1)
	if n > uint64(maxInt) {
		return maxInt
	}
	return int(n)
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func joinedSet(values map[string]struct{}) string {
	if len(values) == 0 {
		return "external"
	}
	out := make([]string, 0, len(values))
	for v := range values {
		out = append(out, v)
	}
	sortStrings(out)
	return strings.Join(out, ",")
}

func sortStrings(values []string) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j] < values[j-1]; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}
