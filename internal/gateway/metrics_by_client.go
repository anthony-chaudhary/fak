package gateway

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// byClientLabel folds a perfSource client onto the closed perfledger vocabulary
// so the by-client families stay bounded at len(perfledger.Clients) series.
func byClientLabel(client string) string {
	if client == "" {
		return perfledger.ClientUnknown
	}
	for _, known := range perfledger.Clients {
		if client == known {
			return known
		}
	}
	return perfledger.ClientOther
}

// observeByClientLocked books one non-synthetic turn's uncached and cached
// prompt tokens against its client class. Caller holds inferenceMu.
func (m *gatewayMetrics) observeByClientLocked(src perfSource, promptTok, cachedTok int) {
	if src.synthetic {
		return
	}
	client := byClientLabel(src.client)
	if promptTok > 0 {
		if m.inferPromptTokensByClient == nil {
			m.inferPromptTokensByClient = map[string]uint64{}
		}
		m.inferPromptTokensByClient[client] += uint64(promptTok)
	}
	if cachedTok > 0 {
		if m.inferCachedTokensByClient == nil {
			m.inferCachedTokensByClient = map[string]uint64{}
		}
		m.inferCachedTokensByClient[client] += uint64(cachedTok)
	}
}

func copyClientCounts(src map[string]uint64) map[string]uint64 {
	out := make(map[string]uint64, len(src))
	for k, v := range src {
		out[k] = v
	}
	return out
}

// writeInferenceByClientMetrics renders every known client row (0 allowed) in
// perfledger.Clients order, so rate() sees each series from the first scrape.
func writeInferenceByClientMetrics(b *strings.Builder, snap inferenceSnapshot) {
	families := []struct {
		name, help string
		counts     map[string]uint64
	}{
		{"fak_gateway_inference_prompt_tokens_by_client_total", "Uncached prompt (input) tokens — the fak_gateway_inference_prompt_tokens_total axis — cut by the requesting harness (closed perfledger client vocabulary; empty client reads unknown). Excludes sender-marked synthetic turns (fak_gateway_inference_synthetic_turns_total), which the unlabeled total includes, so rows sum to at most the unlabeled family.", snap.promptTokByClient},
		{"fak_gateway_inference_cached_prompt_tokens_by_client_total", "Provider prompt-cache read tokens — the fak_gateway_inference_cached_prompt_tokens_total axis — cut by the requesting harness (closed perfledger client vocabulary; empty client reads unknown). Excludes sender-marked synthetic turns, which the unlabeled total includes. Per-harness cache hit rate = cached / (cached + uncached by_client).", snap.cachedTokByClient},
	}
	for _, fam := range families {
		writeHelpType(b, fam.name, fam.help, "counter")
		for _, client := range perfledger.Clients {
			fmt.Fprintf(b, "%s{client=\"%s\"} %d\n", fam.name, client, fam.counts[client])
		}
	}
}
