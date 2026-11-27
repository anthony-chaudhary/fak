package gateway

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/kvbudget"
)

const (
	HeaderContextTier   = "X-Fak-Context-Tier"
	HeaderTierCapTokens = "X-Fak-Tier-Cap-Tokens"
	HeaderAvailTiers    = "X-Fak-Available-Tiers"
)

func (s *Server) handleFakCapacityTiers(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	if s == nil {
		writeJSON(w, http.StatusOK, struct {
			Enabled bool `json:"enabled"`
		}{Enabled: false})
		return
	}
	s.admissionMu.RLock()
	ctl := s.admissionCtl
	s.admissionMu.RUnlock()

	if ctl == nil {
		writeJSON(w, http.StatusOK, struct {
			Enabled bool `json:"enabled"`
		}{Enabled: false})
		return
	}
	snap := ctl.TierPoolSnapshot()
	if !snap.Enabled {
		writeJSON(w, http.StatusOK, struct {
			Enabled bool `json:"enabled"`
		}{Enabled: false})
		return
	}
	writeJSON(w, http.StatusOK, snap)
}

// FormatAvailableTiersHeader formats tier availability into a compact header
// string like "master=0@131072,worker=4@32768,micro=32@4096", where the count
// is each tier's AvailableNow (guaranteed floor remaining plus elastic borrow
// headroom).
func FormatAvailableTiersHeader(tiers []kvbudget.TierStatus) string {
	if len(tiers) == 0 {
		return ""
	}
	parts := make([]string, 0, len(tiers))
	for _, t := range tiers {
		parts = append(parts, fmt.Sprintf("%s=%d@%d", t.Name, t.AvailableNow, t.MaxContext))
	}
	return strings.Join(parts, ",")
}

// WriteTierHeaders sets X-Fak-Context-Tier, X-Fak-Tier-Cap-Tokens, and
// X-Fak-Available-Tiers on an HTTP response.
func WriteTierHeaders(w http.ResponseWriter, snap kvbudget.TierPoolSnapshot, assignedTier string, capTokens int) {
	if w == nil {
		return
	}
	if assignedTier != "" {
		w.Header().Set(HeaderContextTier, assignedTier)
	}
	if capTokens > 0 {
		w.Header().Set(HeaderTierCapTokens, strconv.Itoa(capTokens))
	}
	if avail := FormatAvailableTiersHeader(snap.Tiers); avail != "" {
		w.Header().Set(HeaderAvailTiers, avail)
	}
}

// WriteTierShedResponse sets X-Fak-Available-Tiers and emits an HTTP 429 JSON
// response carrying structured downgrade/compaction advice.
func WriteTierShedResponse(w http.ResponseWriter, advice TierAdvice) {
	if w == nil {
		return
	}
	if avail := FormatAvailableTiersHeader(advice.AvailableTiers); avail != "" {
		w.Header().Set(HeaderAvailTiers, avail)
	}
	writeJSON(w, http.StatusTooManyRequests, map[string]any{
		"error":   advice.Code,
		"message": advice.Message,
		"advice":  advice,
	})
}
