package gateway

import (
	"encoding/json"
	"net/http"
	"sort"
	"strings"
	"time"
)

// ---------------------------------------------------------------------------
// OpenAI-compatible /v1/moderations.
//
// As with embeddings, the gateway implements the moderations SURFACE (request
// shape, the full OpenAI category vocabulary, per-item batch results) with a
// deterministic, self-contained backend: a lexical classifier that scans each
// input for category keywords. It is an honest baseline — NOT a learned safety
// model — so a result is explainable, reproducible, and works on-host with no GPU
// or network. A stock OpenAI client parses the response unchanged. Swapping the
// lexical scorer for a learned classifier is an engine/model-lane concern behind
// the same wire.
// ---------------------------------------------------------------------------

// moderationFlagThreshold is the category score at or above which an input is
// flagged. Each keyword hit contributes moderationHitWeight; two hits in a
// category clear the bar.
const (
	moderationFlagThreshold = 0.5
	moderationHitWeight     = 0.34
)

// ModerationsRequest is the inbound POST /v1/moderations body. Input is raw to
// accept either a bare string or an array of strings (batch) — the same wire
// polymorphism as embeddings. The optional model field is echoed back.
type ModerationsRequest struct {
	Model string          `json:"model,omitempty"`
	Input json.RawMessage `json:"input"`
}

// ModerationsResponse is the OpenAI-compatible envelope: one ModerationResult per
// input item, in request order (per-item batch results).
type ModerationsResponse struct {
	ID      string             `json:"id"`
	Model   string             `json:"model"`
	Results []ModerationResult `json:"results"`
}

// ModerationResult is one input's classification. Categories is the boolean
// per-category flag map; CategoryScores is the [0,1] score map. Both carry the
// FULL OpenAI category key set (scores default to 0) so a client keying on a
// specific category never reads a missing field.
type ModerationResult struct {
	Flagged        bool               `json:"flagged"`
	Categories     map[string]bool    `json:"categories"`
	CategoryScores map[string]float64 `json:"category_scores"`
}

// moderationCategory pairs a stable OpenAI category key with the lexicon that
// drives its deterministic score.
type moderationCategory struct {
	key      string
	keywords []string
}

// moderationLexicon is the deterministic keyword backend. It declares the full
// OpenAI category vocabulary; a category with no keywords always scores 0 but is
// still present in the response so the shape is complete. The lists are a modest,
// explainable baseline — not an exhaustive safety classifier.
var moderationLexicon = []moderationCategory{
	{"sexual", []string{"porn", "explicit sex", "nsfw"}},
	{"hate", []string{"racial slur", "subhuman", "ethnic cleansing"}},
	{"harassment", []string{"i will hurt you", "worthless idiot", "shut up loser"}},
	{"self-harm", []string{"kill myself", "suicide", "self harm", "cut myself"}},
	{"sexual/minors", []string{"child porn", "underage sex"}},
	{"hate/threatening", []string{"exterminate them", "kill all"}},
	{"violence/graphic", []string{"dismember", "decapitate", "blood everywhere"}},
	{"self-harm/intent", []string{"i want to die", "going to end it"}},
	{"self-harm/instructions", []string{"how to overdose", "ways to self harm"}},
	{"harassment/threatening", []string{"i will find you", "you will pay"}},
	{"violence", []string{"kill him", "shoot them", "stab", "bomb the"}},
}

// handleModerations serves POST /v1/moderations. Self-contained (no model
// round-trip): each input is scored against the lexical backend and returned as a
// per-item result, so a batched (array) request yields one classification per
// item in order.
func (s *Server) handleModerations(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	var req ModerationsRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return
	}
	items, ok := normalizeTextInput(req.Input)
	if !ok || len(items) == 0 {
		writeErr(w, http.StatusBadRequest, "moderations requires a non-empty `input` (string or array of strings)")
		return
	}
	if len(items) > embedMaxItems {
		writeErr(w, http.StatusBadRequest, "moderations `input` exceeds the per-request batch limit")
		return
	}

	results := make([]ModerationResult, len(items))
	for i, item := range items {
		results[i] = moderate(item)
	}

	model := strings.TrimSpace(req.Model)
	if model == "" {
		model = "fak-moderation-lexical"
	}
	writeJSON(w, http.StatusOK, ModerationsResponse{
		ID:      "modr-fak-" + itoa(uint64(time.Now().UnixNano())),
		Model:   model,
		Results: results,
	})
}

// moderate scores one input string against the lexical backend and folds the
// per-category scores into the OpenAI result shape. flagged is true iff ANY
// category reaches the threshold.
func moderate(input string) ModerationResult {
	hay := strings.ToLower(input)
	categories := make(map[string]bool, len(moderationLexicon))
	scores := make(map[string]float64, len(moderationLexicon))
	flagged := false
	for _, cat := range moderationLexicon {
		score := categoryScore(hay, cat.keywords)
		scores[cat.key] = score
		hit := score >= moderationFlagThreshold
		categories[cat.key] = hit
		if hit {
			flagged = true
		}
	}
	return ModerationResult{Flagged: flagged, Categories: categories, CategoryScores: scores}
}

// categoryScore returns a deterministic [0,1] score: each distinct matched keyword
// adds moderationHitWeight, capped at 1.0. No match scores 0.
func categoryScore(hay string, keywords []string) float64 {
	hits := 0
	for _, kw := range keywords {
		if strings.Contains(hay, kw) {
			hits++
		}
	}
	score := float64(hits) * moderationHitWeight
	if score > 1.0 {
		score = 1.0
	}
	return score
}

// moderationCategoryKeys returns the stable, sorted category vocabulary — a
// helper for tests asserting the response shape is complete.
func moderationCategoryKeys() []string {
	keys := make([]string, 0, len(moderationLexicon))
	for _, c := range moderationLexicon {
		keys = append(keys, c.key)
	}
	sort.Strings(keys)
	return keys
}
