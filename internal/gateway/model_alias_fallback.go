package gateway

// model_alias_fallback.go — facade-level alias-variant fallback (ticket
// claude-alias-fallback, audit finding 2). A harness that pins one model across
// every tier can still emit variant spellings of that model — the `[1m]` / `-1m`
// extended-window markers Claude Code settings and vendor guides put on the model
// id — and a facade that forwards the id verbatim dead-ends the session: the id is
// a registered alias nowhere on the serving path, so the turn returns empty or the
// upstream refuses an id it does not serve (the audited DOA deaths).
//
// The facade has no data-driven alias table of its own (the router's routing.json
// owns that table); its known-alias universe is the ids it was configured to
// serve: the configured model, the planner's model, the roster's bound ids, and
// the modelreg equivalents of the configured/planner ids. The fallback is
// deliberately VARIANT-SCOPED: an id that carries a variant marker is
// resolved-or-refused, while every other id keeps the facade's existing
// pass-through contract — the client's label rides unchanged and need not name
// the served model (the Anthropic-wire adapter tests pin that contract).

import (
	"bytes"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const (
	// HeaderAliasFallback annotates a response whose requested model id was
	// rewritten by the alias-variant fallback. The value is
	// "<requested> -> <served>". It is set before the first response byte, so
	// both the buffered and the SSE surface carry it.
	HeaderAliasFallback = "X-Fak-Alias-Fallback"

	// ErrCodeAliasModelNotFound is the typed code the fallback fails closed
	// with: a variant-shaped id whose base resolves to no model this facade
	// serves. A client branches on the code instead of parsing prose.
	ErrCodeAliasModelNotFound = "model_not_found"
)

// aliasVariantMarkers are the variant spellings the fallback strips, tried
// repeatedly so a stacked id (`x[1m]-1m`) still resolves to `x`. Matching is
// case-insensitive (`[1M]` and `-1M` are the same markers — the DeepSeek
// Anthropic path already normalizes `[1m]` this way); the base keeps the case
// the client sent and is matched case-insensitively downstream.
var aliasVariantMarkers = []string{"[1m]", "-1m"}

// stripAliasVariantMarkers removes every trailing variant marker from id and
// reports whether at least one was removed. A bare marker id (`[1m]`) strips to
// "" — still reported as marker-shaped, so the caller fail-closes it instead of
// forwarding garbage downstream.
func stripAliasVariantMarkers(id string) (string, bool) {
	stripped := id
	any := false
	for {
		lower := strings.ToLower(stripped)
		cut := -1
		for _, m := range aliasVariantMarkers {
			if strings.HasSuffix(lower, m) {
				cut = len(m)
				break
			}
		}
		if cut < 0 {
			return stripped, any
		}
		stripped = strings.TrimSpace(stripped[:len(stripped)-cut])
		any = true
	}
}

// resolveAliasVariant is the pure core of the fallback: requested is the client's
// model label, canonical maps an id onto the exact model id the facade serves
// ("" when it names none). It returns the id to serve, whether a variant
// fallback was applied, and whether the label resolves at all:
//
//   - empty or a known id: served as sent, no fallback;
//   - an unknown id with NO variant marker: served as sent (the existing
//     pass-through contract — not this seam's concern);
//   - an unknown VARIANT-SHAPED id whose base resolves: served = the base's
//     canonical id, fallback applied;
//   - an unknown VARIANT-SHAPED id whose base resolves to nothing: ok=false —
//     the caller fail-closes with a typed error instead of letting the id ride
//     downstream and die there.
func resolveAliasVariant(requested string, canonical func(string) string) (served string, fell bool, ok bool) {
	requested = strings.TrimSpace(requested)
	if requested == "" {
		return requested, false, true
	}
	if canonical(requested) != "" {
		return requested, false, true
	}
	base, hadMarker := stripAliasVariantMarkers(requested)
	if !hadMarker {
		return requested, false, true
	}
	served = canonical(strings.TrimSpace(base))
	if served == "" {
		return requested, false, false
	}
	return served, true, true
}

// canonicalServedAlias maps id onto the exact model id this facade serves, or ""
// when it names none. Exact configured/planner/roster spellings win; a modelreg
// equivalent (two ids that load the same artifact, e.g. qwen38:27b and
// qwen38:27b-q4) resolves to the configured/planner id it equivalence-matches, so
// the rewrite target is always a spelling the facade itself put into service.
func (s *Server) canonicalServedAlias(id string) string {
	if s == nil {
		return ""
	}
	id = strings.TrimSpace(id)
	if id == "" {
		return ""
	}
	if served := strings.TrimSpace(s.model); served != "" && strings.EqualFold(id, served) {
		return served
	}
	if s.roster != nil {
		for _, b := range s.roster.Bindings {
			if served := strings.TrimSpace(b.Model); served != "" && strings.EqualFold(id, served) {
				return served
			}
		}
	}
	if s.planner != nil {
		if served := strings.TrimSpace(s.planner.Model()); served != "" && strings.EqualFold(id, served) {
			return served
		}
	}
	if served := strings.TrimSpace(s.model); served != "" && sameServedModel(id, served) {
		return served
	}
	if s.planner != nil {
		if served := strings.TrimSpace(s.planner.Model()); served != "" && sameServedModel(id, served) {
			return served
		}
	}
	return ""
}

// normalizeAnthropicRequestModel applies the alias-variant fallback to the
// request's model label in place, before anything binds to it: the conversation
// key, the roster/chat-route binding, the dual-planner routing, the turn echo,
// and the passthrough raw bytes all read req.Model downstream. A variant of a
// served alias resolves to its base and the response is annotated with
// HeaderAliasFallback; a variant-shaped id with no resolvable base fails closed
// with a typed model_not_found error here rather than riding downstream to die
// as an empty turn. Omitted and non-variant ids are untouched.
//
// When the fallback fires, req.Raw's top-level model value is spliced to the
// base too (spliceAnthropicModel) so the anthropic→anthropic passthrough — the
// one shape that forwards the client's bytes verbatim — carries the resolved id
// as well. The splice is fail-safe identity: on any ambiguity req.Raw keeps the
// client's bytes and only the decoded label is rewritten.
func (s *Server) normalizeAnthropicRequestModel(w http.ResponseWriter, req *agent.AnthropicMessagesRequest) bool {
	if req == nil {
		return true
	}
	requested := strings.TrimSpace(req.Model)
	if requested == "" {
		return true
	}
	served, fell, ok := resolveAliasVariant(requested, s.canonicalServedAlias)
	if !ok {
		base, _ := stripAliasVariantMarkers(requested)
		base = strings.TrimSpace(base)
		writeErrCodeFields(w, http.StatusBadRequest, ErrCodeAliasModelNotFound,
			fmt.Sprintf("requested model %q carries an alias-variant marker but its base %q resolves to no model this facade serves; drop the variant marker or name a served model", requested, base),
			map[string]any{"requested_model": requested, "resolved_base": base})
		return false
	}
	if !fell {
		return true
	}
	req.Model = served
	w.Header().Set(HeaderAliasFallback, requested+" -> "+served)
	if len(req.Raw) > 0 {
		if out, spliced := spliceAnthropicModel(req.Raw, served); spliced {
			req.Raw = out
		}
	}
	return true
}

// spliceAnthropicModel replaces the string value of the top-level "model" key in
// an Anthropic /v1/messages body with model, by a byte splice that touches ONLY
// that string — every other byte (and so the cache_control prefix) is preserved
// verbatim. Candidate key occurrences are tried in order and a splice ships only
// if the body re-decodes with model as its top-level model, so a "model" literal
// inside a client string or a nested schema can never be mistaken for the real
// key. It returns ok=false (caller leaves the body unchanged) when no occurrence
// qualifies — fail-safe identity, never a cache-busting rewrite.
func spliceAnthropicModel(raw []byte, model string) ([]byte, bool) {
	key := []byte(`"model"`)
	offset := 0
	for {
		ki := bytes.Index(raw[offset:], key)
		if ki < 0 {
			return nil, false
		}
		ki += offset
		offset = ki + 1
		if out, ok := spliceModelKeyAt(raw, ki, model); ok {
			return out, true
		}
	}
}

// spliceModelKeyAt tries one splice at the "model" key occurrence starting at ki.
func spliceModelKeyAt(raw []byte, ki int, model string) ([]byte, bool) {
	i := ki + len(`"model"`)
	for i < len(raw) && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r') {
		i++
	}
	if i >= len(raw) || raw[i] != ':' {
		return nil, false
	}
	i++
	for i < len(raw) && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r') {
		i++
	}
	if i >= len(raw) || raw[i] != '"' {
		return nil, false
	}
	i++
	start := i
	for i < len(raw) && raw[i] != '"' {
		if raw[i] == '\\' { // escaped byte: not a plain model id — bail
			return nil, false
		}
		i++
	}
	if i >= len(raw) {
		return nil, false
	}
	var b bytes.Buffer
	b.Grow(len(raw))
	b.Write(raw[:start])
	b.WriteString(model)
	b.Write(raw[i:])
	out := b.Bytes()
	dec, err := agent.DecodeAnthropicMessagesRequest(out)
	if err != nil || strings.TrimSpace(dec.Model) != model {
		return nil, false
	}
	return out, true
}
