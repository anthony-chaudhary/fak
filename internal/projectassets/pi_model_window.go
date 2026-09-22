package projectassets

// pi_model_windows.go — PER-MODEL served context windows for the Pi `fak` provider
// catalog. Before this file the Pi writers resolved ONE served window (whatever the
// backend's /v1/models last advertised, on this host the Qwen 131072) and applied the
// derived safe budget to EVERY model entry in the catalog. Because PiSafeContextBudget
// halves the window, a catalog whose last-observed window was the Qwen one advertised
// contextWindow = 65536 for models that are not Qwen at all — including
// `deepseek-ai/DeepSeek-V4.1-Flash`, whose real served window is ~1M. The router
// default for DeepSeek therefore reported 66k, which is the Qwen resident target, not a
// DeepSeek number.
//
// The fix is to resolve the served window PER MODEL. The routing-mode catalog is
// genuinely multi-model (router aliases resolve to different backends with different
// real windows), so a single flat window cannot be correct for more than one entry.
//
// Precedence for a model's served window:
//
//  1. An explicit per-model registry entry (exact id, then alias, then pattern) below.
//  2. The caller's explicit served window (the `--window` flag), which remains an
//     operator override for a model the registry does not name.
//  3. DefaultPiServedWindow, so an unknown model still degrades to the historical number.
//
// The operator-directed cap for DeepSeek V4.1 Flash is expressed by declaring its served
// window as PiOperatorDeepSeekResidentMax*2 (1000000), so the doctrine halving in
// PiSafeContextBudget yields a 500000 resident target. The model's real cap is ~1M; this
// table records the OPERATOR WINDOW, not a claim about the upstream's hard limit.

// PiOperatorDeepSeekResidentMax is the operator-directed resident maximum (tokens) for
// DeepSeek V4.1 Flash on the Pi `fak` provider. The upstream model's real window is ~1M;
// the operator pinned the Pi resident target to 500k explicitly rather than letting the
// catalog inherit the Qwen 131072 prior (which produced the 65536 report).
const PiOperatorDeepSeekResidentMax = 500000

// PiDeepSeekFlashServedWindow is the served window recorded for DeepSeek V4.1 Flash so
// that the doctrine halving lands exactly on PiOperatorDeepSeekResidentMax. It is written
// as twice the operator maximum so the two numbers cannot drift apart.
const PiDeepSeekFlashServedWindow = PiOperatorDeepSeekResidentMax * PiSafeWindowDenominator / PiSafeWindowNumerator

// piModelWindow is one registry row: the model identity plus the served window to assume
// for it.
type piModelWindow struct {
	match  func(string) bool
	window int
}

// piModelWindowRegistry is the per-model served-window table, evaluated in order. Exact
// ids come first, then provider-qualified and alias forms, then globs. It is deliberately
// small and explicit: a wrong entry silently resizes an agent's context, so an unnamed
// model falls through to the caller's window/default rather than being guessed at.
var piModelWindowRegistry = []piModelWindow{
	// DeepSeek V4.1 Flash: the router default on this host. Operator-directed resident
	// maximum of 500k (see PiOperatorDeepSeekResidentMax).
	{match: func(id string) bool {
		return piMatchModel(id, "deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-v41-flash", "deepseek-v4-flash")
	}, window: PiDeepSeekFlashServedWindow},
	// DeepSeek V4 Pro: same family, served by Nebius in the routing ladder.
	{match: func(id string) bool { return piMatchModel(id, "DeepSeek-V4-Pro-0813", "deepseek-v4-pro") }, window: PiDeepSeekFlashServedWindow},
	// GLM 5.3 family (Nebius/Modal rungs).
	{match: func(id string) bool { return piMatchModel(id, "zai-org/GLM-5.3-Flash", "glm-5.3-flash") }, window: 131072},
	// Qwen 3.8 27B on Strix Halo: the window the Halo llama.cpp backend advertises.
	{match: func(id string) bool {
		return piMatchModel(id, "Qwen3.8-27B-UD-Q2_K_XL", "qwen38", "qwen38:27b-q4", "qwen38:27b")
	}, window: DefaultPiServedWindow},
	// Any remaining qwen-* id is a Qwen-family route; keep the historical window.
	{match: func(id string) bool { return piHasModelPrefix(id, "qwen") }, window: DefaultPiServedWindow},
}

// piMatchModel reports whether id matches any of the given candidate ids, ignoring case
// and surrounding space. The last path segment of a provider-qualified id is also compared
// so `provider/model` and bare `model` resolve to the same registry row.
func piMatchModel(id string, candidates ...string) bool {
	id = normalizePiModelKey(id)
	if id == "" {
		return false
	}
	leaf := id
	if i := lastPathSegment(id); i != "" {
		leaf = i
	}
	for _, c := range candidates {
		key := normalizePiModelKey(c)
		if key == "" {
			continue
		}
		if id == key || leaf == normalizePiModelKey(lastPathSegmentOrSelf(key)) {
			return true
		}
	}
	return false
}

// piHasModelPrefix reports whether the id's leaf begins with prefix (case-insensitive).
func piHasModelPrefix(id, prefix string) bool {
	leaf := lastPathSegmentOrSelf(normalizePiModelKey(id))
	prefix = normalizePiModelKey(prefix)
	if leaf == "" || prefix == "" {
		return false
	}
	return len(leaf) >= len(prefix) && leaf[:len(prefix)] == prefix
}

// PiModelServedWindow resolves the served context window to assume for a model id.
//
// Precedence:
//  1. the per-model registry (exact id / alias / glob), so DeepSeek V4.1 Flash gets its
//     operator-directed window instead of inheriting the Qwen one;
//  2. explicitWindow when it is positive (the caller's `--window` override for a model the
//     registry does not name);
//  3. DefaultPiServedWindow for an unknown model or an unset override, preserving the
//     historical behaviour for ids fak has never heard of.
//
// A non-positive result is never returned: the floor is DefaultPiServedWindow.
func PiModelServedWindow(modelID string, explicitWindow int) int {
	if row, ok := lookupPiModelWindow(modelID); ok {
		// An explicit override wins for a model the operator named directly, but only when
		// it is at least as large as the registry window: a `--window` smaller than a
		// known-real window is an intentional conservative pin and is honored.
		if explicitWindow > 0 && explicitWindow < row {
			return explicitWindow
		}
		return row
	}
	if explicitWindow > 0 {
		return explicitWindow
	}
	return DefaultPiServedWindow
}

// lookupPiModelWindow returns the registry window for a model id, reporting whether a row
// matched. Matching is case-insensitive and tolerates a provider-qualified id.
func lookupPiModelWindow(modelID string) (int, bool) {
	key := normalizePiModelKey(modelID)
	if key == "" {
		return 0, false
	}
	for _, row := range piModelWindowRegistry {
		if row.match(modelID) {
			return row.window, true
		}
	}
	return 0, false
}

// PiModelContextBudget derives the safe Pi envelope for a specific model id using the
// per-model served window, so each catalog entry carries its own resident target instead
// of one flat number applied to every model in the catalog.
func PiModelContextBudget(modelID string, explicitWindow int) PiContextBudget {
	return PiSafeContextBudget(PiModelServedWindow(modelID, explicitWindow))
}

// normalizePiModelKey lowercases and trims a model id for registry comparison.
func normalizePiModelKey(raw string) string {
	out := make([]rune, 0, len(raw))
	started := false
	for _, r := range raw {
		if !started && (r == ' ' || r == '\t' || r == '\n' || r == '\r') {
			continue
		}
		started = true
		if r >= 'A' && r <= 'Z' {
			r += 'a' - 'A'
		}
		out = append(out, r)
	}
	// Trim trailing whitespace.
	for len(out) > 0 {
		last := out[len(out)-1]
		if last == ' ' || last == '\t' || last == '\n' || last == '\r' {
			out = out[:len(out)-1]
			continue
		}
		break
	}
	return string(out)
}

// lastPathSegment returns the substring after the final '/' in id, or "" when there is no
// slash.
func lastPathSegment(id string) string {
	for i := len(id) - 1; i >= 0; i-- {
		if id[i] == '/' {
			return id[i+1:]
		}
	}
	return ""
}

// lastPathSegmentOrSelf returns the last path segment of id, or id itself when it has none.
func lastPathSegmentOrSelf(id string) string {
	if leaf := lastPathSegment(id); leaf != "" {
		return leaf
	}
	return id
}
