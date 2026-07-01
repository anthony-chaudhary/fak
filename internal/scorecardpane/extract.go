package scorecardpane

// extract.go — the tolerant payload-walk helpers ported from the Python tool's
// find_int / find_grade / find_score, with find_value added for the continuous
// scorecard migration. Each scorecard nests its debt integer under
// corpus.<debt> (most) or doc.<debt> (doc-appeal); these searches keep the fold
// from caring which nesting a given scorecard chose. The payloads are decoded with
// encoding/json into map[string]any, so numbers arrive as float64 — asInt rejects
// non-integral floats and booleans, matching the Python isinstance(int) guard.

// asInt returns the value as an int when it is an integral JSON number (not a
// bool). JSON has no int type, so json.Unmarshal yields float64; an integral
// float64 is the int the Python contract stored.
func asInt(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		i := int(n)
		if float64(i) == n {
			return i, true
		}
	case int:
		return n, true
	}
	return 0, false
}

// asFloat returns the value as a float64 when it is a JSON number (not a bool).
func asFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	}
	return 0, false
}

func asBool(v any) bool {
	b, ok := v.(bool)
	return ok && b
}

func asString(v any) string {
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return s
}

// findInt is the first int value stored under key anywhere in the payload, with the
// corpus/doc nests preferred (matching the Python find_int precedence: try the two
// nests, then the top level, then a deep walk).
func findInt(payload any, key string) *int {
	switch p := payload.(type) {
	case map[string]any:
		for _, nest := range []string{"corpus", "doc"} {
			if sub, ok := p[nest].(map[string]any); ok {
				if i, ok := asInt(sub[key]); ok {
					return &i
				}
			}
		}
		if i, ok := asInt(p[key]); ok {
			return &i
		}
		for _, v := range p {
			if got := findInt(v, key); got != nil {
				return got
			}
		}
	case []any:
		for _, v := range p {
			if got := findInt(v, key); got != nil {
				return got
			}
		}
	}
	return nil
}

// findGrade is the portfolio grade a scorecard reports at corpus/doc/top level, if
// any (no deep walk — a per-item grade must not stand in for the corpus letter).
func findGrade(payload any) *string {
	p, ok := payload.(map[string]any)
	if !ok {
		return nil
	}
	for _, nest := range []string{"corpus", "doc"} {
		if sub, ok := p[nest].(map[string]any); ok {
			if g, ok := sub["grade"].(string); ok {
				return &g
			}
		}
	}
	if g, ok := p["grade"].(string); ok {
		return &g
	}
	return nil
}

// findScore is the corpus/doc/top-level aggregate score stored under key, if any.
// Scoped to corpus/doc/top ONLY (unlike findInt's deep walk) so a page's score
// never stands in for the corpus aggregate. Used to derive the TRUE grade for the
// legacy scorecards that emit a 0-100 score but no corpus letter.
func findScore(payload any, key string) *float64 {
	if key == "" {
		return nil
	}
	p, ok := payload.(map[string]any)
	if !ok {
		return nil
	}
	for _, nest := range []string{"corpus", "doc"} {
		if sub, ok := p[nest].(map[string]any); ok {
			if f, ok := asFloat(sub[key]); ok {
				return &f
			}
		}
	}
	if f, ok := asFloat(p[key]); ok {
		return &f
	}
	return nil
}

// findValue is the corpus/doc/top-level continuous aggregate value, if any. It is
// intentionally scoped like findScore so a per-KPI value never stands in for the
// scorecard-level value.
func findValue(payload any) *float64 {
	return findScore(payload, "value")
}
