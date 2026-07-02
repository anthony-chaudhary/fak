package dispatchstatus

import (
	"encoding/json"
	"fmt"
	"strconv"
)

// The card fold operates on raw JSON-object documents (the shape every collector
// arm produces), so these helpers give the fold the same tolerant reads the
// Python card gets for free from dict.get / _int.

func orEmpty(m map[string]any) map[string]any {
	if m == nil {
		return map[string]any{}
	}
	return m
}

func mp(v any) map[string]any {
	if m, ok := v.(map[string]any); ok {
		return m
	}
	return map[string]any{}
}

func listOf(v any) []any {
	if l, ok := v.([]any); ok {
		return l
	}
	return nil
}

func copyMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// setDefault mirrors dict.setdefault: assign only when the key is absent.
func setDefault(m map[string]any, key string, v any) {
	if _, ok := m[key]; !ok {
		m[key] = v
	}
}

// intOf converts any JSON-borne number (or numeric string) to int, mirroring
// dispatch_status._int's tolerance.
func intOf(v any) (int, bool) {
	switch x := v.(type) {
	case int:
		return x, true
	case int64:
		return int(x), true
	case float64:
		return int(x), true
	case float32:
		return int(x), true
	case json.Number:
		if i, err := x.Int64(); err == nil {
			return int(i), true
		}
		if f, err := x.Float64(); err == nil {
			return int(f), true
		}
		return 0, false
	case string:
		if i, err := strconv.Atoi(x); err == nil {
			return i, true
		}
		return 0, false
	case bool:
		if x {
			return 1, true
		}
		return 0, true
	default:
		return 0, false
	}
}

// intOrNil returns the value as an int, or nil when it is not a number — the
// "int | None" shape the payload carries for cap/live/routed/unrouted.
func intOrNil(v any) any {
	if n, ok := intOf(v); ok {
		return n
	}
	return nil
}

func intDefault(v any, d int) int {
	if n, ok := intOf(v); ok {
		return n
	}
	return d
}

func boolOf(v any) bool {
	b, _ := v.(bool)
	return b
}

func strOf(v any) string {
	s, _ := v.(string)
	return s
}

// laneIssueCount mirrors the Python lane fold: a lane's issue count is the
// length of its issues list (whether the lane is a dict with "issues" or a bare
// list), or the value itself when it is already a number.
func laneIssueCount(info any) int {
	if m, ok := info.(map[string]any); ok {
		info = m["issues"]
	}
	if l, ok := info.([]any); ok {
		return len(l)
	}
	return intDefault(info, 0)
}

func nilWhen(cond bool, v int) any {
	if cond {
		return nil
	}
	return v
}

func nilWhenAny(cond bool, v any) any {
	if cond {
		return nil
	}
	return v
}

// pyAtom renders a scalar the way a Python f-string renders it — "None" for an
// absent value, "True"/"False" for booleans, integral floats without the ".0" —
// so the Go card's human render stays line-compatible with the Python card the
// fixture-parity pinning diffs against.
func pyAtom(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case string:
		return x
	case float64:
		return strconv.FormatFloat(x, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(x), 'f', -1, 32)
	case int:
		return strconv.Itoa(x)
	case int64:
		return strconv.FormatInt(x, 10)
	case json.Number:
		return x.String()
	default:
		return fmt.Sprintf("%v", x)
	}
}
