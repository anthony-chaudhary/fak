package main

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func accountFromMap(m map[string]any) dispatchtick.Account {
	return dispatchtick.Account{
		Tag:   dispatchMapString(m, "tag"),
		Tier:  m["tier"],
		Model: dispatchMapString(m, "model"),
		Dir:   firstString(dispatchMapString(m, "dir"), dispatchMapString(m, "config_dir")),
	}
}

func dispatchtickWorkKind(backend string) string {
	b, err := dispatchtick.NormalizeBackend(backend)
	if err != nil {
		return dispatchtick.DefaultWorkKind("claude")
	}
	return dispatchtick.DefaultWorkKind(b)
}

func stringSlice(v any) []string {
	var out []string
	if arr, ok := v.([]any); ok {
		for _, x := range arr {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
	}
	return out
}

func envMap(kvs []string) map[string]string {
	out := map[string]string{}
	for _, kv := range kvs {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			out[kv[:i]] = kv[i+1:]
		}
	}
	return out
}

// envSliceFromMap stays as a package-local name for its many call sites; the
// one shared implementation lives in internal/procguard (#1419).
func envSliceFromMap(env map[string]string) []string {
	return procguard.EnvSlice(env)
}

func mapAt(m map[string]any, key string) map[string]any {
	if v, ok := m[key].(map[string]any); ok {
		return v
	}
	return map[string]any{}
}

func dispatchMapString(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func dispatchMapBool(m map[string]any, key string) bool {
	v, _ := m[key].(bool)
	return v
}

func dispatchStringValue(v any) string {
	if s, ok := v.(string); ok {
		return strings.TrimSpace(s)
	}
	return ""
}

func dispatchBoolValue(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case string:
		switch strings.ToLower(strings.TrimSpace(x)) {
		case "1", "true", "yes", "y", "on":
			return true
		}
	}
	return false
}

func dispatchIntValue(v any) int {
	if n := intPtrFromAny(v); n != nil {
		return *n
	}
	return 0
}

func dispatchMapInt(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case json.Number:
		n, _ := v.Int64()
		return int(n)
	}
	return 0
}

func intPtrFromAny(v any) *int {
	switch x := v.(type) {
	case int:
		return &x
	case int64:
		n := int(x)
		return &n
	case float64:
		n := int(x)
		return &n
	case json.Number:
		if n, err := x.Int64(); err == nil {
			i := int(n)
			return &i
		}
	case string:
		if n, err := strconv.Atoi(strings.TrimSpace(x)); err == nil {
			return &n
		}
	}
	return nil
}

func anySlice(v any) []any {
	if arr, ok := v.([]any); ok {
		return arr
	}
	if arr, ok := v.([]map[string]any); ok {
		out := make([]any, 0, len(arr))
		for _, item := range arr {
			out = append(out, item)
		}
		return out
	}
	return nil
}

func nonEmptyLines(s string) []string {
	rows := []string{}
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.TrimSpace(line) != "" {
			rows = append(rows, line)
		}
	}
	return rows
}

func sortedSet(in map[int]bool) []int {
	out := make([]int, 0, len(in))
	for n := range in {
		out = append(out, n)
	}
	sort.Ints(out)
	return out
}

func sortedStringSet(in map[string]bool) []string {
	out := make([]string, 0, len(in))
	for s := range in {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

func dispatchAnyOSBase(path string) string {
	path = strings.TrimRight(strings.ReplaceAll(strings.TrimSpace(path), "\\", "/"), "/")
	if path == "" {
		return ""
	}
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[i+1:]
	}
	return path
}

func parseAccountTier(s string) any {
	if s == "" {
		return nil
	}
	if n, err := strconv.Atoi(s); err == nil {
		return n
	}
	return s
}
