package adjudicator

import (
	"strings"
)

func commandText(args map[string]any) string {
	for _, key := range []string{"command", "cmd", "shell", "script"} {
		if s, ok := stringArg(args, key); ok {
			return s
		}
	}
	for _, key := range []string{"argv", "args"} {
		if joined, ok := stringSliceArg(args, key); ok {
			return joined
		}
	}
	return ""
}

func stringArg(args map[string]any, key string) (string, bool) {
	if args == nil {
		return "", false
	}
	v, ok := args[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok && strings.TrimSpace(s) != ""
}

func stringSliceArg(args map[string]any, key string) (string, bool) {
	if args == nil {
		return "", false
	}
	v, ok := args[key]
	if !ok {
		return "", false
	}
	switch xs := v.(type) {
	case []string:
		if len(xs) == 0 {
			return "", false
		}
		return strings.Join(xs, " "), true
	case []any:
		parts := make([]string, 0, len(xs))
		for _, x := range xs {
			s, ok := x.(string)
			if !ok {
				return "", false
			}
			parts = append(parts, s)
		}
		if len(parts) == 0 {
			return "", false
		}
		return strings.Join(parts, " "), true
	default:
		return "", false
	}
}

func boolArg(args map[string]any, key string) (bool, bool) {
	if args == nil {
		return false, false
	}
	v, ok := args[key]
	if !ok {
		return false, false
	}
	b, ok := v.(bool)
	return b, ok
}
