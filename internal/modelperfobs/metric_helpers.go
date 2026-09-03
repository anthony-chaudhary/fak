package modelperfobs

import "strings"

func deviceMetricUnavailable(s string) bool {
	s = strings.TrimSpace(strings.ToLower(s))
	return s == "" || s == "n/a" || s == "not supported" || s == "[not supported]"
}
