package resume

import "sort"

type CauseShare struct {
	Cause string `json:"dominant_cause,omitempty"`
	Share string `json:"cause_share,omitempty"`
}

// DominantCause folds launch causes after the last rearm marker. Exact ties are
// deterministic (lexical), so every runtime can emit the same settlement evidence.
func DominantCause(causes []string) CauseShare {
	counts := map[string]int{}
	total := 0
	for _, c := range causes {
		if c != "" {
			counts[c]++
			total++
		}
	}
	if total == 0 {
		return CauseShare{}
	}
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	best := keys[0]
	for _, k := range keys[1:] {
		if counts[k] > counts[best] {
			best = k
		}
	}
	return CauseShare{best, itoaCause(counts[best]) + "/" + itoaCause(total)}
}
func itoaCause(n int) string {
	if n == 0 {
		return "0"
	}
	b := []byte{}
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
