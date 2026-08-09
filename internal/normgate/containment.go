package normgate

// Containment returns the fraction of the smaller distinct-token set present
// in the larger set. It is symmetric, ignores duplicate tokens, and returns
// zero when either input has no tokens.
//
// Callers are responsible for choosing tokenization and normalization suitable
// for their corpus; keeping this scorer token-agnostic prevents hidden policy.
func Containment(a, b []string) float64 {
	left := tokenSet(a)
	right := tokenSet(b)
	if len(left) == 0 || len(right) == 0 {
		return 0
	}
	if len(left) > len(right) {
		left, right = right, left
	}
	shared := 0
	for token := range left {
		if _, ok := right[token]; ok {
			shared++
		}
	}
	return float64(shared) / float64(len(left))
}

func tokenSet(tokens []string) map[string]struct{} {
	set := make(map[string]struct{}, len(tokens))
	for _, token := range tokens {
		if token != "" {
			set[token] = struct{}{}
		}
	}
	return set
}
