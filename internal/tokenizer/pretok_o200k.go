package tokenizer

import "unicode"

// preTokenizeO200k implements the o200k Split regex that Inkling 2 ships in its
// tokenizer.json (and that llama.cpp selects for tokenizer.ggml.pre=inkling2):
//
//	[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//	  | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//	  | \p{N}{1,3}
//	  |  ?[^\s\p{L}\p{N}]+[\r\n/]*
//	  | \s*[\r\n]+
//	  | \s+(?!\S)
//	  | \s+
//
// The letter alternatives split camelCase and ALLCAPS words and keep combining
// marks attached, which the Qwen/GLM scanners do not. Go's RE2 has no lookahead,
// so this is a hand-written scanner with the regex's backtracking order.
func preTokenizeO200k(text string) []string { return preTokenizeWith(text, matchO200kAt) }

var (
	o200kUpper = []*unicode.RangeTable{unicode.Lu, unicode.Lt, unicode.Lm, unicode.Lo, unicode.M}
	o200kLower = []*unicode.RangeTable{unicode.Ll, unicode.Lm, unicode.Lo, unicode.M}
)

func isO200kUpper(r rune) bool { return unicode.IsOneOf(o200kUpper, r) }
func isO200kLower(r rune) bool { return unicode.IsOneOf(o200kLower, r) }

// isO200kPrefix is [^\r\n\p{L}\p{N}]; combining marks are allowed.
func isO200kPrefix(r rune) bool { return r != '\r' && r != '\n' && !isL(r) && !isN(r) }

func isO200kPunct(r rune) bool { return !isWS(r) && !isL(r) && !isN(r) }

// matchO200kAt returns the rune length of the first o200k alternative matching at i.
func matchO200kAt(rs []rune, i int) int {
	n := len(rs)
	r := rs[i]

	// 1-2. Letter words. Each alternative tries the optional prefix first, as the
	// regex backtracks into the no-prefix branch only after the prefixed one fails.
	for _, word := range [](func([]rune, int) int){o200kWordLowerTail, o200kWordUpperHead} {
		for _, prefixed := range []bool{true, false} {
			j := i
			if prefixed {
				if !isO200kPrefix(r) || j+1 >= n {
					continue
				}
				j++
			}
			if end := word(rs, j); end > j {
				return end + o200kContraction(rs, end) - i
			}
		}
	}

	// 3. \p{N}{1,3}
	if isN(r) {
		j := i
		for j < n && j-i < 3 && isN(rs[j]) {
			j++
		}
		return j - i
	}

	// 4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
	{
		j := i
		if r == ' ' && j+1 < n && isO200kPunct(rs[j+1]) {
			j++
		}
		start := j
		for j < n && isO200kPunct(rs[j]) {
			j++
		}
		if j > start {
			for j < n && (rs[j] == '\r' || rs[j] == '\n' || rs[j] == '/') {
				j++
			}
			return j - i
		}
	}

	if !isWS(r) {
		return 0
	}
	j := i
	lastNL := -1
	for j < n && isWS(rs[j]) {
		if rs[j] == '\r' || rs[j] == '\n' {
			lastNL = j
		}
		j++
	}
	// 5. \s*[\r\n]+ ends at the last newline of the whitespace run.
	if lastNL >= 0 {
		return lastNL - i + 1
	}
	// 6. \s+(?!\S) leaves the final space for the following word; 7. \s+.
	run := j - i
	if j < n && run >= 2 {
		return run - 1
	}
	return run
}

// o200kWordLowerTail matches [Upper]*[Lower]+ at j and returns its end, or j.
// The greedy upper run gives back runes until a lower-class rune can start the
// tail; Lm/Lo/M belong to both classes, so that is where the tail may begin.
func o200kWordLowerTail(rs []rune, j int) int {
	n := len(rs)
	a := j
	for a < n && isO200kUpper(rs[a]) {
		a++
	}
	for x := a; x >= j; x-- {
		y := x
		for y < n && isO200kLower(rs[y]) {
			y++
		}
		if y > x {
			return y
		}
	}
	return j
}

// o200kWordUpperHead matches [Upper]+[Lower]* at j and returns its end, or j.
func o200kWordUpperHead(rs []rune, j int) int {
	n := len(rs)
	a := j
	for a < n && isO200kUpper(rs[a]) {
		a++
	}
	if a == j {
		return j
	}
	for a < n && isO200kLower(rs[a]) {
		a++
	}
	return a
}

// o200kContraction matches (?i:'s|'t|'re|'ve|'m|'ll|'d) at k and returns its length.
func o200kContraction(rs []rune, k int) int {
	n := len(rs)
	if k+1 >= n || rs[k] != '\'' {
		return 0
	}
	c1 := unicode.ToLower(rs[k+1])
	switch c1 {
	case 's', 't', 'm', 'd':
		return 2
	case 'r', 'v', 'l':
		if k+2 < n {
			c2 := unicode.ToLower(rs[k+2])
			if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
				return 3
			}
		}
	}
	return 0
}
