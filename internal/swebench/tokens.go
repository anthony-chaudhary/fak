package swebench

import "unicode"

// EstimateTokens is a dependency-free token estimate used only to SIZE the
// workload geometry (how big a problem statement / patch / tool result is, so the
// cost arms get a realistic per-instance prefix and result length). It is NOT the
// tokenizer the model arms decode with — those use real token ids through the
// engine — so a small bias here only nudges the geometry, never the measured
// wall-clock. The estimate blends a chars/4 floor (the usual English+code rule of
// thumb) with a word count, which tracks BPE tokenizers well enough for sizing:
// most whitespace-delimited words are 1 token, punctuation and long identifiers
// split into a few. Empty string -> 0.
func EstimateTokens(s string) int {
	if s == "" {
		return 0
	}
	words := 0
	inWord := false
	punct := 0
	for _, r := range s {
		switch {
		case unicode.IsSpace(r):
			inWord = false
		case unicode.IsLetter(r) || unicode.IsNumber(r):
			if !inWord {
				words++
			}
			inWord = true
		default:
			// punctuation / symbols tend to be their own BPE token in code
			punct++
			inWord = false
		}
	}
	chars := len([]rune(s))
	byChars := chars / 4
	byWords := words + punct/2
	// Average the two estimators; never report 0 for non-empty input.
	est := (byChars + byWords) / 2
	if est < 1 {
		est = 1
	}
	return est
}
