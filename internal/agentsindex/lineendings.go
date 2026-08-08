package agentsindex

import "bytes"

// EOL is a line-ending sequence found in text.
type EOL string

const (
	EOLLF   EOL = "\n"
	EOLCRLF EOL = "\r\n"
)

// Detect reports the dominant line ending and whether both LF and CRLF occur.
// Ties prefer CRLF, avoiding a platform-dependent answer.
func Detect(b []byte) (EOL, bool) {
	crlf := bytes.Count(b, []byte("\r\n"))
	lf := bytes.Count(b, []byte("\n")) - crlf
	mixed := crlf > 0 && lf > 0
	if crlf > 0 && crlf >= lf {
		return EOLCRLF, mixed
	}
	return EOLLF, mixed
}

// Reconcile preserves an existing text file's line-ending convention. Mixed
// files heal to their dominant ending. Binary input is returned unchanged.
// When existing has no line endings, policy is used (LF for an invalid policy).
func Reconcile(existing, replacement []byte, policy EOL) []byte {
	if bytes.IndexByte(existing, 0) >= 0 || bytes.IndexByte(replacement, 0) >= 0 {
		return bytes.Clone(replacement)
	}
	ending, _ := Detect(existing)
	if !bytes.Contains(existing, []byte("\n")) {
		ending = policy
		if ending != EOLCRLF && ending != EOLLF {
			ending = EOLLF
		}
	}
	normalized := bytes.ReplaceAll(replacement, []byte("\r\n"), []byte("\n"))
	if ending == EOLCRLF {
		normalized = bytes.ReplaceAll(normalized, []byte("\n"), []byte("\r\n"))
	}
	return normalized
}
