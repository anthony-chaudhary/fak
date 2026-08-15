package canon

import (
	"bytes"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// The closed semantic classes below support context-aware PII policy. Detection still
// defaults to protecting every class; callers may exempt named classes only at a trusted
// request boundary.
const (
	PIIClassEmail       = "email"
	PIIClassPhone       = "phone"
	PIIClassNationalID  = "national_id"
	PIIClassPaymentCard = "payment_card"
	PIIClassIBAN        = "iban"
)

type classifiedPIIPattern struct {
	class string
	re    *regexp.Regexp
}

// Keep the original deterministic patterns and order: class-aware policy must not change
// the detector's established false-positive/false-negative envelope.
var classifiedPIIPatterns = []classifiedPIIPattern{
	{PIIClassEmail, regexp.MustCompile(`[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}`)},
	{PIIClassNationalID, regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`)},
	{PIIClassPhone, regexp.MustCompile(`(?:\+\d{1,3}[ .\-]?)?\(?\d{3}\)?[ .\-]\d{3}[ .\-]\d{4}\b`)},
	{PIIClassPaymentCard, regexp.MustCompile(`\b\d{4}[ \-]\d{4}[ \-]\d{4}[ \-]\d{1,7}\b`)},
	{PIIClassIBAN, regexp.MustCompile(`\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b`)},
}

// PIIPatterns is the stable exported detector surface.
var PIIPatterns = func() []*regexp.Regexp {
	out := make([]*regexp.Regexp, 0, len(classifiedPIIPatterns))
	for _, p := range classifiedPIIPatterns {
		out = append(out, p.re)
	}
	return out
}()

// combinedPII is built FROM PIIPatterns so detection and default redaction cannot drift.
var combinedPII = func() *regexp.Regexp {
	alts := make([]string, len(PIIPatterns))
	for i, re := range PIIPatterns {
		alts[i] = "(?:" + re.String() + ")"
	}
	return regexp.MustCompile(strings.Join(alts, "|"))
}()

// RedactPII masks general-PII spans in place and preserves all surrounding bytes.
func RedactPII(body []byte) (redacted []byte, masked int) {
	return redactPIILocations(body, combinedPII.FindAllIndex(body, -1))
}

// RawPIIComplete reports whether the raw body contains a redactable PII span. Scan may
// still report PII when only a normalized/decoded view contains one; that case must seal.
func RawPIIComplete(body []byte) bool { return combinedPII.Match(body) }

// KnownPIIClass reports whether class is in the closed policy taxonomy.
func KnownPIIClass(class string) bool {
	for _, p := range classifiedPIIPatterns {
		if p.class == class {
			return true
		}
	}
	return false
}

// PIIClasses returns the raw, high-confidence classes found in body.
func PIIClasses(body []byte) map[string]bool {
	found := make(map[string]bool)
	for _, p := range classifiedPIIPatterns {
		if p.re.Match(body) {
			found[p.class] = true
		}
	}
	return found
}

// HasPIIExcept reports whether a raw span belongs to a class outside exemptClasses.
func HasPIIExcept(body []byte, exemptClasses map[string]bool) bool {
	for _, p := range classifiedPIIPatterns {
		if !exemptClasses[p.class] && p.re.Match(body) {
			return true
		}
	}
	return false
}

// RedactPIIClasses masks only named classes. Normgate uses this to remove declared-public
// spans temporarily, then re-runs the full canonical scanner to prove nothing protected or
// obfuscated remains.
func RedactPIIClasses(body []byte, classes map[string]bool) (redacted []byte, masked int) {
	return redactClassSet(body, func(class string) bool { return classes[class] })
}

// RedactPIIExcept preserves context-declared public classes and masks protected classes.
func RedactPIIExcept(body []byte, exemptClasses map[string]bool) (redacted []byte, masked int) {
	if len(exemptClasses) == 0 {
		return RedactPII(body)
	}
	return redactClassSet(body, func(class string) bool { return !exemptClasses[class] })
}

func redactClassSet(body []byte, include func(string) bool) ([]byte, int) {
	var locs [][]int
	for _, p := range classifiedPIIPatterns {
		if include(p.class) {
			locs = append(locs, p.re.FindAllIndex(body, -1)...)
		}
	}
	return redactPIILocations(body, locs)
}

func redactPIILocations(body []byte, locs [][]int) ([]byte, int) {
	if len(locs) == 0 {
		return body, 0
	}
	sort.Slice(locs, func(i, j int) bool { return locs[i][0] < locs[j][0] })
	merged := locs[:0]
	for _, loc := range locs {
		if len(merged) == 0 || loc[0] > merged[len(merged)-1][1] {
			merged = append(merged, loc)
		} else if loc[1] > merged[len(merged)-1][1] {
			merged[len(merged)-1][1] = loc[1]
		}
	}
	var out bytes.Buffer
	prev := 0
	for _, loc := range merged {
		out.Write(body[prev:loc[0]])
		out.Write(piiMask(loc[1] - loc[0]))
		prev = loc[1]
	}
	out.Write(body[prev:])
	return out.Bytes(), len(merged)
}

func piiMask(n int) []byte { return []byte(fmt.Sprintf("[redacted:pii:%dB]", n)) }
