// Package canon is the de-obfuscating canonicalizer + lexical threat detector,
// factored out of internal/normgate so it is ONE primitive, tested ONCE, and
// reusable by every gate that needs to scan bytes for a hidden secret or
// injection on a normalized view — not just the write-time admitter.
//
// The problem it closes: a raw ASCII regex/substring detector (ctxmmu v0.1) is
// defeated by trivial obfuscation — char-spacing, base64, hex, homoglyph,
// fullwidth, zero-width, bidi-reverse — so a payload that "reads" as
// "ignore previous instructions" to a model walks straight past the matcher
// (~100% evasion, measured on a private transcript-derived corpus). canon
// canonicalizes a COPY of the bytes into several views (normalized, decoded,
// reversed, de-separated) and runs the detectors over THOSE, so the obfuscation
// is undone before the match.
//
// canon is a pure leaf: it depends only on the standard library, holds no state,
// and makes no policy decision (Quarantine vs Transform vs Allow is the caller's
// job — see normgate's provenance gate). It only answers the factual question
// "do the canonical views reveal a secret or an injection marker?". That keeps it
// safe to reuse from BOTH the write-time gate (normgate) AND the read-time
// re-screen of a reloaded core image (recall), so a session recorded under a weak
// gate is re-screened by today's canonical detector when its pages are paged back
// in — the "tightened gate catches it on reload" property.
package canon

import (
	"encoding/base64"
	"encoding/hex"
	"regexp"
	"strings"
)

// SecretPatterns is the broadened secret vocabulary: distinctive credential
// prefixes plus a keyword-proximity rule for bare high-entropy tokens. The
// prefixes keep false-positive risk low; the proximity rule requires a key-like
// keyword so it does not flag a benign 40-hex git SHA. Exported so a caller (e.g.
// an IFC egress sink-gate) can audit or extend the set.
var SecretPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)sk-[a-z0-9]{16,}`),
	regexp.MustCompile(`(?i)sk-(proj|live|ant)-[a-z0-9_-]{12,}`),
	regexp.MustCompile(`A(KIA|SIA|GPA|IDA|ROA)[0-9A-Z]{12,}`), // AWS perm + STS + others
	regexp.MustCompile(`AIza[0-9A-Za-z_-]{20,}`),              // Google API key
	regexp.MustCompile(`gh[pousr]_[A-Za-z0-9]{20,}`),          // GitHub classic
	regexp.MustCompile(`github_pat_[0-9A-Za-z_]{20,}`),        // GitHub fine-grained
	regexp.MustCompile(`xox[baprse]-?[A-Za-z0-9-]{10,}`),      // Slack (+ xoxe)
	regexp.MustCompile(`-----BEGIN [A-Z ]*PRIVATE KEY-----`),
	regexp.MustCompile(`eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}`), // JWT
	regexp.MustCompile(`(?i)(token|secret|api[_-]?key|bearer|password|passwd)["'\s:=]{1,4}[A-Za-z0-9+/_-]{20,}`),
}

// InjectionMarkers are the prompt-injection lexical tells, matched on the
// lower-cased canonical views (so spacing/case/obfuscation is already undone).
var InjectionMarkers = []string{
	"ignore previous instructions", "ignore all previous", "disregard the above",
	"you are now", "system override", "new instructions:", "###system",
	"reveal your system prompt", "exfiltrate",
}

// DistinctiveSqueezed are the markers safe to match after FULL de-separation
// (alnum-only). Generic ones ("you are now","new instructions:") are excluded —
// squeezing the whole body could match those in benign prose.
var DistinctiveSqueezed = []string{
	"ignorepreviousinstructions", "ignoreallprevious", "disregardtheabove",
	"systemoverride", "revealyoursystemprompt", "exfiltrate", "ignorealLconstraints",
}

// homoglyphs maps common Cyrillic/Greek look-alikes to their ASCII letter.
var homoglyphs = map[rune]rune{
	'а': 'a', 'е': 'e', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'y', 'х': 'x', 'і': 'i',
	'ѕ': 's', 'ԁ': 'd', 'ո': 'n', 'ν': 'v', 'ӏ': 'l', 'ј': 'j', 'ԛ': 'q', 'ԝ': 'w',
	'А': 'A', 'Е': 'E', 'О': 'O', 'Р': 'P', 'С': 'C', 'У': 'Y', 'Х': 'X', 'І': 'I',
	'ο': 'o', 'α': 'a', 'ρ': 'p', 'ϲ': 'c', 'ε': 'e', 'ι': 'i', 'υ': 'u', 'τ': 't',
	'Ѕ': 'S', 'Ј': 'J', 'М': 'M', 'Н': 'H', 'Т': 'T', 'В': 'B', 'К': 'K',
}

// isInvisible reports whether a rune is a zero-width / formatting / bidi control
// an attacker can splice between letters to defeat a substring match.
func isInvisible(r rune) bool {
	switch {
	case r == 0x200B || r == 0x200C || r == 0x200D || r == 0xFEFF || r == 0x2060 || r == 0x00AD:
		return true // zero-width + soft hyphen
	case r >= 0xFE00 && r <= 0xFE0F: // variation selectors
		return true
	case r >= 0xE0100 && r <= 0xE01EF: // variation selectors supplement
		return true
	case r == 0x202A || r == 0x202B || r == 0x202C || r == 0x202D || r == 0x202E: // bidi
		return true
	case r >= 0x2066 && r <= 0x2069: // isolates
		return true
	case r == 0x200E || r == 0x200F: // LRM/RLM
		return true
	}
	return false
}

// Normalize canonicalizes a body: drop invisibles, fold fullwidth + ideographic
// space, map homoglyphs. (Bidi-reversed text is handled separately by also
// scanning the reversed rune sequence — see Views.)
func Normalize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if isInvisible(r) {
			continue
		}
		if r >= 0xFF01 && r <= 0xFF5E { // fullwidth ASCII
			r -= 0xFEE0
		} else if r == 0x3000 { // ideographic space
			r = ' '
		} else if m, ok := homoglyphs[r]; ok {
			r = m
		}
		b.WriteRune(r)
	}
	return b.String()
}

func reverseRunes(s string) string {
	rs := []rune(s)
	for i, j := 0, len(rs)-1; i < j; i, j = i+1, j-1 {
		rs[i], rs[j] = rs[j], rs[i]
	}
	return string(rs)
}

// squeezeAlnum keeps only [a-z0-9] (lower-cased) — collapses spacing/dotting/
// piping/zero-width/emoji separators between letters.
func squeezeAlnum(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

var b64tok = regexp.MustCompile(`[A-Za-z0-9+/]{16,}={0,2}`)
var hextok = regexp.MustCompile(`(?:0x)?[0-9a-fA-F]{16,}`)

// Decoded returns base64/hex decodings of long tokens in s, concatenated, so the
// detectors can be re-run over the cleartext an attacker hid.
func Decoded(s string) string {
	var out strings.Builder
	for _, tok := range b64tok.FindAllString(s, -1) {
		if d, err := base64.StdEncoding.DecodeString(strings.TrimRight(tok, "=") + strings.Repeat("=", (4-len(strings.TrimRight(tok, "="))%4)%4)); err == nil && len(d) > 3 {
			out.Write(d)
			out.WriteByte(' ')
		}
	}
	for _, tok := range hextok.FindAllString(s, -1) {
		h := strings.TrimPrefix(tok, "0x")
		if len(h)%2 == 1 {
			h = h[:len(h)-1]
		}
		if d, err := hex.DecodeString(h); err == nil && len(d) > 3 {
			out.Write(d)
			out.WriteByte(' ')
		}
	}
	return out.String()
}

// Findings is the factual result of a canonical scan. No verdict, no policy — the
// caller decides what to DO with a hit (quarantine, transform, refuse a sink).
type Findings struct {
	Secret    bool
	Injection bool
}

// Any reports whether the scan revealed any threat.
func (f Findings) Any() bool { return f.Secret || f.Injection }

// Scan canonicalizes body and reports whether its canonical views reveal a secret
// or an injection marker. This is the de-obfuscating detector both normgate (write
// time) and recall (read-time re-screen) share, so there is exactly one
// canonicalization to audit and one to keep correct.
func Scan(body []byte) Findings {
	raw := string(body)
	norm := Normalize(raw)
	dec := Decoded(raw) + " " + Decoded(norm)
	rev := reverseRunes(norm)

	var f Findings

	for _, v := range []string{norm, dec, rev} {
		for _, re := range SecretPatterns {
			if re.MatchString(v) {
				f.Secret = true
				break
			}
		}
		if f.Secret {
			break
		}
	}

	for _, v := range []string{strings.ToLower(norm), strings.ToLower(dec), strings.ToLower(rev)} {
		for _, m := range InjectionMarkers {
			if strings.Contains(v, m) {
				f.Injection = true
				break
			}
		}
		if f.Injection {
			break
		}
	}
	if !f.Injection {
		// de-separated (squeeze) pass for distinctive markers only.
		for _, v := range []string{squeezeAlnum(norm), squeezeAlnum(dec), squeezeAlnum(rev)} {
			for _, m := range DistinctiveSqueezed {
				if strings.Contains(v, strings.ToLower(m)) {
					f.Injection = true
					break
				}
			}
			if f.Injection {
				break
			}
		}
	}
	return f
}
