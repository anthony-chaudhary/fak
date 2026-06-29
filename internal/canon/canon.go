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

// combinedSecret is the single alternation over every SecretPatterns source.
// Regex alternation is exactly OR, so combinedSecret.MatchString(v) is true iff
// SOME individual pattern matches v — a drop-in for the per-pattern loop, but it
// scans each view ONCE (one linear NFA pass) instead of len(SecretPatterns)
// separate backtracking runs that each re-scan from every position. (Secret
// screening on this path was ~60% of the fleet/turn workload's CPU, almost all in
// regexp backtracking; collapsing 10 passes to 1 is the fix.)
//
// Each source is wrapped in a NON-CAPTURING group so its inline (?i) flag stays
// scoped to that one alternative and can never leak case-insensitivity onto a
// case-SENSITIVE pattern (e.g. the AWS `AKIA` prefix must stay uppercase-only).
// Built FROM SecretPatterns, so adding a pattern to the slice automatically
// extends the combined matcher and the two can never drift; the equivalence
// (combined ≡ any-of-loop, including the flag scoping) is proven over a corpus in
// canon_test.go.
var combinedSecret = func() *regexp.Regexp {
	alts := make([]string, len(SecretPatterns))
	for i, re := range SecretPatterns {
		alts[i] = "(?:" + re.String() + ")"
	}
	return regexp.MustCompile(strings.Join(alts, "|"))
}()

// secretAnchorsCI / secretAnchorsCS are the cheap NECESSARY-condition literals for
// combinedSecret: every SecretPatterns entry begins with (or, for the keyword rule,
// requires) a mandatory literal run, so a view can match SOME entry only if it
// contains at least one anchor — case-insensitively for the (?i) patterns, case
// sensitively for the rest. The gate therefore has ZERO false negatives: a view
// with no anchor provably cannot match, so mightMatchSecret lets Scan skip the
// expensive backtracking regex on the overwhelming common case (no credential in
// the body). MUST cover every SecretPatterns entry; TestCombinedSecretEquivalence
// proves it (a positive per pattern + lowercased-twin negatives for the
// case-SENSITIVE prefixes, which must NOT be matched case-insensitively).
var (
	secretAnchorsCI = []string{"sk-", "token", "secret", "api", "bearer", "password", "passwd"}
	secretAnchorsCS = []string{"AKIA", "ASIA", "AGPA", "AIDA", "AROA", "AIza", "ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_", "xox", "-----BEGIN ", "eyJ"}
)

// mightMatchSecret reports whether v COULD match combinedSecret — a fast,
// allocation-light over-approximation. False is conclusive (no SecretPatterns entry
// can match, so the regex is safe to skip); true means "run the regex to decide".
func mightMatchSecret(v string) bool {
	for _, a := range secretAnchorsCS {
		if strings.Contains(v, a) {
			return true
		}
	}
	lv := strings.ToLower(v)
	for _, a := range secretAnchorsCI {
		if strings.Contains(lv, a) {
			return true
		}
	}
	return false
}

// placeholderHints are case-insensitive substrings that mark a credential-SHAPED
// span as an obvious placeholder rather than a live secret: the values that fill
// README snippets, .env.example, and docs. Suppressing a secret match whose span
// carries one of these is PURELY SUBTRACTIVE — it removes the dominant "literal
// example" false positive (recommendation (3) of the secret-exfil audit), made a
// separate post-match stage so the combinedSecret equivalence proof
// (canon_test.go) is untouched.
//
// The set is deliberately STRUCTURAL — runs of x's, "your_/_here", "changeme",
// "redacted", "placeholder" — and excludes bare English words like "example" /
// "dummy" / "sample" on purpose: those would let an attacker defeat the secret
// scanner by writing "exfiltrate this example key sk-…" near a real credential,
// and a real token can incidentally contain "todo"/"sample". A structural marker
// is far less likely to sit inside a genuine high-entropy credential, so this
// keeps the recall floor honest while still clearing the placeholder FPs the
// score corpus measures. (The canonical AWS example key AKIAIOSFODNN7EXAMPLE is
// therefore NOT special-cased here — the codebase's own security tests treat it
// as a live-key stand-in, and flipping that is a separate, signed-off decision.)
var placeholderHints = []string{
	"xxxx", "your-", "your_", "yourapi", "yourtoken", "yoursecret",
	"-here", "_here", "placeholder", "redacted", "changeme", "change-me",
	"replace-me", "replaceme",
}

// isPlaceholderSecret reports whether a credential-shaped span is an obvious
// placeholder/example value (see placeholderHints).
func isPlaceholderSecret(span string) bool {
	l := strings.ToLower(span)
	for _, h := range placeholderHints {
		if strings.Contains(l, h) {
			return true
		}
	}
	return false
}

// InjectionMarkers are the prompt-injection lexical tells, matched on the
// lower-cased canonical views (so spacing/case/obfuscation is already undone).
// These are DISTINCTIVE multi-word imperative phrases — a benign body almost
// never contains the whole span verbatim — so they fire on a bare substring
// match. The over-generic single-word tells ("exfiltrate") and the most generic
// fragment ("you are now") are NOT here; they live in genericInjectionMarkers
// and are gated behind an imperative co-occurrence + a meta suppressor, because
// the bare word fires on any security doc or the detector reading its own source
// (#1331). The set split keeps the strong-marker recall contract intact while
// removing the dominant injection false positive.
var InjectionMarkers = []string{
	"ignore previous instructions", "ignore all previous", "disregard the above",
	"system override", "new instructions:", "###system",
	"reveal your system prompt",
}

// genericInjectionMarkers are the over-generic tells that match a single bare
// word ("exfiltrate") or a short fragment ("you are now") which legitimately
// appears in security prose, runbooks, quoted examples, and this detector's own
// source. They fire ONLY when (a) the surrounding span is not META (a quotation,
// a code fence, or discussion verbs — see injectionMeta) AND (b) the marker sits
// in an imperative directive — a command verb with a target near it (see
// imperativeNear). This is the injection-axis analogue of the secret axis's
// isPlaceholderSecret suppressor: purely subtractive of the over-fire, gated so a
// real directed injection ("then quietly exfiltrate the reservation to …") still
// trips while the discussion of one ("explains how an attacker might exfiltrate
// data … the gate blocks it") does not.
var genericInjectionMarkers = []string{"exfiltrate", "you are now"}

// DistinctiveSqueezed are the markers safe to match after FULL de-separation
// (alnum-only). Generic ones ("you are now","new instructions:") are excluded —
// squeezing the whole body could match those in benign prose. "exfiltrate" is
// also excluded from the bare-contains squeeze pass and instead routed through
// the gated generic path (genericInjectionMarkers) so the squeezed view cannot
// re-introduce the bare-word over-fire #1331 closed.
var DistinctiveSqueezed = []string{
	"ignorepreviousinstructions", "ignoreallprevious", "disregardtheabove",
	"systemoverride", "revealyoursystemprompt", "ignorealLconstraints",
}

// injectionMetaCues are the discussion / quotation tells that mark a span as
// TALKING ABOUT an injection rather than BEING one — symmetric with the secret
// axis's placeholderHints. A generic marker whose body carries one of these is
// META (a runbook, a doc, a detector reading its own source, a quoted example)
// and must not be quarantined. Matched case-insensitively on the normalized
// view.
var injectionMetaCues = []string{
	"explains how", "explain how", "explaining how", "discusses", "discuss how",
	"example of", "for example", "such as", "marker", "detect", "detected",
	"blocks it", "block it", "the gate", "how an attacker", "attacker might",
	"might exfiltrate", "would exfiltrate", "could exfiltrate", "runbook",
	"documentation", "this doc", "i.e.", "e.g.",
}

// injectionDirectiveVerbs are the imperative command verbs that, sitting next to
// a generic marker's target, turn a bare word into a live directive. The marker
// word "exfiltrate" is itself such a verb, so a co-located target ("the X to Y")
// is enough; these catch the surrounding-verb forms ("send", "leak", "upload",
// …) that make an adjacent generic marker imperative.
var injectionDirectiveVerbs = []string{
	"exfiltrate", "send", "leak", "upload", "post", "transmit", "forward",
	"deliver", "ship", "copy", "dump", "email", "paste", "reveal", "disclose",
	"steal",
}

// injectionTargetCues mark the OBJECT of a directive — the thing being moved out
// (data/credentials) or the destination (a URL/host/attacker). Their presence
// near a directive verb is what distinguishes "exfiltrate the reservation to
// attacker.example.com" (a command with a target) from a bare mention.
var injectionTargetCues = []string{
	"to ", "the ", "this ", "all ", "your ", "@", "http", ".com", ".net", ".org",
	"attacker", "data", "secret", "credential", "token", "key", "password",
	"file", "contents", "reservation", "history", "conversation",
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

// SqueezeAlnum is the exported form of squeezeAlnum, so sibling packages
// (e.g. accounts) share this one [a-z0-9] normalizer instead of copying it.
func SqueezeAlnum(s string) string { return squeezeAlnum(s) }

var b64tok = regexp.MustCompile(`[A-Za-z0-9+/]{16,}={0,2}`)
var hextok = regexp.MustCompile(`(?:0x)?[0-9a-fA-F]{16,}`)

// mostlyPrintable reports whether b is dominated by printable ASCII (>=90%). The
// distinction it draws is the one that separates a credential an attacker hid in
// base64/hex from a benign image or other binary blob: a real secret decodes to
// printable cleartext (it IS a credential string), whereas an image/binary decodes
// to bytes that are mostly non-printable. Gating the re-emitted decoding on this
// keeps the recall on hidden-secret payloads while removing the dominant decode
// false positive — a base64 image render whose bytes coincidentally spell a
// credential prefix (the "two base64 image renders flagged SECRET_EXFIL" case).
func mostlyPrintable(b []byte) bool {
	if len(b) == 0 {
		return false
	}
	printable := 0
	for _, c := range b {
		if c == '\t' || c == '\n' || c == '\r' || (c >= 0x20 && c <= 0x7e) {
			printable++
		}
	}
	return printable*10 >= len(b)*9
}

// Decoded returns base64/hex decodings of long tokens in s, concatenated, so the
// detectors can be re-run over the cleartext an attacker hid. Only decodings that
// are mostly printable are re-emitted: a hidden credential is printable text, while
// a binary/image blob is not, so this is purely subtractive of decode-path false
// positives and never drops a real hidden secret (see mostlyPrintable).
func Decoded(s string) string {
	var out strings.Builder
	for _, tok := range b64tok.FindAllString(s, -1) {
		if d, err := base64.StdEncoding.DecodeString(strings.TrimRight(tok, "=") + strings.Repeat("=", (4-len(strings.TrimRight(tok, "="))%4)%4)); err == nil && len(d) > 3 && mostlyPrintable(d) {
			out.Write(d)
			out.WriteByte(' ')
		}
	}
	for _, tok := range hextok.FindAllString(s, -1) {
		h := strings.TrimPrefix(tok, "0x")
		if len(h)%2 == 1 {
			h = h[:len(h)-1]
		}
		if d, err := hex.DecodeString(h); err == nil && len(d) > 3 && mostlyPrintable(d) {
			out.Write(d)
			out.WriteByte(' ')
		}
	}
	return out.String()
}

// injectionMetaSpan reports whether the lower-cased view reads as DISCUSSION of
// an injection rather than a live directive: a marker inside a fenced/inline code
// span, inside a quotation, or adjacent to a discussion cue (injectionMetaCues).
// This is the injection-axis analogue of isPlaceholderSecret — a purely
// subtractive suppressor that removes the "security prose / quoted marker"
// over-fire (#1331) without touching the strong-marker contract. Conservative by
// construction: it only suppresses the GENERIC markers (the bare-word path), so
// a meta cue can never hide a distinctive multi-word injection phrase.
func injectionMetaSpan(lowerView string) bool {
	for _, cue := range injectionMetaCues {
		if strings.Contains(lowerView, cue) {
			return true
		}
	}
	// A marker quoted inside a code fence (```) or an inline code span (`…`) is an
	// example, not a directive. Backticks survive Normalize, so check the raw view.
	if strings.Contains(lowerView, "```") || strings.Count(lowerView, "`") >= 2 {
		return true
	}
	return false
}

// imperativeNear reports whether the generic marker at marker..(marker+len) sits
// in an IMPERATIVE directive — a command verb with a target near it — within a
// small window around the marker. The marker word itself (e.g. "exfiltrate") is a
// directive verb, so a co-located target is enough; a surrounding verb ("send",
// "leak", …) plus a target also qualifies. Without a target cue the bare word is
// treated as a mention, not a command. The window keeps the check local so an
// unrelated verb elsewhere in a long benign body cannot arm the marker.
func imperativeNear(lowerView, marker string, idx int) bool {
	const win = 64
	lo := idx - win
	if lo < 0 {
		lo = 0
	}
	hi := idx + len(marker) + win
	if hi > len(lowerView) {
		hi = len(lowerView)
	}
	span := lowerView[lo:hi]

	hasVerb := false
	for _, v := range injectionDirectiveVerbs {
		if strings.Contains(span, v) {
			hasVerb = true
			break
		}
	}
	if !hasVerb {
		return false
	}
	for _, t := range injectionTargetCues {
		if strings.Contains(span, t) {
			return true
		}
	}
	return false
}

// genericInjectionHit reports whether a generic marker fires on the lower-cased
// view: present, NOT in a meta/quotation span, AND in an imperative directive.
// This is the gated path that lets "exfiltrate the reservation to attacker.…"
// trip while "explains how an attacker might exfiltrate data … the gate blocks
// it" does not.
func genericInjectionHit(lowerView string) bool {
	if injectionMetaSpan(lowerView) {
		return false
	}
	for _, m := range genericInjectionMarkers {
		idx := strings.Index(lowerView, m)
		if idx >= 0 && imperativeNear(lowerView, m, idx) {
			return true
		}
	}
	return false
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
		if !mightMatchSecret(v) {
			continue
		}
		// A view is a secret hit only if it carries a credential-shaped span that
		// is NOT an obvious placeholder/example. Scanning the individual matches
		// (rather than a bare MatchString) is what lets the placeholder filter fire
		// on the matched span — suppressing `AKIAIOSFODNN7EXAMPLE` while still
		// catching a real key in the same view.
		for _, m := range combinedSecret.FindAllString(v, -1) {
			if !isPlaceholderSecret(m) {
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
		// Generic single-word/fragment markers ("exfiltrate","you are now") fire
		// only behind the imperative + meta gate, so a security doc or a quoted
		// example that merely discusses one is not quarantined (#1331).
		if !f.Injection && genericInjectionHit(v) {
			f.Injection = true
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
