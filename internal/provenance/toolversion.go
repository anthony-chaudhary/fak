// toolversion.go — the live tool-version witness (#5677, parent #2136).
//
// The trust half of this package (provenance.go) answers "where did this BYTE
// come from?"; the manifest half (runmanifest.go) answers "under exactly what
// recorded scope was this run produced?". This half answers the third provenance
// question, the one about the machine the run actually executed on: "does the
// executable, plugin launcher, or adapter SELECTED AT RUNTIME match the version
// the registry pinned?"
//
// THE INVARIANT IT ENFORCES: a tool satisfies a pin only when the version field
// in the tool's OWN live self-report PARSES to the pinned value. The historical
// shortcut — `strings.Contains(banner, pin)` — is not evidence of a match, and
// it fails in three ways that all read as green:
//
//   - PATH CONFUSION. `/opt/toolchain-1.2.3/bin/mytool version 2.0.0` contains
//     "1.2.3", so a pin of 1.2.3 "matches" a tool whose live version is 2.0.0.
//     The pinned bytes are in the INSTALL PATH, not the version field.
//   - LONGER-TOKEN CONFUSION. `mytool version 1.2.30` contains "1.2.3", so a pin
//     of 1.2.3 "matches" a strictly different release.
//   - INTERPRETER CONFUSION. `mytool 2.0.0 (Python 3.11.4)` carries a second,
//     borrowed version that is not the producer's own field.
//
// Parsing kills all three structurally rather than by adding more substring
// special-cases: 1.2.30 parses to components [1 2 30], which is not equal to
// [1 2 3] under any comparison, and a path-shaped token is never a candidate for
// the version field at all.
//
// FAIL-CLOSED AND STATE-PRESERVING. Measured absence, command failure,
// successful-but-versionless output, exact match, and mismatch are five DISTINCT
// states, never collapsed into a bool. Collapsing them is how "we could not
// establish the version" silently becomes "the version is fine": only VersionMatch
// satisfies a pin, and every other state is a refusal that names itself.
//
// CONSTRAINT PINS ARE A DIFFERENT KIND. `>=1.2.0`, `^1.2.3`, `1.2.*` express a
// RANGE, and running a range through exact-version matching is a category error
// that can only produce a wrong answer. They are classified separately and
// refused here (VersionConstraintPin) rather than silently string-compared; range
// satisfaction is a different contract that this exact-identity witness does not
// claim to decide.
//
// EVIDENCE IS RETAINED, NOT TRUSTED. The complete raw self-report rides along in
// the witness and its receipt so an auditor can re-derive the verdict by hand —
// but it is never the comparison. Retaining evidence and using it as identity are
// different things, and conflating them is the defect this file exists to close.
//
// Like the rest of the package this is a pure, stdlib-only library: it never runs
// a command. The CALLER measures (locates the executable, runs its version probe,
// captures stdout/stderr and the exit error) and hands the observation here as a
// VersionProbe; this file only classifies. That split is what keeps the decision
// testable without a filesystem and keeps provenance free of os/exec.
package provenance

import (
	"encoding/json"
	"strings"
)

// ToolVersion is the TYPED, parsed form of a producer's own version field — the
// unit of comparison this file exists to introduce. Nums holds the dotted numeric
// components in order (1.2.3 -> [1 2 3]); Pre holds the lowercased
// prerelease/build suffix after the first '-' or '+' ("1.2.3-rc1" -> "rc1"), which
// distinguishes a release candidate from its release.
type ToolVersion struct {
	Nums []int  `json:"nums"`
	Pre  string `json:"pre,omitempty"`
}

// ParseToolVersion parses one TOKEN into a typed version, reporting whether it is
// a version at all. It accepts an optional leading "v" (node prints "v20.11.1"),
// trims wrapping punctuation (a banner may parenthesize the field), and requires
// leading dotted components to be non-negative decimal integers. Dotted platform
// or distro release tags (e.g. "windows.3", "el9.x86_64") following the numeric
// components are treated as release metadata rather than syntax errors (#11792).
// A token that merely CONTAINS digits ("tool-1.2.3", "go1.22.3") is not a version
// and is reported as such instead of being mined for a substring.
//
// It is deliberately permissive about arity (a single component parses) because an
// operator-authored pin may legitimately be "14"; the BANNER scanner applies the
// stricter dotted rule. See parseLiveVersion.
func ParseToolVersion(s string) (ToolVersion, bool) {
	t := strings.Trim(strings.TrimSpace(s), "()[]{}<>,;:\"'")
	if t == "" {
		return ToolVersion{}, false
	}
	if t[0] == 'v' || t[0] == 'V' {
		t = t[1:]
	}
	core, pre := t, ""
	if i := strings.IndexAny(t, "-+"); i >= 0 {
		core, pre = t[:i], strings.ToLower(t[i+1:])
	}
	if core == "" {
		return ToolVersion{}, false
	}
	parts := strings.Split(core, ".")
	nums := make([]int, 0, len(parts))
	for i := 0; i < len(parts); i++ {
		p := parts[i]
		if p == "" {
			return ToolVersion{}, false
		}
		if hasNonDigit(p) {
			// A platform or prerelease tag must follow at least one numeric component
			// and must start with a letter (distinguishing it from numeric components with suffixes).
			if len(nums) < 1 || !isLetter(p[0]) {
				return ToolVersion{}, false
			}
			for j := i; j < len(parts); j++ {
				if !isReleaseIdent(parts[j]) {
					return ToolVersion{}, false
				}
			}
			// Dotted prerelease tags (e.g. .rc1, .beta, .alpha, .preview) populate Pre
			// rather than being discarded as platform tags.
			if isPrereleaseTag(p) {
				preTag := strings.ToLower(strings.Join(parts[i:], "."))
				if pre == "" {
					pre = preTag
				} else {
					pre = preTag + "-" + pre
				}
			}
			break
		}
		// Bound the component length so a pathological token cannot overflow the
		// accumulator; no real version component approaches nine digits.
		if len(p) > 9 {
			return ToolVersion{}, false
		}
		n := 0
		for j := 0; j < len(p); j++ {
			n = n*10 + int(p[j]-'0')
		}
		nums = append(nums, n)
	}
	if len(nums) == 0 {
		return ToolVersion{}, false
	}
	return ToolVersion{Nums: nums, Pre: pre}, true
}

func isLetter(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func isPrereleaseTag(s string) bool {
	low := strings.ToLower(s)
	for _, prefix := range []string{"rc", "alpha", "beta", "preview", "pre", "dev"} {
		if strings.HasPrefix(low, prefix) {
			return true
		}
	}
	return false
}

func isReleaseIdent(s string) bool {
	if s == "" || isWildcard(s) {
		return false
	}
	hasAlphaNum := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
			hasAlphaNum = true
			continue
		}
		if c == '_' || c == '-' {
			continue
		}
		return false
	}
	return hasAlphaNum
}

func isWildcard(s string) bool {
	return s == "*" || (len(s) == 1 && (s[0] == 'x' || s[0] == 'X'))
}

func hasNonDigit(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return true
		}
	}
	return false
}

// Equal is the identity comparison: same ARITY, same components elementwise, same
// prerelease. Differing arity is a mismatch rather than a zero-padded match, which
// is what makes "a longer version token cannot satisfy a shorter pin" true by
// construction — 1.2.3.4 does not satisfy a pin of 1.2.3, and neither does 1.2.30.
func (v ToolVersion) Equal(other ToolVersion) bool {
	if len(v.Nums) != len(other.Nums) || v.Pre != other.Pre {
		return false
	}
	for i := range v.Nums {
		if v.Nums[i] != other.Nums[i] {
			return false
		}
	}
	return true
}

// String renders the NORMALIZED parsed version — the canonical form that is
// compared and recorded, not the raw token it came from. An empty version (the
// unparsed zero value) renders "".
func (v ToolVersion) String() string {
	if len(v.Nums) == 0 {
		return ""
	}
	var b strings.Builder
	for i, n := range v.Nums {
		if i > 0 {
			b.WriteByte('.')
		}
		b.WriteString(itoa(n))
	}
	if v.Pre != "" {
		b.WriteByte('-')
		b.WriteString(v.Pre)
	}
	return b.String()
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var buf [12]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

// PinKind separates the two REGISTERED pin shapes, so a range is never fed to an
// exact-identity comparison. The zero value is the fail-closed invalid kind.
type PinKind uint8

const (
	// PinInvalid is a pin that is neither an exact version nor a recognizable
	// constraint — the fail-closed default; it can satisfy nothing.
	PinInvalid PinKind = iota
	// PinExact is a single concrete version, the only kind this witness decides.
	PinExact
	// PinConstraint is a range/wildcard pin (">=1.2.0", "^1.2.3", "1.2.*",
	// ">=1.0 <2.0", "1.x || 2.x"). Represented separately and NEVER exact-matched.
	PinConstraint
)

// String renders the pin kind for the receipt.
func (k PinKind) String() string {
	switch k {
	case PinExact:
		return "exact"
	case PinConstraint:
		return "constraint"
	}
	return "invalid"
}

// ClassifyPin types a registered pin. A pin carrying a comparison operator, a
// wildcard component, or multiple clauses is a CONSTRAINT; a pin that parses whole
// is EXACT; anything else is invalid. Classification happens before any comparison
// so a constraint can never reach the exact-equality path.
func ClassifyPin(pin string) PinKind {
	t := strings.TrimSpace(pin)
	if t == "" {
		return PinInvalid
	}
	// Multi-clause / whitespace-separated ranges: ">=1.0 <2.0", "1.x || 2.x".
	if strings.ContainsAny(t, " \t|,") {
		return PinConstraint
	}
	switch t[0] {
	case '>', '<', '=', '^', '~', '*', '!':
		return PinConstraint
	}
	// Wildcard components: "1.2.*", "1.x".
	probe := t
	if probe[0] == 'v' || probe[0] == 'V' {
		probe = probe[1:]
	}
	for _, seg := range strings.Split(probe, ".") {
		if seg == "*" || strings.EqualFold(seg, "x") {
			return PinConstraint
		}
	}
	if _, ok := ParseToolVersion(t); ok {
		return PinExact
	}
	return PinInvalid
}

// VersionProbe is the MEASURED observation of one tool's live self-report — the
// caller's raw evidence, handed in rather than gathered here. Found and Err are
// the caller's measurement outcome, not a guess: Found=false means the executable
// was looked for and not located (measured absence), and a non-empty Err means the
// probe command was executed and failed.
type VersionProbe struct {
	// Tool is the registered tool name the pin belongs to.
	Tool string `json:"tool"`
	// Path is the resolved executable/launcher path. EVIDENCE ONLY — it is
	// recorded for audit and never scanned for the version, which is precisely the
	// path-confusion hole this file closes.
	Path string `json:"path,omitempty"`
	// Raw is the COMPLETE raw self-report (stdout+stderr of the version probe),
	// retained verbatim as non-authoritative evidence.
	Raw string `json:"raw"`
	// Found reports whether the executable was located at all.
	Found bool `json:"found"`
	// Err is the probe command's failure, empty when the command succeeded.
	Err string `json:"err,omitempty"`
}

// VersionState is the outcome of the live-version check. Every non-match state is
// a distinct, named refusal: collapsing them is how an unestablished version turns
// into a silent pass. The zero value is the fail-closed unknown.
type VersionState uint8

const (
	// VersionUnknown is the fail-closed zero value — nothing was decided.
	VersionUnknown VersionState = iota
	// VersionAbsent is MEASURED absence: the executable was looked for and not found.
	VersionAbsent
	// VersionProbeFailed means the version command ran and failed.
	VersionProbeFailed
	// VersionNoVersion means the command SUCCEEDED but its output carries no
	// parseable version field — a real, distinct state, not a mismatch.
	VersionNoVersion
	// VersionMatch is the only satisfying state: the parsed live version equals the
	// parsed exact pin.
	VersionMatch
	// VersionMismatch means both sides parsed and are different.
	VersionMismatch
	// VersionConstraintPin means the pin is a range, which this exact-identity
	// witness does not decide (and must never string-match).
	VersionConstraintPin
	// VersionPinInvalid means the registered pin is not a usable version at all.
	VersionPinInvalid
)

// String renders the state as its stable receipt token.
func (s VersionState) String() string {
	switch s {
	case VersionAbsent:
		return "absent"
	case VersionProbeFailed:
		return "probe_failed"
	case VersionNoVersion:
		return "no_version"
	case VersionMatch:
		return "match"
	case VersionMismatch:
		return "mismatch"
	case VersionConstraintPin:
		return "constraint_pin"
	case VersionPinInvalid:
		return "pin_invalid"
	}
	return "unknown"
}

// VersionWitness is the public result shape: the typed verdict, the normalized
// parsed readings that produced it, and the retained raw evidence behind them.
type VersionWitness struct {
	Tool string `json:"tool"`
	// Pin is the registered pin verbatim; PinKind types it.
	Pin     string  `json:"pin"`
	PinKind PinKind `json:"-"`
	// Live is the NORMALIZED parsed live version, empty when none was established.
	Live string `json:"live,omitempty"`
	// LiveToken is the exact banner token Live was parsed from — the audit link
	// between the raw evidence and the normalized reading.
	LiveToken string       `json:"live_token,omitempty"`
	State     VersionState `json:"-"`
	Reason    string       `json:"reason"`
	// Raw and Path are retained NON-AUTHORITATIVE evidence: an auditor can
	// re-derive the verdict from them, but neither ever decided it.
	Raw  string `json:"raw"`
	Path string `json:"path,omitempty"`
}

// Satisfied is the single boolean gate: ONLY an exact parsed match satisfies a
// pin. Every refusal — absence, failure, versionless output, mismatch, a
// constraint pin, an invalid pin — is false.
func (w VersionWitness) Satisfied() bool { return w.State == VersionMatch }

// Established reports whether the live version was actually determined (match or
// mismatch). False is the "could not establish" answer: the check produced no
// reading, so nothing may be concluded from it in either direction.
func (w VersionWitness) Established() bool {
	return w.State == VersionMatch || w.State == VersionMismatch
}

// versionReceipt is the on-disk shape: the witness plus its two typed tokens
// rendered as stable strings, so the receipt is machine-readable without importing
// this package's enums.
type versionReceipt struct {
	Tool      string `json:"tool"`
	Pin       string `json:"pin"`
	PinKind   string `json:"pin_kind"`
	State     string `json:"state"`
	Satisfied bool   `json:"satisfied"`
	Live      string `json:"live,omitempty"`
	LiveToken string `json:"live_token,omitempty"`
	Reason    string `json:"reason"`
	Raw       string `json:"raw"`
	Path      string `json:"path,omitempty"`
}

// Receipt renders the witness as the deterministic, machine-readable record of one
// live-version check — the accepted case and the refusal case in the same shape,
// so a reviewer diffs them instead of reading prose.
func (w VersionWitness) Receipt() []byte {
	b, _ := json.MarshalIndent(versionReceipt{
		Tool:      w.Tool,
		Pin:       w.Pin,
		PinKind:   w.PinKind.String(),
		State:     w.State.String(),
		Satisfied: w.Satisfied(),
		Live:      w.Live,
		LiveToken: w.LiveToken,
		Reason:    w.Reason,
		Raw:       w.Raw,
		Path:      w.Path,
	}, "", "  ")
	return b
}

// VerifyToolVersion decides whether a measured live self-report satisfies a
// registered pin, fail-closed, in the contract's fixed order:
//
//  1. The pin is TYPED first. A constraint or an invalid pin never reaches the
//     comparison, so a range can never be exact-string-matched.
//  2. Measured absence and probe failure are reported as themselves — the tool
//     never spoke, so there is no self-report to parse.
//  3. The version FIELD is parsed out of the raw banner. Versionless output is its
//     own state, distinct from a mismatch.
//  4. Only then are the two NORMALIZED PARSED versions compared by exact equality.
//
// The raw self-report and resolved path ride along in the witness as retained
// evidence in every branch, including the refusals.
func VerifyToolVersion(pin string, p VersionProbe) VersionWitness {
	w := VersionWitness{
		Tool:    strings.TrimSpace(p.Tool),
		Pin:     strings.TrimSpace(pin),
		PinKind: ClassifyPin(pin),
		Raw:     p.Raw,
		Path:    p.Path,
	}
	switch w.PinKind {
	case PinConstraint:
		w.State = VersionConstraintPin
		w.Reason = "pin " + w.Pin + " is a constraint, not an exact version: range satisfaction is not decided by the exact-version witness"
		return w
	case PinInvalid:
		w.State = VersionPinInvalid
		w.Reason = "pin " + quote(w.Pin) + " is not a usable version"
		return w
	}
	if !p.Found {
		w.State = VersionAbsent
		w.Reason = "executable for " + quote(w.Tool) + " was not found: live version could not be established"
		return w
	}
	if strings.TrimSpace(p.Err) != "" {
		w.State = VersionProbeFailed
		w.Reason = "version probe for " + quote(w.Tool) + " failed: " + strings.TrimSpace(p.Err)
		return w
	}
	live, token, ok := parseLiveVersion(p.Raw)
	if !ok {
		w.State = VersionNoVersion
		w.Reason = "version probe for " + quote(w.Tool) + " succeeded but reported no parseable version field"
		return w
	}
	w.Live, w.LiveToken = live.String(), token
	want, _ := ParseToolVersion(w.Pin) // PinExact guarantees this parses
	if live.Equal(want) {
		w.State = VersionMatch
		w.Reason = "live version " + w.Live + " matches pin " + want.String()
		return w
	}
	w.State = VersionMismatch
	w.Reason = "live version " + w.Live + " does not match pin " + want.String()
	return w
}

// parseLiveVersion extracts the producer's OWN version field from a raw banner and
// returns it parsed, alongside the exact token it came from.
//
// The scan is deliberately narrow, because every widening is a confusion hole:
//
//   - PATH-SHAPED TOKENS ARE SKIPPED ENTIRELY. A token holding a path separator or
//     a drive letter is an install location, never the version field — this is what
//     stops a pinned version living in `/opt/toolchain-1.2.3/bin` from satisfying a
//     tool whose live version is something else.
//   - THE FIRST parseable token wins. A producer states its own version before any
//     borrowed one, so `mytool 2.0.0 (Python 3.11.4)` reads 2.0.0 and an
//     interpreter version can never be mistaken for the producer's.
//   - AT LEAST TWO DOTTED COMPONENTS are required. A bare integer in a banner is
//     far more often a build number, a year, or a copyright date than a version;
//     requiring the dotted form keeps those from being read as the version field
//     and reported as an established reading.
//
// A banner whose version is fused to a name ("go version go1.22.3") reports NO
// version rather than being mined for a substring: refusing to establish is the
// correct fail-closed answer, and it stays visible as VersionNoVersion.
func parseLiveVersion(raw string) (ToolVersion, string, bool) {
	for _, line := range strings.Split(raw, "\n") {
		for _, tok := range strings.Fields(line) {
			if pathShaped(tok) {
				continue
			}
			v, ok := ParseToolVersion(tok)
			if !ok || len(v.Nums) < 2 {
				continue
			}
			return v, tok, true
		}
	}
	return ToolVersion{}, "", false
}

// pathShaped reports whether a token is a filesystem path rather than a version
// field: it holds a separator, or it opens with a Windows drive letter.
func pathShaped(tok string) bool {
	if strings.ContainsAny(tok, "/\\") {
		return true
	}
	if len(tok) >= 2 && tok[1] == ':' {
		c := tok[0]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
			return true
		}
	}
	return false
}

// quote renders a value for a refusal message, keeping empty readable.
func quote(s string) string {
	if s == "" {
		return `""`
	}
	return `"` + s + `"`
}
