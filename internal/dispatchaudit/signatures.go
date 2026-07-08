package dispatchaudit

// signatures.go ports the log-signature failure detectors from the
// host-scheduled Python tool `tools/dispatch_log_audit.py` into the Go
// dispatchaudit taxonomy, so one analyzer covers BOTH waste (the Worker/Classify
// fold) and the log-signature failures the Python feeder detected (issue #3337,
// epic #3327 Track B). This removes the dependency on a Python cron for the
// signature lens.
//
// The CORE here is a PURE text fold: raw log text IN, a closed-vocabulary
// SignatureFinding list OUT. It touches no disk, no clock, no GitHub — the I/O
// that reads .dispatch-runs/*.log lives in shell.go (ScanDirSignatures), and the
// gh dedup/file step reuses the existing filer.go substrate via AsFinding.
//
// Closed reason vocabulary (SignatureClass) ↔ Python detector parity:
//
//	panic-traceback      SigPanicTraceback    a `panic:` (Go) or `Traceback (most
//	                                          recent call last):` (Python) at the
//	                                          START of a worker-log line; quoted
//	                                          grep/JSON echoes are skipped.
//	hook-failure-storm   SigHookFailureStorm  ≥HookMin `hook: <name> Failed` lines
//	                                          in one session.
//	off-trunk-storm      SigOffTrunkStorm     repeated OFF_TRUNK guard refusals
//	                                          (a bare mention / quoted repo line is
//	                                          not a refusal).
//	auth-wall            SigAuthWall          a "Not logged in" / credit-balance /
//	                                          usage-limit / login-required streak.
//	banner-only-noop     SigBannerOnlyNoOp    a log whose only content is the
//	                                          fak-guard startup banner (the #1275
//	                                          class) — nothing else ran.
//
// The SignatureClass string VALUES are the Python detector keys verbatim, so a
// signature_key ("<class>::<backend>::<normalized-message>") minted here dedups
// against an issue the Python tool already filed during the transition.
//
// Two deliberate divergences from the Python tool, both documented parity notes:
//   - Backend default: a log with no .backend sidecar resolves to `claude`
//     (matching `dispatch_log_audit.backend_of_log`), NOT BackendUnknown, so the
//     cross-tool signature_key stays stable.
//   - Evidence: this repo forbids replaying raw worker transcript text (see the
//     raw-commit-claim quarantine). Sample lines are captured for unit tests but
//     the fileable Finding carries only STRUCTURED evidence (class, backend,
//     count, session count, log names) — never raw worker output.
import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// SignatureClass is the closed log-signature failure vocabulary. Its values are
// the Python detector keys verbatim (see the file doc) so signature keys are
// cross-tool stable.
type SignatureClass string

const (
	SigPanicTraceback   SignatureClass = "panic-traceback"
	SigHookFailureStorm SignatureClass = "hook-failure-storm"
	SigOffTrunkStorm    SignatureClass = "off-trunk-storm"
	SigAuthWall         SignatureClass = "auth-wall"
	SigBannerOnlyNoOp   SignatureClass = "banner-only-noop"
)

// SignatureClasses returns the closed vocabulary in worst-first severity order.
func SignatureClasses() []SignatureClass {
	return []SignatureClass{
		SigPanicTraceback, SigHookFailureStorm, SigOffTrunkStorm,
		SigAuthWall, SigBannerOnlyNoOp,
	}
}

// SignatureReason is the one-line, raw-text-free reason for a class ("" when the
// token is not in the closed vocabulary — the caller should treat that as a bug,
// never emit it).
func SignatureReason(c SignatureClass) string {
	switch c {
	case SigPanicTraceback:
		return "worker crashed with a panic / traceback"
	case SigHookFailureStorm:
		return "repeated hook-handler failures in one session"
	case SigOffTrunkStorm:
		return "repeated OFF_TRUNK guard refusals"
	case SigAuthWall:
		return "auth / credit wall streak (login or quota)"
	case SigBannerOnlyNoOp:
		return "banner-only worker: startup banner and no work"
	default:
		return ""
	}
}

// signatureTitle is the human title fragment for a class (used in issue titles).
func signatureTitle(c SignatureClass) string {
	switch c {
	case SigPanicTraceback:
		return "worker panic / traceback"
	case SigHookFailureStorm:
		return "hook-handler failure storm"
	case SigOffTrunkStorm:
		return "OFF_TRUNK guard-refusal storm"
	case SigAuthWall:
		return "auth / credit wall streak"
	case SigBannerOnlyNoOp:
		return "banner-only / empty no-op worker"
	default:
		return "log-signature failure"
	}
}

// SignatureThresholds parameterizes the tunable floors. Defaults mirror the
// Python DEFAULTS (hook_min=3).
type SignatureThresholds struct {
	// HookMin is the `hook: … Failed` lines/session floor to call a storm.
	HookMin int
}

// DefaultSignatureThresholds is the calibrated default (Python parity).
func DefaultSignatureThresholds() SignatureThresholds {
	return SignatureThresholds{HookMin: 3}
}

const sigMsgKeep = 160 // trailing chars of a message surviving into the key.

var (
	// digits / hex address → `#` so `[7]` and `[12]` share a signature.
	reSigNumHex = regexp.MustCompile(`0x[0-9a-fA-F]+|\d+`)
	reSigWS     = regexp.MustCompile(`\s+`)
	// a ripgrep-style `path:line:` prefix — a worker echoing repo files that
	// mention panic:/OFF_TRUNK is NOT a real emit.
	reSigQuote = regexp.MustCompile(`^[.\\/]?[\w./\\-]+:\d+:`)

	reSigHook          = regexp.MustCompile(`^\s*hook:\s*(\S+)\s+Failed\b`)
	reSigPyExc         = regexp.MustCompile(`^\s*([A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit)\b.*)$`)
	reSigOffTrunkRefse = regexp.MustCompile(`(?i)refus|blocked|denied|reason|guard`)

	sigAuthPatterns = []struct {
		label string
		re    *regexp.Regexp
	}{
		{"not logged in", regexp.MustCompile(`(?i)not logged in`)},
		{"credit balance too low", regexp.MustCompile(`(?i)credit balance`)},
		{"usage limit reached", regexp.MustCompile(`(?i)usage limit`)},
		{"login required", regexp.MustCompile("(?i)please run\\s*/login|run `?/login`?")},
	}
)

// normalizeSigMessage collapses a raw message to a stable signature token:
// lowercase, every run of digits / hex → `#`, whitespace squeezed, truncated.
func normalizeSigMessage(s string) string {
	s = reSigNumHex.ReplaceAllString(strings.ToLower(strings.TrimSpace(s)), "#")
	s = reSigWS.ReplaceAllString(s, " ")
	if r := []rune(s); len(r) > sigMsgKeep {
		return string(r[:sigMsgKeep])
	}
	return s
}

// signatureKey is the dedup identity of a finding: class + backend + normalized
// message (the exact shape the Python tool stamps).
func signatureKey(class SignatureClass, backend Backend, message string) string {
	return string(class) + "::" + string(backend) + "::" + normalizeSigMessage(message)
}

func looksLikeQuote(line string) bool { return reSigQuote.MatchString(line) }

// sigHit is one detector's raw finding over a single log's text.
type sigHit struct {
	Message string
	Count   int
	Sample  []string
}

func matchHookFailures(text string) []sigHit {
	var sample []string
	n := 0
	for _, ln := range splitLines(text) {
		if reSigHook.MatchString(ln) {
			if len(sample) < 5 {
				sample = append(sample, strings.TrimSpace(ln))
			}
			n++
		}
	}
	if n == 0 {
		return nil
	}
	return []sigHit{{Message: "hook handler failures", Count: n, Sample: sample}}
}

func matchPanic(text string) []sigHit {
	lines := splitLines(text)
	var out []sigHit
	for i, raw := range lines {
		line := strings.TrimSpace(raw)
		switch {
		case strings.HasPrefix(line, "panic:"):
			out = append(out, sigHit{Message: line, Count: 1, Sample: stackSample(lines, i)})
		case strings.HasPrefix(line, "Traceback (most recent call last):"):
			msg := "python traceback"
			if exc := pyExc(lines, i); exc != "" {
				msg += ": " + exc
			}
			out = append(out, sigHit{Message: msg, Count: 1, Sample: stackSample(lines, i)})
		}
	}
	return collapseHits(out)
}

func stackSample(lines []string, i int) []string {
	const span = 4
	end := i + span
	if end > len(lines) {
		end = len(lines)
	}
	var out []string
	for _, ln := range lines[i:end] {
		if strings.TrimSpace(ln) != "" {
			out = append(out, strings.TrimRightFunc(ln, unicode.IsSpace))
		}
	}
	return out
}

// pyExc returns the exception headline a Python traceback ends on
// (`KeyError: 'x'`), scanned forward so identical exceptions share a signature.
func pyExc(lines []string, i int) string {
	end := i + 40
	if end > len(lines) {
		end = len(lines)
	}
	for _, ln := range lines[i+1 : end] {
		m := reSigPyExc.FindStringSubmatch(ln)
		if m != nil && !strings.HasPrefix(strings.TrimLeft(ln, " \t"), "File ") {
			s := strings.TrimSpace(m[1])
			if r := []rune(s); len(r) > 120 {
				return string(r[:120])
			}
			return s
		}
	}
	return ""
}

// collapseHits merges hits that normalize to the same message within one log:
// sum counts, keep the first sample, preserve first-seen order.
func collapseHits(found []sigHit) []sigHit {
	order := []string{}
	by := map[string]*sigHit{}
	for _, f := range found {
		k := normalizeSigMessage(f.Message)
		if cur, ok := by[k]; ok {
			cur.Count += f.Count
			continue
		}
		h := f
		by[k] = &h
		order = append(order, k)
	}
	out := make([]sigHit, 0, len(order))
	for _, k := range order {
		out = append(out, *by[k])
	}
	return out
}

func matchOffTrunk(text string) []sigHit {
	var hits []string
	for _, raw := range splitLines(text) {
		line := strings.TrimSpace(raw)
		if !strings.Contains(strings.ToUpper(line), "OFF_TRUNK") {
			continue
		}
		if looksLikeQuote(line) || !reSigOffTrunkRefse.MatchString(line) {
			continue // a quoted repo line / a bare mention is not a guard refusal
		}
		if r := []rune(line); len(r) > 200 {
			line = string(r[:200])
		}
		hits = append(hits, line)
	}
	if len(hits) == 0 {
		return nil
	}
	return []sigHit{{Message: "OFF_TRUNK guard refusal", Count: len(hits), Sample: capSample(hits, 5)}}
}

func matchAuthWall(text string) []sigHit {
	lines := splitLines(text)
	var out []sigHit
	for _, p := range sigAuthPatterns {
		var hits []string
		for _, ln := range lines {
			t := strings.TrimSpace(ln)
			if p.re.MatchString(ln) && !looksLikeQuote(t) {
				hits = append(hits, t)
			}
		}
		if len(hits) > 0 {
			out = append(out, sigHit{Message: "auth wall: " + p.label, Count: len(hits), Sample: capSample(hits, 3)})
		}
	}
	return out
}

// matchBannerOnly fires when a log's only content is the fak-guard startup
// banner (the #1275 class): at least one banner line, and NO real (non-banner,
// non-blank) line. An error/panic/commit line is not a banner line, so any of
// those keeps a log out of this class.
func matchBannerOnly(text string) []sigHit {
	banner := 0
	real := 0
	for _, raw := range splitLines(text) {
		if strings.TrimSpace(raw) == "" || reSpawnHeader.MatchString(raw) {
			continue // blank / the `# fak-spawn` header are neither banner nor work
		}
		if isBannerLine(raw) {
			banner++
		} else {
			real++
		}
	}
	if banner == 0 || real > 0 {
		return nil
	}
	return []sigHit{{
		Message: "banner-only worker: startup banner and no work",
		Count:   1,
		Sample:  []string{"log carried only the fak-guard startup banner"},
	}}
}

func capSample(in []string, n int) []string {
	if len(in) > n {
		return append([]string(nil), in[:n]...)
	}
	return append([]string(nil), in...)
}

func splitLines(text string) []string {
	return strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
}

// sigDetector is one entry of the closed detector table.
type sigDetector struct {
	class     SignatureClass
	severity  int
	perLogMin int // storm floor within a single session
	minTotal  int // aggregated-across-logs floor before a finding is fileable
	match     func(string) []sigHit
}

func signatureDetectors(th SignatureThresholds) []sigDetector {
	hookMin := th.HookMin
	if hookMin <= 0 {
		hookMin = DefaultSignatureThresholds().HookMin
	}
	return []sigDetector{
		{SigPanicTraceback, 100, 1, 1, matchPanic},
		{SigHookFailureStorm, 80, hookMin, hookMin, matchHookFailures},
		{SigOffTrunkStorm, 60, 1, 2, matchOffTrunk},
		{SigAuthWall, 50, 1, 3, matchAuthWall},
		{SigBannerOnlyNoOp, 40, 1, 2, matchBannerOnly},
	}
}

// SigLog is one already-read worker log — the PURE input to the signature fold.
type SigLog struct {
	Name    string  // resolve-*.log base name
	Backend Backend // resolved backend (claude default when no sidecar)
	Text    string  // the (byte-capped) log text
}

// SignatureFinding is a fingerprinted, fileable log-signature failure.
type SignatureFinding struct {
	Class    SignatureClass `json:"class"`
	Severity int            `json:"severity"`
	Backend  Backend        `json:"backend"`
	Message  string         `json:"message"`
	Count    int            `json:"count"`
	Logs     []string       `json:"logs,omitempty"`
	Sample   []string       `json:"sample,omitempty"`

	minTotal int // filing floor (unexported; not serialized)
}

// SignatureKey is the cross-tool dedup identity.
func (sf SignatureFinding) SignatureKey() string {
	return signatureKey(sf.Class, sf.Backend, sf.Message)
}

// Fingerprint is the stable 16-hex hash over the signature key (namespaced so it
// can never collide with an outcome finding's fingerprint).
func (sf SignatureFinding) Fingerprint() string {
	sum := sha256.Sum256([]byte("sig:" + sf.SignatureKey()))
	return hex.EncodeToString(sum[:])[:16]
}

// ScanLogText runs every detector over ONE log's text, applying the per-log
// storm floor. PURE: text in, per-log findings out.
func ScanLogText(name string, backend Backend, text string, th SignatureThresholds) []SignatureFinding {
	if backend == "" {
		backend = BackendClaude
	}
	var out []SignatureFinding
	for _, det := range signatureDetectors(th) {
		for _, h := range det.match(text) {
			if h.Count < det.perLogMin {
				continue
			}
			out = append(out, SignatureFinding{
				Class:    det.class,
				Severity: det.severity,
				Backend:  backend,
				Message:  h.Message,
				Count:    h.Count,
				Logs:     []string{name},
				Sample:   h.Sample,
				minTotal: det.minTotal,
			})
		}
	}
	return out
}

// AggregateSignatures collapses per-log findings into one candidate per
// signature key: sum counts, union logs, bounded sample. Deterministic on the
// input order (callers feed name-sorted logs).
func AggregateSignatures(findings []SignatureFinding) []SignatureFinding {
	order := []string{}
	by := map[string]*SignatureFinding{}
	for _, f := range findings {
		key := f.SignatureKey()
		c := by[key]
		if c == nil {
			cp := SignatureFinding{
				Class: f.Class, Severity: f.Severity, Backend: f.Backend,
				Message: f.Message, minTotal: f.minTotal,
			}
			by[key] = &cp
			order = append(order, key)
			c = &cp
		}
		c.Count += f.Count
		for _, lg := range f.Logs {
			if !contains(c.Logs, lg) {
				c.Logs = append(c.Logs, lg)
			}
		}
		for _, s := range f.Sample {
			if len(c.Sample) < 6 && !contains(c.Sample, s) {
				c.Sample = append(c.Sample, s)
			}
		}
	}
	out := make([]SignatureFinding, 0, len(order))
	for _, k := range order {
		out = append(out, *by[k])
	}
	return out
}

func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

// FoldSignatures is the PURE entry point: logs in, the fileable candidate list
// out (aggregated across logs, filtered by the per-signature min-total floor,
// sorted worst-first with a stable tiebreak). Deterministic.
func FoldSignatures(logs []SigLog, th SignatureThresholds) []SignatureFinding {
	var all []SignatureFinding
	for _, lg := range logs {
		all = append(all, ScanLogText(lg.Name, lg.Backend, lg.Text, th)...)
	}
	agg := AggregateSignatures(all)
	kept := agg[:0]
	for _, c := range agg {
		if c.Count >= c.minTotal {
			kept = append(kept, c)
		}
	}
	sort.SliceStable(kept, func(i, j int) bool {
		if kept[i].Severity != kept[j].Severity {
			return kept[i].Severity > kept[j].Severity
		}
		if kept[i].Count != kept[j].Count {
			return kept[i].Count > kept[j].Count
		}
		return kept[i].SignatureKey() < kept[j].SignatureKey()
	})
	return kept
}

// AsFinding bridges a SignatureFinding into the existing fileable Finding type so
// `--file-issues` dedups + files it with the same substrate as the outcome
// findings. It carries ONLY structured evidence — never raw worker sample text.
func (sf SignatureFinding) AsFinding() Finding {
	first := ""
	if len(sf.Logs) > 0 {
		first = sf.Logs[0]
	}
	ev := "class=" + string(sf.Class) +
		" backend=" + safeEvidenceToken(string(sf.Backend)) +
		" count=" + strconv.Itoa(sf.Count) +
		" sessions=" + strconv.Itoa(len(sf.Logs))
	detail := SignatureReason(sf.Class) + " — " + strconv.Itoa(sf.Count) +
		" occurrence(s) across " + strconv.Itoa(len(sf.Logs)) + " session(s)"
	return Finding{
		Fingerprint:    sf.Fingerprint(),
		Backend:        sf.Backend,
		CodeSite:       sf.SignatureKey(),
		Log:            first,
		Title:          "dispatch-log-audit: " + signatureTitle(sf.Class) + " on " + string(sf.Backend) + " (" + strconv.Itoa(sf.Count) + "×)",
		Detail:         detail,
		Evidence:       ev,
		SignatureClass: sf.Class,
	}
}
