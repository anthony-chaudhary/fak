// Package conflationscore is the Go port of tools/conflation_scorecard.py -- the
// anti-conflation / provenance-honesty stick.
//
// The lesson it generalizes (from the compaction-metrics fix): every number or status a
// surface reports should declare its PROVENANCE -- is it a fact fak AUTHORED/WITNESSED, or a
// value fak OBSERVED from an external party and merely RELAYS? -- and a bad OBSERVED value
// must never be attributed to a fak ACTION unless a WITNESSED signal proves the fault is
// fak's. Conflating the two makes a provider-side miss read as "fak broke the cache", which
// erodes trust in the one number fak can stand behind.
//
// It is a TREE-READING scorecard (no data dir): the rendered fact-strings ARE the data, read
// from the Go source, so it cannot be gamed by editing a JSON file -- only by fixing the
// prose. The scoring fold/grade/markdown machinery lives in pkg/scorecard; this package holds
// only the conflation-specific token tables, extraction, and KPIs.
package conflationscore

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id, byte-identical to the Python card so the fold and any
// consumer cannot tell a Go card from its Python predecessor.
const Schema = "fak-conflation-scorecard/1"

// DebtKey is the headline integer the control-pane folds (scorecard_control_pane.py reads
// corpus.conflation_debt).
const DebtKey = "conflation_debt"

// CleanFloor is the disciplined tree's expected debt; the live-tree smoke pins it.
const CleanFloor = 0

// ReportingSurfaces are the fact-reporting Go files this scorecard gardens. Each renders
// operator-facing numbers/statuses (metric help, exit summaries). Add a surface here when a
// new reporting file lands; the scorecard then holds it to the same provenance discipline.
var ReportingSurfaces = []string{
	"internal/gateway/metrics.go",
	"cmd/fak/guard.go",
}

// externalValueTokens mark a reported value as coming from an EXTERNAL party fak relays. A
// help/summary string mentioning one is reporting an OBSERVED value and must label it so.
var externalValueTokens = []string{
	"cache_read_input_tokens",
	"cache_creation_input_tokens",
	"provider cache",
	"upstream",
	"remote prompt cache",
	"provider-reported",
	"provider billed",
	"what the provider billed",
}

// observedQualifiers honestly label a reported external value as relayed (the OBSERVED side).
var observedQualifiers = []string{
	"OBSERVED", "provider-reported", "relayed", "relays", "not a fak claim",
	"not a claim about what the provider", "attribute nothing to itself",
	"attributes nothing to itself", "the provider's call", "the provider's number",
	// honest phrasings already in the tree that genuinely scope a value as external/
	// provider-side without the literal OBSERVED token -- recognizing these is seeing real
	// honesty, not weakening the detector:
	"provider-side", "performance evidence", "distinct from the local",
	"distinct from local",
}

// witnessedQualifiers label a self-authored counter (the WITNESSED side).
var witnessedQualifiers = []string{
	"WITNESSED", "fak authored", "fak SENT", "what fak SENT", "byte-identical",
}

// attributionPhrases BLAME a fak action for an outcome. Applied to an external value without a
// co-located disambiguating marker, each is a false-attribution defect. Ported verbatim from
// the Python ATTRIBUTION_PHRASES; Go RE2 handles the (?:...) groups, and (?i) gives the
// re.IGNORECASE the Python search used.
var attributionPhrases = []*regexp.Regexp{
	regexp.MustCompile(`(?i)the cache broke`),
	regexp.MustCompile(`(?i)broke the cache`),
	regexp.MustCompile(`(?i)the splice is producing a body the provider re-bills`),
	regexp.MustCompile(`(?i)the splice broke`),
	regexp.MustCompile(`(?i)every fire is re-billing`),
	regexp.MustCompile(`(?i)compaction (?:is )?costing money`),
	regexp.MustCompile(`(?i)means the cache broke`),
}

// disambiguationMarkers near an attribution phrase make it honest (describing, not asserting,
// the failure). A marker on the same string neutralizes an attribution phrase there.
var disambiguationMarkers = []string{
	"conflation", "is the conflation", "not something fak", "does NOT control",
	"does not control", "is NOT that bug", "is not that bug", "reading the crater",
	"reading a low", "unless", "provider-side", "only bail_reason", "the ONLY fak-fault",
}

// faultSignalNamed matches prose that names a single fak-fault signal (the "only X>0 is our
// bug" pattern), ported from the Python re.search with re.I.
var faultSignalNamed = regexp.MustCompile(`(?i)only\b.{0,40}\bis\s+(?:fak's|the\s+\w+\s+)?bug`)

// help/summary extraction regexes, ported from extract_help_strings. The Go string-literal
// backslashes mirror the Python raw strings; ([^"\\]|\\.)* captures a Go double-quoted string
// body with escapes.
var (
	reWriteHelp = regexp.MustCompile(`write(?:HelpType|Counter)\(\s*[^,]+,\s*"[^"]*",\s*"((?:[^"\\]|\\.)*)"`)
	reFprintf   = regexp.MustCompile(`Fprintf\(\s*&?\w+\s*,\s*"((?:[^"\\]|\\.)*)"`)
)

// ExtractHelpStrings pulls the operator-facing fact-strings out of a reporting source file:
// the help arguments of writeHelpType/writeCounter and the format strings of the exit-summary
// Fprintf calls. These are the strings a human reads, so they are what the KPIs grade.
func ExtractHelpStrings(text string) []string {
	var out []string
	for _, m := range reWriteHelp.FindAllStringSubmatch(text, -1) {
		out = append(out, unescape(m[1]))
	}
	for _, m := range reFprintf.FindAllStringSubmatch(text, -1) {
		s := unescape(m[1])
		low := strings.ToLower(s)
		if strings.Contains(s, "fak guard:") || strings.Contains(low, "compaction") || strings.Contains(low, "cache") {
			out = append(out, s)
		}
	}
	return out
}

func unescape(s string) string {
	s = strings.ReplaceAll(s, `\"`, `"`)
	s = strings.ReplaceAll(s, `\n`, " ")
	s = strings.ReplaceAll(s, `\t`, " ")
	return s
}

// kpiProvenanceLabeled (HARD): every help/summary string reporting an EXTERNAL value must
// carry an OBSERVED-side qualifier so a reader knows fak is relaying, not asserting, it.
func kpiProvenanceLabeled(surfaces map[string][]string) scorecard.KPI {
	var defects []string
	totalExternal := 0
	for _, path := range sortedKeys(surfaces) {
		sid := lastSegment(path)
		for _, s := range surfaces[path] {
			if scorecard.HasAny(s, externalValueTokens) {
				totalExternal++
				if !scorecard.HasAny(s, observedQualifiers) {
					defects = append(defects, sid+": external value reported without an OBSERVED/provider-relayed qualifier: \""+scorecard.Clip(s, 90)+"\"")
				}
			}
		}
	}
	score := 100.0
	if len(defects) > 0 {
		denom := totalExternal
		if denom < 1 {
			denom = 1
		}
		score = 100.0 * (1 - float64(len(defects))/float64(denom))
		if score < 0 {
			score = 0
		}
	}
	return scorecard.KPI{
		Key: "provenance_labeled", Group: "honesty", Score: score,
		Detail:  detailCounts(totalExternal-len(defects), totalExternal, "external-value strings carry a provenance label"),
		Defects: defects,
	}
}

// kpiNoFalseAttribution (HARD): no help/summary prose may blame a fak ACTION for a bad
// OBSERVED value unless a co-located marker disambiguates it.
func kpiNoFalseAttribution(surfaces map[string][]string) scorecard.KPI {
	var defects []string
	for _, path := range sortedKeys(surfaces) {
		sid := lastSegment(path)
		for _, s := range surfaces[path] {
			for _, pat := range attributionPhrases {
				if pat.MatchString(s) {
					if !scorecard.HasAny(s, disambiguationMarkers) {
						defects = append(defects, sid+": attributes an observed miss to a fak action with no disambiguation: \""+scorecard.Clip(s, 90)+"\"")
					}
					break
				}
			}
		}
	}
	score := 100.0
	detail := "no observed-miss-blamed-on-fak prose"
	if len(defects) > 0 {
		score = 0.0
		detail = plural(len(defects), "false-attribution string")
	}
	return scorecard.KPI{
		Key: "no_false_attribution", Group: "honesty", Score: score,
		Detail: detail, Defects: defects,
	}
}

// kpiFaultSignalIsolated (SOFT): a metric family mixing witnessed + observed values should
// name exactly one fault signal the help points at (the "only X>0 is our bug" pattern), so a
// reader knows which single signal means the fault is genuinely fak's.
func kpiFaultSignalIsolated(surfaces map[string][]string) scorecard.KPI {
	var soft []string
	for _, path := range sortedKeys(surfaces) {
		sid := lastSegment(path)
		blob := strings.Join(surfaces[path], " ")
		hasObserved := scorecard.HasAny(blob, externalValueTokens)
		hasWitnessed := scorecard.HasAny(blob, witnessedQualifiers)
		if hasObserved && hasWitnessed {
			low := strings.ToLower(blob)
			namesFault := faultSignalNamed.MatchString(blob) ||
				strings.Contains(low, "fak-fault") || strings.Contains(low, "prefix_mismatch")
			if !namesFault {
				soft = append(soft, sid+": mixes WITNESSED + OBSERVED values but names no single fak-fault signal -- a reader can't tell which signal means our bug")
			}
		}
	}
	score := 100.0
	detail := "fault signal isolated where families mix provenance"
	if len(soft) > 0 {
		// No "(s)" suffix here -- the Python detail was a bare count + phrase
		// (conflation_scorecard.py:204), unlike the false-attribution detail which had "(s)".
		detail = fmt.Sprintf("%d mixed family without an isolated fault signal", len(soft))
		score = 70.0
	}
	return scorecard.KPI{
		Key: "fault_signal_isolated", Group: "honesty", Score: score,
		Detail: detail, Soft: soft,
	}
}

// Build reads the reporting surfaces, runs the three KPIs, and folds them into the
// control-pane payload via the shared kernel. root is the repo root.
func Build(root string) scorecard.Payload {
	surfaces := map[string][]string{}
	externalSeen := 0
	for _, rel := range ReportingSurfaces {
		text := scorecard.SafeRead(filepath.Join(root, filepath.FromSlash(rel)))
		strs := ExtractHelpStrings(text)
		surfaces[rel] = strs
		for _, s := range strs {
			if scorecard.HasAny(s, externalValueTokens) {
				externalSeen++
			}
		}
	}
	kpis := []scorecard.KPI{
		kpiProvenanceLabeled(surfaces),
		kpiNoFalseAttribution(surfaces),
		kpiFaultSignalIsolated(surfaces),
	}
	debt := 0
	for _, k := range kpis {
		debt += len(k.Defects)
	}
	finding := "every reported fact labels its provenance and blames no provider-side miss on fak"
	next := "hold -- re-run after a new reporting surface lands"
	if debt > 0 {
		finding = plural(debt, "conflation defect") + ": a reported value is unlabeled or a provider miss is attributed to a fak action"
		next = "fix " + worstKPI(kpis) + ": label external values OBSERVED / correct the attribution"
	}
	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStrict,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		ExtraCorpus: map[string]any{
			"surfaces":             len(ReportingSurfaces),
			"external_values_seen": externalSeen,
		},
	})
	p.Workspace = root
	return p
}

// --- small local helpers (string shaping unique to this card's prose) --------------------

func sortedKeys(m map[string][]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// ReportingSurfaces order is the natural one; preserve it rather than alphabetizing so the
	// defect order matches the Python dict-insertion iteration.
	ordered := make([]string, 0, len(keys))
	seen := map[string]bool{}
	for _, rel := range ReportingSurfaces {
		if _, ok := m[rel]; ok {
			ordered = append(ordered, rel)
			seen[rel] = true
		}
	}
	for _, k := range keys {
		if !seen[k] {
			ordered = append(ordered, k)
		}
	}
	return ordered
}

func lastSegment(path string) string {
	if i := strings.LastIndexAny(path, "/\\"); i >= 0 {
		return path[i+1:]
	}
	return path
}

func detailCounts(have, total int, suffix string) string {
	return fmt.Sprintf("%d/%d %s", have, total, suffix)
}

// plural renders "N noun" with a trailing "(s)" when N != 1, matching the Python f-strings
// that wrote "{n} conflation defect(s)".
func plural(n int, noun string) string {
	s := fmt.Sprintf("%d %s", n, noun)
	if n != 1 {
		s += "(s)"
	}
	return s
}

func worstKPI(kpis []scorecard.KPI) string {
	worst := kpis[0]
	for _, k := range kpis[1:] {
		if len(k.Defects) > len(worst.Defects) {
			worst = k
		}
	}
	return worst.Key
}
