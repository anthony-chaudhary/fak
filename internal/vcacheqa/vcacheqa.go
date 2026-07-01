package vcacheqa

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cachewitness"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/strmatch"
	"github.com/anthony-chaudhary/fak/internal/vcachestar"
)

// ---------------------------------------------------------------------------
// Pillar 1a: Honesty lint (Law A2 elision detector).
// ---------------------------------------------------------------------------

// elisionPhrases name the Law-A2 violation this lint exists to catch: live
// (non-test) source reasoning that a re-send can be SKIPPED because the
// provider is assumed to already hold the bytes. Any of these phrases inside a
// comment or string literal on a non-test .go file under the scanned package is
// a violation — correctness must never be conditioned on a warmth belief.
var elisionPhrases = []string{
	"provider already has",
	"provider has it cached",
	"skip resend",
	"skip re-send",
	"don't resend",
	"do not resend",
	"since it's cached we can skip",
	"because it's warm we can skip",
	"assume the provider has",
	"elide the context because",
	"elide context because",
}

// HonestyDefect is one Law-A2 elision finding: the file/line and the offending
// text, so a reviewer can jump straight to the violation.
type HonestyDefect struct {
	Path string
	Line int
	Text string
}

// HonestyLint AST-scans every non-test .go file under pkgDir (an internal/<gate>
// package directory) for the elision phrases above, appearing either in a
// comment or in a string literal. It mirrors internal/architest's idiom
// (go/parser.ParseDir + go/ast.Inspect over CallExpr/comment groups) rather
// than a text grep, so a match is always attributable to a specific AST node's
// position. An empty result means the gate's live code never reasons "the
// provider probably has it, so we can skip re-sending" — Law A2 held.
func HonestyLint(pkgDir string) ([]HonestyDefect, error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, pkgDir, func(fi fs.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("vcacheqa: parse %s: %w", pkgDir, err)
	}
	var defects []HonestyDefect
	for _, p := range pkgs {
		for path, f := range p.Files {
			// Comments: the most common place a shortcut gets rationalized in prose.
			for _, cg := range f.Comments {
				text := cg.Text()
				if phrase, ok := strmatch.FirstContained(strings.ToLower(text), elisionPhrases); ok {
					pos := fset.Position(cg.Pos())
					defects = append(defects, HonestyDefect{Path: path, Line: pos.Line, Text: phrase})
				}
			}
			// String literals: a log/reason string can carry the same rationalization.
			ast.Inspect(f, func(n ast.Node) bool {
				lit, ok := n.(*ast.BasicLit)
				if !ok || lit.Kind != token.STRING {
					return true
				}
				v, err := strconv.Unquote(lit.Value)
				if err != nil {
					return true
				}
				if phrase, ok := strmatch.FirstContained(strings.ToLower(v), elisionPhrases); ok {
					pos := fset.Position(lit.Pos())
					defects = append(defects, HonestyDefect{Path: path, Line: pos.Line, Text: phrase})
				}
				return true
			})
		}
	}
	sort.Slice(defects, func(i, j int) bool {
		if defects[i].Path != defects[j].Path {
			return defects[i].Path < defects[j].Path
		}
		return defects[i].Line < defects[j].Line
	})
	return defects, nil
}

// ---------------------------------------------------------------------------
// Pillar 1b: Forced cache-MISS test helper (drives vcachestar.FoldTelemetry).
// ---------------------------------------------------------------------------

// ForcedMissResult is the outcome of firing a believed-warm zero-read miss
// through the real demote+byte-diff mechanism.
type ForcedMissResult struct {
	Fold vcachestar.FoldResult
	// Demoted must be true: a gate that manifests "believed warm" but the
	// provider reports zero cache_read MUST demote to cold, never claim a hit
	// that did not happen. This is the lethal "manifest says HIT, provider
	// says MISS" case the issue names explicitly.
	Demoted bool
	// DivergedBytes is true when FirstDivergeByteOffset >= 0, i.e. the fold
	// actually located a byte-level divergence (not just a token-count claim).
	DivergedBytes bool
}

// ForceCacheMiss constructs a believed-warm Belief from lastPrefix/
// lastPrefixBytes and feeds it real provider telemetry reporting a ZERO
// cache_read against currentPrefix/currentPrefixBytes — the forced-MISS
// scenario every M1-M5 gate's own test must be able to fire. It delegates the
// actual reconciliation to vcachestar.FoldTelemetry (the existing
// demote+byte-diff mechanism) rather than reimplementing it, so this helper
// can never drift from the real gate logic it is meant to exercise.
func ForceCacheMiss(lastPrefix, currentPrefix []cachemeta.PromptSegment, lastPrefixBytes, currentPrefixBytes []byte) ForcedMissResult {
	belief := vcachestar.Belief{
		Warm:            true,
		LastPrefix:      lastPrefix,
		LastPrefixBytes: lastPrefixBytes,
	}
	telemetry := vcachestar.Telemetry{
		CacheReadInputTokens: 0, // the forced MISS: provider reports no cache read at all
		UncachedInputTokens:  sumTokens(currentPrefix),
		CurrentPrefix:        currentPrefix,
		CurrentPrefixBytes:   currentPrefixBytes,
	}
	fold := vcachestar.FoldTelemetry(belief, telemetry)
	return ForcedMissResult{
		Fold:          fold,
		Demoted:       fold.Demoted,
		DivergedBytes: fold.FirstDivergeByteOffset >= 0,
	}
}

// sumTokens totals a prefix's segment token counts (the same shape
// vcachestar's own unexported tokenSum helper computes), used as
// ForceCacheMiss's UncachedInputTokens fallback so a caller need not
// pre-compute a token total by hand.
func sumTokens(segs []cachemeta.PromptSegment) int64 {
	var n int64
	for _, s := range segs {
		n += s.Tokens
	}
	return n
}

// ---------------------------------------------------------------------------
// Pillar 2: Non-forgeable witness (journal.Row-shaped hash chain).
// ---------------------------------------------------------------------------

// WitnessRow is byte-schema-identical to internal/journal.Row's JSON shape
// (field names/order match exactly) so a row built here round-trips through
// internal/journal.VerifyRows/ReadRows/Verify without any adapter. The gate
// under test is the PRODUCER (it calls Chain to build its own audit rows);
// the independent reader calls journal.VerifyRows over the same rows and gets
// the identical verdict journal.Verify would give a tampered on-disk journal —
// no number is trusted from the producer's own claim.
type WitnessRow = journal.Row

// Chain builds one hash-chained WitnessRow following a gate's decision. prev
// is the previous row's Hash ("" for the first row in a chain). The hash
// algorithm MIRRORS internal/journal's private chainHash exactly (sha256 over
// prev + unit-separator-delimited Seq/TSUnixNano/Kind/Tool/TraceID/Verdict/
// Reason/By/Taint/ArgsDigest/ResultDigest, in that declared order) because
// internal/journal exports no public row-builder (only Emit(abi.Event), which
// couples a caller to the full abi.Event/Verdict machinery) — reusing the
// journal.Row TYPE and journal.VerifyRows CHECK is the load-bearing reuse;
// this function only fills in that type's chain fields identically so the
// independent journal.VerifyRows call succeeds on an honest chain and fails
// on a tampered one, exactly like a real on-disk journal.
func Chain(prev string, seq uint64, tsUnixNano int64, kind, gate, verdict, reason, argsDigest, resultDigest string) WitnessRow {
	row := journal.Row{
		Seq:          seq,
		TSUnixNano:   tsUnixNano,
		Kind:         kind,
		Tool:         gate,
		Verdict:      verdict,
		Reason:       reason,
		ArgsDigest:   argsDigest,
		ResultDigest: resultDigest,
		PrevHash:     prev,
	}
	row.Hash = chainHash(prev, row)
	return row
}

// chainHash MIRRORS internal/journal's private chainHash byte-for-byte (same
// sha256 pre-image shape: prev + unit-separator-delimited fields in the same
// declared order). Documented at Chain's doc comment: this duplication exists
// ONLY because internal/journal exports no public row constructor; the
// verification path (VerifyRows) is the real, unmodified, independently-owned
// journal.go code, never reimplemented here.
func chainHash(prev string, r journal.Row) string {
	h := sha256.New()
	h.Write([]byte(prev))
	fmt.Fprintf(h, "\x1f%d\x1f%d\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s",
		r.Seq, r.TSUnixNano, r.Kind, r.Tool, r.TraceID, r.Verdict,
		r.Reason, r.By, r.Taint, r.ArgsDigest, r.ResultDigest)
	return hex.EncodeToString(h.Sum(nil))
}

// VerifyWitness is a thin, explicitly-named pass-through to the REAL
// independent-reader check (internal/journal.VerifyRows). It exists only so a
// gate's own test reads "vcacheqa.VerifyWitness" at the call site rather than
// reaching into internal/journal directly — the semantics are exactly
// journal.VerifyRows's, unmodified.
func VerifyWitness(rows []WitnessRow) (int, error) {
	return journal.VerifyRows(rows)
}

// ---------------------------------------------------------------------------
// Pillar 3: Provenance fence (OBSERVED vs WITNESSED, cachewitness vocabulary).
// ---------------------------------------------------------------------------

// externalValueTokens/observedQualifiers/witnessedQualifiers mirror
// internal/conflationscore's token tables (same phrasing, same intent: an
// external/provider-relayed value must carry an OBSERVED-side qualifier, and a
// fak-authored number must carry a WITNESSED-side qualifier). Kept as a small,
// local copy rather than an import of conflationscore's unexported tables, so
// this package can fence a NEW gate surface without editing conflationscore's
// owned files (the collision-avoidance the issue asks for); the vocabulary
// itself is drawn from cachewitness.Provenance (Witnessed/Observed/Modeled),
// the one already-fixed source of truth for the three provenance labels.
var (
	externalValueTokens = []string{
		"cache_read_input_tokens", "cache_creation_input_tokens",
		"provider cache", "upstream", "remote prompt cache",
		"provider-reported", "provider billed",
	}
	observedQualifiers = []string{
		string(cachewitness.Observed), "provider-reported", "relayed", "relays",
		"provider-side", "the provider's",
	}
	witnessedQualifiers = []string{
		string(cachewitness.Witnessed), "fak authored", "fak SENT", "byte-identical",
	}
)

// ProvenanceDefect is one unlabeled-owner finding: a reported fact-string that
// names an external/provider value or a fak-authored value with no
// disambiguating OBSERVED/WITNESSED-side qualifier.
type ProvenanceDefect struct {
	Surface string
	Text    string
	Reason  string
}

// ProvenanceFence checks a gate's own reported fact-strings (help text, log
// reasons, summary lines — whatever the caller extracts and hands in, keyed by
// surface name) against the OBSERVED/WITNESSED vocabulary. It is the same
// check internal/conflationscore already runs over internal/gateway/metrics.go
// and cmd/fak/guard.go, generalized so a NEW gate surface can self-check
// without conflationscore.ReportingSurfaces needing to name it (that list is
// owned by a concurrently-developed package this issue must not edit).
func ProvenanceFence(surfaces map[string][]string) []ProvenanceDefect {
	var defects []ProvenanceDefect
	for _, surface := range sortedSurfaceKeys(surfaces) {
		for _, s := range surfaces[surface] {
			isExternal := hasAny(s, externalValueTokens)
			isObserved := hasAny(s, observedQualifiers)
			isWitnessed := hasAny(s, witnessedQualifiers)
			if isExternal && !isObserved && !isWitnessed {
				defects = append(defects, ProvenanceDefect{
					Surface: surface, Text: s,
					Reason: "external/provider-relayed value reported with no OBSERVED (or WITNESSED) qualifier",
				})
			}
		}
	}
	return defects
}

func hasAny(s string, tokens []string) bool {
	low := strings.ToLower(s)
	for _, t := range tokens {
		if strings.Contains(low, strings.ToLower(t)) {
			return true
		}
	}
	return false
}

func sortedSurfaceKeys(m map[string][]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// ---------------------------------------------------------------------------
// Pillar 4: Determinism (same inputs -> same verdict; no clock/IO in the kernel).
// ---------------------------------------------------------------------------

// DeterminismDefect names a gate decision function that produced two different
// verdicts over the identical input.
type DeterminismDefect struct {
	Reason string
}

// CheckDeterminism re-invokes decide(in) twice over the SAME input and
// compares the two outputs via equal (caller-supplied, since a gate's verdict
// type is not known here). A pure decision kernel (no wall-clock read, no file
// IO, no map-iteration-order leak) must return byte-identical verdicts both
// times; any difference is reported as a defect. Returns nil (clean) when the
// two runs agree.
func CheckDeterminism[T any](decide func() T, equal func(a, b T) bool) *DeterminismDefect {
	a := decide()
	b := decide()
	if !equal(a, b) {
		return &DeterminismDefect{Reason: "two invocations over the identical input produced different verdicts -- the decision kernel is not pure (clock/IO/nondeterministic iteration leaked in)"}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Contract: the single call an M1-M5 child makes to run all four pillars.
// ---------------------------------------------------------------------------

// GateReport folds all four pillars into one result for a single gate.
type GateReport struct {
	Gate              string
	HonestyDefects    []HonestyDefect
	ForcedMiss        ForcedMissResult
	WitnessOK         bool
	WitnessErr        error
	ProvenanceDefects []ProvenanceDefect
	DeterminismDefect *DeterminismDefect
}

// OK reports whether the gate cleared every pillar: no honesty-lint defects,
// the forced-MISS scenario demoted as required, the witness chain verifies,
// no unlabeled-owner provenance defects, and the decision kernel is
// deterministic.
func (r GateReport) OK() bool {
	return len(r.HonestyDefects) == 0 &&
		r.ForcedMiss.Demoted &&
		r.WitnessOK && r.WitnessErr == nil &&
		len(r.ProvenanceDefects) == 0 &&
		r.DeterminismDefect == nil
}

// GatePkgDir resolves internal/<gate> under root for HonestyLint scanning —
// the one path-joining convention every M1-M5 child's own architest-style test
// needs, so each gate's test does not hand-roll filepath.Join(root,
// "internal", gate) itself.
func GatePkgDir(root, gate string) string {
	return filepath.Join(root, "internal", gate)
}
