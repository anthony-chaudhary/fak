// layout-contract check — the lg-layoutcontract recurrence guard over the
// declarative tensor layout contract registry
// (internal/compute/layoutcontract). This is the registry's first consumer:
// nothing outside its own package imports it before this verb.
//
// WHY THIS EXISTS. The ledger rows LML-04 and LML-22 (published-vs-reference
// layout divergence) were caught post-hoc because the layout fact lived only in
// kernel comments. The registry pins each family's closed fact set — storage
// order, scale layout, padding rule, packing path, and a path:line@sha source
// ref — so that any future move or rewrite surfaces here as a STALE PIN to
// re-pin instead of as silent wrong numbers on a hardware run. The four checks
// below make that true mechanically:
//
//  1. integrity          every entry Validate()s against the registry's closed
//     invariants (closed enums, no half-registered pair);
//  2. family-coverage    the registered family set is cross-checked against
//     docs/data/packing-coverage.json (schema
//     fak.packingmatrix.v1); deltas are reported EXPLICITLY
//     as gaps, never folded into a false "complete";
//  3. source-pins        every SourceRef resolves to an existing file whose
//     line count still reaches the pinned line, and a
//     transform pin's line still cites its forward
//     function;
//  4. transform-roundtrip every registered pair is driven forward then
//     inverse on the same deterministic payloads its
//     package-level witness uses, requiring byte identity.
//
// The outcome vocabulary is closed: pass | gaps | fail. Gaps are coverage
// deltas (exit 0, reported); violations are contract breaks (exit 1). Refusals
// — unknown family, missing or schema-mismatched matrix — exit 2 with a
// sentinel error, because a check that cannot be completed must not read as
// "no gaps found". Every check takes its inputs by parameter (LayoutOptions),
// so planted-defect tests drive injected contract lists and fixture trees
// without mutating the live registry.
package devcmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute/layoutcontract"
	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// layoutCheckSchema is the closed machine-output schema of the check report.
const layoutCheckSchema = "fak.layoutcontract.check.v1"

// Closed outcome vocabulary.
const (
	layoutOutcomePass = "pass"
	layoutOutcomeGaps = "gaps"
	layoutOutcomeFail = "fail"
)

// Closed per-check status values.
const (
	layoutStatusOK       = "ok"
	layoutStatusGaps     = "gaps"
	layoutStatusViolated = "violated"
)

// Closed check names, in execution order.
const (
	layoutCheckIntegrity = "integrity"
	layoutCheckCoverage  = "family-coverage"
	layoutCheckPins      = "source-pins"
	layoutCheckRoundTrip = "transform-roundtrip"
)

// layoutCheckNames is the closed ordered set of check names in the report.
var layoutCheckNames = []string{layoutCheckIntegrity, layoutCheckCoverage, layoutCheckPins, layoutCheckRoundTrip}

// Gap kinds, closed.
const (
	layoutGapMatrixOnly   = "matrix-only"
	layoutGapRegistryOnly = "registry-only"
)

var (
	// ErrLayoutIntegrity wraps a registry entry that violates a closed
	// invariant (Validate() failure, including a half-registered transform
	// pair and a duplicate family tag).
	ErrLayoutIntegrity = errors.New("layout-contract: registry entry violates a closed invariant")
	// ErrLayoutStalePin wraps a SourceRef that no longer resolves to its
	// cited definition site.
	ErrLayoutStalePin = errors.New("layout-contract: source pin does not resolve to its cited definition")
	// ErrLayoutRoundTrip wraps a transform pair whose inverse o forward is
	// not the byte identity on a legal shape.
	ErrLayoutRoundTrip = errors.New("layout-contract: transform pair is not byte-identical under inverse o forward")
	// ErrLayoutNoWitness wraps a registered pair the check cannot drive:
	// no round-trip witness shape is registered for its family.
	ErrLayoutNoWitness = errors.New("layout-contract: registered transform pair has no round-trip witness shape")
	// ErrLayoutMatrixMissing wraps the packing coverage matrix being absent
	// or unreadable at the expected path.
	ErrLayoutMatrixMissing = errors.New("layout-contract: packing coverage matrix unavailable")
	// ErrLayoutMatrixSchema wraps the matrix declaring a foreign schema or
	// carrying malformed cells.
	ErrLayoutMatrixSchema = errors.New("layout-contract: packing coverage matrix schema mismatch")
)

// packingMatrixPath and packingMatrixSchema locate the landed data leaf this
// guard cross-checks against (ticket lg-packing-matrix).
const (
	packingMatrixPath   = "docs/data/packing-coverage.json"
	packingMatrixSchema = "fak.packingmatrix.v1"
)

// LayoutOptions injects every input the checks need. Zero value checks the
// live registry from the discovered repository root.
type LayoutOptions struct {
	// Root is the repository root SourceRefs and the packing matrix resolve
	// against; empty discovers it via devindex.FindRoot(".").
	Root string
	// Family scopes every check to one registered family; empty checks all.
	// Coverage deltas are then restricted to rows involving that family.
	Family string
	// Contracts replaces the live registry; nil uses layoutcontract.All().
	// Injected lists are checked as given, so a planted defect never mutates
	// the global registry.
	Contracts []layoutcontract.Contract
	// SkipSourcePins disables the source-pin freshness check. RunLayoutContract
	// sets it when the operator re-bases the check onto an explicit --root: the
	// live registry's SourceRefs point into the repository that registered
	// them, so a foreign tree can neither prove nor refute pin freshness and a
	// coverage-only re-base must not read registry-only gaps as contract
	// breaks. Injected contract lists (planted-defect drills) never set the
	// skip, so their pins stay enforced against the given root.
	SkipSourcePins bool
}

// LayoutGapRow is one registry<->matrix family delta.
type LayoutGapRow struct {
	Kind   string `json:"kind"`
	Family string `json:"family"`
}

// LayoutViolation is one failed check row. Err carries the sentinel-wrapped
// cause so callers can assert with errors.Is; it is never serialized.
type LayoutViolation struct {
	Check  string `json:"check"`
	Family string `json:"family,omitempty"`
	Pin    string `json:"pin,omitempty"`
	Reason string `json:"reason"`
	Err    error  `json:"-"`
}

// LayoutCheckRow is one named check's closed status plus a one-line detail.
type LayoutCheckRow struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Detail string `json:"detail,omitempty"`
}

// LayoutReport is the closed machine output of the check.
type LayoutReport struct {
	Schema     string            `json:"schema"`
	Outcome    string            `json:"outcome"`
	Scope      string            `json:"scope"`
	Root       string            `json:"root,omitempty"`
	Families   []string          `json:"families"`
	Checks     []LayoutCheckRow  `json:"checks"`
	Gaps       []LayoutGapRow    `json:"gaps,omitempty"`
	Violations []LayoutViolation `json:"violations,omitempty"`
}

// layoutCheckError aggregates every violation of one check run. Its Unwrap
// chain exposes each violation's sentinel error, so errors.Is reports true
// for any sentinel a violation carries.
type layoutCheckError struct {
	violations []LayoutViolation
}

func (e *layoutCheckError) Error() string {
	return fmt.Sprintf("layout-contract: %d violation(s)", len(e.violations))
}

func (e *layoutCheckError) Unwrap() []error {
	out := make([]error, len(e.violations))
	for i, v := range e.violations {
		out[i] = v.Err
	}
	return out
}

// layoutWitness drives one family's registered pair through the SAME uniform
// Shape/payload convention as the registry's own round-trip witness
// (internal/compute/layoutcontract/roundtrip_test.go): Q4_K takes the raw
// row-major GGUF payload (rows*(cols/256)*144 bytes), NVFP4 takes the linear
// block-scale array the swizzle permutes (rows*cols bytes). A family with a
// Present() pair but no witness entry fails closed with ErrLayoutNoWitness
// instead of skipping silently.
type layoutWitness struct {
	shapes  []layoutcontract.Shape
	payload func(s layoutcontract.Shape) int
}

var defaultLayoutWitnesses = map[string]layoutWitness{
	"Q4_K": {
		shapes: []layoutcontract.Shape{
			{Rows: 5, Cols: 256, Width: 1}, // identity width, one super-block per row
			{Rows: 5, Cols: 512, Width: 4}, // two super-blocks, tail group of 1 row
			{Rows: 3, Cols: 256, Width: 8}, // short tail group (3 rows mod 8)
			{Rows: 8, Cols: 768, Width: 8}, // full 8-row groups, three super-blocks
		},
		payload: func(s layoutcontract.Shape) int { return s.Rows * (s.Cols / 256) * 144 },
	},
	"NVFP4": {
		shapes: []layoutcontract.Shape{
			{Rows: 1, Cols: 1},     // smallest legal payload, fully padded tile
			{Rows: 128, Cols: 4},   // exactly one 128x4 tile, no padding
			{Rows: 130, Cols: 7},   // ragged tail row and column tiles
			{Rows: 256, Cols: 100}, // two row tiles, ragged columns
		},
		payload: func(s layoutcontract.Shape) int { return s.Rows * s.Cols },
	},
}

// layoutDeterministicPayload sizes one pseudo-random payload with the exact
// seed convention of the registry's round-trip witness, so the guard drives
// the same bytes the package-level test drives.
func layoutDeterministicPayload(family string, n int) []byte {
	r := rand.New(rand.NewSource(int64(len(family)*1000 + n)))
	b := make([]byte, n)
	r.Read(b)
	return b
}

// layoutMatrixTag maps a registry family tag onto the packing matrix spelling
// of the same family where they differ. The alias table is CLOSED: an
// unmatched spelling surfaces as a gap row instead of silently counting as
// covered, so a new matrix spelling forces a witnessed decision here.
var layoutMatrixAlias = map[string]string{
	"mxfp4": "mxfp4_e8m0",
}

func layoutMatrixTag(family string) string {
	norm := strings.ToLower(family)
	if alias, ok := layoutMatrixAlias[norm]; ok {
		return alias
	}
	return norm
}

// layoutReadPin resolves one SourceRef against root. It returns the pinned
// line's text when the file exists and the line is in range.
func layoutReadPin(root string, ref layoutcontract.SourceRef) (line string, ok bool, reason string) {
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(ref.Path)))
	if err != nil {
		return "", false, fmt.Sprintf("file %s does not exist or is unreadable", ref.Path)
	}
	lines := layoutSplitLines(data)
	if ref.Line < 1 || ref.Line > len(lines) {
		return "", false, fmt.Sprintf("pinned line %d exceeds the file's %d lines", ref.Line, len(lines))
	}
	return lines[ref.Line-1], true, ""
}

// layoutSplitLines counts physical lines the way a re-pin would: CRLF folds
// to LF and a single trailing newline does not mint an extra line.
func layoutSplitLines(data []byte) []string {
	s := strings.ReplaceAll(string(data), "\r\n", "\n")
	s = strings.TrimSuffix(s, "\n")
	if s == "" {
		return nil
	}
	return strings.Split(s, "\n")
}

// layoutForwardName extracts the forward function name from a Transform.Name
// like "bench.RepackQ4KInterleaved / bench.UnrepackQ4KInterleaved": the first
// half of the pair, package qualifier stripped. An empty result disables the
// pin-content check for that entry.
func layoutForwardName(name string) string {
	half := name
	if i := strings.Index(name, "/"); i >= 0 {
		half = name[:i]
	}
	half = strings.TrimSpace(half)
	if i := strings.LastIndex(half, "."); i >= 0 {
		half = half[i+1:]
	}
	return half
}

// loadPackingMatrix returns the sorted distinct family tags of the landed
// packing coverage matrix (cells plus excluded_pairs). Missing or
// schema-mismatched matrices refuse with sentinels rather than reading as
// "no gaps found".
func loadPackingMatrix(root string) ([]string, error) {
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(packingMatrixPath)))
	if err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrLayoutMatrixMissing, packingMatrixPath, err)
	}
	var m struct {
		Schema string `json:"schema"`
		Cells  []struct {
			Family string `json:"family"`
		} `json:"cells"`
		ExcludedPairs []struct {
			Family string `json:"family"`
		} `json:"excluded_pairs"`
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("%w: %s: %v", ErrLayoutMatrixSchema, packingMatrixPath, err)
	}
	if m.Schema != packingMatrixSchema {
		return nil, fmt.Errorf("%w: %s declares schema %q, want %q", ErrLayoutMatrixSchema, packingMatrixPath, m.Schema, packingMatrixSchema)
	}
	set := map[string]bool{}
	for _, c := range m.Cells {
		if c.Family != "" {
			set[strings.ToLower(c.Family)] = true
		}
	}
	for _, e := range m.ExcludedPairs {
		if e.Family != "" {
			set[strings.ToLower(e.Family)] = true
		}
	}
	out := make([]string, 0, len(set))
	for f := range set {
		out = append(out, f)
	}
	sort.Strings(out)
	return out, nil
}

// CheckLayoutContracts runs the four checks and returns the closed report.
// A nil error means outcome pass|gaps; a returned error is either a refusal
// (sentinel-wrapped, no report) or a *layoutCheckError carrying every
// violation's sentinel for errors.Is.
func CheckLayoutContracts(opts LayoutOptions) (*LayoutReport, error) {
	root := opts.Root
	if root == "" {
		root = devindex.FindRoot(".")
	}
	contracts := opts.Contracts
	if contracts == nil {
		contracts = layoutcontract.All()
	}

	rep := &LayoutReport{Schema: layoutCheckSchema, Scope: "all", Root: root}
	var violations []LayoutViolation
	addViolation := func(check, family, pin, reason string, sentinel error) {
		violations = append(violations, LayoutViolation{
			Check:  check,
			Family: family,
			Pin:    pin,
			Reason: reason,
			Err:    fmt.Errorf("%w: %s", sentinel, reason),
		})
	}

	if opts.Family != "" {
		known := false
		for _, c := range contracts {
			if c.Family == opts.Family {
				known = true
				break
			}
		}
		if !known {
			return nil, fmt.Errorf("%w: %q", layoutcontract.ErrUnknownFamily, opts.Family)
		}
		rep.Scope = opts.Family
	}
	inScope := func(family string) bool { return opts.Family == "" || family == opts.Family }

	families := []string{}
	for _, c := range contracts {
		if inScope(c.Family) {
			families = append(families, c.Family)
		}
	}
	sort.Strings(families)
	rep.Families = families

	// 1. integrity: closed invariants per entry, plus the closed-set rule a
	// registry is a SET of families (duplicate tags violate it).
	integrityBad := 0
	seen := map[string]int{}
	for _, c := range contracts {
		if !inScope(c.Family) {
			continue
		}
		seen[c.Family]++
		if seen[c.Family] > 1 {
			integrityBad++
			addViolation(layoutCheckIntegrity, c.Family, "", "duplicate family tag in registry", ErrLayoutIntegrity)
			continue
		}
		if err := c.Validate(); err != nil {
			integrityBad++
			addViolation(layoutCheckIntegrity, c.Family, "", err.Error(), ErrLayoutIntegrity)
		}
	}

	// 2. family-coverage: registry vs packing matrix deltas, explicit.
	matrixFams, err := loadPackingMatrix(root)
	if err != nil {
		return nil, err
	}
	regSet := map[string]bool{}
	for _, c := range contracts {
		if inScope(c.Family) {
			regSet[layoutMatrixTag(c.Family)] = true
		}
	}
	matrixSet := map[string]bool{}
	for _, f := range matrixFams {
		matrixSet[f] = true
	}
	scopedTag := ""
	if opts.Family != "" {
		scopedTag = layoutMatrixTag(opts.Family)
	}
	gaps := []LayoutGapRow{}
	for _, f := range matrixFams {
		if regSet[f] {
			continue
		}
		if scopedTag != "" && f != scopedTag {
			continue
		}
		gaps = append(gaps, LayoutGapRow{Kind: layoutGapMatrixOnly, Family: f})
	}
	for tag := range regSet {
		if matrixSet[tag] {
			continue
		}
		gaps = append(gaps, LayoutGapRow{Kind: layoutGapRegistryOnly, Family: tag})
	}
	sort.Slice(gaps, func(i, j int) bool {
		if gaps[i].Kind != gaps[j].Kind {
			return gaps[i].Kind < gaps[j].Kind
		}
		return gaps[i].Family < gaps[j].Family
	})

	// 3. source-pins: entry Source plus every Present() Transform.Source.
	// Skipped under SkipSourcePins (explicit --root re-base): see the
	// LayoutOptions field comment for why a foreign root cannot falsify a pin.
	pinsChecked, pinsBad := 0, 0
	if !opts.SkipSourcePins {
		for _, c := range contracts {
			if !inScope(c.Family) {
				continue
			}
			pinsChecked++
			if _, ok, reason := layoutReadPin(root, c.Source); !ok {
				pinsBad++
				addViolation(layoutCheckPins, c.Family, c.Source.String(), reason, ErrLayoutStalePin)
			}
			if !c.Transform.Present() {
				continue
			}
			pinsChecked++
			line, ok, reason := layoutReadPin(root, c.Transform.Source)
			if !ok {
				pinsBad++
				addViolation(layoutCheckPins, c.Family, c.Transform.Source.String(), reason, ErrLayoutStalePin)
				continue
			}
			if name := layoutForwardName(c.Transform.Name); name != "" && !strings.Contains(line, name) {
				pinsBad++
				addViolation(layoutCheckPins, c.Family, c.Transform.Source.String(),
					fmt.Sprintf("pinned line no longer cites the forward function %s", name), ErrLayoutStalePin)
			}
		}
	}

	// 4. transform-roundtrip: forward then inverse on the witness shapes,
	// byte identity required.
	pairsDriven, shapesDriven, rtBad := 0, 0, 0
	for _, c := range contracts {
		if !inScope(c.Family) || !c.Transform.Present() {
			continue
		}
		w, known := defaultLayoutWitnesses[c.Family]
		if !known || len(w.shapes) == 0 {
			rtBad++
			addViolation(layoutCheckRoundTrip, c.Family, "",
				"no round-trip witness shape registered for this family", ErrLayoutNoWitness)
			continue
		}
		pairsDriven++
		for _, s := range w.shapes {
			raw := layoutDeterministicPayload(c.Family, w.payload(s))
			shape := fmt.Sprintf("rows=%d cols=%d width=%d", s.Rows, s.Cols, s.Width)
			fwd, err := c.Transform.Forward(s, raw)
			if err != nil {
				rtBad++
				addViolation(layoutCheckRoundTrip, c.Family, "",
					fmt.Sprintf("forward failed at %s: %v", shape, err), ErrLayoutRoundTrip)
				continue
			}
			back, err := c.Transform.Inverse(s, fwd)
			if err != nil {
				rtBad++
				addViolation(layoutCheckRoundTrip, c.Family, "",
					fmt.Sprintf("inverse failed at %s: %v", shape, err), ErrLayoutRoundTrip)
				continue
			}
			if !bytes.Equal(back, raw) {
				rtBad++
				addViolation(layoutCheckRoundTrip, c.Family, "",
					fmt.Sprintf("inverse o forward is not the byte identity at %s (%d bytes)", shape, len(raw)), ErrLayoutRoundTrip)
				continue
			}
			shapesDriven++
		}
	}

	rep.Gaps = gaps
	rep.Violations = violations
	coverageStatus := layoutStatusOK
	if len(gaps) > 0 {
		coverageStatus = layoutStatusGaps // gaps are deltas, not violations
	}
	rep.Checks = []LayoutCheckRow{
		{Name: layoutCheckIntegrity, Status: layoutStatusOf(integrityBad), Detail: fmt.Sprintf("%d/%d entries validate", len(families)-integrityBad, len(families))},
		{Name: layoutCheckCoverage, Status: coverageStatus, Detail: fmt.Sprintf("%d matrix-only, %d registry-only", countGapKind(gaps, layoutGapMatrixOnly), countGapKind(gaps, layoutGapRegistryOnly))},
		{Name: layoutCheckPins, Status: layoutStatusOf(pinsBad), Detail: layoutPinsDetail(opts.SkipSourcePins, pinsChecked-pinsBad, pinsChecked)},
		{Name: layoutCheckRoundTrip, Status: layoutStatusOf(rtBad), Detail: fmt.Sprintf("%d pair(s) driven across %d shape(s), byte-identical", pairsDriven, shapesDriven)},
	}
	switch {
	case len(violations) > 0:
		rep.Outcome = layoutOutcomeFail
	case len(gaps) > 0:
		rep.Outcome = layoutOutcomeGaps
	default:
		rep.Outcome = layoutOutcomePass
	}
	if len(violations) > 0 {
		return rep, &layoutCheckError{violations: violations}
	}
	return rep, nil
}

func layoutStatusOf(bad int) string {
	if bad > 0 {
		return layoutStatusViolated
	}
	return layoutStatusOK
}

// layoutPinsDetail renders the source-pins check row. A skipped check reports
// ok with the skip reason so the closed report never reads a re-based
// coverage run as a pin verdict.
func layoutPinsDetail(skipped bool, good, total int) string {
	if skipped {
		return "skipped: --root re-bases coverage; pins stay pinned-repo facts"
	}
	return fmt.Sprintf("%d/%d pins resolve within range", good, total)
}

func countGapKind(gaps []LayoutGapRow, kind string) int {
	n := 0
	for _, g := range gaps {
		if g.Kind == kind {
			n++
		}
	}
	return n
}

// RunLayoutContract is the CLI entry point:
//
//	layout-contract check [--json] [--family F] [--root PATH]
//
// Exit 0 on pass|gaps, 1 on any violation, 2 on a refusal (unknown family,
// missing or schema-mismatched matrix, usage error). An explicit --root
// re-bases the coverage matrix only; the live registry's source pins stay
// pinned-repo facts, so pin freshness is skipped on a re-based run instead of
// reading registry-only coverage gaps as contract breaks.
func RunLayoutContract(stdout, stderr io.Writer, argv []string) int {
	usage := "usage: fak-dev layout-contract check [--json] [--family F] [--root PATH]"
	if len(argv) == 0 || argv[0] != "check" {
		fmt.Fprintln(stderr, usage)
		return 2
	}
	asJSON := false
	family := ""
	root := ""
	for i := 1; i < len(argv); i++ {
		switch {
		case argv[i] == "--json":
			asJSON = true
		case argv[i] == "--family":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, usage)
				return 2
			}
			i++
			family = argv[i]
		case strings.HasPrefix(argv[i], "--family="):
			family = argv[i][len("--family="):]
		case argv[i] == "--root":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, usage)
				return 2
			}
			i++
			root = argv[i]
		case strings.HasPrefix(argv[i], "--root="):
			root = argv[i][len("--root="):]
		default:
			fmt.Fprintf(stderr, "%s\nunknown argument %q\n", usage, argv[i])
			return 2
		}
	}

	rep, err := CheckLayoutContracts(LayoutOptions{Root: root, Family: family, SkipSourcePins: root != ""})
	if err != nil {
		if errors.Is(err, layoutcontract.ErrUnknownFamily) ||
			errors.Is(err, ErrLayoutMatrixMissing) ||
			errors.Is(err, ErrLayoutMatrixSchema) {
			fmt.Fprintf(stderr, "LAYOUT_CONTRACT_REFUSED: %v\n", err)
			return 2
		}
		writeLayoutReport(stdout, rep, asJSON)
		return 1
	}
	writeLayoutReport(stdout, rep, asJSON)
	return 0
}

// writeLayoutReport renders the closed report: JSON under --json, otherwise a
// human table ending in the closed outcome token.
func writeLayoutReport(stdout io.Writer, rep *LayoutReport, asJSON bool) {
	if asJSON {
		b, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Fprintln(stdout, string(b))
		return
	}
	fmt.Fprintf(stdout, "layout-contract check: scope=%s root=%s\n", rep.Scope, rep.Root)
	for _, c := range rep.Checks {
		fmt.Fprintf(stdout, "  %-20s %-8s %s\n", c.Name, c.Status, c.Detail)
	}
	for _, g := range rep.Gaps {
		fmt.Fprintf(stdout, "  gap %-13s %s\n", g.Kind+":", g.Family)
	}
	for _, v := range rep.Violations {
		pin := ""
		if v.Pin != "" {
			pin = " pin=" + v.Pin
		}
		fmt.Fprintf(stdout, "  VIOLATION %s%s%s: %s\n", v.Check, v.Family, pin, v.Reason)
	}
	switch rep.Outcome {
	case layoutOutcomeFail:
		fmt.Fprintf(stdout, "layout-contract check: FAIL (%d violation(s))\n", len(rep.Violations))
	case layoutOutcomeGaps:
		fmt.Fprintf(stdout, "layout-contract check: GAPS (%d coverage gap(s), 0 violations)\n", len(rep.Gaps))
	default:
		fmt.Fprintln(stdout, "layout-contract check: PASS")
	}
}
