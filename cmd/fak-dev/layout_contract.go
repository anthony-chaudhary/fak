package main

// layoutcontract — read-only CLI surface over the declarative tensor-layout
// contract registry (internal/compute/layoutcontract). Port workers and CI use
// it to verify that a kernel's expected packed/tile layout matches what the
// loader produces BEFORE a port silently pays a runtime repack. No subcommand
// mutates state. Exit codes: 0 match/success, 1 mismatch, 2 usage or unknown
// family.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute/layoutcontract"
)

const (
	layoutContractFamiliesSchema = "fak.layoutcontract-families/1"
	layoutContractShowSchema     = "fak.layoutcontract/1"
	layoutContractCheckSchema    = "fak.layoutcontract-check/1"
	layoutContractOrphansSchema  = "fak.layoutcontract-orphans/1"

	layoutContractVerdictMatch    = "match"
	layoutContractVerdictMismatch = "mismatch"
	layoutContractVerdictUnknown  = "unknown"
)

var errLayoutContractUsage = errors.New("layoutcontract usage")

// layoutContractObservedFields is the closed observed-layout vocabulary:
// storage_order, scale_layout, interleave_width, padding_rule, packing_stage.
// Any key outside it refuses the input instead of being ignored.
var layoutContractObservedFields = []string{
	"storage_order",
	"scale_layout",
	"interleave_width",
	"padding_rule",
	"packing_stage",
}

const layoutContractUsage = `usage: fak-dev layoutcontract <subcommand> [flags]

  families                      list registered quantized kernel families (stable order)
  show FAMILY                   emit the full contract as JSON (schema fak.layoutcontract/1)
  check FAMILY --observed FILE  compare an observed layout description against the
                                registered contract; exit 0 match / 1 mismatch /
                                2 usage-or-unknown-family
  orphans [--root DIR]          report registered families with no production call
                                site (presence-vs-invokability over the bounded
                                seam roots)

Read-only: no subcommand mutates state. Observed files use the closed field
vocabulary storage_order, scale_layout, interleave_width, padding_rule,
packing_stage; absent fields are reported as unknown, never coerced to match.`

func runLayoutContract(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, layoutContractUsage)
		return 2
	}
	switch argv[0] {
	case "families":
		return runLayoutContractFamilies(stdout, stderr, argv[1:])
	case "show":
		return runLayoutContractShow(stdout, stderr, argv[1:])
	case "check":
		return runLayoutContractCheck(stdout, stderr, argv[1:])
	case "orphans":
		return runLayoutContractOrphans(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, layoutContractUsage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak-dev layoutcontract: unknown subcommand %q\n", argv[0])
		fmt.Fprintln(stderr, layoutContractUsage)
		return 2
	}
}

func writeLayoutContractJSON(stdout io.Writer, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	_, err = stdout.Write(append(b, '\n'))
	return err
}

// singleLayoutContractPositional enforces the closed argument grammar of the
// family-taking subcommands: exactly one bare positional, no flags.
func singleLayoutContractPositional(stderr io.Writer, sub string, argv []string) (string, bool) {
	for _, arg := range argv {
		if strings.HasPrefix(arg, "-") && arg != "-" {
			fmt.Fprintf(stderr, "fak-dev layoutcontract %s: unexpected flag %q\n", sub, arg)
			return "", false
		}
	}
	if len(argv) != 1 || argv[0] == "" {
		fmt.Fprintf(stderr, "fak-dev layoutcontract %s: exactly one FAMILY argument is required\n", sub)
		return "", false
	}
	return argv[0], true
}

// refuseUnknownFamily renders the fail-closed ErrUnknownFamily refusal and
// reports whether the caller should exit 2.
func refuseUnknownFamily(stderr io.Writer, sub, family string, err error) (exit int, done bool) {
	if errors.Is(err, layoutcontract.ErrUnknownFamily) {
		fmt.Fprintf(stderr, "fak-dev layoutcontract %s: ErrUnknownFamily: %q is not a registered family; see 'fak-dev layoutcontract families'\n", sub, family)
		return 2, true
	}
	return 0, false
}

// --- families ---------------------------------------------------------------

type layoutContractFamiliesReceipt struct {
	Schema   string   `json:"schema"`
	Families []string `json:"families"`
}

func runLayoutContractFamilies(stdout, stderr io.Writer, argv []string) int {
	if len(argv) != 0 {
		fmt.Fprintf(stderr, "fak-dev layoutcontract families takes no arguments (got %d)\n", len(argv))
		return 2
	}
	receipt := layoutContractFamiliesReceipt{
		Schema:   layoutContractFamiliesSchema,
		Families: layoutcontract.Families(),
	}
	if err := writeLayoutContractJSON(stdout, receipt); err != nil {
		fmt.Fprintf(stderr, "fak-dev layoutcontract families: encode: %v\n", err)
		return 2
	}
	return 0
}

// --- show -------------------------------------------------------------------

type layoutContractSourceRefView struct {
	Path string `json:"path"`
	Line int    `json:"line"`
	SHA  string `json:"sha"`
	Ref  string `json:"ref"`
}

func newLayoutContractSourceRefView(r layoutcontract.SourceRef) layoutContractSourceRefView {
	return layoutContractSourceRefView{Path: r.Path, Line: r.Line, SHA: r.SHA, Ref: r.String()}
}

type layoutContractTransformView struct {
	Present bool                         `json:"present"`
	Name    string                       `json:"name,omitempty"`
	Source  *layoutContractSourceRefView `json:"source,omitempty"`
}

type layoutContractShowReceipt struct {
	Schema          string                      `json:"schema"`
	Family          string                      `json:"family"`
	WeightOrder     string                      `json:"weight_order"`
	InterleaveWidth int                         `json:"interleave_width"`
	ScaleLayout     string                      `json:"scale_layout"`
	PaddingRule     string                      `json:"padding_rule"`
	PackingPath     string                      `json:"packing_path"`
	Source          layoutContractSourceRefView `json:"source"`
	Transform       layoutContractTransformView `json:"transform"`
}

func runLayoutContractShow(stdout, stderr io.Writer, argv []string) int {
	family, ok := singleLayoutContractPositional(stderr, "show", argv)
	if !ok {
		return 2
	}
	c, err := layoutcontract.Get(family)
	if err != nil {
		if _, done := refuseUnknownFamily(stderr, "show", family, err); done {
			return 2
		}
		fmt.Fprintf(stderr, "fak-dev layoutcontract show: %v\n", err)
		return 1
	}
	if err := c.Validate(); err != nil {
		// A corrupt registry entry is a data failure, not a usage error.
		fmt.Fprintf(stderr, "fak-dev layoutcontract show: registry entry failed validation: %v\n", err)
		return 1
	}
	transform := layoutContractTransformView{Present: c.Transform.Present()}
	if transform.Present {
		transform.Name = c.Transform.Name
		src := newLayoutContractSourceRefView(c.Transform.Source)
		transform.Source = &src
	}
	receipt := layoutContractShowReceipt{
		Schema:          layoutContractShowSchema,
		Family:          c.Family,
		WeightOrder:     string(c.WeightOrder),
		InterleaveWidth: c.InterleaveWidth,
		ScaleLayout:     string(c.ScaleLayout),
		PaddingRule:     c.PaddingRule,
		PackingPath:     string(c.PackingPath),
		Source:          newLayoutContractSourceRefView(c.Source),
		Transform:       transform,
	}
	if err := writeLayoutContractJSON(stdout, receipt); err != nil {
		fmt.Fprintf(stderr, "fak-dev layoutcontract show: encode: %v\n", err)
		return 2
	}
	return 0
}

// --- check ------------------------------------------------------------------

// observedLayout is one observed layout description. Every field is optional;
// a nil pointer means the observer did not claim the field, which the verdict
// step reports as unknown rather than matching it away.
type observedLayout struct {
	StorageOrder    *string `json:"storage_order"`
	ScaleLayout     *string `json:"scale_layout"`
	InterleaveWidth *int    `json:"interleave_width"`
	PaddingRule     *string `json:"padding_rule"`
	PackingStage    *string `json:"packing_stage"`
}

// parseObservedLayout decodes an observed-layout file, refusing closed-
// vocabulary violations and type mismatches with errLayoutContractUsage. It
// returns the raw bytes alongside the decoded view so the receipt can pin the
// exact input without a second read.
func parseObservedLayout(path string) (observedLayout, []byte, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return observedLayout{}, nil, fmt.Errorf("%w: read %s: %v", errLayoutContractUsage, path, err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return observedLayout{}, nil, fmt.Errorf("%w: decode %s: %v", errLayoutContractUsage, path, err)
	}
	closed := make(map[string]bool, len(layoutContractObservedFields))
	for _, f := range layoutContractObservedFields {
		closed[f] = true
	}
	for key := range raw {
		if !closed[key] {
			return observedLayout{}, nil, fmt.Errorf("%w: %s carries key %q outside the closed observed vocabulary %v",
				errLayoutContractUsage, path, key, layoutContractObservedFields)
		}
	}
	var obs observedLayout
	if err := json.Unmarshal(b, &obs); err != nil {
		return observedLayout{}, nil, fmt.Errorf("%w: decode %s: %v", errLayoutContractUsage, path, err)
	}
	return obs, b, nil
}

type layoutContractVerdict struct {
	Field    string `json:"field"`
	Expected string `json:"expected"`
	Observed string `json:"observed,omitempty"`
	Verdict  string `json:"verdict"`
	Note     string `json:"note,omitempty"`
}

type layoutContractCheckSummary struct {
	Match    int `json:"match"`
	Mismatch int `json:"mismatch"`
	Unknown  int `json:"unknown"`
}

type layoutContractCheckReceipt struct {
	Schema         string                     `json:"schema"`
	Family         string                     `json:"family"`
	ObservedFile   string                     `json:"observed_file"`
	ObservedSHA256 string                     `json:"observed_sha256"`
	Verdicts       []layoutContractVerdict    `json:"verdicts"`
	Summary        layoutContractCheckSummary `json:"summary"`
	Result         string                     `json:"result"`
}

// compareObservedLayout reduces one observed description against one
// registered contract into per-field verdicts in the fixed closed-vocabulary
// order. Absent observed fields are unknown, never coerced to match; a
// present value the registry cannot pin (host-resolved interleave width) is
// likewise unknown with a note saying why.
func compareObservedLayout(c layoutcontract.Contract, obs observedLayout) []layoutContractVerdict {
	stringVerdict := func(field, want string, got *string) layoutContractVerdict {
		v := layoutContractVerdict{Field: field, Expected: want}
		switch {
		case got == nil:
			v.Verdict = layoutContractVerdictUnknown
			v.Note = "observed field absent"
		case *got == want:
			v.Verdict = layoutContractVerdictMatch
			v.Observed = *got
		default:
			v.Verdict = layoutContractVerdictMismatch
			v.Observed = *got
		}
		return v
	}
	width := c.InterleaveWidth
	widthVerdict := layoutContractVerdict{Field: "interleave_width", Expected: strconv.Itoa(width)}
	switch {
	case obs.InterleaveWidth == nil:
		widthVerdict.Verdict = layoutContractVerdictUnknown
		widthVerdict.Note = "observed field absent"
	case width == 0:
		// 0 in the registry means the width is resolved at runtime from host
		// CPU features, so the registry pins no expected width to confirm
		// against.
		widthVerdict.Verdict = layoutContractVerdictUnknown
		widthVerdict.Observed = strconv.Itoa(*obs.InterleaveWidth)
		widthVerdict.Note = "registry interleave width is host-resolved (0); the observed value cannot be confirmed from the registry"
	case *obs.InterleaveWidth == width:
		widthVerdict.Verdict = layoutContractVerdictMatch
		widthVerdict.Observed = strconv.Itoa(width)
	default:
		widthVerdict.Verdict = layoutContractVerdictMismatch
		widthVerdict.Observed = strconv.Itoa(*obs.InterleaveWidth)
	}
	return []layoutContractVerdict{
		stringVerdict("storage_order", string(c.WeightOrder), obs.StorageOrder),
		stringVerdict("scale_layout", string(c.ScaleLayout), obs.ScaleLayout),
		widthVerdict,
		stringVerdict("padding_rule", c.PaddingRule, obs.PaddingRule),
		stringVerdict("packing_stage", string(c.PackingPath), obs.PackingStage),
	}
}

func summarizeLayoutContractVerdicts(vs []layoutContractVerdict) (layoutContractCheckSummary, string) {
	s := layoutContractCheckSummary{}
	for _, v := range vs {
		switch v.Verdict {
		case layoutContractVerdictMatch:
			s.Match++
		case layoutContractVerdictMismatch:
			s.Mismatch++
		default:
			s.Unknown++
		}
	}
	result := "match"
	if s.Mismatch > 0 {
		result = "mismatch"
	}
	return s, result
}

func runLayoutContractCheck(stdout, stderr io.Writer, argv []string) int {
	family, observedPath := "", ""
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			fmt.Fprintln(stdout, layoutContractUsage)
			return 0
		case arg == "--observed":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "fak-dev layoutcontract check: --observed needs a value")
				return 2
			}
			observedPath = argv[i+1]
			i++
		case strings.HasPrefix(arg, "--observed="):
			observedPath = strings.TrimPrefix(arg, "--observed=")
		case strings.HasPrefix(arg, "-") && arg != "-":
			fmt.Fprintf(stderr, "fak-dev layoutcontract check: unexpected flag %q\n", arg)
			return 2
		default:
			if family != "" {
				fmt.Fprintln(stderr, "fak-dev layoutcontract check: at most one FAMILY argument")
				return 2
			}
			family = arg
		}
	}
	if family == "" || observedPath == "" {
		fmt.Fprintln(stderr, layoutContractUsage)
		return 2
	}
	c, err := layoutcontract.Get(family)
	if err != nil {
		if _, done := refuseUnknownFamily(stderr, "check", family, err); done {
			return 2
		}
		fmt.Fprintf(stderr, "fak-dev layoutcontract check: %v\n", err)
		return 1
	}
	obs, observedBytes, err := parseObservedLayout(observedPath)
	if err != nil {
		// Unreadable, malformed, or closed-vocabulary-violating input is a
		// usage error: 1 is reserved for a genuine layout mismatch.
		fmt.Fprintf(stderr, "fak-dev layoutcontract check: %v\n", err)
		return 2
	}
	sum := sha256.Sum256(observedBytes)
	verdicts := compareObservedLayout(c, obs)
	summary, result := summarizeLayoutContractVerdicts(verdicts)
	receipt := layoutContractCheckReceipt{
		Schema:         layoutContractCheckSchema,
		Family:         c.Family,
		ObservedFile:   observedPath,
		ObservedSHA256: "sha256:" + hex.EncodeToString(sum[:]),
		Verdicts:       verdicts,
		Summary:        summary,
		Result:         result,
	}
	if err := writeLayoutContractJSON(stdout, receipt); err != nil {
		fmt.Fprintf(stderr, "fak-dev layoutcontract check: encode: %v\n", err)
		return 2
	}
	if result == "mismatch" {
		return 1
	}
	return 0
}

// --- orphans ----------------------------------------------------------------

// layoutContractSeamRoots is the bounded set of production seams scanned for
// registry call sites: the GGUF weight loader, the model assembly paths, and
// the compute upload/packing paths. The registry's own package directory is
// excluded (self-references are not production call sites), as are test files
// (production invokability only).
var layoutContractSeamRoots = []string{
	"internal/ggufload",
	"internal/model",
	"internal/compute",
}

const layoutContractSelfDir = "internal/compute/layoutcontract"

// layoutContractRegistryRefRe matches a qualified use of the registry package
// (layoutcontract.X). An import path alone does not qualify: the package must
// actually be invoked.
var layoutContractRegistryRefRe = regexp.MustCompile(`\blayoutcontract\.`)

func layoutContractFamilyTagRe(tag string) *regexp.Regexp {
	return regexp.MustCompile(`\b` + regexp.QuoteMeta(tag) + `\b`)
}

type layoutContractOrphanRow struct {
	Family    string   `json:"family"`
	CallSites []string `json:"call_sites"`
	Orphan    bool     `json:"orphan"`
}

type layoutContractOrphansReceipt struct {
	Schema       string                    `json:"schema"`
	Root         string                    `json:"root"`
	HeadSHA      string                    `json:"head_sha,omitempty"`
	SeamRoots    []string                  `json:"seam_roots"`
	Excluded     []string                  `json:"excluded"`
	FilesScanned int                       `json:"files_scanned"`
	Families     []layoutContractOrphanRow `json:"families"`
	Orphans      []string                  `json:"orphans"`
}

// scanLayoutContractCallSites walks the bounded seam roots under root and
// attributes each qualifying seam .go file (non-test, referencing the
// layoutcontract package) to the family tags it mentions. Deterministic:
// lexical walk order, sorted call-site lists, no clock or environment reads
// beyond the filesystem.
func scanLayoutContractCallSites(root string) (map[string][]string, int, error) {
	callSites := map[string][]string{}
	filesScanned := 0
	for _, seam := range layoutContractSeamRoots {
		dir := filepath.Join(root, filepath.FromSlash(seam))
		err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if d.IsDir() {
				rel, _ := filepath.Rel(root, path)
				if filepath.ToSlash(rel) == layoutContractSelfDir {
					return fs.SkipDir
				}
				return nil
			}
			name := d.Name()
			if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
				return nil
			}
			b, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			filesScanned++
			if !layoutContractRegistryRefRe.Match(b) {
				return nil
			}
			rel, _ := filepath.Rel(root, path)
			rel = filepath.ToSlash(rel)
			for _, tag := range layoutcontract.Families() {
				if layoutContractFamilyTagRe(tag).Match(b) {
					callSites[tag] = append(callSites[tag], rel)
				}
			}
			return nil
		})
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				continue // a seam root absent in a fixture tree scans empty
			}
			return nil, 0, err
		}
	}
	for tag := range callSites {
		sort.Strings(callSites[tag])
	}
	return callSites, filesScanned, nil
}

// layoutContractHeadSHA best-effort resolves the HEAD short SHA of root (""
// outside a git tree — fixtures audit cleanly without one).
func layoutContractHeadSHA(root string) string {
	out, err := exec.Command("git", "-C", root, "rev-parse", "HEAD").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// detectLayoutContractRepoRoot walks up from the working directory to the
// nearest go.mod; falls back to "." so a missing tree reports an IO error
// rather than guessing.
func detectLayoutContractRepoRoot() string {
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	dir := wd
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "."
		}
		dir = parent
	}
}

func runLayoutContractOrphans(stdout, stderr io.Writer, argv []string) int {
	root := ""
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			fmt.Fprintln(stdout, layoutContractUsage)
			return 0
		case arg == "--root":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "fak-dev layoutcontract orphans: --root needs a value")
				return 2
			}
			root = argv[i+1]
			i++
		case strings.HasPrefix(arg, "--root="):
			root = strings.TrimPrefix(arg, "--root=")
		default:
			fmt.Fprintf(stderr, "fak-dev layoutcontract orphans: unexpected argument %q\n", arg)
			return 2
		}
	}
	if root == "" {
		root = detectLayoutContractRepoRoot()
	}
	if fi, err := os.Stat(root); err != nil || !fi.IsDir() {
		fmt.Fprintf(stderr, "fak-dev layoutcontract orphans: root %s is not a directory\n", root)
		return 2
	}
	callSites, filesScanned, err := scanLayoutContractCallSites(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev layoutcontract orphans: scan: %v\n", err)
		return 2
	}
	families := layoutcontract.Families() // sorted
	rows := make([]layoutContractOrphanRow, 0, len(families))
	var orphans []string
	for _, fam := range families {
		sites := callSites[fam]
		if sites == nil {
			sites = []string{}
		}
		orphan := len(sites) == 0
		if orphan {
			orphans = append(orphans, fam)
		}
		rows = append(rows, layoutContractOrphanRow{Family: fam, CallSites: sites, Orphan: orphan})
	}
	receipt := layoutContractOrphansReceipt{
		Schema:       layoutContractOrphansSchema,
		Root:         root,
		HeadSHA:      layoutContractHeadSHA(root),
		SeamRoots:    layoutContractSeamRoots,
		Excluded:     []string{layoutContractSelfDir},
		FilesScanned: filesScanned,
		Families:     rows,
		Orphans:      orphans,
	}
	if err := writeLayoutContractJSON(stdout, receipt); err != nil {
		fmt.Fprintf(stderr, "fak-dev layoutcontract orphans: encode: %v\n", err)
		return 2
	}
	return 0
}
