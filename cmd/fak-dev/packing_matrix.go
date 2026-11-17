package main

// packing-matrix — bounded read-only reader over the loader packing coverage
// matrix (docs/data/packing-coverage.json, schema fak.packingmatrix.v1), the
// baseline metric of the tensor-layout-mismatch prevention campaign (ticket
// lg-packing-matrix). It reports how many shipped kernel-family x device cells
// deliver their expected layout at load time (native) versus how many pay a
// runtime transpose/swizzle/repack on the inference hot path (runtime-repack),
// lists the defect cells with their first evidence line, and surfaces the
// matrix's known gaps so the next agent sees the problem without hand-digging.
// No subcommand mutates state. Exit codes: 0 report emitted, 1 a --fail-on
// mode is present, 2 usage error or typed refusal (PACKING_MATRIX_NOT_FOUND,
// PACKING_MATRIX_SCHEMA_MISMATCH, PACKING_MATRIX_MODE_UNKNOWN,
// PACKING_MATRIX_DUPLICATE_CELL, PACKING_MATRIX_EMPTY_CELLS).

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	packingMatrixReportSchema = "fak.packingmatrix.report.v1"
	packingMatrixSourcePath   = "docs/data/packing-coverage.json"
	packingMatrixSourceSchema = "fak.packingmatrix.v1"

	packingModeNative        = "native"
	packingModeRuntimeRepack = "runtime-repack"
	packingModeUnknown       = "unknown"

	packingVerdictClean  = "clean"
	packingVerdictDefect = "runtime-repack-present"

	packingMatrixDiscoverDepth = 6
)

// closedPackingModes is the closed mode vocabulary of the matrix; any other
// value refuses the document instead of being counted as native.
var closedPackingModes = []string{packingModeNative, packingModeRuntimeRepack, packingModeUnknown}

var (
	errPackingMatrixNotFound  = errors.New("PACKING_MATRIX_NOT_FOUND")
	errPackingMatrixSchema    = errors.New("PACKING_MATRIX_SCHEMA_MISMATCH")
	errPackingMatrixMode      = errors.New("PACKING_MATRIX_MODE_UNKNOWN")
	errPackingMatrixDuplicate = errors.New("PACKING_MATRIX_DUPLICATE_CELL")
	errPackingMatrixEmpty     = errors.New("PACKING_MATRIX_EMPTY_CELLS")
)

const packingMatrixUsage = `usage: fak-dev packing-matrix show [--root PATH] [--json] [--fail-on MODE]

Read the loader packing coverage matrix (docs/data/packing-coverage.json,
schema fak.packingmatrix.v1) and report the tensor-layout-mismatch campaign
baseline: native ratio, runtime-repack defect cells, unknown cells, excluded
pairs and known gaps. Read-only; nothing mutates state.

  --root PATH     repository root containing docs/data/packing-coverage.json
                  (default: discover by walking up from the working directory)
  --json          emit the machine report (schema fak.packingmatrix.report.v1)
  --fail-on MODE  exit 1 when any cell is in MODE (closed modes: native,
                  runtime-repack, unknown); gate use

Exit codes: 0 report emitted, 1 a --fail-on mode is present, 2 usage error or
typed refusal (PACKING_MATRIX_NOT_FOUND, PACKING_MATRIX_SCHEMA_MISMATCH,
PACKING_MATRIX_MODE_UNKNOWN, PACKING_MATRIX_DUPLICATE_CELL,
PACKING_MATRIX_EMPTY_CELLS).`

// packingMatrixCell is one (family, device) row of the source matrix.
type packingMatrixCell struct {
	Family   string   `json:"family"`
	Device   string   `json:"device"`
	Mode     string   `json:"mode"`
	Evidence []string `json:"evidence"`
}

// packingMatrixExcluded documents a (family, device) pair with no shipped
// kernel path, kept so a later new path shows up as a NEW cell.
type packingMatrixExcluded struct {
	Family string `json:"family"`
	Device string `json:"device"`
	Reason string `json:"reason"`
}

// packingMatrixDoc mirrors the closed top-level fields of
// fak.packingmatrix.v1; forward-compatible extra fields are ignored.
type packingMatrixDoc struct {
	Schema      string                  `json:"schema"`
	GeneratedAt string                  `json:"generated_at"`
	AuditHead   string                  `json:"audit_head"`
	Cells       []packingMatrixCell     `json:"cells"`
	Excluded    []packingMatrixExcluded `json:"excluded_pairs"`
	KnownGaps   []string                `json:"known_gaps"`
}

type packingMatrixTotals struct {
	Cells         int `json:"cells"`
	Native        int `json:"native"`
	RuntimeRepack int `json:"runtime_repack"`
	Unknown       int `json:"unknown"`
}

// packingMatrixDefectRow names one flagged cell plus its first evidence line.
type packingMatrixDefectRow struct {
	Family        string `json:"family"`
	Device        string `json:"device"`
	FirstEvidence string `json:"first_evidence,omitempty"`
}

// packingMatrixReport is the closed machine output of the reader.
type packingMatrixReport struct {
	Schema      string                   `json:"schema"`
	Root        string                   `json:"root"`
	Source      string                   `json:"source"`
	AuditHead   string                   `json:"audit_head,omitempty"`
	GeneratedAt string                   `json:"generated_at,omitempty"`
	Totals      packingMatrixTotals      `json:"totals"`
	NativeRatio string                   `json:"native_ratio"`
	Defects     []packingMatrixDefectRow `json:"defect_cells"`
	Unknowns    []packingMatrixDefectRow `json:"unknown_cells"`
	Excluded    int                      `json:"excluded_pairs"`
	KnownGaps   []string                 `json:"known_gaps,omitempty"`
	Verdict     string                   `json:"verdict"`
}

func runPackingMatrix(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] != "show" {
		fmt.Fprintln(stderr, packingMatrixUsage)
		return 2
	}
	root := ""
	asJSON := false
	failOn := ""
	rest := argv[1:]
	for i := 0; i < len(rest); i++ {
		switch {
		case rest[i] == "--root":
			if i+1 >= len(rest) {
				fmt.Fprintln(stderr, packingMatrixUsage)
				return 2
			}
			i++
			root = rest[i]
		case strings.HasPrefix(rest[i], "--root="):
			root = strings.TrimPrefix(rest[i], "--root=")
		case rest[i] == "--json":
			asJSON = true
		case rest[i] == "--fail-on":
			if i+1 >= len(rest) {
				fmt.Fprintln(stderr, packingMatrixUsage)
				return 2
			}
			i++
			failOn = rest[i]
		case strings.HasPrefix(rest[i], "--fail-on="):
			failOn = strings.TrimPrefix(rest[i], "--fail-on=")
		case rest[i] == "-h" || rest[i] == "--help":
			fmt.Fprintln(stdout, packingMatrixUsage)
			return 0
		default:
			fmt.Fprintf(stderr, "%s\nunknown argument %q\n", packingMatrixUsage, rest[i])
			return 2
		}
	}
	if failOn != "" && !isClosedPackingMode(failOn) {
		fmt.Fprintf(stderr, "PACKING_MATRIX_USAGE: --fail-on takes one of %s\n", strings.Join(closedPackingModes, ", "))
		fmt.Fprintln(stderr, packingMatrixUsage)
		return 2
	}

	file, resolvedRoot, err := resolvePackingMatrixFile(root)
	if err != nil {
		fmt.Fprintf(stderr, "%v: %v\n", errPackingMatrixNotFound, err)
		return 2
	}
	doc, err := loadPackingMatrixDoc(file)
	if err != nil {
		fmt.Fprintf(stderr, "%v\n", err)
		return 2
	}

	rep := buildPackingMatrixReport(resolvedRoot, doc)
	writePackingMatrixReport(stdout, rep, asJSON)
	if failOn != "" && countPackingMatrixMode(doc, failOn) > 0 {
		return 1
	}
	return 0
}

func isClosedPackingMode(mode string) bool {
	for _, m := range closedPackingModes {
		if mode == m {
			return true
		}
	}
	return false
}

// resolvePackingMatrixFile locates the matrix. An explicit root is used as
// given; otherwise the search walks UP from the working directory (bounded),
// never scanning sibling trees.
func resolvePackingMatrixFile(root string) (string, string, error) {
	if root != "" {
		file := filepath.Join(root, packingMatrixSourcePath)
		if _, err := os.Stat(file); err != nil {
			return "", "", fmt.Errorf("%s not found under --root %s", packingMatrixSourcePath, root)
		}
		return file, root, nil
	}
	wd, err := os.Getwd()
	if err != nil {
		return "", "", err
	}
	dir := wd
	for depth := 0; depth <= packingMatrixDiscoverDepth; depth++ {
		file := filepath.Join(dir, packingMatrixSourcePath)
		if _, err := os.Stat(file); err == nil {
			return file, dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", "", fmt.Errorf("%s not found walking up from %s", packingMatrixSourcePath, wd)
}

// loadPackingMatrixDoc reads and validates the matrix against its closed
// contract: exact schema, non-empty cells, closed mode vocabulary, cited
// evidence, unique (family, device) pairs.
func loadPackingMatrixDoc(file string) (*packingMatrixDoc, error) {
	raw, err := os.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("%v: %w", errPackingMatrixNotFound, err)
	}
	var doc packingMatrixDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("%v: %s: %v", errPackingMatrixSchema, file, err)
	}
	if doc.Schema != packingMatrixSourceSchema {
		return nil, fmt.Errorf("%v: %s declares schema %q, want %q", errPackingMatrixSchema, file, doc.Schema, packingMatrixSourceSchema)
	}
	if len(doc.Cells) == 0 {
		return nil, fmt.Errorf("%v: %s carries no cells", errPackingMatrixEmpty, file)
	}
	seen := make(map[[2]string]bool, len(doc.Cells))
	for _, c := range doc.Cells {
		if c.Family == "" || c.Device == "" {
			return nil, fmt.Errorf("%v: cell missing family or device in %s", errPackingMatrixSchema, file)
		}
		if !isClosedPackingMode(c.Mode) {
			return nil, fmt.Errorf("%v: cell %s x %s declares mode %q outside %s", errPackingMatrixMode, c.Family, c.Device, c.Mode, strings.Join(closedPackingModes, "/"))
		}
		if len(c.Evidence) == 0 {
			return nil, fmt.Errorf("%v: cell %s x %s cites no evidence", errPackingMatrixSchema, c.Family, c.Device)
		}
		key := [2]string{c.Family, c.Device}
		if seen[key] {
			return nil, fmt.Errorf("%v: duplicate cell %s x %s", errPackingMatrixDuplicate, c.Family, c.Device)
		}
		seen[key] = true
	}
	return &doc, nil
}

func countPackingMatrixMode(doc *packingMatrixDoc, mode string) int {
	n := 0
	for _, c := range doc.Cells {
		if c.Mode == mode {
			n++
		}
	}
	return n
}

// buildPackingMatrixReport folds the validated document into the closed
// report. Cell order follows the source file, which is the campaign's
// canonical ordering.
func buildPackingMatrixReport(root string, doc *packingMatrixDoc) *packingMatrixReport {
	totals := packingMatrixTotals{Cells: len(doc.Cells)}
	defects := []packingMatrixDefectRow{}
	unknowns := []packingMatrixDefectRow{}
	for _, c := range doc.Cells {
		switch c.Mode {
		case packingModeNative:
			totals.Native++
		case packingModeRuntimeRepack:
			totals.RuntimeRepack++
			defects = append(defects, packingMatrixDefectRow{Family: c.Family, Device: c.Device, FirstEvidence: c.Evidence[0]})
		case packingModeUnknown:
			totals.Unknown++
			unknowns = append(unknowns, packingMatrixDefectRow{Family: c.Family, Device: c.Device, FirstEvidence: c.Evidence[0]})
		}
	}
	ratio := "0/0"
	if totals.Cells > 0 {
		ratio = fmt.Sprintf("%d/%d", totals.Native, totals.Cells)
	}
	verdict := packingVerdictClean
	if totals.RuntimeRepack > 0 {
		verdict = packingVerdictDefect
	}
	return &packingMatrixReport{
		Schema:      packingMatrixReportSchema,
		Root:        root,
		Source:      packingMatrixSourcePath,
		AuditHead:   doc.AuditHead,
		GeneratedAt: doc.GeneratedAt,
		Totals:      totals,
		NativeRatio: ratio,
		Defects:     defects,
		Unknowns:    unknowns,
		Excluded:    len(doc.Excluded),
		KnownGaps:   doc.KnownGaps,
		Verdict:     verdict,
	}
}

// writePackingMatrixReport renders the closed report: JSON under --json,
// otherwise a compact human table ending in the verdict token.
func writePackingMatrixReport(stdout io.Writer, rep *packingMatrixReport, asJSON bool) {
	if asJSON {
		b, _ := json.MarshalIndent(rep, "", "  ")
		fmt.Fprintln(stdout, string(b))
		return
	}
	head := rep.AuditHead
	if len(head) > 12 {
		head = head[:12]
	}
	fmt.Fprintf(stdout, "packing-matrix show: root=%s audit_head=%s generated_at=%s\n", rep.Root, head, rep.GeneratedAt)
	t := rep.Totals
	fmt.Fprintf(stdout, "  cells=%d native=%d runtime-repack=%d unknown=%d native-ratio=%s\n", t.Cells, t.Native, t.RuntimeRepack, t.Unknown, rep.NativeRatio)
	for _, d := range rep.Defects {
		fmt.Fprintf(stdout, "  DEFECT %s x %s: %s\n", d.Family, d.Device, d.FirstEvidence)
	}
	for _, u := range rep.Unknowns {
		fmt.Fprintf(stdout, "  UNKNOWN %s x %s: %s\n", u.Family, u.Device, u.FirstEvidence)
	}
	fmt.Fprintf(stdout, "  EXCLUDED pairs=%d\n", rep.Excluded)
	if len(rep.KnownGaps) > 0 {
		fmt.Fprintln(stdout, "  KNOWN GAPS:")
		for _, g := range rep.KnownGaps {
			fmt.Fprintf(stdout, "    - %s\n", g)
		}
	}
	fmt.Fprintf(stdout, "packing-matrix show: VERDICT %s\n", rep.Verdict)
}
