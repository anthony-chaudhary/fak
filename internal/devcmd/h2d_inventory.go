// h2d-inventory — static inventory of host->device (H2D) copy call sites on the
// per-decode-token STATE paths of the public engine: sampler/token-input state,
// grammar/constrained-decoding mask state, and KV-append metadata.
//
// The ad-hoc method this replaces is the one-off benchmark-note measurement of
// sampler/grammar/mask state copies (PHL-145-style DeepInfer per-cursor-state
// masks): a number measured once in prose, that cannot be recomputed after a
// refactor, cannot flag a NEW H2D site added to a decode state path, and cannot
// be checked against another model shape or batch size. This inventory converts
// it into a version-controlled, recomputable, trendable metric:
//
//   - a closed inventory of the H2D call sites on the per-token state paths
//     (one row per seam), each pinned to file + function + source pattern;
//   - cadence classification per row (per-token / per-layer / per-request) with
//     a byte-size factor where derivable (state size x batch), so any model
//     shape turns the inventory into copies/token and bytes/token;
//   - verification against the LIVE source (go/ast): a missing function or a
//     vanished pattern is reported as DRIFT and fails the run — the metric can
//     no longer silently rot the way a prose number does;
//   - discovery of UNCOVERED H2D primitives (fcuda_h2d / UploadClass /
//     uploadHostF32 / cudaMemcpy ...) in the state-path files that no inventory
//     row pins — the new-regression signal;
//   - zero-copy witness rows: state seams that cross WITHOUT an H2D memcpy
//     (scalar kernel args, host-resident mask application) are inventoried too,
//     so converting one of them into a memcpy shows up as drift or an uncovered
//     marker instead of a silent launch-latency regression;
//   - a committed snapshot JSON under docs/benchmarks/inventory/ (the same
//     pattern graph-audit uses) makes the figure trendable across trunk.
//
// Measured mode lives in internal/model (FAK_H2D_CENSUS=1): the shared
// activation-upload funnel and the paged block-table sync count themselves when
// a real device forward runs, so future hardware receipts carry measured
// copies/bytes beside this static estimate.

package devcmd

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// H2DClass is the cadence class of an inventoried H2D call site. Per-token and
// per-layer classes scale with the matching model-shape counter and sum into
// the steady-state copies/token + bytes/token; per-request rows are dirty-gated
// request-scoped syncs reported separately (they do not sum into /token).
type H2DClass string

const (
	H2DPerToken   H2DClass = "per-token"
	H2DPerLayer   H2DClass = "per-layer"
	H2DPerRequest H2DClass = "per-request"
)

// H2DBytesExpr names the byte-size factor of one copy at a seam, derived from
// the model shape. "" means zero bytes cross (a zero-copy or host-resident
// seam: the state crosses via kernel args or stays on the host).
type H2DBytesExpr string

const (
	H2DBytesNone          H2DBytesExpr = ""
	H2DBytesHiddenRowF32  H2DBytesExpr = "hidden_row_f32"  // 4 x hidden_size
	H2DBytesVocabF32      H2DBytesExpr = "vocab_f32"       // 4 x vocab
	H2DBytesBlockTableF32 H2DBytesExpr = "block_table_f32" // 4 x ceil(seq_len/block_tokens)
)

// H2DCallSite is one version-controlled inventory row. Pattern must occur in
// the enclosing function's source text (or anywhere in the file when Func is
// empty) — verification re-checks this against the live tree on every run.
// Copies is the number of H2D copies issued per fire of the seam (0 = the seam
// fires but no H2D memcpy happens there: a zero-copy witness). BatchScales says
// whether the fire count scales with the decode batch (per-lane seams do;
// request-scoped table syncs do not).
type H2DCallSite struct {
	ID          string       `json:"id"`
	File        string       `json:"file"`
	Func        string       `json:"func"` // "" = file-level
	Pattern     string       `json:"pattern"`
	Class       H2DClass     `json:"class"`
	Copies      int          `json:"copies"`
	BytesExpr   H2DBytesExpr `json:"bytes_expr,omitempty"`
	BatchScales bool         `json:"batch_scales,omitempty"`
	Note        string       `json:"note"`
}

// h2dInventory is the closed set of H2D call sites on the per-decode-token
// STATE paths (sampler/token-input, grammar/constrained mask state, KV-append
// metadata) of the public engine's reference decode. Rows 1-2 are the sampler
// token-input uploads (host fallback when the embedding table is not
// device-resident); row 3 documents that the grammar/schema/logit-bias mask
// state is HOST-RESIDENT (zero H2D) — the drift anchor for an on-device mask
// regression; rows 4-6 are the KV-append metadata seams that already cross
// without a memcpy (scalar pos / page ids / seq-len via kernel args); row 7 is
// the only dirty-gated request-scoped H2D (paged block table); row 8 is the
// split-hybrid full-logits re-upload.
var h2dInventory = []H2DCallSite{
	{ID: "sampler-embed-row-hal", File: "internal/model/hal.go", Func: "tokenHALOutput",
		Pattern: `"hal-token-input"`, Class: H2DPerToken, Copies: 1, BytesExpr: H2DBytesHiddenRowF32,
		Note: "embedding-row upload when the device embedding table is not resident (host-fallback stage); zero when embed_tokens is device-resident"},
	{ID: "sampler-embed-row-batch", File: "internal/model/qwen35_batch_decode.go", Func: "qwen35DecodeBatchRequests",
		Pattern: `"qwen35-decode-batch-embedding-row"`, Class: H2DPerToken, Copies: 1, BytesExpr: H2DBytesHiddenRowF32, BatchScales: true,
		Note: "gathered embedding-row upload for the independent Qwen3.5 decode batch (packed/oversized tables); one row buffer per lane, or one cohort [lanes,hidden] panel on shared-panel backends"},
	{ID: "mask-host-resident", File: "internal/model/constraint.go", Func: "sampleConstrained",
		Pattern: "c.Mask.MaskLogits(history, eff)", Class: H2DPerToken, Copies: 0,
		Note: "grammar/JSON-schema/logit-bias mask state applies HOST-SIDE over read-back logits: zero H2D. An on-device mask would add 4 x vocab x batch B/tok — this row is the drift anchor for that regression"},
	{ID: "kv-append-scalar-pos", File: "internal/compute/cuda_kv.go", Func: "growAppend",
		Pattern: "C.fcuda_kv_write(", Class: H2DPerLayer, Copies: 0,
		Note: "KV-append position crosses as a SCALAR KERNEL ARG (graph-stable append), not a memcpy; per-layer zero-copy witness"},
	{ID: "paged-kv-page-ids", File: "internal/compute/cuda_paged_kv.go", Func: "appendRowsLocked",
		Pattern: "fcuda_paged_kv_write_pages_f32(", Class: H2DPerLayer, Copies: 0,
		Note: "paged KV page ids cross via the C adapter's kernel-argument copy, not a memcpy"},
	{ID: "paged-kv-seqlen-store", File: "internal/compute/cuda_paged_kv.go", Func: "syncDeviceTableLocked",
		Pattern: "C.fcuda_paged_store_int(", Class: H2DPerToken, Copies: 0,
		Note: "paged seq-len crosses via a 1-thread store kernel (value in kernel args), not a memcpy"},
	{ID: "paged-kv-block-table", File: "internal/compute/cuda_paged_kv.go", Func: "syncDeviceTableLocked",
		Pattern: "C.fcuda_h2d(k.dTable", Class: H2DPerRequest, Copies: 1, BytesExpr: H2DBytesBlockTableF32,
		Note: "dirty-gated block-table H2D on page allocation: at most one upload per table change, <= 4 x ceil(seq_len/block_tokens) bytes/request"},
	{ID: "split-logits-reupload", File: "internal/model/hal.go", Func: "tokenHALOutput",
		Pattern: `"hal-split-logits"`, Class: H2DPerToken, Copies: 1, BytesExpr: H2DBytesVocabF32,
		Note: "hybrid CPU/GPU split only: the full post-host-suffix logits vector is re-uploaded to the device after the host suffix"},
}

// H2DShape parameterizes the inventory math. Batch defaults to 1; SeqLen and
// BlockTokens bound the paged block-table estimate.
type H2DShape struct {
	Name        string `json:"name,omitempty"`
	HiddenSize  int    `json:"hidden_size"`
	Vocab       int    `json:"vocab"`
	Layers      int    `json:"layers"`
	Batch       int    `json:"batch,omitempty"`
	SeqLen      int    `json:"seq_len,omitempty"`
	BlockTokens int    `json:"block_tokens,omitempty"`
}

// normalize fills defaulted counters and validates. It returns the normalized
// copy so callers keep the input untouched.
func (s H2DShape) normalize() (H2DShape, error) {
	out := s
	if out.HiddenSize <= 0 {
		return out, fmt.Errorf("model shape needs hidden_size >= 1 (got %d)", out.HiddenSize)
	}
	if out.Vocab <= 0 {
		return out, fmt.Errorf("model shape needs vocab >= 1 (got %d)", out.Vocab)
	}
	if out.Layers <= 0 {
		return out, fmt.Errorf("model shape needs layers >= 1 (got %d)", out.Layers)
	}
	if out.Batch == 0 {
		out.Batch = 1
	}
	if out.Batch < 0 {
		return out, fmt.Errorf("batch=%d out of range [1, inf]", out.Batch)
	}
	if out.SeqLen == 0 {
		out.SeqLen = 4096
	}
	if out.SeqLen < 1 {
		return out, fmt.Errorf("seq_len=%d out of range [1, inf]", out.SeqLen)
	}
	if out.BlockTokens == 0 {
		out.BlockTokens = 64
	}
	if out.BlockTokens < 1 {
		return out, fmt.Errorf("block_tokens=%d out of range [1, inf]", out.BlockTokens)
	}
	return out, nil
}

// h2dPresets are the named shapes. tiny-4 exists for shape-math regression
// tests; small-batch-4x16k mirrors the campaign's target operating point
// (small-batch decode, long context, paged KV).
var h2dPresets = map[string]H2DShape{
	"tiny-4":            {Name: "tiny-4", HiddenSize: 512, Vocab: 1024, Layers: 4, Batch: 1, SeqLen: 128, BlockTokens: 32},
	"small-batch-4x16k": {Name: "small-batch-4x16k", HiddenSize: 4096, Vocab: 151936, Layers: 32, Batch: 4, SeqLen: 16384, BlockTokens: 64},
}

// bytesPerCopy evaluates the row's byte-size factor at the shape.
func bytesPerCopy(site H2DCallSite, shape H2DShape) int64 {
	switch site.BytesExpr {
	case H2DBytesHiddenRowF32:
		return int64(shape.HiddenSize) * 4
	case H2DBytesVocabF32:
		return int64(shape.Vocab) * 4
	case H2DBytesBlockTableF32:
		entries := (shape.SeqLen + shape.BlockTokens - 1) / shape.BlockTokens
		return int64(entries) * 4
	default:
		return 0
	}
}

// firesPerToken is how many times the seam fires per decoded token for the
// shape (0 for per-request rows, which are reported separately).
func firesPerToken(site H2DCallSite, shape H2DShape) int {
	batch := 1
	if site.BatchScales {
		batch = shape.Batch
	}
	switch site.Class {
	case H2DPerToken:
		return batch
	case H2DPerLayer:
		return shape.Layers * batch
	default:
		return 0
	}
}

// H2DTotals is the steady-state per-token arithmetic plus the request-scoped
// dirty-gated upper bound.
type H2DTotals struct {
	CopiesPerToken    int   `json:"copies_per_token"`
	BytesPerToken     int64 `json:"bytes_per_token"`
	RequestSyncCopies int   `json:"request_sync_copies"`
	RequestSyncBytes  int64 `json:"request_sync_bytes"`
}

// computeH2DTotals turns the inventory into copies/token + bytes/token for the
// given shape. Zero-copy rows contribute nothing; per-request rows feed only
// the request-sync bound.
func computeH2DTotals(inv []H2DCallSite, shape H2DShape) H2DTotals {
	var t H2DTotals
	for _, s := range inv {
		fires := firesPerToken(s, shape)
		t.CopiesPerToken += s.Copies * fires
		t.BytesPerToken += int64(s.Copies*fires) * bytesPerCopy(s, shape)
		if s.Class == H2DPerRequest {
			t.RequestSyncCopies += s.Copies
			t.RequestSyncBytes += int64(s.Copies) * bytesPerCopy(s, shape)
		}
	}
	return t
}

// goFileScan is the parsed view of one Go file: function-name -> source spans
// (byte offsets; methods of different receivers sharing a name merge into one
// span list, which is safe because inventory patterns are specific enough not
// to collide across receivers).
type h2dGoFileScan struct {
	path        string
	src         []byte
	funcSpans   map[string][][2]int
	parseErr    error
	fileLevelOK bool
}

func scanH2DGoFile(root, rel string) (*h2dGoFileScan, error) {
	full := filepath.Join(root, filepath.FromSlash(rel))
	sc := &h2dGoFileScan{path: full, funcSpans: map[string][][2]int{}}
	src, err := os.ReadFile(full)
	if err != nil {
		return sc, err
	}
	sc.src = src
	fset := token.NewFileSet()
	fn, perr := parser.ParseFile(fset, full, src, 0)
	if perr != nil {
		sc.parseErr = perr
		return sc, nil
	}
	sc.fileLevelOK = true
	for _, decl := range fn.Decls {
		fd, ok := decl.(*ast.FuncDecl)
		if !ok || fd.Name == nil {
			continue
		}
		start := fset.Position(fd.Pos()).Offset
		end := fset.Position(fd.End()).Offset
		sc.funcSpans[fd.Name.Name] = append(sc.funcSpans[fd.Name.Name], [2]int{start, end})
	}
	return sc, nil
}

// funcText returns the concatenated source text of every function named name in
// the scan ("" when the name is absent).
func (sc *h2dGoFileScan) funcText(name string) (string, bool) {
	spans, ok := sc.funcSpans[name]
	if !ok {
		return "", false
	}
	var b strings.Builder
	for _, sp := range spans {
		b.Write(sc.src[sp[0]:sp[1]])
	}
	return b.String(), true
}

// lineStarts precomputes byte offsets of line beginnings for line-based scans.
func (sc *h2dGoFileScan) lineStarts() []int {
	starts := []int{0}
	for i, b := range sc.src {
		if b == '\n' {
			starts = append(starts, i+1)
		}
	}
	return starts
}

// H2DDrift is one inventory row that no longer matches the live source.
type H2DDrift struct {
	Site        H2DCallSite `json:"site"`
	Reason      string      `json:"reason"` // file-missing | parse-error | func-missing | pattern-absent
	Occurrences int         `json:"occurrences"`
}

// verifyH2DInventory re-checks every inventory row against the live tree and
// returns both the drift findings and the per-row occurrence counts.
func verifyH2DInventory(root string, inv []H2DCallSite) ([]H2DDrift, map[string]int) {
	scanMemo := map[string]*h2dGoFileScan{}
	get := func(rel string) *h2dGoFileScan {
		if sc, ok := scanMemo[rel]; ok {
			return sc
		}
		sc, _ := scanH2DGoFile(root, rel)
		scanMemo[rel] = sc
		return sc
	}
	var drift []H2DDrift
	occ := map[string]int{}
	for _, site := range inv {
		sc := get(site.File)
		if sc.src == nil {
			drift = append(drift, H2DDrift{Site: site, Reason: "file-missing"})
			continue
		}
		if sc.parseErr != nil {
			drift = append(drift, H2DDrift{Site: site, Reason: "parse-error"})
			continue
		}
		var text string
		if site.Func == "" {
			text = string(sc.src)
		} else if ft, ok := sc.funcText(site.Func); ok {
			text = ft
		} else {
			drift = append(drift, H2DDrift{Site: site, Reason: "func-missing"})
			continue
		}
		n := strings.Count(text, site.Pattern)
		occ[site.ID] = n
		if n == 0 {
			drift = append(drift, H2DDrift{Site: site, Reason: "pattern-absent"})
		}
	}
	return drift, occ
}

// h2dDiscoverFiles is the version-controlled state-path file set the
// uncovered-site scan covers. Extend it deliberately when the decode state
// paths gain new host-glue files.
var h2dDiscoverFiles = []string{
	"internal/model/hal.go",
	"internal/model/qwen35_batch_decode.go",
	"internal/model/constraint.go",
	"internal/compute/cuda_kv.go",
	"internal/compute/cuda_paged_kv.go",
}

// h2dMarkers are the high-signal H2D primitives. A hit inside a file region
// with NO inventory coverage for its line is an UNCOVERED candidate — the
// new-regression signal.
var h2dMarkers = []string{
	"C.fcuda_h2d(",
	"UploadClass(",
	"uploadHostF32(",
	"cudaMemcpy(",
}

// H2DUncovered is an H2D-primitive hit the inventory does not pin.
type H2DUncovered struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Marker  string `json:"marker"`
	Snippet string `json:"snippet"`
}

// discoverH2DUncovered scans the state-path files for H2D markers whose lines
// no inventory row covers (a line is covered when it falls inside an
// inventoried function span, or the file carries a file-level inventory row).
func discoverH2DUncovered(root string, inv []H2DCallSite) []H2DUncovered {
	type coverage struct {
		fileLevel bool
		spans     [][2]int // line-based [startLine, endLine]
	}
	coveredBy := map[string]*coverage{}
	scanMemo := map[string]*h2dGoFileScan{}
	for _, rel := range h2dDiscoverFiles {
		sc, err := scanH2DGoFile(root, rel)
		if err != nil || sc.src == nil || sc.parseErr != nil {
			continue
		}
		scanMemo[rel] = sc
		starts := sc.lineStarts()
		lineOf := func(off int) int {
			lo, hi := 0, len(starts)-1
			for lo <= hi {
				mid := (lo + hi) / 2
				if starts[mid] <= off {
					lo = mid + 1
				} else {
					hi = mid - 1
				}
			}
			return lo // 1-based line number
		}
		cov := coveredBy[rel]
		if cov == nil {
			cov = &coverage{}
			coveredBy[rel] = cov
		}
		for _, site := range inv {
			if site.File != rel {
				continue
			}
			if site.Func == "" {
				cov.fileLevel = true
				continue
			}
			for _, sp := range sc.funcSpans[site.Func] {
				cov.spans = append(cov.spans, [2]int{lineOf(sp[0]), lineOf(sp[1])})
			}
		}
	}
	inSpan := func(spans [][2]int, line int) bool {
		for _, sp := range spans {
			if line >= sp[0] && line <= sp[1] {
				return true
			}
		}
		return false
	}
	var out []H2DUncovered
	for _, rel := range h2dDiscoverFiles {
		sc, ok := scanMemo[rel]
		if !ok {
			continue
		}
		cov := coveredBy[rel]
		lines := strings.Split(string(sc.src), "\n")
		for i, line := range lines {
			lineNo := i + 1
			if (cov != nil && cov.fileLevel) || (cov != nil && inSpan(cov.spans, lineNo)) {
				continue
			}
			for _, m := range h2dMarkers {
				if strings.Contains(line, m) {
					snippet := strings.TrimSpace(line)
					if len(snippet) > 100 {
						snippet = snippet[:100] + "…"
					}
					out = append(out, H2DUncovered{File: rel, Line: lineNo, Marker: m, Snippet: snippet})
					break // one report per line
				}
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		return out[i].Line < out[j].Line
	})
	return out
}

// H2DSiteRow is an inventory row joined with its live occurrence count and its
// shape-parameterized contribution.
type H2DSiteRow struct {
	H2DCallSite
	Occurrences    int   `json:"occurrences"`
	FiresPerToken  int   `json:"fires_per_token"`
	CopiesPerToken int   `json:"copies_per_token"`
	BytesPerToken  int64 `json:"bytes_per_token"`
}

// H2DInventoryDoc is the --json / snapshot document.
type H2DInventoryDoc struct {
	Schema    string         `json:"schema"`
	Root      string         `json:"root"`
	HeadSHA   string         `json:"head_sha,omitempty"`
	Shape     H2DShape       `json:"shape"`
	Totals    H2DTotals      `json:"totals"`
	Sites     []H2DSiteRow   `json:"sites"`
	Drift     []H2DDrift     `json:"drift"`
	Uncovered []H2DUncovered `json:"uncovered"`
}

const h2dUsage = `usage: fak-dev h2d-inventory [--root DIR] [--preset NAME | --model-shape JSON] [--no-discover] [--snapshot PATH] [--json]

Inventory the host->device copy call sites on the per-decode-token state paths
(sampler/token-input, grammar/mask state, KV-append metadata): verifies the
version-controlled call-site inventory against the live source, computes
copies/token and bytes/token for a model shape, lists uncovered H2D candidates,
and optionally writes the committed snapshot JSON.

Exit codes: 0 PASS, 1 drift detected, 2 usage/IO error.`

// headSHAAt returns the short git HEAD sha of root ("unknown" when unavailable).
func headSHAAt(root string) string {
	out, err := exec.Command("git", "-C", root, "rev-parse", "--short", "HEAD").Output()
	if err != nil {
		return "unknown"
	}
	return strings.TrimSpace(string(out))
}

// RunH2DInventory implements the fak-dev h2d-inventory verb.
func RunH2DInventory(stdout, stderr io.Writer, argv []string) int {
	root := repoRoot()
	preset := "tiny-4"
	shapeJSON := ""
	noDiscover := false
	snapshotPath := ""
	asJSON := false
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		switch {
		case arg == "-h" || arg == "--help":
			fmt.Fprintln(stdout, h2dUsage)
			return 0
		case arg == "--root":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "h2d-inventory: --root needs a value")
				return 2
			}
			i++
			root = argv[i]
		case strings.HasPrefix(arg, "--root="):
			root = strings.TrimPrefix(arg, "--root=")
		case arg == "--preset":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "h2d-inventory: --preset needs a value")
				return 2
			}
			i++
			preset = argv[i]
		case strings.HasPrefix(arg, "--preset="):
			preset = strings.TrimPrefix(arg, "--preset=")
		case arg == "--model-shape":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "h2d-inventory: --model-shape needs a JSON value")
				return 2
			}
			i++
			shapeJSON = argv[i]
		case strings.HasPrefix(arg, "--model-shape="):
			shapeJSON = strings.TrimPrefix(arg, "--model-shape=")
		case arg == "--no-discover":
			noDiscover = true
		case arg == "--snapshot":
			if i+1 >= len(argv) {
				fmt.Fprintln(stderr, "h2d-inventory: --snapshot needs a PATH")
				return 2
			}
			i++
			snapshotPath = argv[i]
		case strings.HasPrefix(arg, "--snapshot="):
			snapshotPath = strings.TrimPrefix(arg, "--snapshot=")
		case arg == "--json":
			asJSON = true
		default:
			fmt.Fprintln(stderr, h2dUsage)
			return 2
		}
	}

	var shape H2DShape
	switch {
	case shapeJSON != "":
		if err := json.Unmarshal([]byte(shapeJSON), &shape); err != nil {
			fmt.Fprintf(stderr, "h2d-inventory: parse --model-shape: %v\n", err)
			return 2
		}
		if shape.Name == "" {
			shape.Name = "custom"
		}
	default:
		ps, ok := h2dPresets[preset]
		if !ok {
			names := make([]string, 0, len(h2dPresets))
			for n := range h2dPresets {
				names = append(names, n)
			}
			sort.Strings(names)
			fmt.Fprintf(stderr, "h2d-inventory: unknown preset %q (known: %s)\n", preset, strings.Join(names, ", "))
			return 2
		}
		shape = ps
	}
	normed, err := shape.normalize()
	if err != nil {
		fmt.Fprintf(stderr, "h2d-inventory: invalid shape: %v\n", err)
		return 2
	}
	shape = normed

	drift, occ := verifyH2DInventory(root, h2dInventory)
	totals := computeH2DTotals(h2dInventory, shape)
	var uncovered []H2DUncovered
	if !noDiscover {
		uncovered = discoverH2DUncovered(root, h2dInventory)
	}

	doc := H2DInventoryDoc{
		Schema:  "fak.dev.h2d-inventory.v1",
		Root:    "", // repo-root-relative; keeps the committed snapshot reproducible (graph-audit convention)
		HeadSHA: headSHAAt(root),
		Shape:   shape,
		Totals:  totals,
	}
	for _, s := range h2dInventory {
		fires := firesPerToken(s, shape)
		row := H2DSiteRow{
			H2DCallSite:    s,
			Occurrences:    occ[s.ID],
			FiresPerToken:  fires,
			CopiesPerToken: s.Copies * fires,
			BytesPerToken:  int64(s.Copies*fires) * bytesPerCopy(s, shape),
		}
		if s.Class == H2DPerRequest {
			row.BytesPerToken = 0 // request-scoped; see totals.request_sync_bytes
		}
		doc.Sites = append(doc.Sites, row)
	}
	doc.Drift = drift
	doc.Uncovered = uncovered

	if snapshotPath != "" {
		data, err := json.MarshalIndent(doc, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "h2d-inventory: marshal snapshot: %v\n", err)
			return 2
		}
		if err := os.MkdirAll(filepath.Dir(snapshotPath), 0o755); err != nil {
			fmt.Fprintf(stderr, "h2d-inventory: mkdir snapshot dir: %v\n", err)
			return 2
		}
		if err := os.WriteFile(snapshotPath, append(data, '\n'), 0o644); err != nil {
			fmt.Fprintf(stderr, "h2d-inventory: write snapshot: %v\n", err)
			return 2
		}
	}

	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(doc); err != nil {
			fmt.Fprintf(stderr, "h2d-inventory: encode: %v\n", err)
			return 2
		}
	} else {
		writeH2DInventoryText(stdout, &doc)
	}

	if len(drift) > 0 {
		return 1
	}
	return 0
}

func formatBytes(n int64) string {
	if n >= 1024*1024 {
		return fmt.Sprintf("%.1f MiB", float64(n)/(1024*1024))
	}
	if n >= 1024 {
		return fmt.Sprintf("%.1f KiB", float64(n)/1024)
	}
	return strconv.FormatInt(n, 10) + " B"
}

func writeH2DInventoryText(w io.Writer, d *H2DInventoryDoc) {
	s := d.Shape
	fmt.Fprintf(w, "h2d-inventory root=%s shape=%s(hidden=%d vocab=%d layers=%d batch=%d seq=%d blk=%d) head=%s\n",
		d.Root, s.Name, s.HiddenSize, s.Vocab, s.Layers, s.Batch, s.SeqLen, s.BlockTokens, d.HeadSHA)
	fmt.Fprintf(w, "%-24s %-12s %5s %10s %8s %8s\n", "call site", "class", "fires", "B/fire", "copies", "B/tok")
	for _, r := range d.Sites {
		label := r.ID
		if len(label) > 24 {
			label = label[:21] + "..."
		}
		fireB := "-"
		if r.Copies > 0 {
			fireB = strconv.FormatInt(bytesPerCopy(r.H2DCallSite, s), 10)
		}
		btok := "-"
		if r.Copies > 0 && r.Class != H2DPerRequest {
			btok = strconv.FormatInt(r.BytesPerToken, 10)
		}
		cp := "-"
		if r.Copies > 0 {
			cp = strconv.Itoa(r.CopiesPerToken)
		}
		fmt.Fprintf(w, "%-24s %-12s %5d %10s %8s %8s\n", label, r.Class, r.FiresPerToken, fireB, cp, btok)
	}
	fmt.Fprintf(w, "TOTAL estimated H2D copies / decoded token: %d\n", d.Totals.CopiesPerToken)
	fmt.Fprintf(w, "TOTAL estimated H2D bytes / decoded token: %d (%s)\n", d.Totals.BytesPerToken, formatBytes(d.Totals.BytesPerToken))
	if d.Totals.RequestSyncCopies > 0 {
		fmt.Fprintf(w, "request-scoped (dirty-gated) H2D: %d copy(ies), <= %d B/request (%s)\n",
			d.Totals.RequestSyncCopies, d.Totals.RequestSyncBytes, formatBytes(d.Totals.RequestSyncBytes))
	}
	if len(d.Drift) > 0 {
		fmt.Fprintf(w, "DRIFT (%d inventory row(s) no longer match the live source):\n", len(d.Drift))
		for _, dr := range d.Drift {
			fmt.Fprintf(w, "  %-58s %-14s %s\n", dr.Site.File+":"+dr.Site.Func+"["+dr.Site.Pattern+"]", dr.Reason, dr.Site.Note)
		}
	}
	if len(d.Uncovered) > 0 {
		fmt.Fprintf(w, "UNCOVERED H2D candidates (%d, informational — review and extend the inventory if they are decode state-path glue):\n", len(d.Uncovered))
		for _, u := range d.Uncovered {
			fmt.Fprintf(w, "  %s:%d [%s] %s\n", u.File, u.Line, u.Marker, u.Snippet)
		}
	}
}
