// Package roofline folds the GLM-5.2 DGX-drive run artifacts into a single
// current-vs-80%target-vs-ceiling dashboard — one row per drive lane, keyed to
// the theoretical-ceiling note.
//
// The problem it solves. The GLM-5.2 lab drive (epic #3073) is a fleet of
// concurrent lane tickets, each aimed at 80 % of a roofline ceiling documented
// in docs/notes/GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md. The ceiling doc
// carries the *targets*; the run artifacts under experiments/benchmark/runs
// carry the *measurements*. Nobody today folds the two into one legible
// "where are we vs where the hardware caps out" view. This package is that
// fold, and it is deliberately laptop-composable: no GPU, no network — it reads
// committed JSON artifacts and emits a deterministic table.
//
// The honesty contract. A lane's CURRENT column is only ever filled from a
// measured run artifact — never from the ceiling doc, never inferred. The
// CEILING and 80%-TARGET columns are transcribed roofline ESTIMATES from the
// doc (the doc labels them so). The target is a pure, testable function of the
// ceiling: target = 0.8 × ceiling. A lane with no matching measured artifact is
// marked PENDING, not zero and not an invented number. A synthetic-weights
// kernel-wiring witness (an "optimistic lower bound", explicitly NOT the real
// 753B model) never fills a real-model lane's current — it is surfaced
// separately so it is neither trusted nor silently dropped.
package roofline

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/maputil"

	"github.com/anthony-chaudhary/fak/internal/numfmt"
)

// Repo-relative locations the generator reads and writes.
const (
	// RunsRel is the per-machine benchmark run tree the fold reads.
	RunsRel = "experiments/benchmark/runs/by-machine"
	// CeilingDocRel is the roofline authority the ceilings are keyed to.
	CeilingDocRel = "docs/notes/GLM52-DGX-THEORETICAL-CEILING-2026-07-06.md"
	// DocOutRel is the committed dashboard this package generates.
	DocOutRel = "docs/notes/GLM52-DGX-ROOFLINE-DASHBOARD.md"
)

// Kind is the throughput metric a lane's current is filled from.
type Kind string

const (
	KindDecodeSingle Kind = "decode-single" // batch=1 single-stream decode tok/s
	KindAggregate    Kind = "aggregate"     // batched aggregate decode tok/s at concurrency
	KindPrefill      Kind = "prefill"       // prompt prefill tok/s
	KindCPUOffload   Kind = "cpu-offload"   // server-2 cpu-offload GLM-5.2 tok/s (host-bound)
)

// Status is the fold verdict for one lane.
type Status string

const (
	// StatusMeasured means a real-model artifact filled the current column.
	StatusMeasured Status = "MEASURED"
	// StatusPending means no matching measured artifact exists yet.
	StatusPending Status = "PENDING"
)

// Ceiling is the practical roofline for a lane, transcribed from the ceiling
// doc (§3/§4). It is an ESTIMATE range, never a measurement. HostBound marks a
// lane whose ceiling obeys a different (host-bandwidth) law with no numeric GPU
// roofline — the server-2 cpu-offload path (§3.4).
type Ceiling struct {
	Lo, Hi    float64 // practical-ceiling range in tok/s (Hi==Lo for a point estimate)
	HostBound bool    // true => "different law" (§3.4), no numeric 0.8× target
	Regime    string  // the roofline regime, e.g. "memory-bandwidth bound (batch=1 decode)"
	DocRef    string  // section anchor in the ceiling doc, e.g. "§3.1 / §4"
}

// TargetLo is the low end of the 80 % drive target = 0.8 × ceiling.
func (c Ceiling) TargetLo() float64 { return 0.8 * c.Lo }

// TargetHi is the high end of the 80 % drive target = 0.8 × ceiling.
func (c Ceiling) TargetHi() float64 { return 0.8 * c.Hi }

// Lane is one drive lane from the ceiling doc §6, with its roofline ceiling and
// the artifact metric that fills its current.
type Lane struct {
	ID          string  // "A", "B", "D", "G-offload"
	Name        string  // human name, e.g. "single-stream decode"
	Node        string  // "server-3" (80 GiB resident) | "server-2" (40 GiB cpu-offload)
	Metric      string  // the column label, e.g. "single-stream decode tok/s"
	Kind        Kind    // which measured metric fills this lane's current
	RequireReal bool    // only a real-753B measurement may fill current
	Ceiling     Ceiling // roofline ceiling (from the doc)
}

// Lanes returns the canonical GLM-5.2 DGX lane spec — the four current-vs-ceiling
// rows of the ceiling doc §4, with ceilings transcribed from §3. Every ceiling is
// a roofline ESTIMATE from the doc; only the current column is ever measured.
func Lanes() []Lane {
	return []Lane{
		{
			ID: "A", Name: "single-stream decode", Node: "server-3",
			Metric: "single-stream decode tok/s", Kind: KindDecodeSingle, RequireReal: true,
			Ceiling: Ceiling{Lo: 150, Hi: 200, Regime: "memory-bandwidth bound (batch=1)", DocRef: "§3.1 / §4"},
		},
		{
			ID: "B", Name: "aggregate decode @ concurrency", Node: "server-3",
			Metric: "aggregate decode tok/s", Kind: KindAggregate, RequireReal: true,
			Ceiling: Ceiling{Lo: 11000, Hi: 14000, Regime: "compute-bound (concurrency ~64–128, BF16)", DocRef: "§3.2 / §4"},
		},
		{
			ID: "D", Name: "prefill", Node: "server-3",
			Metric: "prefill tok/s", Kind: KindPrefill, RequireReal: true,
			Ceiling: Ceiling{Lo: 11000, Hi: 14000, Regime: "compute-bound (~64 GFLOP/token)", DocRef: "§3.3 / §4"},
		},
		{
			ID: "G-offload", Name: "GLM-5.2 cpu-offload", Node: "server-2",
			Metric: "cpu-offload tok/s", Kind: KindCPUOffload, RequireReal: true,
			Ceiling: Ceiling{HostBound: true, Regime: "host memory-bandwidth + host GEMM (different law)", DocRef: "§3.4 / §4"},
		},
	}
}

// Measurement is one throughput number extracted from an artifact.
type Measurement struct {
	Kind    Kind    // which metric this is
	TokS    float64 // tokens per second
	Witness string  // short provenance, e.g. "-sm layer, batch=1, UD-Q4_K_M"
}

// Artifact is a normalized GLM-5.2 run artifact — the fold's raw input. The
// loader maps each result.json schema onto this shape. A lane's current is only
// ever filled from an Artifact.Meas value, never from the ceiling doc.
type Artifact struct {
	RunID     string        // run_id
	MachineID string        // machine_id
	Timestamp string        // ISO timestamp
	Path      string        // repo-relative artifact path
	Real753B  bool          // measures the real 753B GLM-5.2 checkpoint
	Synthetic bool          // synthetic reduced-weights kernel-wiring witness (NOT the 753B)
	Failed    bool          // a real attempt that produced no usable number (wedge/abort)
	FailNote  string        // one line on why (when Failed)
	Meas      []Measurement // measured throughput values (empty when Failed)
}

// Row is one folded dashboard row: a lane plus the measured current (if any).
type Row struct {
	Lane       Lane
	Status     Status
	Current    float64 // valid iff Status == StatusMeasured
	Witness    string  // measurement provenance (when measured)
	WitnessRun string  // run_id that filled current (when measured)
}

// Dashboard is the whole fold: the lane rows plus the artifacts that were seen
// but deliberately not counted toward a ceiling.
type Dashboard struct {
	CeilingDoc      string     // repo-relative ceiling-doc path (the cited authority)
	CeilingDocTitle string     // the doc's front-matter title
	Rows            []Row      // one per lane
	Synthetic       []Artifact // synthetic-weights witnesses (surfaced, not counted)
	FailedReal      []Artifact // real attempts that produced no usable number
	GLMArtifacts    int        // count of GLM-5.2 artifacts loaded
}

// Measured reports how many lanes have a measured current.
func (d Dashboard) Measured() int {
	n := 0
	for _, r := range d.Rows {
		if r.Status == StatusMeasured {
			n++
		}
	}
	return n
}

// Pending reports how many lanes are PENDING.
func (d Dashboard) Pending() int { return len(d.Rows) - d.Measured() }

// Fold matches artifacts to lanes and returns one row per lane. The current is
// the best (highest tok/s) matching measurement; a lane with no match is
// PENDING. Matching is deterministic: artifacts are folded in run_id order and
// ties keep the lexicographically-first run_id.
func Fold(lanes []Lane, arts []Artifact) []Row {
	ordered := append([]Artifact(nil), arts...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].RunID < ordered[j].RunID })

	rows := make([]Row, 0, len(lanes))
	for _, ln := range lanes {
		row := Row{Lane: ln, Status: StatusPending}
		best := math.Inf(-1)
		for _, a := range ordered {
			if a.Failed {
				continue
			}
			if ln.RequireReal && !a.Real753B {
				continue
			}
			for _, m := range a.Meas {
				if m.Kind != ln.Kind {
					continue
				}
				if m.TokS > best {
					best = m.TokS
					row.Status = StatusMeasured
					row.Current = m.TokS
					row.Witness = m.Witness
					row.WitnessRun = a.RunID
				}
			}
		}
		rows = append(rows, row)
	}
	return rows
}

// Generate loads the GLM-5.2 artifacts under root, reads the ceiling doc for its
// authority title (erroring if the doc is missing — the dashboard is meaningless
// without the ceiling it is keyed to), folds the lanes, and returns the
// dashboard. root is the repository root.
func Generate(root string) (Dashboard, error) {
	docPath := filepath.Join(root, filepath.FromSlash(CeilingDocRel))
	docRaw, err := os.ReadFile(docPath)
	if err != nil {
		return Dashboard{}, fmt.Errorf("ceiling doc not found at %s: %w", CeilingDocRel, err)
	}
	arts, err := LoadArtifacts(root)
	if err != nil {
		return Dashboard{}, err
	}
	d := Dashboard{
		CeilingDoc:      CeilingDocRel,
		CeilingDocTitle: frontMatterTitle(string(docRaw)),
		Rows:            Fold(Lanes(), arts),
		GLMArtifacts:    len(arts),
	}
	for _, a := range arts {
		switch {
		case a.Synthetic:
			d.Synthetic = append(d.Synthetic, a)
		case a.Failed:
			d.FailedReal = append(d.FailedReal, a)
		}
	}
	sort.SliceStable(d.Synthetic, func(i, j int) bool { return d.Synthetic[i].RunID < d.Synthetic[j].RunID })
	sort.SliceStable(d.FailedReal, func(i, j int) bool { return d.FailedReal[i].RunID < d.FailedReal[j].RunID })
	return d, nil
}

// LoadArtifacts walks the run tree under root and returns the normalized GLM-5.2
// artifacts, sorted by run_id. Non-GLM runs are skipped. Unparseable files are
// skipped, not fatal — a peer's malformed artifact must not sink the fold.
func LoadArtifacts(root string) ([]Artifact, error) {
	runsDir := filepath.Join(root, filepath.FromSlash(RunsRel))
	var out []Artifact
	err := filepath.WalkDir(runsDir, func(path string, de fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable subtrees rather than abort the whole walk
		}
		if de.IsDir() || de.Name() != "result.json" {
			return nil
		}
		raw, rerr := os.ReadFile(path)
		if rerr != nil {
			return nil
		}
		var m map[string]any
		if json.Unmarshal(raw, &m) != nil {
			return nil
		}
		if !isGLM(raw) {
			return nil
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			rel = path
		}
		a := normalize(m, raw, filepath.ToSlash(rel))
		if a.RunID == "" || a.RunID == filepath.ToSlash(rel) {
			// No run_id in the artifact: the run-dir name is a cleaner identifier
			// than the full path (e.g. "20260627T000000Z-glm52-cpu-wedge").
			a.RunID = filepath.Base(filepath.Dir(path))
		}
		out = append(out, a)
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].RunID < out[j].RunID })
	return out, nil
}

// normalize maps one parsed result.json onto an Artifact, dispatching on the
// shape of the record (a verdict A/B, a context curve, or an engine wedge).
func normalize(m map[string]any, raw []byte, rel string) Artifact {
	lower := strings.ToLower(string(raw))
	a := Artifact{
		RunID:     str(m, "run_id"),
		MachineID: str(m, "machine_id"),
		Timestamp: str(m, "timestamp"),
		Path:      rel,
	}
	if a.RunID == "" {
		a.RunID = rel // fall back to path so the row is still identifiable
	}
	a.Synthetic = strings.Contains(lower, "synthetic") || strings.Contains(lower, "not-the-753b")
	a.Real753B = !a.Synthetic // a GLM-5.2 record that is not synthetic measures the real checkpoint

	switch {
	case hasKey(m, "verdict"):
		// L1 row/layer A/B: the winning single-stream decode config is the better of the two.
		v, _ := m["verdict"].(map[string]any)
		layer, okL := num(v, "decode_toks_layer")
		row, okR := num(v, "decode_toks_row")
		best, witness := layer, "-sm layer"
		if okR && (!okL || row > best) {
			best, witness = row, "-sm row"
		}
		if okL || okR {
			a.Meas = append(a.Meas, Measurement{Kind: KindDecodeSingle, TokS: best, Witness: witness + ", batch=1"})
		}
	case hasKey(m, "curve"):
		// Context-scaling curve: best collected decode + prefill points.
		curve, _ := m["curve"].([]any)
		var bestDecode, bestPrefill float64
		var okD, okP bool
		for _, p := range curve {
			pt, ok := p.(map[string]any)
			if !ok || str(pt, "status") != "collected" {
				continue
			}
			if d, ok := num(pt, "decode_tok_s"); ok && (!okD || d > bestDecode) {
				bestDecode, okD = d, true
			}
			if pf, ok := num(pt, "prefill_tok_s"); ok && (!okP || pf > bestPrefill) {
				bestPrefill, okP = pf, true
			}
		}
		if okD {
			a.Meas = append(a.Meas, Measurement{Kind: KindDecodeSingle, TokS: bestDecode, Witness: "best collected context point"})
		}
		if okP {
			a.Meas = append(a.Meas, Measurement{Kind: KindPrefill, TokS: bestPrefill, Witness: "best collected context point"})
		}
	case hasKey(m, "engines") || boolField(m, "ok") == false && hasKey(m, "headline"):
		// A CPU/offload serve that produced no usable throughput (wedge/abort).
		a.Failed = true
		a.FailNote = firstSentence(str(m, "headline"))
	}
	if len(a.Meas) == 0 && !a.Failed {
		// A real record we could not extract a number from is honest PENDING input,
		// not a measurement — mark it failed-to-extract so it is surfaced, not counted.
		if a.Real753B {
			a.Failed = true
			a.FailNote = "no throughput number could be extracted from this artifact"
		}
	}
	return a
}

// Markdown renders the dashboard as a committed doc.
func (d Dashboard) Markdown() string {
	var b bytes.Buffer
	b.WriteString("---\n")
	b.WriteString("title: \"GLM-5.2 GPU-server roofline dashboard: current vs 80%-target vs ceiling, one row per lane\"\n")
	b.WriteString("description: \"A deterministic fold of the committed GLM-5.2 lab run artifacts (experiments/benchmark/runs) against the roofline ceilings, one row per drive lane. CURRENT is measured-only; a lane with no measured artifact is PENDING; ceilings are transcribed roofline estimates from the ceiling note. Regenerate, do not hand-edit.\"\n")
	b.WriteString("---\n\n")

	b.WriteString("# GLM-5.2 GPU-server roofline dashboard — current vs 80% target vs ceiling\n\n")
	b.WriteString("> **Generated, not hand-authored.** This table is folded from the committed run\n")
	b.WriteString("> artifacts under `" + RunsRel + "/` by `internal/roofline` (a pure-Go, laptop-composable\n")
	b.WriteString("> generator — no GPU). Regenerate with\n")
	b.WriteString("> `ROOFLINE_WRITE_DOC=1 go test ./internal/roofline/ -run TestGenerateRealDashboardDoc -count=1`.\n>\n")
	b.WriteString("> **Honesty contract.** The **Current** column is filled *only* from a measured\n")
	b.WriteString("> real-753B run artifact — never from the ceiling doc, never inferred. A lane with no\n")
	b.WriteString("> matching measured artifact is **PENDING** (not zero, not invented). The **Ceiling** and\n")
	b.WriteString("> **80% target** columns are roofline **ESTIMATES** transcribed from the ceiling note; the\n")
	b.WriteString("> target is exactly `0.8 × ceiling`.\n\n")

	fmt.Fprintf(&b, "**Ceiling authority:** [`%s`](%s) — %s\n\n", d.CeilingDoc, filepath.Base(d.CeilingDoc), d.CeilingDocTitle)
	fmt.Fprintf(&b, "**Folded from %d GLM-5.2 artifact(s):** %d lane(s) MEASURED, %d PENDING.\n\n", d.GLMArtifacts, d.Measured(), d.Pending())

	b.WriteString("| Lane | Metric | Node | Current | 80% target | Practical ceiling | Regime | Status | Witness |\n")
	b.WriteString("|------|--------|------|--------:|-----------:|------------------:|--------|:------:|---------|\n")
	for _, r := range d.Rows {
		current, target, ceiling := "— (PENDING)", "host-bound (§3.4)", "host-bound (§3.4)"
		if !r.Lane.Ceiling.HostBound {
			target = rng(r.Lane.Ceiling.TargetLo(), r.Lane.Ceiling.TargetHi())
			ceiling = rng(r.Lane.Ceiling.Lo, r.Lane.Ceiling.Hi)
		}
		witness := "—"
		if r.Status == StatusMeasured {
			current = fmt.Sprintf("**%.1f** tok/s", r.Current)
			witness = fmt.Sprintf("`%s` (%s)", r.WitnessRun, r.Witness)
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
			r.Lane.ID, r.Lane.Name, r.Lane.Node, current, target, ceiling,
			r.Lane.Ceiling.Regime, r.Status, witness)
	}
	b.WriteString("\nCeiling ranges are `lo–hi` (tok/s); the 80% target column is the mechanical `0.8 ×` of each.\n")

	if len(d.Synthetic) > 0 {
		b.WriteString("\n## Synthetic kernel-wiring witnesses (NOT the 753B — not counted toward any ceiling)\n\n")
		b.WriteString("These runs measure the CUDA decode/prefill *wiring* on reduced synthetic weights. The\n")
		b.WriteString("artifacts label their own numbers an \"optimistic lower bound\" on the work the real 753B\n")
		b.WriteString("MoE must do — i.e. an optimistic *upper* bound on real-model speed. They are listed for\n")
		b.WriteString("transparency and deliberately do **not** fill any lane's Current column.\n\n")
		writeArtifactTable(&b, "Synthetic measurement(s)", "--------------------------",
			d.Synthetic, func(a Artifact) string { return measList(a.Meas) })
	}

	if len(d.FailedReal) > 0 {
		b.WriteString("\n## Real-model attempts with no usable number (surfaced, still PENDING)\n\n")
		b.WriteString("Real GLM-5.2 attempts that were committed as artifacts but produced no clean throughput\n")
		b.WriteString("(a wedge or an abort). They keep the relevant lane PENDING — a failed attempt is not a\n")
		b.WriteString("measurement — but are surfaced so the gap is not silent.\n\n")
		writeArtifactTable(&b, "Why no number", "---------------",
			d.FailedReal, func(a Artifact) string { return a.FailNote })
	}

	b.WriteString("\n## Notes\n\n")
	b.WriteString("- **Lanes** are the four current-vs-ceiling rows of the ceiling note §4 (A single-stream\n")
	b.WriteString("  decode, B aggregate decode, D prefill — all server-3 resident — and G-offload, the\n")
	b.WriteString("  server-2 cpu-offload path, whose ceiling is a different host-bound law with no numeric\n")
	b.WriteString("  GPU roofline). The other §6 lanes (E arch/kernel, F GGUF-header ground-truth, G-fit\n")
	b.WriteString("  smaller-quant resident, H-stock stock-engine baseline, B-cache warm-cache, C harness)\n")
	b.WriteString("  have no single numeric roofline ceiling and are not rows here.\n")
	b.WriteString("- The server-2 cpu-offload baseline (0.23–2.62 tok/s) cited in the ceiling note §3.4 comes\n")
	b.WriteString("  from off-tree 2026-06-25/28 runs; it is **not** a committed artifact under `" + RunsRel + "/`,\n")
	b.WriteString("  so this generator does not fill it — G-offload stays PENDING until a real artifact lands.\n")

	return b.String()
}

// --- small helpers ---------------------------------------------------------

// writeArtifactTable writes one artifact side-table: the `Run | Machine | <third>`
// header, its dash separator, and one row per artifact. Only the third column varies
// between the synthetic-witness table and the failed-real table, so it is passed as a
// heading plus a cell function while the run/machine columns are stated once — a
// column added to these tables now lands in both instead of drifting between them.
// `sep` is each caller's own dash run, passed verbatim so the regenerated dashboard
// stays byte-identical to the committed doc TestGenerateRealDashboardDoc compares.
func writeArtifactTable(b *bytes.Buffer, third, sep string, arts []Artifact, cell func(Artifact) string) {
	fmt.Fprintf(b, "| Run | Machine | %s |\n", third)
	fmt.Fprintf(b, "|-----|---------|%s|\n", sep)
	for _, a := range arts {
		fmt.Fprintf(b, "| `%s` | %s | %s |\n", a.RunID, a.MachineID, cell(a))
	}
}

func isGLM(raw []byte) bool {
	l := strings.ToLower(string(raw))
	return strings.Contains(l, "glm52") || strings.Contains(l, "glm-5.2") || strings.Contains(l, "glm_moe_dsa")
}

// frontMatterTitle returns the first front-matter `title:` value, unquoted. It
// stops at the closing `---` fence so a later body line can never be mistaken
// for the title.
func frontMatterTitle(doc string) string {
	lines := strings.Split(doc, "\n")
	for i, line := range lines {
		t := strings.TrimSpace(line)
		if i > 0 && t == "---" {
			break // front matter closed
		}
		if strings.HasPrefix(t, "title:") {
			v := strings.TrimSpace(strings.TrimPrefix(t, "title:"))
			return strings.Trim(v, "\"")
		}
	}
	return ""
}

func rng(lo, hi float64) string {
	if lo == hi {
		return numfmt.UpToOneDecimal(lo)
	}
	return numfmt.UpToOneDecimal(lo) + "–" + numfmt.UpToOneDecimal(hi)
}

func measList(ms []Measurement) string {
	parts := make([]string, 0, len(ms))
	for _, m := range ms {
		parts = append(parts, fmt.Sprintf("%s %.1f tok/s", m.Kind, m.TokS))
	}
	if len(parts) == 0 {
		return "—"
	}
	return strings.Join(parts, "; ")
}

func firstSentence(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '.'); i >= 0 {
		return strings.TrimSpace(s[:i+1])
	}
	if len(s) > 160 {
		return s[:160] + "…"
	}
	return s
}

func hasKey(m map[string]any, k string) bool { _, ok := m[k]; return ok }

// str is maputil.Str: read a decoded-JSON key as a string, "" when absent or not a string.
// One definition of that rule (internal/maputil), not a copy per decoder.
func str(m map[string]any, k string) string { return maputil.Str(m, k) }

func num(m map[string]any, k string) (float64, bool) {
	switch x := m[k].(type) {
	case float64:
		return x, true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

// boolField returns the bool at k, or true when absent/non-bool (so an absent
// "ok" is not read as a failure). Callers test explicitly for false.
func boolField(m map[string]any, k string) bool {
	if v, ok := m[k].(bool); ok {
		return v
	}
	return true
}
