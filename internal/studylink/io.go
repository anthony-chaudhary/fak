package studylink

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func ReadLedger(path string) (Ledger, error) {
	var ledger Ledger
	b, err := os.ReadFile(path)
	if err != nil {
		return ledger, fmt.Errorf("studylink: read ledger: %w", err)
	}
	if err := json.Unmarshal(b, &ledger); err != nil {
		return ledger, fmt.Errorf("studylink: decode ledger: %w", err)
	}
	return ledger, nil
}

func WriteLedger(path string, ledger Ledger) error {
	if err := ValidateStructure(ledger, nil, ""); err != nil {
		return err
	}
	b, err := json.MarshalIndent(ledger, "", "  ")
	if err != nil {
		return fmt.Errorf("studylink: encode ledger: %w", err)
	}
	b = append(b, '\n')
	return atomicWrite(path, b)
}

func WriteSummary(path string, summary Summary) error {
	return atomicWrite(path, RenderSummary(summary))
}

func RenderSummary(summary Summary) []byte {
	var b bytes.Buffer
	fmt.Fprintln(&b, "# vLLM → FAK evidence join")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "Deterministic join over **%d clusters** (**%d actionable**) from the compact vLLM index. The FAK side was searched from the complete captured study-forge corpus (%d records at `%s`) and repository paths were checked at `%s`.\n\n", summary.Total, summary.Actionable, summary.Sources.ForgeRecordCount, summary.Sources.ForgeCutoff, summary.Sources.RepositoryRevision)
	fmt.Fprintln(&b, "## Reproduce")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "```bash")
	fmt.Fprintln(&b, "fak study-link build \\")
	fmt.Fprintf(&b, "  --index %s \\\n", summary.Sources.IndexPath)
	fmt.Fprintf(&b, "  --forge /path/to/%s \\\n", filepath.Base(summary.Sources.ForgePath))
	fmt.Fprintf(&b, "  --adjacency %s \\\n", summary.Sources.AdjacencyPath)
	fmt.Fprintln(&b, "  --repo . \\")
	fmt.Fprintln(&b, "  --out docs/research/vllm-fak-join-2026-08-27/ledger.json \\")
	fmt.Fprintln(&b, "  --summary docs/research/vllm-fak-join-2026-08-27/README.md")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "fak study-link validate \\")
	fmt.Fprintln(&b, "  --ledger docs/research/vllm-fak-join-2026-08-27/ledger.json \\")
	fmt.Fprintf(&b, "  --index %s \\\n", summary.Sources.IndexPath)
	fmt.Fprintf(&b, "  --forge /path/to/%s \\\n", filepath.Base(summary.Sources.ForgePath))
	fmt.Fprintf(&b, "  --adjacency %s --repo .\n", summary.Sources.AdjacencyPath)
	fmt.Fprintln(&b, "```")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "The forge input must be a terminal `complete` capture. A truncated `gh issue list --limit 1000` export is rejected by construction because it has no complete study-forge receipt.")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Counts")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| disposition | count |")
	fmt.Fprintln(&b, "|---|---:|")
	for _, disposition := range dispositionOrder {
		fmt.Fprintf(&b, "| `%s` | %d |\n", disposition, summary.Counts[disposition])
	}
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "`landed` and `open_exact` require a uniquely scoped exact witness: either a conservative implementation-title match or an explicit checked-in vLLM ticket-map/history anchor. `landed` additionally requires captured closed state, an existing tracked path, and issue-linked commit history. Reproducible lexical candidates remain `partial`/manual review; the generator never promotes them semantically.")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Strong samples")
	fmt.Fprintln(&b)
	renderSamples(&b, summary.StrongSamples)
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Ambiguous samples")
	fmt.Fprintln(&b)
	renderSamples(&b, summary.AmbiguousSamples)
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "## Manual review (%d)\n\n", len(summary.ManualReview))
	fmt.Fprintln(&b, "Every `partial` and `conflict` join is listed; no ambiguous candidate is silently promoted.")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| cluster | disposition | reason | reproducible match sample |")
	fmt.Fprintln(&b, "|---|---|---|---|")
	for _, row := range summary.ManualReview {
		matches := strings.Join(firstStrings(row.Matches, 3), "<br>")
		matches += fmt.Sprintf("<br>total=%d full=%s", row.MatchCount, row.FullMatchesDigest)
		fmt.Fprintf(&b, "| `%s` | `%s` | %s | %s |\n", mdCell(row.ClusterID), row.Disposition, mdCell(row.Reason), mdCell(matches))
	}
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Source checksums")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- compact index: `%s`\n", summary.Sources.IndexSHA256)
	fmt.Fprintf(&b, "- complete FAK forge corpus: `%s`\n", summary.Sources.ForgeSHA256)
	fmt.Fprintf(&b, "- adjacency manifest: `%s`\n", summary.Sources.AdjacencySHA256)
	fmt.Fprintf(&b, "- ledger source revision: `%s`\n", summary.Sources.ForgeRevision)
	return b.Bytes()
}

func renderSamples(b *bytes.Buffer, samples []SampleJoin) {
	if len(samples) == 0 {
		fmt.Fprintln(b, "None.")
		return
	}
	fmt.Fprintln(b, "| cluster | disposition | artifacts |")
	fmt.Fprintln(b, "|---|---|---|")
	for _, sample := range samples {
		fmt.Fprintf(b, "| `%s` | `%s` | %s |\n", mdCell(sample.ClusterID), sample.Disposition, mdCell(strings.Join(sample.Artifacts, "<br>")))
	}
}

func mdCell(s string) string { return strings.ReplaceAll(strings.ReplaceAll(s, "|", "\\|"), "\n", " ") }

func atomicWrite(path string, data []byte) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("studylink: create output directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".studylink-*.tmp")
	if err != nil {
		return fmt.Errorf("studylink: create temp output: %w", err)
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("studylink: write output: %w", err)
	}
	if err := tmp.Chmod(0o644); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(name, path); err != nil {
		return fmt.Errorf("studylink: replace output: %w", err)
	}
	return nil
}

func SortedCounts(counts map[Disposition]int) []Disposition {
	out := append([]Disposition(nil), dispositionOrder...)
	sort.SliceStable(out, func(i, j int) bool { return i < j })
	return out
}
