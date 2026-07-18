package toolproc

// repeatstream.go — the STREAMING front for #5121 (parent #2822; sibling of the
// hermetic ingester in repeatingest.go): feed NATIVE Codex rollout logs, as files
// on disk, into the repeat classifier — and, for an immutable-read call, resolve
// the read target's CONTENT digest so the invalidation-after-mutation key model
// (proved in repeatclass_test.go / repeatreuse_test.go) is exercised on real data.
//
// THE DIVISION OF LABOR. repeatingest.go stays pure and hermetic (bytes in,
// records out, no I/O of its own). This file owns the two impure edges the CLI
// needs and nothing more:
//
//   1. FILES. IngestRolloutFiles opens each rollout JSONL in argument order and
//      streams it through IngestRollout. A file that cannot be opened is an
//      ERROR (the operator named it; silently skipping would fabricate a smaller
//      workload), while a malformed LINE inside a file stays tolerated, exactly
//      as the ingester documents.
//
//   2. DIGESTS. AttachReadDigests re-normalizes each ingested record and, when
//      the registry classifies it as an immutable read with a resolved path,
//      fills CallRecord.Digest from a caller-supplied DigestFn. FileDigest is
//      the real-filesystem DigestFn: SHA-256 of the file's current content,
//      "" when the file is unreadable — the conservative path-only fold, never
//      an invented identity. The digest is of the file AS IT IS NOW: a read of
//      a path whose content has since changed forms a NEW identity and is never
//      folded into (or served from) the stale group — the #2917 key model.
//
// RenderRepeatReport is the human form of the classifier's RepeatReport (the
// --json machine form is the report itself); it prints sizes and canonical
// commands only — never an output body, never a secret (Normalize redacted the
// argument line before anything here runs).

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

// DigestFn resolves a read path (canonical, forward-slashed) to its content or
// identity digest — or "" when no digest can be witnessed (unreadable, absent).
// A nil DigestFn skips enrichment entirely.
type DigestFn func(path string) string

// digestHexLen bounds the rendered digest: 16 hex chars (64 bits) of SHA-256 —
// an identity key, not a security boundary; short enough to stay legible in a
// report row.
const digestHexLen = 16

// FileDigest returns the real-filesystem DigestFn: SHA-256 ("sha256:<16hex>")
// of the file's CURRENT content. Relative paths resolve against root ("" =>
// the process working directory). Any failure to read yields "" — the record
// keeps the conservative path-only identity rather than gaining a fabricated
// digest.
func FileDigest(root string) DigestFn {
	return func(path string) string {
		if path == "" {
			return ""
		}
		p := filepath.FromSlash(path)
		if !filepath.IsAbs(p) && root != "" {
			p = filepath.Join(root, p)
		}
		f, err := os.Open(p)
		if err != nil {
			return ""
		}
		defer f.Close()
		h := sha256.New()
		if _, err := io.Copy(h, f); err != nil {
			return ""
		}
		return "sha256:" + hex.EncodeToString(h.Sum(nil))[:digestHexLen]
	}
}

// AttachReadDigests returns records with CallRecord.Digest filled for every
// record the registry classifies as an immutable read with a resolved path and
// no digest already observed. Non-read records, digest-bearing records, and
// records whose path yields "" pass through unchanged. digest == nil is a no-op.
// The input slice is not mutated.
func AttachReadDigests(records []CallRecord, cfg RepeatConfig, digest DigestFn) []CallRecord {
	out := make([]CallRecord, len(records))
	copy(out, records)
	if digest == nil {
		return out
	}
	memo := map[string]string{} // path -> digest, once per distinct path per pass
	for i := range out {
		if out[i].Digest != "" {
			continue
		}
		nc := Normalize(out[i], cfg)
		if nc.Class != CmdImmutableRead || nc.Path == "" {
			continue
		}
		d, ok := memo[nc.Path]
		if !ok {
			d = digest(nc.Path)
			memo[nc.Path] = d
		}
		if d != "" {
			out[i].Digest = d
		}
	}
	return out
}

// IngestRolloutFiles streams each named rollout JSONL through IngestRollout, in
// argument order, and returns the concatenated CallRecords. A file that cannot
// be opened or read is an error naming the file — the workload must never be
// silently under-counted — while malformed lines inside a readable file remain
// tolerated by the ingester.
func IngestRolloutFiles(paths []string) ([]CallRecord, error) {
	var all []CallRecord
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return nil, fmt.Errorf("rollout %s: %w", p, err)
		}
		all = append(all, IngestRollout(f)...)
		f.Close()
	}
	return all, nil
}

// RenderRepeatReport prints the human form of a RepeatReport: the headline
// totals, the per-class inventory, and the top rows by avoidable saving
// (topN <= 0 renders every group). Only canonical text and sizes are printed.
func RenderRepeatReport(w io.Writer, rep RepeatReport, topN int) {
	fmt.Fprintf(w, "repeats: records=%d groups=%d output_bytes=%d avoidable_spawns=%d avoidable_input_bytes=%d\n",
		rep.Totals.Records, rep.Totals.Groups, rep.Totals.OutputBytes,
		rep.Totals.AvoidableSpawns, rep.Totals.AvoidableInputBytes)

	classes := make([]string, 0, len(rep.Totals.PerClass))
	for c := range rep.Totals.PerClass {
		classes = append(classes, string(c))
	}
	sort.Strings(classes)
	fmt.Fprint(w, "per_class:")
	for _, c := range classes {
		fmt.Fprintf(w, " %s=%d", c, rep.Totals.PerClass[RepeatClass(c)])
	}
	fmt.Fprintln(w)

	n := len(rep.Groups)
	if topN > 0 && topN < n {
		n = topN
	}
	for _, g := range rep.Groups[:n] {
		fmt.Fprintf(w, "  %-16s %-17s count=%-6d dups=%d+%d bytes=%-10d avoid=%d/%dB %q",
			g.Class, g.Reuse, g.Count, g.ExactDups, g.NearDups,
			g.OutputBytes, g.AvoidableSpawns, g.AvoidableInputBytes, g.Canonical)
		if g.Digest != "" {
			fmt.Fprintf(w, " digest=%s", g.Digest)
		}
		fmt.Fprintln(w)
	}
	if n < len(rep.Groups) {
		fmt.Fprintf(w, "  ... %d more groups (use --json for the full report)\n", len(rep.Groups)-n)
	}
}
