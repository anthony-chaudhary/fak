package hooks

import (
	"path"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sotamatrix"
)

// gate_priorart.go — the PRIOR_ART advisory gate. When a staged commit touches a KERNEL file
// that a sotamatrix row claims as its FileGlobs, this gate prints the SOTA reference for that
// operation and suggests a `Prior-art:` trailer — so an agent doesn't re-derive from scratch
// what llama.cpp / Marlin / FlashInfer already solved (the documented failure mode the
// sotamatrix package exists to kill). It is ADVISORY: it runs in WARN mode by default
// (Gate.DefaultMode = "warn") and never blocks a commit.
//
// SOURCE OF TRUTH: the gate reads sotamatrix.Operations() and NEVER hard-codes a SOTA
// reference — the same discipline gate_tierdeclared.go uses against the architest tier table.
// sotamatrix is the single maintained datum; this gate is one of its readers. Adding a kernel
// op (and its prior-art reference) means adding ONE row there, not editing this gate.

// matchKernelGlob reports whether path matches a sotamatrix FileGlob. It normalizes backslashes
// to forward slashes in p, then matches against glob. The matrix globs are all ONE directory
// level deep (e.g. "internal/model/awq*.go", "internal/metalgemm/*", "internal/model/*.metal"),
// so two forms cover them:
//   - a trailing "/*" glob ("internal/metalgemm/*") matches any path under that prefix dir;
//   - every other glob is matched with path.Match, which handles single-segment wildcards like
//     "internal/model/awq*.go" or "internal/model/*.metal" correctly (the "*" never spans a "/").
func matchKernelGlob(p, glob string) bool {
	p = strings.ReplaceAll(p, "\\", "/")
	// A "dir/*" glob: accept any path directly under that prefix directory. path.Match's "*"
	// will not cross a "/", so a deeper path under the dir still matches this prefix form, and a
	// one-level file matches too — both are "touched a file in this kernel dir".
	if strings.HasSuffix(glob, "/*") {
		prefix := strings.TrimSuffix(glob, "*") // keeps the trailing "/"
		return strings.HasPrefix(p, prefix)
	}
	ok, err := path.Match(glob, p)
	return err == nil && ok
}

// priorArtSuppressTrailer is the token an author adds (in a touched file, a comment, or a doc
// line) to silence the advisory: it documents that prior art WAS consulted. A pre-commit gate
// sees the staged diff, not the commit message, so the suppression is keyed on an added line
// rather than a message trailer.
const priorArtSuppressTrailer = "prior-art:"

// gatePriorArt emits ONE advisory PRIOR_ART finding per distinct sotamatrix op whose FileGlobs
// match a touched path, unless the staged diff itself adds a line containing "Prior-art:"
// (case-insensitive) anywhere — that lets an author who documents the prior art silence it.
// Findings are deduped by op slug (touching three files of one op gives one finding) and sorted
// by slug for determinism.
func gatePriorArt(d *StagedDiff) ([]Finding, error) {
	// Suppression: any added line carrying the "Prior-art:" token quiets the whole gate. The
	// author has attested to consulting prior art, which is exactly what the advisory asks for.
	for _, al := range d.AddedLines() {
		if strings.Contains(strings.ToLower(al.Text), priorArtSuppressTrailer) {
			return nil, nil
		}
	}

	ops := sotamatrix.Operations()
	// matched: slug -> (op, first touched path that matched it), deduped by slug.
	type hit struct {
		op   sotamatrix.Op
		file string
	}
	matched := map[string]hit{}
	for _, raw := range d.StagedPaths {
		p := strings.ReplaceAll(raw, "\\", "/")
		for _, op := range ops {
			if _, seen := matched[op.Slug]; seen {
				continue
			}
			for _, glob := range op.FileGlobs {
				if matchKernelGlob(p, glob) {
					matched[op.Slug] = hit{op: op, file: p}
					break
				}
			}
		}
	}

	slugs := make([]string, 0, len(matched))
	for s := range matched {
		slugs = append(slugs, s)
	}
	sort.Strings(slugs)

	var findings []Finding
	for _, s := range slugs {
		h := matched[s]
		findings = append(findings, Finding{
			Gate:   "PRIOR_ART",
			File:   h.file,
			Line:   0,
			Detail: priorArtDetail(h.op),
		})
	}
	return findings, nil
}

// priorArtDetail renders the one-line advisory for an op. It keeps the SOTA, route, and primary
// link (the actionable parts) and trims the oracle so the whole line stays under ~240 chars.
func priorArtDetail(op sotamatrix.Op) string {
	head := `kernel op "` + op.Title + `" — check prior art before scratch-building: ` + op.SOTA +
		" (route=" + string(op.Route) + ", read " + op.PrimaryLink
	tail := `). Add a "Prior-art:" trailer or comment to silence.`
	const budget = 240
	// Oracle is the trimmable part: include "; oracle: <oracle>" only if it fits the budget.
	if op.Oracle != "" {
		withOracle := head + "; oracle: " + op.Oracle + tail
		if len(withOracle) <= budget {
			return withOracle
		}
		// Try a trimmed oracle so the witness hint survives where it can.
		room := budget - len(head) - len("; oracle: ") - len(tail)
		if room > 12 { // only bother if a meaningful fragment fits
			return head + "; oracle: " + op.Oracle[:room] + tail
		}
	}
	return head + tail
}
