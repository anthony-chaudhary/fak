package hooks

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

type PublicLeakTreeReport struct {
	OK       bool      `json:"ok"`
	Mode     string    `json:"mode"`
	Count    int       `json:"count"`
	Findings []Finding `json:"findings"`
	Misses   []Finding `json:"misses"`
}

func ScanStagedPublicLeak(root string) ([]Finding, error) {
	d, err := ReadStagedDiff(root)
	if err != nil {
		return nil, err
	}
	return gatePublicLeak(d)
}

func ScanRangePublicLeak(root, revRange string) ([]Finding, error) {
	out, code, err := realRunner(context.Background(), root, "diff", "--unified=0", "--no-color", "--diff-filter=ACMR", revRange)
	if err != nil {
		return nil, err
	}
	if code != 0 {
		return nil, ErrCouldNotRun
	}
	tip := revRange
	if _, after, ok := strings.Cut(revRange, ".."); ok {
		tip = after
	}
	d := &StagedDiff{
		Root:        root,
		run:         realRunner,
		ctx:         context.Background(),
		Treeish:     tip + ":",
		AddedByFile: parseUnifiedAddedLines(out),
		fileCache:   map[string]fileEntry{},
	}
	return gatePublicLeak(d)
}

func AuditPublicLeakTree(root string) (PublicLeakTreeReport, error) {
	t, err := ReadTrackedTree(root)
	if err != nil {
		return PublicLeakTreeReport{}, err
	}
	findings, mode := scanPublicLeakTreeFiles(t.Paths, oneShotPublicLeakTreeReader(t))
	return PublicLeakTreeReport{OK: len(findings) == 0, Mode: mode, Count: len(findings), Findings: findings, Misses: findings}, nil
}

func PublicLeakTreeText(report PublicLeakTreeReport) string {
	if report.OK {
		return "public-scrub: tree clean (no AUDIT_NEEDLES in tracked files)\n"
	}
	return strings.Join(PublicLeakFindingLines(report.Findings, "tracked tree"), "\n") + "\n"
}

func PublicLeakFindingLines(findings []Finding, where string) []string {
	lines := make([]string, 0, len(findings))
	for _, f := range findings {
		loc := f.File
		if loc == "" {
			loc = where
			if where == "the commit message" {
				loc = "message"
			}
		}
		if f.Line > 0 {
			loc = fmt.Sprintf("%s:%d", loc, f.Line)
		}
		lines = append(lines, fmt.Sprintf("PUBLIC_LEAK %s %s", loc, f.Detail))
	}
	return lines
}

// oneShotPublicLeakTreeReader is for the fresh, exclusively owned tree in
// AuditPublicLeakTree, not a tree shared with other hygiene gates. Keep the first
// observation (including a failed read) until its final use: the sidecar can also
// be tracked, and an unmerged index can list the same path more than once. Bodies
// used only once need not stay reachable for the rest of the whole-tree scan.
func oneShotPublicLeakTreeReader(t *TrackedTree) func(string) ([]byte, bool) {
	remaining := make(map[string]int, len(t.Paths)+1)
	remaining[privateNeedlesRel]++
	for _, path := range t.Paths {
		remaining[path]++
	}
	return func(path string) ([]byte, bool) {
		body, ok := t.FileBytes(path)
		remaining[path]--
		if remaining[path] == 0 {
			delete(t.fileCache, path)
		}
		return body, ok
	}
}

func scanPublicLeakTree(t *TrackedTree) ([]Finding, string) {
	return scanPublicLeakTreeFiles(t.Paths, t.FileBytes)
}

func scanPublicLeakTreeFiles(paths []string, fileBytes func(string) ([]byte, bool)) ([]Finding, string) {
	mode := "shape-only"
	var needles []string
	if b, ok := fileBytes(privateNeedlesRel); ok {
		needles = treeExportAuditNeedles(b)
		if len(needles) > 0 {
			mode = "full"
		}
	}
	var findings []Finding
	for _, f := range paths {
		norm := strings.ReplaceAll(f, "\\", "/")
		if selfReferentialLeak[norm] {
			continue
		}
		findings = append(findings, publicLeakLineFindings(norm, 0, norm, needles)...)
		body, ok := fileBytes(f)
		if !ok {
			continue
		}
		if bytes.IndexByte(body, 0) >= 0 {
			continue
		}
		lineNo := 1
		for line := range strings.SplitSeq(string(body), "\n") {
			findings = append(findings, publicLeakLineFindings(line, lineNo, f, needles)...)
			lineNo++
		}
	}
	return findings, mode
}

func treeExportAuditNeedles(raw []byte) []string {
	var priv struct {
		ExportAuditNeedles []string `json:"export_audit_needles"`
	}
	if err := json.Unmarshal(raw, &priv); err != nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, n := range priv.ExportAuditNeedles {
		if n == "" || seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	return out
}
