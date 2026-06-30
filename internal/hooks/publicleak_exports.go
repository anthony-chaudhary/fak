package hooks

import (
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
	d := &StagedDiff{
		Root:        root,
		run:         realRunner,
		ctx:         context.Background(),
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
	findings, mode := scanPublicLeakTree(t)
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

func scanPublicLeakTree(t *TrackedTree) ([]Finding, string) {
	mode := "shape-only"
	var needles []string
	if b, ok := t.FileBytes(privateNeedlesRel); ok {
		needles = treeExportAuditNeedles(b)
		if len(needles) > 0 {
			mode = "full"
		}
	}
	var findings []Finding
	for _, f := range t.Paths {
		norm := strings.ReplaceAll(f, "\\", "/")
		if selfReferentialLeak[norm] {
			continue
		}
		body, ok := t.FileBytes(f)
		if !ok {
			continue
		}
		if strings.ContainsRune(string(body), '\x00') {
			continue
		}
		for i, line := range strings.Split(string(body), "\n") {
			payloadL := strings.ToLower(line)
			for _, n := range needles {
				if strings.Contains(payloadL, strings.ToLower(n)) {
					findings = append(findings, Finding{
						Gate: "PUBLIC_LEAK", File: f, Line: i + 1,
						Detail: "[" + n + "]  " + preview(line),
					})
				}
			}
			for _, rx := range auditRegexes {
				if rx.re.MatchString(line) {
					findings = append(findings, Finding{
						Gate: "PUBLIC_LEAK", File: f, Line: i + 1,
						Detail: "[" + rx.label + "]  " + preview(line),
					})
				}
			}
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
