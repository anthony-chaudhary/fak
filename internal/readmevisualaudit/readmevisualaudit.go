package readmevisualaudit

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const Schema = "fleet-readme-visual-audit/1"

var (
	readmeGlobs      = []string{"**/README.md", "**/readme.md"}
	excludeFragments = []string{"/.pytest_cache/", "/_registry/", "/node_modules/", "/.git/", "/vendor/", "/testdata/"}
	boxGlyphs        = "│┌┐└┘├┤┬┴┼─━┃┏┓┗┛┣┫┳┻╋═║╔╗╚╝╠╣╦╩╬▶◀▲▼►◄→←↑↓↳↴↦⟶⟵⇒⇐⇨█▉▊▋▌▍▎▏▕▁▂▃▄▅▆▇░▒▓▬■"
	asciiArrows      = []string{"-->", "==>", "<--", "<==", "+--", "--+", "|--", "--|", ".->", "o--"}
	fenceRE          = regexp.MustCompile("(?s)```([^\\n`]*)\\n(.*?)```")
	imgRE            = regexp.MustCompile(`!\[([^\]]*)\]\(([^)\s]+)`)
	imgExtRE         = regexp.MustCompile(`(?i)\.(svg|png|jpe?g|gif|webp|avif)(?:[?#].*)?$`)
	badgeMarkers     = []string{"shields.io", "img.shields", "badge", "colab.research.google.com/assets", "/badges/", "buymeacoffee", "ko-fi", "/actions/workflows/"}
)

type AuditOne struct {
	HasVisual   bool     `json:"has_visual"`
	Kinds       []string `json:"kinds"`
	Mermaid     bool     `json:"mermaid"`
	Images      []string `json:"images"`
	ASCIIBlocks int      `json:"ascii_blocks"`
}

type Check struct {
	Check  string `json:"check"`
	Status string `json:"status"`
	Detail string `json:"detail"`
}

type Report struct {
	Schema     string         `json:"schema"`
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
	Workspace  string         `json:"workspace"`
	Counts     map[string]int `json:"counts"`
	Checks     []Check        `json:"checks"`
}

func Fences(text string) [][2]string {
	matches := fenceRE.FindAllStringSubmatch(text, -1)
	out := make([][2]string, 0, len(matches))
	for _, m := range matches {
		out = append(out, [2]string{strings.TrimSpace(m[1]), m[2]})
	}
	return out
}

func HasMermaid(text string) bool {
	for _, f := range Fences(text) {
		if strings.HasPrefix(strings.ToLower(f[0]), "mermaid") {
			return true
		}
	}
	return false
}

func DiagramImages(text string) []string {
	var out []string
	for _, m := range imgRE.FindAllStringSubmatch(text, -1) {
		target := m[2]
		low := strings.ToLower(target)
		badge := false
		for _, marker := range badgeMarkers {
			if strings.Contains(low, marker) {
				badge = true
				break
			}
		}
		if badge {
			continue
		}
		if imgExtRE.MatchString(low) || strings.HasPrefix(low, "visuals/") || strings.HasPrefix(low, "../visuals/") || strings.HasPrefix(low, "../../visuals/") {
			out = append(out, target)
		}
	}
	return out
}

func ASCIIDiagramBlocks(text string) int {
	count := 0
	for _, f := range Fences(text) {
		hits := 0
		for _, line := range strings.Split(f[1], "\n") {
			if strings.ContainsAny(line, boxGlyphs) || strmatch.ContainsAny(line, asciiArrows...) {
				hits++
				if hits >= 2 {
					count++
					break
				}
			}
		}
	}
	return count
}

func AuditText(text string) AuditOne {
	mermaid := HasMermaid(text)
	images := DiagramImages(text)
	asciiN := ASCIIDiagramBlocks(text)
	var kinds []string
	if mermaid {
		kinds = append(kinds, "mermaid")
	}
	if len(images) > 0 {
		kinds = append(kinds, fmt.Sprintf("image×%d", len(images)))
	}
	if asciiN > 0 {
		kinds = append(kinds, fmt.Sprintf("ascii×%d", asciiN))
	}
	return AuditOne{HasVisual: len(kinds) > 0, Kinds: kinds, Mermaid: mermaid, Images: images, ASCIIBlocks: asciiN}
}

func ListReadmes(root string) []string {
	cmd := exec.Command("git", "-C", root, "ls-files", "*README.md", "*readme.md")
	windowgate.ConfigureBackgroundCommand(cmd)
	if out, err := cmd.Output(); err == nil && strings.TrimSpace(string(out)) != "" {
		var rels []string
		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if line != "" && !excluded(line) {
				rels = append(rels, filepath.ToSlash(line))
			}
		}
		sort.Strings(rels)
		return rels
	}
	seen := map[string]bool{}
	for _, pat := range readmeGlobs {
		matches, _ := filepath.Glob(filepath.Join(root, pat))
		for _, p := range matches {
			if rel, err := filepath.Rel(root, p); err == nil && !excluded(rel) {
				seen[filepath.ToSlash(rel)] = true
			}
		}
	}
	var rels []string
	for rel := range seen {
		rels = append(rels, rel)
	}
	sort.Strings(rels)
	return rels
}

func Collect(workspace string) Report {
	root, _ := filepath.Abs(workspace)
	rels := ListReadmes(root)
	if len(rels) == 0 {
		return BuildPayload(root, nil, "no tracked README.md found (run from repo ROOT)")
	}
	var checks []Check
	for _, rel := range rels {
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			checks = append(checks, Check{Check: rel, Status: "FAIL", Detail: "cannot read: " + err.Error()})
			continue
		}
		v := AuditText(string(data))
		if v.HasVisual {
			checks = append(checks, Check{Check: rel, Status: "OK", Detail: "has " + strings.Join(v.Kinds, ", ")})
		} else {
			checks = append(checks, Check{Check: rel, Status: "FAIL", Detail: "text-only: no mermaid, diagram image, or ASCII diagram"})
		}
	}
	return BuildPayload(root, checks, "")
}

func BuildPayload(workspace string, checks []Check, errorText string) Report {
	counts := map[string]int{"OK": 0, "WARN": 0, "FAIL": 0, "ADVISORY": 0}
	var fails []Check
	for _, c := range checks {
		counts[c.Status]++
		if c.Status == "FAIL" {
			fails = append(fails, c)
		}
	}
	total := len(checks)
	var ok bool
	var verdict, finding, reason, nextAction string
	if errorText != "" {
		ok, verdict, finding = false, "AUDIT_ERROR", "tooling_error"
		reason = errorText
		nextAction = "run from the repo ROOT so `git ls-files *README.md` resolves, then re-run"
	} else if len(fails) > 0 {
		ok, verdict, finding = false, "ACTION", "readmes_text_only"
		names := make([]string, 0, min(len(fails), 6))
		for i, f := range fails {
			if i >= 6 {
				break
			}
			names = append(names, f.Check)
		}
		more := ""
		if len(fails) > 6 {
			more = fmt.Sprintf(" (+%d more)", len(fails)-6)
		}
		reason = fmt.Sprintf("%d/%d README(s) are text-only: %s%s", len(fails), total, strings.Join(names, ", "), more)
		nextAction = "add one diagram-class visual to each FAIL - a ```mermaid block (GitHub-rendered pages), an embedded visuals/*.png, or a fenced ASCII diagram (Jekyll-served docs/*) - accurate to the page, no new claims"
	} else {
		ok, verdict, finding = true, "OK", "all_readmes_visual"
		reason = fmt.Sprintf("all %d tracked README(s) carry a diagram-class visual", total)
		nextAction = "no action; re-run after adding a README or stripping a diagram"
	}
	return Report{Schema: Schema, OK: ok, Verdict: verdict, Finding: finding, Reason: reason, NextAction: nextAction, Workspace: workspace, Counts: counts, Checks: checks}
}

func Render(report Report) string {
	counts := report.Counts
	total := counts["OK"] + counts["WARN"] + counts["FAIL"] + counts["ADVISORY"]
	lines := []string{
		fmt.Sprintf("readme-visual audit: %s (%s)", report.Verdict, report.Finding),
		fmt.Sprintf("readmes: %d/%d have a visual · fail=%d", counts["OK"], total, counts["FAIL"]),
		"next: " + report.NextAction,
	}
	for _, c := range report.Checks {
		mark := "  ?  "
		if c.Status == "OK" {
			mark = "  ok "
		} else if c.Status == "FAIL" {
			mark = " FAIL"
		}
		lines = append(lines, fmt.Sprintf("%s  %-48s %s", mark, c.Check, c.Detail))
	}
	return strings.Join(lines, "\n")
}

func MarshalJSON(report Report) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}

func excluded(rel string) bool {
	slug := "/" + filepath.ToSlash(rel)
	for _, frag := range excludeFragments {
		if strings.Contains(slug, frag) {
			return true
		}
	}
	return false
}
