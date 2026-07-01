package branchrole

import (
	"bufio"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	PublicDocClassUnclassified        = "unclassified"
	PublicDocClassPublicFrontDoor     = "public-front-door"
	PublicDocClassReleaseArtifact     = "release-artifact"
	PublicDocClassContributorWorkflow = "contributor-workflow"
	PublicDocClassAdjudicationFixture = "adjudication-fixture"
	PublicDocClassAuditDoc            = "audit-doc"
)

var publicDocRefPatterns = []*regexp.Regexp{
	regexp.MustCompile(`github\.com/anthony-chaudhary/fak/(blob|tree)/main\b`),
	regexp.MustCompile(`raw\.githubusercontent\.com/anthony-chaudhary/fak/main\b`),
	regexp.MustCompile(`colab\.research\.google\.com/github/anthony-chaudhary/fak/blob/main\b`),
	regexp.MustCompile(`badge\.svg\?branch=main\b`),
	regexp.MustCompile(`github\.com/anthony-chaudhary/fak/releases/latest\b`),
	regexp.MustCompile(`go\s+install\s+github\.com/anthony-chaudhary/fak/cmd/fak@(latest|main|master)\b`),
	regexp.MustCompile(`\bgit\s+(fetch|merge|push)\s+origin[/ ](main|master)\b`),
	regexp.MustCompile(`\borigin/(main|master)\b`),
	regexp.MustCompile("(?i)\\bwork directly on (the trunk )?\\(?`?\\bmain\\b`?\\)?"),
	regexp.MustCompile("(?i)\\bcommits?\\b[^\\n]{0,40}`?\\bmain\\b`?"),
}

// PublicDocFinding is one branch-shaped public/contributor doc reference.
type PublicDocFinding struct {
	Path  string
	Line  int
	Class string
	Text  string
}

// AuditPublicDocRefs scans the public front-door and contributor docs that #1700
// tracks. It is intentionally narrower than AuditHardcodedRefs: generic prose
// such as "main agent" is out of scope unless it is a branch-shaped URL, install
// command, or contributor workflow instruction.
func AuditPublicDocRefs(root string) ([]PublicDocFinding, error) {
	if root == "" {
		var err error
		root, err = FindRoot("")
		if err != nil {
			return nil, err
		}
	}
	files, err := publicDocAuditFiles(root)
	if err != nil {
		return nil, err
	}
	var findings []PublicDocFinding
	for _, rel := range files {
		rows, scanErr := scanPublicDocRefFile(filepath.Join(root, filepath.FromSlash(rel)), rel)
		if scanErr != nil {
			return nil, scanErr
		}
		findings = append(findings, rows...)
	}
	return findings, nil
}

func publicDocAuditFiles(root string) ([]string, error) {
	exact := []string{
		"README.md",
		"INSTALL.md",
		"GETTING-STARTED.md",
		"START-HERE.md",
		"AGENTS.md",
		"CONTRIBUTING.md",
		".github/copilot-instructions.md",
		"docs/fak/deployment-guide.md",
		"docs/branch-regime.md",
		"docs/branch-regime-hardcoded-ref-audit.md",
		"docs/branch-regime-shadow-cutover.md",
		"docs/branch-regime-public-front-door-audit.md",
	}
	seen := map[string]bool{}
	var out []string
	for _, rel := range exact {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); err == nil {
			seen[rel] = true
			out = append(out, rel)
		}
	}
	integrations := filepath.Join(root, "docs", "integrations")
	if _, err := os.Stat(integrations); err == nil {
		if err := filepath.WalkDir(integrations, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			if filepath.Ext(path) != ".md" {
				return nil
			}
			rel, relErr := filepath.Rel(root, path)
			if relErr != nil {
				return relErr
			}
			rel = filepath.ToSlash(rel)
			if !seen[rel] {
				seen[rel] = true
				out = append(out, rel)
			}
			return nil
		}); err != nil {
			return nil, err
		}
	}
	sort.Strings(out)
	return out, nil
}

func scanPublicDocRefFile(path, rel string) ([]PublicDocFinding, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []PublicDocFinding
	scanner := bufio.NewScanner(f)
	lineNo := 0
	for scanner.Scan() {
		lineNo++
		text := scanner.Text()
		if !publicDocRefLine(text) {
			continue
		}
		out = append(out, PublicDocFinding{
			Path:  rel,
			Line:  lineNo,
			Class: ClassifyPublicDocRef(rel, text),
			Text:  strings.TrimSpace(text),
		})
	}
	return out, scanner.Err()
}

func publicDocRefLine(line string) bool {
	for _, pattern := range publicDocRefPatterns {
		if pattern.MatchString(line) {
			return true
		}
	}
	return false
}

// ClassifyPublicDocRef classifies branch-shaped refs in public/contributor docs.
func ClassifyPublicDocRef(path, line string) string {
	p := filepath.ToSlash(path)
	lower := strings.ToLower(line)
	switch {
	case p == "docs/branch-regime.md" ||
		p == "docs/branch-regime-hardcoded-ref-audit.md" ||
		p == "docs/branch-regime-shadow-cutover.md" ||
		p == "docs/branch-regime-public-front-door-audit.md" ||
		strings.HasPrefix(p, "internal/branchrole/"):
		return PublicDocClassAuditDoc
	case strings.Contains(lower, "go install github.com/anthony-chaudhary/fak/cmd/fak@latest") ||
		strings.Contains(lower, "github.com/anthony-chaudhary/fak/releases/latest"):
		return PublicDocClassReleaseArtifact
	case strings.Contains(lower, "go install github.com/anthony-chaudhary/fak/cmd/fak@main") ||
		strings.Contains(lower, "go install github.com/anthony-chaudhary/fak/cmd/fak@master"):
		return PublicDocClassUnclassified
	case strings.HasPrefix(p, "docs/integrations/") &&
		(strings.Contains(lower, "git push origin main") || strings.Contains(lower, "git push origin master")):
		return PublicDocClassAdjudicationFixture
	case p == "docs/fak/deployment-guide.md" && strings.Contains(lower, `"command":"git push origin main"`):
		return PublicDocClassAdjudicationFixture
	case p == "AGENTS.md" || p == "CONTRIBUTING.md" || p == ".github/copilot-instructions.md":
		return PublicDocClassContributorWorkflow
	case publicRepoMainLink(lower):
		return PublicDocClassPublicFrontDoor
	default:
		return PublicDocClassUnclassified
	}
}

func publicRepoMainLink(lower string) bool {
	for _, needle := range []string{
		"github.com/anthony-chaudhary/fak/blob/main",
		"github.com/anthony-chaudhary/fak/tree/main",
		"raw.githubusercontent.com/anthony-chaudhary/fak/main",
		"colab.research.google.com/github/anthony-chaudhary/fak/blob/main",
		"badge.svg?branch=main",
	} {
		if strings.Contains(lower, needle) {
			return true
		}
	}
	return false
}
