package devcmd

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type indexPolicyFinding struct {
	Reason string `json:"reason"`
	File   string `json:"file"`
	Line   int    `json:"line,omitempty"`
	Detail string `json:"detail"`
}

type indexPolicyReport struct {
	Schema   string               `json:"schema"`
	Root     string               `json:"root"`
	Findings []indexPolicyFinding `json:"findings"`
	OK       bool                 `json:"ok"`
}

func runIndexPolicy(stdout, stderr io.Writer, root string, args []string, asJSON bool) int {
	if len(args) != 0 {
		fmt.Fprintln(stderr, "usage: fak-dev index policy [--root PATH] [--json]")
		return 2
	}
	paths, err := gitTrackedPaths(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev index policy: %v\n", err)
		return 1
	}
	findings, err := verbTierFindings(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak-dev index policy: %v\n", err)
		return 1
	}
	findings = append(findings, bareDevSpellingFindings(root, paths, parseBareDevAllowlist(bareDevAllowlistRaw))...)
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].Reason != findings[j].Reason {
			return findings[i].Reason < findings[j].Reason
		}
		if findings[i].File != findings[j].File {
			return findings[i].File < findings[j].File
		}
		return findings[i].Line < findings[j].Line
	})
	report := indexPolicyReport{Schema: "fak-dev-index-policy/1", Root: root, Findings: findings, OK: len(findings) == 0}
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak-dev index policy: encode json: %v\n", err)
			return 1
		}
	} else if report.OK {
		fmt.Fprintln(stdout, "index policy OK - runtime verbs are classified and repository callers use the development boundary")
	} else {
		for _, finding := range findings {
			where := finding.File
			if finding.Line > 0 {
				where = fmt.Sprintf("%s:%d", where, finding.Line)
			}
			fmt.Fprintf(stdout, "%s %s: %s\n", finding.Reason, where, finding.Detail)
		}
	}
	if !report.OK {
		return 1
	}
	return 0
}

func gitTrackedPaths(root string) ([]string, error) {
	cmd := windowgate.Command("git", "-C", root, "ls-files", "-z")
	out, err := cmd.Output()
	if err == nil {
		var paths []string
		for _, path := range strings.Split(string(out), "\x00") {
			if path != "" {
				paths = append(paths, path)
			}
		}
		// A directory beneath another checkout can make `git -C root` succeed while
		// listing no files from that parent repository. Treat that as an archive only
		// when root itself has no .git marker.
		if len(paths) > 0 {
			sort.Strings(paths)
			return paths, nil
		}
		if _, statErr := os.Stat(filepath.Join(root, ".git")); statErr == nil || !os.IsNotExist(statErr) {
			return paths, nil
		}
	}
	// A clean `git archive` is the authoritative shared-tree witness and has no .git
	// directory. Walk that immutable snapshot rather than making the policy checker
	// depend on repository metadata that the archive intentionally omits.
	if _, statErr := os.Stat(filepath.Join(root, ".git")); !os.IsNotExist(statErr) {
		return nil, fmt.Errorf("git ls-files: %w", err)
	}
	var paths []string
	walkErr := filepath.WalkDir(root, func(name string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		rel, relErr := filepath.Rel(root, name)
		if relErr != nil {
			return relErr
		}
		paths = append(paths, filepath.ToSlash(rel))
		return nil
	})
	if walkErr != nil {
		return nil, fmt.Errorf("walk archive: %w", walkErr)
	}
	sort.Strings(paths)
	return paths, nil
}
