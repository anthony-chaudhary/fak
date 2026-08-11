package disambiguation

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// PublicManifests is the admission view derived from the repository's existing
// public contracts. Leaves are real internal/<name> and cmd/<name> directories;
// lanes are declarations in dos.toml [lanes] and [lanes.trees].
type PublicManifests struct {
	Leaves []string `json:"leaves"`
	Lanes  []string `json:"lanes"`
}

var (
	tomlSectionRE = regexp.MustCompile(`^\s*\[([^]]+)\]\s*(?:#.*)?$`)
	tomlKeyRE     = regexp.MustCompile(`^\s*([A-Za-z0-9_.-]+)\s*=`)
	tomlQuotedRE  = regexp.MustCompile(`"([A-Za-z0-9_.-]+)"`)
)

// FindRepositoryRoot finds the public repository manifests without consulting
// private configuration or environment-specific state.
func FindRepositoryRoot(start string) (string, error) {
	if strings.TrimSpace(start) == "" {
		start = "."
	}
	root, err := filepath.Abs(start)
	if err != nil {
		return "", err
	}
	if info, statErr := os.Stat(root); statErr == nil && !info.IsDir() {
		root = filepath.Dir(root)
	}
	for {
		if fileExists(filepath.Join(root, "dos.toml")) && dirExists(filepath.Join(root, "internal")) {
			return root, nil
		}
		parent := filepath.Dir(root)
		if parent == root {
			break
		}
		root = parent
	}
	return "", fmt.Errorf("public manifests not found from %q (want dos.toml and internal/)", start)
}

// LoadPublicManifests reads the authoritative public leaf and lane registries.
func LoadPublicManifests(repoRoot string) (PublicManifests, error) {
	leaves := map[string]struct{}{}
	for _, base := range []string{"internal", "cmd"} {
		entries, err := os.ReadDir(filepath.Join(repoRoot, base))
		if err != nil {
			return PublicManifests{}, fmt.Errorf("read public leaf manifest %s/: %w", base, err)
		}
		for _, entry := range entries {
			if entry.IsDir() && validOwnerName(entry.Name()) {
				leaves[strings.ToLower(entry.Name())] = struct{}{}
			}
		}
	}
	lanes, err := loadPublicLanes(filepath.Join(repoRoot, "dos.toml"))
	if err != nil {
		return PublicManifests{}, err
	}
	return PublicManifests{Leaves: sortedKeys(leaves), Lanes: sortedKeys(lanes)}, nil
}

// NewAdmittedIndex is the generator/admission constructor. It refuses to build
// an index until every ownership target exists in the public manifests.
func NewAdmittedIndex(entries []Entry, manifests PublicManifests) (*Index, error) {
	if err := AdmitOwnership(entries, manifests); err != nil {
		return nil, err
	}
	return NewIndex(entries)
}

// AdmitOwnership rejects entries that cannot be routed through both public
// registries. It is called by generation before any output is written.
func AdmitOwnership(entries []Entry, manifests PublicManifests) error {
	leaves := stringSet(manifests.Leaves)
	lanes := stringSet(manifests.Lanes)
	for i, entry := range entries {
		leaf := strings.ToLower(strings.TrimSpace(entry.Owner.Leaf))
		lane := strings.ToLower(strings.TrimSpace(entry.Owner.Lane))
		if _, ok := leaves[leaf]; !ok {
			return fmt.Errorf("entries[%d] %q: owner leaf %q is absent from public leaf manifest", i, entry.Identity.CanonicalTerm, entry.Owner.Leaf)
		}
		if _, ok := lanes[lane]; !ok {
			return fmt.Errorf("entries[%d] %q: dispatch lane %q is absent from public lane manifest", i, entry.Identity.CanonicalTerm, entry.Owner.Lane)
		}
	}
	return nil
}

func loadPublicLanes(path string) (map[string]struct{}, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("read public lane manifest dos.toml: %w", err)
	}
	defer f.Close()
	lanes := map[string]struct{}{}
	section := ""
	arrayKey := ""
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		raw := strings.SplitN(scanner.Text(), "#", 2)[0]
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if match := tomlSectionRE.FindStringSubmatch(line); match != nil {
			section, arrayKey = match[1], ""
			continue
		}
		if section == "lanes.trees" {
			if match := tomlKeyRE.FindStringSubmatch(line); match != nil {
				lanes[strings.ToLower(match[1])] = struct{}{}
			}
			continue
		}
		if section != "lanes" {
			continue
		}
		if match := tomlKeyRE.FindStringSubmatch(line); match != nil {
			switch match[1] {
			case "concurrent", "exclusive", "autopick":
				arrayKey = match[1]
			default:
				arrayKey = ""
			}
		}
		if arrayKey != "" {
			for _, match := range tomlQuotedRE.FindAllStringSubmatch(line, -1) {
				lanes[strings.ToLower(match[1])] = struct{}{}
			}
			if strings.Contains(line, "]") {
				arrayKey = ""
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("parse public lane manifest dos.toml: %w", err)
	}
	if len(lanes) == 0 {
		return nil, fmt.Errorf("public lane manifest dos.toml declares no lanes")
	}
	return lanes, nil
}

func validOwnerName(s string) bool { return s != "" && !strings.HasPrefix(s, ".") }
func fileExists(path string) bool  { info, err := os.Stat(path); return err == nil && !info.IsDir() }
func dirExists(path string) bool   { info, err := os.Stat(path); return err == nil && info.IsDir() }
func stringSet(values []string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, value := range values {
		out[strings.ToLower(strings.TrimSpace(value))] = struct{}{}
	}
	return out
}
func sortedKeys(values map[string]struct{}) []string {
	out := make([]string, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}
