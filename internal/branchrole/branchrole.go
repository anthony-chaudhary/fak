// Package branchrole reads fak's branch-role contract from dos.toml.
package branchrole

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const defaultBranch = "main"

// Roles names the long-lived branch roles used during the dev/main migration.
type Roles struct {
	DevelopmentBranch string
	ReleaseBranch     string
	ReleaseSource     string
	PublicFrontDoor   string
}

// Defaults is the current no-cutover branch regime.
func Defaults() Roles {
	return Roles{
		DevelopmentBranch: defaultBranch,
		ReleaseBranch:     defaultBranch,
		ReleaseSource:     defaultBranch,
		PublicFrontDoor:   defaultBranch,
	}
}

// Load reads [branch_roles] from dos.toml under root. If root is empty or points
// inside the repo, Load walks upward until it finds dos.toml. Returned Roles are
// always populated with Defaults; err reports why the config could not be read.
func Load(root string) (Roles, error) {
	dir, err := FindRoot(root)
	if err != nil {
		return Defaults(), err
	}
	return LoadFile(filepath.Join(dir, "dos.toml"))
}

// FindRoot walks upward from start until it finds dos.toml.
func FindRoot(start string) (string, error) {
	if strings.TrimSpace(start) == "" {
		wd, err := os.Getwd()
		if err != nil {
			return "", err
		}
		start = wd
	}
	abs, err := filepath.Abs(start)
	if err != nil {
		return "", err
	}
	if info, statErr := os.Stat(abs); statErr == nil && !info.IsDir() {
		abs = filepath.Dir(abs)
	}
	for {
		if _, err := os.Stat(filepath.Join(abs, "dos.toml")); err == nil {
			return abs, nil
		}
		parent := filepath.Dir(abs)
		if parent == abs {
			return "", fmt.Errorf("branchrole: dos.toml not found from %s", start)
		}
		abs = parent
	}
}

// LoadFile reads [branch_roles] from a specific dos.toml path.
func LoadFile(path string) (Roles, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Defaults(), err
	}
	roles, _, err := parse(b)
	return roles, err
}

func parse(b []byte) (Roles, bool, error) {
	roles := Defaults()
	found := false
	inRoles := false
	seen := map[string]bool{}
	scanner := bufio.NewScanner(bytes.NewReader(b))
	for scanner.Scan() {
		line := strings.TrimSpace(stripComment(scanner.Text()))
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			inRoles = strings.TrimSpace(strings.Trim(line, "[]")) == "branch_roles"
			if inRoles {
				found = true
			}
			continue
		}
		if !inRoles {
			continue
		}
		eq := strings.Index(line, "=")
		if eq < 0 {
			return roles, found, fmt.Errorf("branchrole: malformed branch_roles line %q", line)
		}
		key := strings.TrimSpace(line[:eq])
		value, ok := parseString(strings.TrimSpace(line[eq+1:]))
		if !ok {
			return roles, found, fmt.Errorf("branchrole: %s must be a quoted string", key)
		}
		if isKnownRoleKey(key) {
			if seen[key] {
				return roles, found, fmt.Errorf("branchrole: duplicate %s", key)
			}
			seen[key] = true
		}
		value = strings.TrimSpace(value)
		if value == "" {
			return roles, found, fmt.Errorf("branchrole: %s cannot be empty", key)
		}
		if isKnownRoleKey(key) && !validBranchName(value) {
			return roles, found, fmt.Errorf("branchrole: %s has invalid branch name %q", key, value)
		}
		switch key {
		case "development_branch":
			roles.DevelopmentBranch = value
		case "release_branch":
			roles.ReleaseBranch = value
		case "release_source":
			roles.ReleaseSource = value
		case "public_front_door":
			roles.PublicFrontDoor = value
		}
	}
	if err := scanner.Err(); err != nil {
		return roles, found, err
	}
	return roles, found, nil
}

func isKnownRoleKey(key string) bool {
	switch key {
	case "development_branch", "release_branch", "release_source", "public_front_door":
		return true
	default:
		return false
	}
}

func validBranchName(name string) bool {
	if name == "" || strings.TrimSpace(name) != name {
		return false
	}
	if strings.HasPrefix(name, "-") || strings.HasPrefix(name, "/") || strings.HasSuffix(name, "/") {
		return false
	}
	if strings.Contains(name, "..") || strings.Contains(name, "//") || strings.Contains(name, "@{") {
		return false
	}
	if strings.HasSuffix(name, ".lock") || name == "@" {
		return false
	}
	for _, part := range strings.Split(name, "/") {
		if part == "" || strings.HasPrefix(part, ".") || strings.HasSuffix(part, ".lock") || strings.HasSuffix(part, ".") {
			return false
		}
	}
	for _, r := range name {
		if r <= 0x20 || strings.ContainsRune(`~^:?*[\\`, r) {
			return false
		}
	}
	return true
}

func stripComment(s string) string {
	var quote byte
	escaped := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if escaped {
			escaped = false
			continue
		}
		if quote != 0 {
			if c == '\\' {
				escaped = true
				continue
			}
			if c == quote {
				quote = 0
			}
			continue
		}
		if c == '"' || c == '\'' {
			quote = c
			continue
		}
		if c == '#' {
			return s[:i]
		}
	}
	return s
}

func parseString(s string) (string, bool) {
	if len(s) < 2 {
		return "", false
	}
	switch s[0] {
	case '"':
		end := quotedEnd(s, '"')
		if end < 0 || strings.TrimSpace(s[end+1:]) != "" {
			return "", false
		}
		v, err := strconv.Unquote(s[:end+1])
		return v, err == nil
	case '\'':
		end := quotedEnd(s, '\'')
		if end < 0 || strings.TrimSpace(s[end+1:]) != "" {
			return "", false
		}
		return s[1:end], true
	default:
		return "", false
	}
}

func quotedEnd(s string, quote byte) int {
	escaped := false
	for i := 1; i < len(s); i++ {
		c := s[i]
		if escaped {
			escaped = false
			continue
		}
		if quote == '"' && c == '\\' {
			escaped = true
			continue
		}
		if c == quote {
			return i
		}
	}
	return -1
}
