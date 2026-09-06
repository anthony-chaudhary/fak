package corelocks

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Coordination status values.
const (
	// StatusConfigured indicates that a valid [coordination] table was declared in dos.toml.
	StatusConfigured = "configured"

	// StatusLegacyUnconfigured indicates that dos.toml exists but carries no
	// [coordination] table, representing a legacy or unconfigured workspace.
	StatusLegacyUnconfigured = "legacy/unconfigured"
)

// ProvenanceKind classifies the workspace root provenance.
type ProvenanceKind string

const (
	// ProvenanceGit indicates an ordinary git repository root.
	ProvenanceGit ProvenanceKind = "git"

	// ProvenanceGitWorktree indicates a linked git worktree.
	ProvenanceGitWorktree ProvenanceKind = "git-worktree"

	// ProvenanceNonGit indicates a workspace outside any git repository.
	ProvenanceNonGit ProvenanceKind = "non-git"
)

// RootProvenance records the filesystem and repository origin of the resolved workspace.
type RootProvenance struct {
	// Root is the absolute path to the workspace root directory containing dos.toml.
	Root string `json:"root"`

	// GitTop is the git toplevel directory, or "" for non-git workspaces.
	GitTop string `json:"git_top,omitempty"`

	// CommonDir is the git common directory (where refs/objects live), or "" for non-git workspaces.
	// For linked worktrees, this points to the main repository's .git directory.
	CommonDir string `json:"common_dir,omitempty"`

	// Kind classifies the root provenance: "git", "git-worktree", or "non-git".
	Kind ProvenanceKind `json:"kind"`
}

// String returns a human-readable representation of root provenance.
func (p RootProvenance) String() string {
	switch p.Kind {
	case ProvenanceGitWorktree:
		return fmt.Sprintf("git-worktree(root=%s, common=%s)", p.Root, p.CommonDir)
	case ProvenanceGit:
		return fmt.Sprintf("git(root=%s)", p.Root)
	default:
		return fmt.Sprintf("non-git(root=%s)", p.Root)
	}
}

// CoordinationAuthority is the resolved coordination authority descriptor.
type CoordinationAuthority struct {
	// WorkspaceID identifies the workspace across coordination peers.
	WorkspaceID string `json:"workspace_id"`

	// AuthorityLocator is the URL, address, or locator of the coordination authority.
	// In legacy/unconfigured workspaces, this is empty (never an invented grant).
	AuthorityLocator string `json:"authority_locator"`

	// ConfigSource is the absolute path to the dos.toml file from which configuration was resolved.
	ConfigSource string `json:"config_source"`

	// RootProvenance details the filesystem and git origin of the workspace root.
	RootProvenance RootProvenance `json:"root_provenance"`

	// Status indicates whether coordination is "configured" or "legacy/unconfigured".
	Status string `json:"status"`

	// Configured reports whether a valid [coordination] table was declared.
	Configured bool `json:"configured"`

	// Version is the declared coordination schema version (1 for current declarations).
	Version int `json:"version,omitempty"`
}

// IsConfigured reports whether coordination is affirmatively configured.
func (a *CoordinationAuthority) IsConfigured() bool {
	return a != nil && a.Configured && a.Status == StatusConfigured
}

// IsLegacy reports whether the workspace is legacy/unconfigured.
func (a *CoordinationAuthority) IsLegacy() bool {
	return a == nil || !a.Configured || a.Status == StatusLegacyUnconfigured
}

var (
	// ErrMalformedDeclaration indicates syntax errors, missing required fields, or invalid values in dos.toml.
	ErrMalformedDeclaration = errors.New("corelocks: malformed coordination declaration")

	// ErrConflictingDeclaration indicates conflicting or duplicate coordination declarations.
	ErrConflictingDeclaration = errors.New("corelocks: conflicting coordination declaration")

	// ErrUnsupportedVersion indicates an unrecognized or unsupported coordination table version.
	ErrUnsupportedVersion = errors.New("corelocks: unsupported coordination version")

	// ErrWorkspaceNotFound indicates no dos.toml could be located by upward walk.
	ErrWorkspaceNotFound = errors.New("corelocks: dos.toml not found")

	// ErrMissingWorkspaceID indicates a non-git workspace lacks an explicit workspace_id.
	ErrMissingWorkspaceID = errors.New("corelocks: non-git workspace requires explicit workspace_id")

	// ErrLegacyUnconfigured is a sentinel for legacy/unconfigured workspaces.
	ErrLegacyUnconfigured = errors.New("corelocks: coordination authority unconfigured (legacy workspace)")
)

// MalformedDeclarationError provides typed context for malformed coordination declarations.
type MalformedDeclarationError struct {
	Path   string
	Reason string
}

func (e *MalformedDeclarationError) Error() string {
	if e.Path != "" {
		return fmt.Sprintf("%s: %s: %s", ErrMalformedDeclaration.Error(), e.Path, e.Reason)
	}
	return fmt.Sprintf("%s: %s", ErrMalformedDeclaration.Error(), e.Reason)
}

func (e *MalformedDeclarationError) Unwrap() error {
	return ErrMalformedDeclaration
}

// ConflictingDeclarationError provides typed context for conflicting coordination declarations.
type ConflictingDeclarationError struct {
	Path   string
	Reason string
}

func (e *ConflictingDeclarationError) Error() string {
	if e.Path != "" {
		return fmt.Sprintf("%s: %s: %s", ErrConflictingDeclaration.Error(), e.Path, e.Reason)
	}
	return fmt.Sprintf("%s: %s", ErrConflictingDeclaration.Error(), e.Reason)
}

func (e *ConflictingDeclarationError) Unwrap() error {
	return ErrConflictingDeclaration
}

// UnsupportedVersionError provides typed context when a coordination table specifies an unsupported version.
type UnsupportedVersionError struct {
	Version int
	Raw     string
	Path    string
}

func (e *UnsupportedVersionError) Error() string {
	if e.Path != "" {
		return fmt.Sprintf("%s: version %s in %s", ErrUnsupportedVersion.Error(), e.Raw, e.Path)
	}
	return fmt.Sprintf("%s: version %s", ErrUnsupportedVersion.Error(), e.Raw)
}

func (e *UnsupportedVersionError) Unwrap() []error {
	return []error{ErrUnsupportedVersion, ErrMalformedDeclaration}
}

// MissingWorkspaceIDError provides typed context when a non-git workspace lacks an explicit workspace_id.
type MissingWorkspaceIDError struct {
	Path string
}

func (e *MissingWorkspaceIDError) Error() string {
	if e.Path != "" {
		return fmt.Sprintf("%s in %s", ErrMissingWorkspaceID.Error(), e.Path)
	}
	return ErrMissingWorkspaceID.Error()
}

func (e *MissingWorkspaceIDError) Unwrap() []error {
	return []error{ErrMissingWorkspaceID, ErrMalformedDeclaration}
}

// ResolveCoordinationAuthority walks upward from start looking for dos.toml,
// and resolves the coordination authority descriptor and root provenance.
// If start is empty, the current working directory is used.
func ResolveCoordinationAuthority(start string) (*CoordinationAuthority, error) {
	return ResolveCoordinationAuthorityWithin(start, "")
}

// ResolveAuthority is an alias for ResolveCoordinationAuthority.
func ResolveAuthority(start string) (*CoordinationAuthority, error) {
	return ResolveCoordinationAuthority(start)
}

// ResolveCoordination is an alias for ResolveCoordinationAuthority.
func ResolveCoordination(start string) (*CoordinationAuthority, error) {
	return ResolveCoordinationAuthority(start)
}

// ResolveCoordinationAuthorityWithin is ResolveCoordinationAuthority bounded by ceiling.
// The upward walk stops after examining ceiling.
func ResolveCoordinationAuthorityWithin(start, ceiling string) (*CoordinationAuthority, error) {
	workspaceRoot, err := findWorkspaceRoot(start, ceiling)
	if err != nil {
		return nil, err
	}
	tomlPath := filepath.Join(workspaceRoot, "dos.toml")
	data, err := os.ReadFile(tomlPath)
	if err != nil {
		return nil, err
	}
	prov := inspectGitProvenance(workspaceRoot)
	prov.Root = workspaceRoot
	return parseCoordinationTable(data, tomlPath, prov)
}

// ResolveCoordinationAuthorityFile reads and resolves coordination configuration
// directly from the specified dos.toml path.
func ResolveCoordinationAuthorityFile(path string) (*CoordinationAuthority, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return nil, err
	}
	workspaceRoot := filepath.Dir(abs)
	prov := inspectGitProvenance(workspaceRoot)
	prov.Root = workspaceRoot
	return parseCoordinationTable(data, abs, prov)
}

// ResolveCoordinationAuthorityBytes parses coordination configuration from byte content
// against an indicated workspace root.
func ResolveCoordinationAuthorityBytes(data []byte, workspaceRoot string) (*CoordinationAuthority, error) {
	if strings.TrimSpace(workspaceRoot) == "" {
		workspaceRoot = "."
	}
	abs, err := filepath.Abs(workspaceRoot)
	if err != nil {
		abs = workspaceRoot
	}
	prov := inspectGitProvenance(abs)
	prov.Root = abs
	configSource := filepath.Join(abs, "dos.toml")
	return parseCoordinationTable(data, configSource, prov)
}

// findWorkspaceRoot walks upward from start looking for dos.toml, halting at ceiling.
func findWorkspaceRoot(start, ceiling string) (string, error) {
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
	if fi, statErr := os.Stat(abs); statErr == nil && !fi.IsDir() {
		abs = filepath.Dir(abs)
	}
	stop := ""
	if ceiling != "" {
		if stopAbs, err := filepath.Abs(ceiling); err == nil {
			stop = stopAbs
		}
	}

	dir := abs
	for {
		tomlPath := filepath.Join(dir, "dos.toml")
		if fi, err := os.Stat(tomlPath); err == nil && !fi.IsDir() {
			return dir, nil
		}
		if stop != "" && sameDir(dir, stop) {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	// Check if start is inside a linked git worktree whose main checkout has dos.toml.
	if prov := inspectGitProvenance(abs); prov.Kind == ProvenanceGitWorktree && prov.CommonDir != "" {
		mainRepo := filepath.Dir(prov.CommonDir)
		tomlPath := filepath.Join(mainRepo, "dos.toml")
		if fi, err := os.Stat(tomlPath); err == nil && !fi.IsDir() {
			return mainRepo, nil
		}
	}

	return "", fmt.Errorf("%w: dos.toml not found at or above %s", ErrWorkspaceNotFound, start)
}

// inspectGitProvenance determines whether dir is part of a git checkout, linked worktree, or non-git tree.
func inspectGitProvenance(dir string) RootProvenance {
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = filepath.Clean(dir)
	}
	if fi, err := os.Stat(abs); err == nil && !fi.IsDir() {
		abs = filepath.Dir(abs)
	}

	curr := abs
	for {
		dotGit := filepath.Join(curr, ".git")
		fi, err := os.Stat(dotGit)
		if err == nil {
			if fi.IsDir() {
				return RootProvenance{
					Root:      curr,
					GitTop:    curr,
					CommonDir: dotGit,
					Kind:      ProvenanceGit,
				}
			}
			// Linked worktree .git pointer file
			b, readErr := os.ReadFile(dotGit)
			if readErr == nil {
				line := strings.TrimSpace(string(b))
				if rest, ok := strings.CutPrefix(line, "gitdir:"); ok {
					gitDir := filepath.FromSlash(strings.TrimSpace(rest))
					if !filepath.IsAbs(gitDir) {
						gitDir = filepath.Join(curr, gitDir)
					}
					gitDir = filepath.Clean(gitDir)
					commondir := gitDir
					if cb, err := os.ReadFile(filepath.Join(gitDir, "commondir")); err == nil {
						cdLine := filepath.FromSlash(strings.TrimSpace(string(cb)))
						if cdLine != "" {
							if !filepath.IsAbs(cdLine) {
								commondir = filepath.Clean(filepath.Join(gitDir, cdLine))
							} else {
								commondir = filepath.Clean(cdLine)
							}
						}
					}
					return RootProvenance{
						Root:      curr,
						GitTop:    curr,
						CommonDir: commondir,
						Kind:      ProvenanceGitWorktree,
					}
				}
			}
		}
		parent := filepath.Dir(curr)
		if parent == curr {
			break
		}
		curr = parent
	}

	return RootProvenance{
		Root: abs,
		Kind: ProvenanceNonGit,
	}
}

func derivedGitWorkspaceID(prov RootProvenance) string {
	if prov.CommonDir != "" {
		mainRepo := filepath.Dir(prov.CommonDir)
		return filepath.Base(mainRepo)
	}
	if prov.Root != "" {
		return filepath.Base(prov.Root)
	}
	return ""
}

type parsedCoordination struct {
	found         bool
	version       int
	versionRaw    string
	hasVersion    bool
	schemaVersion int
	hasSchemaVer  bool
	authority     string
	authorityKey  string
	workspaceID   string
	workspaceKey  string
	seenKeys      map[string]bool
}

func parseCoordinationTable(data []byte, path string, prov RootProvenance) (*CoordinationAuthority, error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	inCoord := false
	coordCount := 0
	parsed := parsedCoordination{seenKeys: make(map[string]bool)}
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		raw := scanner.Text()
		trimmed := strings.TrimSpace(raw)
		if trimmed == "" {
			continue
		}
		if strings.HasPrefix(trimmed, "#") {
			continue
		}

		if trimmed == "[[coordination]]" || strings.HasPrefix(trimmed, "[[coordination") {
			return nil, &MalformedDeclarationError{
				Path:   path,
				Reason: fmt.Sprintf("line %d: [[coordination]] array of tables is not permitted; use [coordination]", lineNum),
			}
		}

		if strings.HasPrefix(trimmed, "[") && !strings.HasPrefix(trimmed, "[[") {
			if !strings.HasSuffix(trimmed, "]") {
				if inCoord || strings.Contains(trimmed, "coordination") {
					return nil, &MalformedDeclarationError{
						Path:   path,
						Reason: fmt.Sprintf("line %d: malformed section header %q", lineNum, trimmed),
					}
				}
				inCoord = false
				continue
			}
			tableName := strings.TrimSpace(strings.Trim(trimmed, "[]"))
			if tableName == "coordination" {
				coordCount++
				if coordCount > 1 {
					return nil, &ConflictingDeclarationError{
						Path:   path,
						Reason: fmt.Sprintf("line %d: duplicate [coordination] table declaration", lineNum),
					}
				}
				inCoord = true
				parsed.found = true
				continue
			} else {
				inCoord = false
				continue
			}
		}

		if !inCoord {
			continue
		}

		lineWithoutComment, unclosedQuote := stripCommentAndCheck(raw)
		if unclosedQuote {
			return nil, &MalformedDeclarationError{
				Path:   path,
				Reason: fmt.Sprintf("line %d: unterminated string literal", lineNum),
			}
		}
		line := strings.TrimSpace(lineWithoutComment)
		if line == "" {
			continue
		}

		eq := strings.Index(line, "=")
		if eq < 0 {
			return nil, &MalformedDeclarationError{
				Path:   path,
				Reason: fmt.Sprintf("line %d: malformed key-value declaration %q", lineNum, line),
			}
		}
		key := strings.TrimSpace(line[:eq])
		val := strings.TrimSpace(line[eq+1:])
		if key == "" {
			return nil, &MalformedDeclarationError{
				Path:   path,
				Reason: fmt.Sprintf("line %d: empty key in [coordination]", lineNum),
			}
		}
		if val == "" {
			return nil, &MalformedDeclarationError{
				Path:   path,
				Reason: fmt.Sprintf("line %d: empty value for key %q in [coordination]", lineNum, key),
			}
		}

		if parsed.seenKeys[key] {
			return nil, &ConflictingDeclarationError{
				Path:   path,
				Reason: fmt.Sprintf("line %d: duplicate key %q in [coordination] table", lineNum, key),
			}
		}
		parsed.seenKeys[key] = true

		switch key {
		case "version":
			v, rawV, ok := parseVersion(val)
			if !ok {
				return nil, &MalformedDeclarationError{
					Path:   path,
					Reason: fmt.Sprintf("line %d: invalid version %q in [coordination]", lineNum, val),
				}
			}
			parsed.version = v
			parsed.versionRaw = rawV
			parsed.hasVersion = true

		case "schema_version":
			v, rawV, ok := parseVersion(val)
			if !ok {
				return nil, &MalformedDeclarationError{
					Path:   path,
					Reason: fmt.Sprintf("line %d: invalid schema_version %q in [coordination]", lineNum, val),
				}
			}
			parsed.schemaVersion = v
			if !parsed.hasVersion {
				parsed.version = v
				parsed.versionRaw = rawV
			}
			parsed.hasSchemaVer = true

		case "authority_locator", "authority", "locator":
			s, ok := parseString(val)
			if !ok {
				return nil, &MalformedDeclarationError{
					Path:   path,
					Reason: fmt.Sprintf("line %d: %s must be a quoted string", lineNum, key),
				}
			}
			s = strings.TrimSpace(s)
			if s == "" {
				return nil, &MalformedDeclarationError{
					Path:   path,
					Reason: fmt.Sprintf("line %d: %s cannot be empty", lineNum, key),
				}
			}
			if parsed.authority != "" && parsed.authority != s {
				return nil, &ConflictingDeclarationError{
					Path:   path,
					Reason: fmt.Sprintf("line %d: conflicting authority declarations: %s=%q vs %s=%q", lineNum, parsed.authorityKey, parsed.authority, key, s),
				}
			}
			parsed.authority = s
			parsed.authorityKey = key

		case "workspace_id", "workspace", "id":
			s, ok := parseString(val)
			if !ok {
				return nil, &MalformedDeclarationError{
					Path:   path,
					Reason: fmt.Sprintf("line %d: %s must be a quoted string", lineNum, key),
				}
			}
			s = strings.TrimSpace(s)
			if s == "" {
				return nil, &MalformedDeclarationError{
					Path:   path,
					Reason: fmt.Sprintf("line %d: %s cannot be empty", lineNum, key),
				}
			}
			if parsed.workspaceID != "" && parsed.workspaceID != s {
				return nil, &ConflictingDeclarationError{
					Path:   path,
					Reason: fmt.Sprintf("line %d: conflicting workspace declarations: %s=%q vs %s=%q", lineNum, parsed.workspaceKey, parsed.workspaceID, key, s),
				}
			}
			parsed.workspaceID = s
			parsed.workspaceKey = key

		default:
			return nil, &MalformedDeclarationError{
				Path:   path,
				Reason: fmt.Sprintf("line %d: unknown key %q in [coordination] table", lineNum, key),
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, &MalformedDeclarationError{
			Path:   path,
			Reason: fmt.Sprintf("reading configuration: %v", err),
		}
	}

	if !parsed.found {
		// Missing [coordination] table: legacy/unconfigured workspace.
		return &CoordinationAuthority{
			WorkspaceID:      derivedGitWorkspaceID(prov),
			AuthorityLocator: "", // never an invented grant
			ConfigSource:     path,
			RootProvenance:   prov,
			Status:           StatusLegacyUnconfigured,
			Configured:       false,
		}, nil
	}

	// [coordination] was declared; validate required fields.
	if !parsed.hasVersion && !parsed.hasSchemaVer {
		return nil, &MalformedDeclarationError{
			Path:   path,
			Reason: "missing required 'version' in [coordination] table",
		}
	}
	if parsed.hasVersion && parsed.hasSchemaVer && parsed.version != parsed.schemaVersion {
		return nil, &ConflictingDeclarationError{
			Path:   path,
			Reason: fmt.Sprintf("conflicting version (%d) and schema_version (%d)", parsed.version, parsed.schemaVersion),
		}
	}
	if parsed.version != 1 {
		return nil, &UnsupportedVersionError{
			Version: parsed.version,
			Raw:     parsed.versionRaw,
			Path:    path,
		}
	}

	if parsed.authority == "" {
		return nil, &MalformedDeclarationError{
			Path:   path,
			Reason: "missing required 'authority_locator' in [coordination] table",
		}
	}

	wsID := parsed.workspaceID
	if wsID == "" {
		if prov.Kind == ProvenanceNonGit {
			return nil, &MissingWorkspaceIDError{Path: path}
		}
		wsID = derivedGitWorkspaceID(prov)
	}

	return &CoordinationAuthority{
		WorkspaceID:      wsID,
		AuthorityLocator: parsed.authority,
		ConfigSource:     path,
		RootProvenance:   prov,
		Status:           StatusConfigured,
		Configured:       true,
		Version:          parsed.version,
	}, nil
}

func stripCommentAndCheck(s string) (string, bool) {
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
			return s[:i], false
		}
	}
	return s, quote != 0
}

func parseVersion(s string) (int, string, bool) {
	s = strings.TrimSpace(s)
	if str, ok := parseString(s); ok {
		n, err := strconv.Atoi(str)
		if err != nil {
			return 0, str, false
		}
		return n, str, true
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, s, false
	}
	return n, strconv.Itoa(n), true
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
