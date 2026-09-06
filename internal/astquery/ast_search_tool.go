package astquery

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// ToolName is the registered tool name for the in-kernel structural AST search tool.
const ToolName = "fak_ast_search"

// ToolMatch represents a single structural AST match found by SearchTool.
type ToolMatch struct {
	File        string            `json:"file"`
	Line        int               `json:"line"`
	Column      int               `json:"column"`
	Offset      int               `json:"offset"`
	EndLine     int               `json:"end_line,omitempty"`
	EndColumn   int               `json:"end_column,omitempty"`
	Pos         token.Position    `json:"pos"`
	EndPos      token.Position    `json:"end_pos,omitempty"`
	SourceLine  string            `json:"source_line"`
	SourceLines []string          `json:"source_lines,omitempty"`
	Text        string            `json:"text"`
	Bindings    map[string]string `json:"bindings"`
}

// ToolResult represents the overall results of SearchTool.
type ToolResult struct {
	Workspace string      `json:"workspace,omitempty"`
	Pattern   string      `json:"pattern"`
	Matches   []ToolMatch `json:"matches"`
	Count     int         `json:"count"`
	Truncated bool        `json:"truncated,omitempty"`
}

// String returns a human-readable representation of the tool result.
func (r *ToolResult) String() string {
	if r == nil || len(r.Matches) == 0 {
		return "(no matches)"
	}
	var sb strings.Builder
	for i, m := range r.Matches {
		if i > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(fmt.Sprintf("%s:%d:%d: %s", m.File, m.Line, m.Column, m.Text))
	}
	if r.Truncated {
		sb.WriteString("\n... (max matches reached)")
	}
	return sb.String()
}

// SearchToolParams represents the input parameters for fak_ast_search tool execution.
type SearchToolParams struct {
	Workspace  string   `json:"workspace,omitempty"`
	Pattern    string   `json:"pattern"`
	Paths      []string `json:"paths,omitempty"`
	MaxMatches int      `json:"max_matches,omitempty"`
}

// Execute executes the structural AST search using the provided parameters.
func (p SearchToolParams) Execute() (*ToolResult, error) {
	return SearchTool(p.Workspace, p.Pattern, p.Paths, p.MaxMatches)
}

// SearchTool executes an in-kernel structural AST search over Go files.
// It parses the Go pattern (supporting metavariables like $VAR and wildcard $_),
// searches Go source files across the workspace or specified paths, ignores
// comments and string literals that mention the code pattern, and returns exact
// token positions, source lines, and metavariable bindings.
func SearchTool(workspace string, pattern string, paths []string, maxMatches int) (*ToolResult, error) {
	trimmedPat := strings.TrimSpace(pattern)
	if trimmedPat == "" {
		return nil, fmt.Errorf("astquery: empty pattern")
	}

	pat, err := parsePattern(trimmedPat)
	if err != nil {
		return nil, err
	}

	if workspace == "" {
		workspace = "."
	}
	absWorkspace, err := filepath.Abs(workspace)
	if err != nil {
		absWorkspace = filepath.Clean(workspace)
	}

	result := &ToolResult{
		Workspace: workspace,
		Pattern:   pattern,
		Matches:   make([]ToolMatch, 0),
	}

	addMatches := func(matches []ToolMatch) bool {
		for _, m := range matches {
			result.Matches = append(result.Matches, m)
			if maxMatches > 0 && len(result.Matches) >= maxMatches {
				result.Truncated = true
				result.Count = len(result.Matches)
				return true
			}
		}
		return false
	}

	if len(paths) == 0 {
		info, statErr := os.Stat(absWorkspace)
		if statErr != nil {
			return nil, statErr
		}
		if !info.IsDir() {
			displayPath := filepath.Base(absWorkspace)
			fileMatches, fileErr := searchFileForTool(absWorkspace, displayPath, pat)
			if fileErr != nil {
				return nil, fileErr
			}
			addMatches(fileMatches)
			result.Count = len(result.Matches)
			return result, nil
		}

		err := filepath.WalkDir(absWorkspace, func(path string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			if d.IsDir() {
				name := d.Name()
				if path != absWorkspace {
					if strings.HasPrefix(name, ".") && name != "." {
						return filepath.SkipDir
					}
					if name == "vendor" {
						return filepath.SkipDir
					}
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") {
				return nil
			}

			displayPath := path
			if rel, relErr := filepath.Rel(absWorkspace, path); relErr == nil && !strings.HasPrefix(rel, "..") {
				displayPath = filepath.ToSlash(rel)
			}

			fileMatches, fileErr := searchFileForTool(path, displayPath, pat)
			if fileErr != nil {
				return nil
			}
			if addMatches(fileMatches) {
				return fs.SkipAll
			}
			return nil
		})
		if err != nil && err != fs.SkipAll {
			return nil, err
		}
		result.Count = len(result.Matches)
		return result, nil
	}

	visited := make(map[string]bool)
	for _, p := range paths {
		var candidates []string
		fullPath := p
		if !filepath.IsAbs(fullPath) {
			fullPath = filepath.Join(absWorkspace, p)
		}
		fullPath = filepath.Clean(fullPath)
		rel, err := filepath.Rel(absWorkspace, fullPath)
		if err != nil || strings.HasPrefix(rel, "..") || rel == ".." {
			return nil, fmt.Errorf("astquery: path outside workspace: %s", p)
		}

		if strings.ContainsAny(p, "*?[") {
			matches, globErr := filepath.Glob(fullPath)
			if globErr == nil && len(matches) > 0 {
				candidates = matches
			} else {
				// Unmatched glob returns zero candidates without hard error
				continue
			}
		} else {
			candidates = []string{fullPath}
		}

		for _, candidate := range candidates {
			info, statErr := os.Stat(candidate)
			if statErr != nil {
				return nil, statErr
			}
			if info.IsDir() {
				err := filepath.WalkDir(candidate, func(path string, d fs.DirEntry, walkErr error) error {
					if walkErr != nil {
						return nil
					}
					if d.IsDir() {
						name := d.Name()
						if path != candidate {
							if strings.HasPrefix(name, ".") && name != "." {
								return filepath.SkipDir
							}
							if name == "vendor" {
								return filepath.SkipDir
							}
						}
						return nil
					}
					if !strings.HasSuffix(path, ".go") {
						return nil
					}
					cleanPath := filepath.Clean(path)
					if visited[cleanPath] {
						return nil
					}
					visited[cleanPath] = true

					displayPath := path
					if rel, relErr := filepath.Rel(absWorkspace, path); relErr == nil && !strings.HasPrefix(rel, "..") {
						displayPath = filepath.ToSlash(rel)
					} else if rel, relErr := filepath.Rel(workspace, path); relErr == nil && !strings.HasPrefix(rel, "..") {
						displayPath = filepath.ToSlash(rel)
					}

					fileMatches, fileErr := searchFileForTool(path, displayPath, pat)
					if fileErr != nil {
						return nil
					}
					if addMatches(fileMatches) {
						return fs.SkipAll
					}
					return nil
				})
				if err != nil && err != fs.SkipAll {
					return nil, err
				}
				if result.Truncated {
					result.Count = len(result.Matches)
					return result, nil
				}
			} else {
				if !strings.HasSuffix(candidate, ".go") {
					continue
				}
				cleanPath := filepath.Clean(candidate)
				if visited[cleanPath] {
					continue
				}
				visited[cleanPath] = true

				displayPath := candidate
				if rel, relErr := filepath.Rel(absWorkspace, candidate); relErr == nil && !strings.HasPrefix(rel, "..") {
					displayPath = filepath.ToSlash(rel)
				} else if rel, relErr := filepath.Rel(workspace, candidate); relErr == nil && !strings.HasPrefix(rel, "..") {
					displayPath = filepath.ToSlash(rel)
				} else {
					displayPath = filepath.ToSlash(p)
				}

				fileMatches, fileErr := searchFileForTool(candidate, displayPath, pat)
				if fileErr != nil {
					return nil, fileErr
				}
				if addMatches(fileMatches) {
					result.Count = len(result.Matches)
					return result, nil
				}
			}
		}
	}

	result.Count = len(result.Matches)
	return result, nil
}

// SearchToolSource runs the structural AST search against an in-memory Go source string.
func SearchToolSource(src, pattern string) (*ToolResult, error) {
	trimmedPat := strings.TrimSpace(pattern)
	if trimmedPat == "" {
		return nil, fmt.Errorf("astquery: empty pattern")
	}
	pat, err := parsePattern(trimmedPat)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "src.go", src, 0)
	if err != nil {
		return nil, fmt.Errorf("astquery: bad source: %w", err)
	}

	matches := searchNode(file, pat, fset)
	lines := strings.Split(src, "\n")
	result := &ToolResult{
		Pattern: pattern,
		Matches: make([]ToolMatch, 0, len(matches)),
		Count:   len(matches),
	}

	for _, m := range matches {
		var srcLine string
		var srcLines []string
		lineIdx := m.Pos.Line - 1
		endLineIdx := m.EndPos.Line - 1
		if endLineIdx < lineIdx {
			endLineIdx = lineIdx
		}
		if lineIdx >= 0 && lineIdx < len(lines) {
			srcLine = strings.TrimRight(lines[lineIdx], "\r")
			if endLineIdx >= len(lines) {
				endLineIdx = len(lines) - 1
			}
			srcLines = make([]string, 0, endLineIdx-lineIdx+1)
			for i := lineIdx; i <= endLineIdx; i++ {
				srcLines = append(srcLines, strings.TrimRight(lines[i], "\r"))
			}
		}

		bindsCopy := make(map[string]string, len(m.Bindings))
		for k, v := range m.Bindings {
			bindsCopy[k] = v
		}

		result.Matches = append(result.Matches, ToolMatch{
			File:        "src.go",
			Line:        m.Pos.Line,
			Column:      m.Pos.Column,
			Offset:      m.Pos.Offset,
			EndLine:     m.EndPos.Line,
			EndColumn:   m.EndPos.Column,
			Pos:         m.Pos,
			EndPos:      m.EndPos,
			SourceLine:  srcLine,
			SourceLines: srcLines,
			Text:        m.Text,
			Bindings:    bindsCopy,
		})
	}

	return result, nil
}

func searchFileForTool(filePath, displayPath string, pat ast.Node) ([]ToolMatch, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filePath, data, 0)
	if err != nil {
		return nil, fmt.Errorf("astquery: bad source %s: %w", filePath, err)
	}

	matches := searchNode(file, pat, fset)
	if len(matches) == 0 {
		return nil, nil
	}

	lines := strings.Split(string(data), "\n")
	toolMatches := make([]ToolMatch, 0, len(matches))

	for _, m := range matches {
		var srcLine string
		var srcLines []string
		lineIdx := m.Pos.Line - 1
		endLineIdx := m.EndPos.Line - 1
		if endLineIdx < lineIdx {
			endLineIdx = lineIdx
		}
		if lineIdx >= 0 && lineIdx < len(lines) {
			srcLine = strings.TrimRight(lines[lineIdx], "\r")
			if endLineIdx >= len(lines) {
				endLineIdx = len(lines) - 1
			}
			srcLines = make([]string, 0, endLineIdx-lineIdx+1)
			for i := lineIdx; i <= endLineIdx; i++ {
				srcLines = append(srcLines, strings.TrimRight(lines[i], "\r"))
			}
		}

		bindsCopy := make(map[string]string, len(m.Bindings))
		for k, v := range m.Bindings {
			bindsCopy[k] = v
		}

		disp := displayPath
		if disp == "" {
			disp = filepath.ToSlash(filePath)
		}

		toolMatches = append(toolMatches, ToolMatch{
			File:        disp,
			Line:        m.Pos.Line,
			Column:      m.Pos.Column,
			Offset:      m.Pos.Offset,
			EndLine:     m.EndPos.Line,
			EndColumn:   m.EndPos.Column,
			Pos:         m.Pos,
			EndPos:      m.EndPos,
			SourceLine:  srcLine,
			SourceLines: srcLines,
			Text:        m.Text,
			Bindings:    bindsCopy,
		})
	}

	return toolMatches, nil
}
