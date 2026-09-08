package astquery

// StructuralSearch searches Go source code against a structural pattern containing
// metavariables (such as $VAR or wildcard $_). It returns matching AST node positions
// (line numbers), matching source text, and metavariable bindings (e.g. {"$ERR": "err"}),
// while ignoring comments and string literals.
func StructuralSearch(src, pattern string) (*ToolResult, error) {
	return SearchToolSource(src, pattern)
}

// SearchPattern matches a structural pattern across Go source code and returns
// the slice of matching ToolMatch items.
func SearchPattern(src, pattern string) ([]ToolMatch, error) {
	res, err := SearchToolSource(src, pattern)
	if err != nil {
		return nil, err
	}
	return res.Matches, nil
}

// SearchWorkspace executes an in-kernel structural AST search across files in the
// workspace or designated paths.
func SearchWorkspace(workspace, pattern string, paths []string, maxMatches int) (*ToolResult, error) {
	return SearchTool(workspace, pattern, paths, maxMatches)
}
