package agentopt

import (
	"bufio"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"sort"
	"strings"
)

// Family 7: Retrieval & knowledge augmentation - syntax-aware code chunking for dense retrieval.

// SyntaxBoundary identifies the syntactic unit type of a code chunk.
type SyntaxBoundary string

const (
	// BoundaryPackage denotes a package declaration and file header unit.
	BoundaryPackage SyntaxBoundary = "package"
	// BoundaryImport denotes an import specification block.
	BoundaryImport SyntaxBoundary = "import"
	// BoundaryFunction denotes a standalone function definition.
	BoundaryFunction SyntaxBoundary = "function"
	// BoundaryMethod denotes a receiver method or class member function.
	BoundaryMethod SyntaxBoundary = "method"
	// BoundaryType denotes a type definition or alias.
	BoundaryType SyntaxBoundary = "type"
	// BoundaryStruct denotes a struct type definition.
	BoundaryStruct SyntaxBoundary = "struct"
	// BoundaryInterface denotes an interface specification.
	BoundaryInterface SyntaxBoundary = "interface"
	// BoundaryClass denotes a class definition or header.
	BoundaryClass SyntaxBoundary = "class"
	// BoundaryBlock denotes a block of declarations or statements.
	BoundaryBlock SyntaxBoundary = "block"
)

// CodeChunk represents a syntactic unit of source code with retrieval breadcrumbs.
type CodeChunk struct {
	FilePath       string         `json:"file_path"`
	PackageName    string         `json:"package_name"`
	EnclosingScope string         `json:"enclosing_scope"`
	StartLine      int            `json:"start_line"`
	EndLine        int            `json:"end_line"`
	Boundary       SyntaxBoundary `json:"boundary"`
	Content        string         `json:"content"`
}

// Breadcrumb returns a human-readable hierarchical navigation string.
func (c CodeChunk) Breadcrumb() string {
	var parts []string
	if c.FilePath != "" {
		parts = append(parts, c.FilePath)
	}
	if c.PackageName != "" {
		parts = append(parts, c.PackageName)
	}
	if c.EnclosingScope != "" {
		parts = append(parts, c.EnclosingScope)
	}
	return strings.Join(parts, " > ")
}

// SyntaxChunker partitions source files along syntactic boundaries for dense retrieval.
type SyntaxChunker struct {
	DefaultPackage string
}

// NewSyntaxChunker creates a new SyntaxChunker.
func NewSyntaxChunker() *SyntaxChunker {
	return &SyntaxChunker{}
}

// ChunkSource splits source code along natural syntactic units without splitting functions across chunks.
func (sc *SyntaxChunker) ChunkSource(filePath string, content string) []CodeChunk {
	if sc == nil {
		sc = &SyntaxChunker{}
	}
	if strings.TrimSpace(content) == "" {
		return nil
	}

	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".go":
		return sc.chunkGo(filePath, content)
	case ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs":
		return sc.chunkTSJS(filePath, content)
	default:
		return sc.chunkGeneric(filePath, content)
	}
}

// chunkGo partitions Go source code using the Go AST parser.
func (sc *SyntaxChunker) chunkGo(filePath string, content string) []CodeChunk {
	lines := strings.Split(content, "\n")
	totalLines := len(lines)

	fset := token.NewFileSet()
	fileNode, err := parser.ParseFile(fset, filePath, content, parser.ParseComments)
	if err != nil && fileNode == nil {
		return sc.chunkGeneric(filePath, content)
	}

	pkgName := ""
	if fileNode != nil && fileNode.Name != nil {
		pkgName = fileNode.Name.Name
	}
	if pkgName == "" {
		pkgName = sc.DefaultPackage
	}

	var chunks []CodeChunk

	// 1. Capture package declaration and imports header
	if fileNode != nil {
		headerStart := 1
		if fileNode.Doc != nil {
			headerStart = fset.Position(fileNode.Doc.Pos()).Line
		}
		headerEnd := fset.Position(fileNode.Name.End()).Line
		hasImports := false
		for _, decl := range fileNode.Decls {
			if gd, ok := decl.(*ast.GenDecl); ok && gd.Tok == token.IMPORT {
				hasImports = true
				impEnd := fset.Position(gd.End()).Line
				if impEnd > headerEnd {
					headerEnd = impEnd
				}
			}
		}

		if headerEnd >= headerStart && headerEnd <= totalLines {
			bKind := BoundaryPackage
			if hasImports {
				bKind = BoundaryImport
			}
			chunks = append(chunks, CodeChunk{
				FilePath:       filePath,
				PackageName:    pkgName,
				EnclosingScope: "",
				StartLine:      headerStart,
				EndLine:        headerEnd,
				Boundary:       bKind,
				Content:        strings.Join(lines[headerStart-1:headerEnd], "\n"),
			})
		}
	}

	// 2. Capture top-level declarations
	for _, decl := range fileNode.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			isMethod := d.Recv != nil && len(d.Recv.List) > 0
			scope := ""
			bKind := BoundaryFunction
			if isMethod {
				bKind = BoundaryMethod
				scope = extractReceiverScope(d.Recv.List[0].Type)
			}

			startLine := fset.Position(d.Pos()).Line
			if d.Doc != nil {
				startLine = fset.Position(d.Doc.Pos()).Line
			}
			endLine := fset.Position(d.End()).Line

			if startLine < 1 {
				startLine = 1
			}
			if endLine > totalLines {
				endLine = totalLines
			}
			if startLine <= endLine {
				chunks = append(chunks, CodeChunk{
					FilePath:       filePath,
					PackageName:    pkgName,
					EnclosingScope: scope,
					StartLine:      startLine,
					EndLine:        endLine,
					Boundary:       bKind,
					Content:        strings.Join(lines[startLine-1:endLine], "\n"),
				})
			}

		case *ast.GenDecl:
			if d.Tok == token.IMPORT {
				continue
			}

			if d.Tok == token.TYPE {
				for _, spec := range d.Specs {
					ts, ok := spec.(*ast.TypeSpec)
					if !ok {
						continue
					}
					bKind := BoundaryType
					switch ts.Type.(type) {
					case *ast.StructType:
						bKind = BoundaryStruct
					case *ast.InterfaceType:
						bKind = BoundaryInterface
					}
					scope := ts.Name.Name

					startLine := fset.Position(ts.Pos()).Line
					if ts.Doc != nil {
						startLine = fset.Position(ts.Doc.Pos()).Line
					} else if d.Doc != nil && len(d.Specs) == 1 {
						startLine = fset.Position(d.Doc.Pos()).Line
					}
					endLine := fset.Position(ts.End()).Line

					if startLine < 1 {
						startLine = 1
					}
					if endLine > totalLines {
						endLine = totalLines
					}
					if startLine <= endLine {
						chunks = append(chunks, CodeChunk{
							FilePath:       filePath,
							PackageName:    pkgName,
							EnclosingScope: scope,
							StartLine:      startLine,
							EndLine:        endLine,
							Boundary:       bKind,
							Content:        strings.Join(lines[startLine-1:endLine], "\n"),
						})
					}
				}
			} else {
				startLine := fset.Position(d.Pos()).Line
				if d.Doc != nil {
					startLine = fset.Position(d.Doc.Pos()).Line
				}
				endLine := fset.Position(d.End()).Line
				if startLine < 1 {
					startLine = 1
				}
				if endLine > totalLines {
					endLine = totalLines
				}
				if startLine <= endLine {
					chunks = append(chunks, CodeChunk{
						FilePath:       filePath,
						PackageName:    pkgName,
						EnclosingScope: "",
						StartLine:      startLine,
						EndLine:        endLine,
						Boundary:       BoundaryBlock,
						Content:        strings.Join(lines[startLine-1:endLine], "\n"),
					})
				}
			}
		}
	}

	sortChunksByStartLine(chunks)
	return chunks
}

// extractReceiverScope extracts the base type name of a method receiver.
func extractReceiverScope(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.StarExpr:
		return extractReceiverScope(t.X)
	case *ast.Ident:
		return t.Name
	case *ast.IndexExpr:
		return extractReceiverScope(t.X)
	case *ast.IndexListExpr:
		return extractReceiverScope(t.X)
	default:
		return ""
	}
}

// chunkTSJS partitions TypeScript and JavaScript source code along syntactic boundaries.
func (sc *SyntaxChunker) chunkTSJS(filePath string, content string) []CodeChunk {
	lines := strings.Split(content, "\n")
	totalLines := len(lines)
	pkgName := sc.deriveTSPackage(filePath, content)

	var chunks []CodeChunk

	type scanState struct {
		inSingleQuote  bool
		inDoubleQuote  bool
		inBacktick     bool
		inBlockComment bool
		braceDepth     int
		parenDepth     int
	}

	state := scanState{}

	// lineInfo stores the structural metrics for each 0-indexed line.
	type lineInfo struct {
		startBrace int
		endBrace   int
		inComment  bool
		trimmed    string
	}

	infos := make([]lineInfo, totalLines)

	for i, line := range lines {
		startBrace := state.braceDepth
		inCommentAtStart := state.inBlockComment
		trimmed := strings.TrimSpace(line)

		inLineComment := false
		for j := 0; j < len(line); j++ {
			ch := line[j]
			var nextCh byte
			if j+1 < len(line) {
				nextCh = line[j+1]
			}

			if inLineComment {
				break
			}
			if state.inBlockComment {
				if ch == '*' && nextCh == '/' {
					state.inBlockComment = false
					j++
				}
				continue
			}
			if state.inSingleQuote {
				if ch == '\\' {
					j++
				} else if ch == '\'' {
					state.inSingleQuote = false
				}
				continue
			}
			if state.inDoubleQuote {
				if ch == '\\' {
					j++
				} else if ch == '"' {
					state.inDoubleQuote = false
				}
				continue
			}
			if state.inBacktick {
				if ch == '\\' {
					j++
				} else if ch == '`' {
					state.inBacktick = false
				} else if ch == '$' && nextCh == '{' {
					state.braceDepth++
					j++
				}
				continue
			}

			if ch == '/' && nextCh == '/' {
				inLineComment = true
				break
			}
			if ch == '/' && nextCh == '*' {
				state.inBlockComment = true
				j++
				continue
			}
			if ch == '\'' {
				state.inSingleQuote = true
				continue
			}
			if ch == '"' {
				state.inDoubleQuote = true
				continue
			}
			if ch == '`' {
				state.inBacktick = true
				continue
			}
			if ch == '(' {
				state.parenDepth++
			} else if ch == ')' {
				if state.parenDepth > 0 {
					state.parenDepth--
				}
			} else if ch == '{' {
				state.braceDepth++
			} else if ch == '}' {
				if state.braceDepth > 0 {
					state.braceDepth--
				}
			}
		}

		infos[i] = lineInfo{
			startBrace: startBrace,
			endBrace:   state.braceDepth,
			inComment:  inCommentAtStart || inLineComment,
			trimmed:    trimmed,
		}
	}

	// Helper to find preceding doc comment start (0-indexed line).
	findDocStart := func(lineIdx int) int {
		curr := lineIdx - 1
		for curr >= 0 {
			t := strings.TrimSpace(lines[curr])
			if strings.HasPrefix(t, "//") || strings.HasPrefix(t, "/*") || strings.HasPrefix(t, "*") || strings.HasSuffix(t, "*/") {
				curr--
			} else {
				break
			}
		}
		return curr + 1
	}

	// Helper to add a chunk.
	addChunk := func(startLine, endLine int, boundary SyntaxBoundary, scope string) {
		if startLine < 1 {
			startLine = 1
		}
		if endLine > totalLines {
			endLine = totalLines
		}
		if startLine <= endLine {
			chunks = append(chunks, CodeChunk{
				FilePath:       filePath,
				PackageName:    pkgName,
				EnclosingScope: scope,
				StartLine:      startLine,
				EndLine:        endLine,
				Boundary:       boundary,
				Content:        strings.Join(lines[startLine-1:endLine], "\n"),
			})
		}
	}

	// Parse declarations line by line.
	i := 0
	for i < totalLines {
		info := infos[i]
		trimmed := info.trimmed

		// Skip blank lines
		if trimmed == "" {
			i++
			continue
		}

		// Imports at top level
		if info.startBrace == 0 && (strings.HasPrefix(trimmed, "import ") || strings.HasPrefix(trimmed, "import{") || strings.HasPrefix(trimmed, "import\"") || strings.HasPrefix(trimmed, "import'")) {
			startLine := i + 1
			endLine := startLine
			for k := i; k < totalLines; k++ {
				endLine = k + 1
				if strings.Contains(lines[k], ";") || (infos[k].endBrace == 0 && !strings.Contains(lines[k], "from")) {
					if strings.Contains(lines[k], ";") || k > i {
						i = k + 1
						break
					}
				}
				if k == totalLines-1 {
					i = totalLines
				}
			}
			addChunk(startLine, endLine, BoundaryImport, "")
			continue
		}

		// Interface at top level
		if info.startBrace == 0 && isInterfaceDecl(trimmed) {
			name := extractDeclName(trimmed, "interface")
			docStart := findDocStart(i)
			startLine := docStart + 1
			endLine := i + 1
			for k := i; k < totalLines; k++ {
				if infos[k].endBrace == 0 && strings.Contains(lines[k], "}") {
					endLine = k + 1
					i = k + 1
					break
				}
				if k == totalLines-1 {
					endLine = totalLines
					i = totalLines
				}
			}
			addChunk(startLine, endLine, BoundaryInterface, name)
			continue
		}

		// Type alias at top level
		if info.startBrace == 0 && isTypeDecl(trimmed) {
			name := extractDeclName(trimmed, "type")
			docStart := findDocStart(i)
			startLine := docStart + 1
			endLine := i + 1
			for k := i; k < totalLines; k++ {
				if infos[k].endBrace == 0 && strings.Contains(lines[k], ";") {
					endLine = k + 1
					i = k + 1
					break
				}
				if k == totalLines-1 {
					endLine = totalLines
					i = totalLines
				}
			}
			addChunk(startLine, endLine, BoundaryType, name)
			continue
		}

		// Enum at top level
		if info.startBrace == 0 && isEnumDecl(trimmed) {
			name := extractDeclName(trimmed, "enum")
			docStart := findDocStart(i)
			startLine := docStart + 1
			endLine := i + 1
			for k := i; k < totalLines; k++ {
				if infos[k].endBrace == 0 && strings.Contains(lines[k], "}") {
					endLine = k + 1
					i = k + 1
					break
				}
				if k == totalLines-1 {
					endLine = totalLines
					i = totalLines
				}
			}
			addChunk(startLine, endLine, BoundaryType, name)
			continue
		}

		// Top-level function
		if info.startBrace == 0 && isFunctionDecl(trimmed) {
			docStart := findDocStart(i)
			startLine := docStart + 1
			endLine := i + 1
			for k := i; k < totalLines; k++ {
				if infos[k].endBrace == 0 && strings.Contains(lines[k], "}") {
					endLine = k + 1
					i = k + 1
					break
				}
				if k == totalLines-1 {
					endLine = totalLines
					i = totalLines
				}
			}
			addChunk(startLine, endLine, BoundaryFunction, "")
			continue
		}

		// Arrow function or function expression assigned to const/let/var
		if info.startBrace == 0 && isVariableFunctionDecl(trimmed) {
			docStart := findDocStart(i)
			startLine := docStart + 1
			endLine := i + 1
			for k := i; k < totalLines; k++ {
				if infos[k].endBrace == 0 && (strings.Contains(lines[k], "}") || strings.Contains(lines[k], ";")) {
					endLine = k + 1
					i = k + 1
					break
				}
				if k == totalLines-1 {
					endLine = totalLines
					i = totalLines
				}
			}
			addChunk(startLine, endLine, BoundaryFunction, "")
			continue
		}

		// Class declaration at top level
		if info.startBrace == 0 && isClassDecl(trimmed) {
			className := extractDeclName(trimmed, "class")
			docStart := findDocStart(i)
			classStartLine := docStart + 1

			classHeaderEnd := i + 1
			classEndLine := totalLines

			// Walk lines inside the class
			k := i
			firstMethodFound := false

			for k < totalLines {
				kInfo := infos[k]
				kTrimmed := kInfo.trimmed

				// Check if this line inside class starts a method
				if kInfo.startBrace == 1 && isClassMethodStart(kTrimmed) {
					if !firstMethodFound {
						firstMethodFound = true
						if classHeaderEnd >= classStartLine {
							addChunk(classStartLine, classHeaderEnd, BoundaryClass, className)
						}
					}

					mDocStart := findDocStart(k)
					mStartLine := mDocStart + 1
					mEndLine := k + 1

					// Find where this method closes back to brace depth 1
					for m := k; m < totalLines; m++ {
						if infos[m].endBrace == 1 && strings.Contains(lines[m], "}") {
							mEndLine = m + 1
							k = m
							break
						}
						if m == totalLines-1 {
							mEndLine = totalLines
							k = totalLines
						}
					}
					addChunk(mStartLine, mEndLine, BoundaryMethod, className)
					classHeaderEnd = k + 1
					k++
					continue
				}

				// If class body closes back to 0
				if kInfo.endBrace == 0 && strings.Contains(lines[k], "}") {
					classEndLine = k + 1
					k++
					break
				}

				if !firstMethodFound {
					classHeaderEnd = k + 1
				}
				k++
			}

			if !firstMethodFound {
				addChunk(classStartLine, classEndLine, BoundaryClass, className)
			}

			i = k
			continue
		}

		i++
	}

	sortChunksByStartLine(chunks)
	return chunks
}

func isInterfaceDecl(s string) bool {
	clean := stripExport(s)
	return strings.HasPrefix(clean, "interface ")
}

func isTypeDecl(s string) bool {
	clean := stripExport(s)
	return strings.HasPrefix(clean, "type ")
}

func isEnumDecl(s string) bool {
	clean := stripExport(s)
	return strings.HasPrefix(clean, "enum ")
}

func isClassDecl(s string) bool {
	clean := stripExport(s)
	return strings.HasPrefix(clean, "class ")
}

func isFunctionDecl(s string) bool {
	clean := stripExport(s)
	if strings.HasPrefix(clean, "function ") || strings.HasPrefix(clean, "function* ") {
		return true
	}
	if strings.HasPrefix(clean, "async function ") || strings.HasPrefix(clean, "async function* ") {
		return true
	}
	return false
}

func isVariableFunctionDecl(s string) bool {
	clean := stripExport(s)
	if strings.HasPrefix(clean, "const ") || strings.HasPrefix(clean, "let ") || strings.HasPrefix(clean, "var ") {
		return strings.Contains(clean, "=>") || strings.Contains(clean, "function")
	}
	return false
}

func isClassMethodStart(s string) bool {
	if s == "" || strings.HasPrefix(s, "//") || strings.HasPrefix(s, "/*") || strings.HasPrefix(s, "*") {
		return false
	}
	// Common member qualifiers
	clean := s
	for {
		changed := false
		for _, prefix := range []string{"public ", "private ", "protected ", "static ", "async ", "override ", "readonly ", "get ", "set "} {
			if strings.HasPrefix(clean, prefix) {
				clean = strings.TrimSpace(strings.TrimPrefix(clean, prefix))
				changed = true
			}
		}
		if !changed {
			break
		}
	}
	// A method must have parentheses for parameter list
	if !strings.Contains(clean, "(") {
		return false
	}
	// Methods open braces or lead into parameter lists
	return strings.Contains(clean, "{") || strings.Contains(clean, "(")
}

func stripExport(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "export default ") {
		return strings.TrimSpace(strings.TrimPrefix(s, "export default "))
	}
	if strings.HasPrefix(s, "export ") {
		return strings.TrimSpace(strings.TrimPrefix(s, "export "))
	}
	return s
}

func extractDeclName(s, keyword string) string {
	clean := stripExport(s)
	idx := strings.Index(clean, keyword)
	if idx == -1 {
		return ""
	}
	rest := strings.TrimSpace(clean[idx+len(keyword):])
	fields := strings.FieldsFunc(rest, func(r rune) bool {
		return r == ' ' || r == '{' || r == '<' || r == '(' || r == ':' || r == '='
	})
	if len(fields) > 0 {
		return fields[0]
	}
	return ""
}

// deriveTSPackage extracts a package name from comments, explicit package statements, or file path.
func (sc *SyntaxChunker) deriveTSPackage(filePath, content string) string {
	if sc.DefaultPackage != "" {
		return sc.DefaultPackage
	}

	scanner := bufio.NewScanner(strings.NewReader(content))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, "//") || strings.HasPrefix(line, "/*") || strings.HasPrefix(line, "*") {
			lower := strings.ToLower(line)
			if idx := strings.Index(lower, "@package"); idx != -1 {
				fields := strings.Fields(line[idx+len("@package"):])
				if len(fields) > 0 {
					return strings.Trim(fields[0], "*/; ")
				}
			}
			if idx := strings.Index(lower, "@module"); idx != -1 {
				fields := strings.Fields(line[idx+len("@module"):])
				if len(fields) > 0 {
					return strings.Trim(fields[0], "*/; ")
				}
			}
		}
		if strings.HasPrefix(line, "package ") && strings.HasSuffix(line, ";") {
			name := strings.TrimPrefix(line, "package ")
			name = strings.TrimSuffix(name, ";")
			return strings.TrimSpace(name)
		}
	}

	dir := filepath.Dir(filePath)
	if dir != "." && dir != "/" && dir != "" {
		return filepath.Base(dir)
	}
	return ""
}

// chunkGeneric provides fallback block-based chunking for unsupported formats.
func (sc *SyntaxChunker) chunkGeneric(filePath string, content string) []CodeChunk {
	lines := strings.Split(content, "\n")
	totalLines := len(lines)
	pkgName := sc.DefaultPackage
	if pkgName == "" {
		dir := filepath.Dir(filePath)
		if dir != "." && dir != "/" && dir != "" {
			pkgName = filepath.Base(dir)
		}
	}

	var chunks []CodeChunk
	start := -1

	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			if start == -1 {
				start = i
			}
		} else {
			if start != -1 {
				chunks = append(chunks, CodeChunk{
					FilePath:       filePath,
					PackageName:    pkgName,
					EnclosingScope: "",
					StartLine:      start + 1,
					EndLine:        i,
					Boundary:       BoundaryBlock,
					Content:        strings.Join(lines[start:i], "\n"),
				})
				start = -1
			}
		}
	}

	if start != -1 {
		chunks = append(chunks, CodeChunk{
			FilePath:       filePath,
			PackageName:    pkgName,
			EnclosingScope: "",
			StartLine:      start + 1,
			EndLine:        totalLines,
			Boundary:       BoundaryBlock,
			Content:        strings.Join(lines[start:totalLines], "\n"),
		})
	}

	return chunks
}

func sortChunksByStartLine(chunks []CodeChunk) {
	sort.SliceStable(chunks, func(i, j int) bool {
		if chunks[i].StartLine == chunks[j].StartLine {
			return chunks[i].EndLine < chunks[j].EndLine
		}
		return chunks[i].StartLine < chunks[j].StartLine
	})
}
