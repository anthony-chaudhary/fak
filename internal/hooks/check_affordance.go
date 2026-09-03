package hooks

import (
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// AffordanceViolation records a site where a refusal or denial construction
// lacks an actionable next-step affordance or NextAction string.
type AffordanceViolation struct {
	Line    int
	Token   string
	Message string
}

var nolintAffordanceRE = regexp.MustCompile(`(?i)//\s*nolint:(?:affordance|bare-denial)\b`)

// CheckAffordanceCompleteness inspects Go code content to ensure that refusal
// constructions (adjudicator refusal rungs, VerdictDeny, or refusal returns)
// include an actionable next-step affordance or NextAction string.
func CheckAffordanceCompleteness(content string) ([]AffordanceViolation, error) {
	if strings.TrimSpace(content) == "" {
		return nil, nil
	}

	fset := token.NewFileSet()
	lineOffset := 0
	file, err := parser.ParseFile(fset, "source.go", content, parser.ParseComments)
	if err != nil {
		trimmed := strings.TrimSpace(content)
		if strings.Contains(err.Error(), "expected 'package'") ||
			strings.HasPrefix(trimmed, "func ") ||
			strings.HasPrefix(trimmed, "type ") ||
			strings.HasPrefix(trimmed, "return ") ||
			strings.HasPrefix(trimmed, "var ") {

			wrapped := "package p\n"
			if !strings.HasPrefix(trimmed, "func ") &&
				!strings.HasPrefix(trimmed, "type ") &&
				!strings.HasPrefix(trimmed, "var ") &&
				!strings.HasPrefix(trimmed, "const ") {
				wrapped += "func _() {\n" + content + "\n}\n"
				lineOffset = 2
			} else {
				wrapped += content
				lineOffset = 1
			}

			fsetSnippet := token.NewFileSet()
			fileSnippet, errSnippet := parser.ParseFile(fsetSnippet, "snippet.go", wrapped, parser.ParseComments)
			if errSnippet == nil {
				fset = fsetSnippet
				file = fileSnippet
				err = nil
			}
		}
	}
	if err != nil {
		return nil, err
	}

	var violations []AffordanceViolation

	ast.Inspect(file, func(n ast.Node) bool {
		if n == nil {
			return true
		}

		// 1. Inspect refusal struct type declarations: type ...Refusal struct { ... }
		if ts, ok := n.(*ast.TypeSpec); ok {
			if isRefusalTypeName(ts.Name.Name) {
				if st, ok := ts.Type.(*ast.StructType); ok {
					if !hasAffordanceField(st) {
						if !isNolintAffordance(fset, file.Comments, ts.Pos()) {
							line := fset.Position(ts.Pos()).Line - lineOffset
							if line < 1 {
								line = 1
							}
							violations = append(violations, AffordanceViolation{
								Line:    line,
								Token:   ts.Name.Name,
								Message: "refusal declaration missing next-action affordance field",
							})
						}
					}
				}
			}
			return true
		}

		// 2. Inspect composite literals: VerdictDeny or Refusal struct creation
		if lit, ok := n.(*ast.CompositeLit); ok {
			typeName := getCompositeLitTypeName(lit.Type)
			isDeny, denyNode := getVerdictDenyField(lit)
			isRefusalStruct := isRefusalTypeName(typeName)

			if isDeny || isRefusalStruct {
				pos := lit.Pos()
				if isDeny && denyNode != nil {
					pos = denyNode.Pos()
				}
				if isNolintAffordance(fset, file.Comments, lit.Pos(), pos) {
					return true
				}

				present, isEmpty := checkAffordanceInLiteral(lit)
				tokenName := extractRefusalToken(lit, typeName)
				line := fset.Position(pos).Line - lineOffset
				if line < 1 {
					line = 1
				}

				if !present {
					msg := "refusal construction missing next-action affordance"
					if isDeny {
						msg = "VerdictDeny refusal missing next-action affordance"
					}
					violations = append(violations, AffordanceViolation{
						Line:    line,
						Token:   tokenName,
						Message: msg,
					})
				} else if isEmpty {
					msg := "refusal construction has empty next-action affordance"
					if isDeny {
						msg = "VerdictDeny refusal has empty next-action affordance"
					}
					violations = append(violations, AffordanceViolation{
						Line:    line,
						Token:   tokenName,
						Message: msg,
					})
				}
			}
			return true
		}

		// 3. Inspect function calls: verdict(VerdictDeny, ...) or NewRefusal(...)
		if call, ok := n.(*ast.CallExpr); ok {
			isRefusalCall, hasAff, isEmp, tokenName := checkAffordanceInCall(call)
			if isRefusalCall {
				if isNolintAffordance(fset, file.Comments, call.Pos()) {
					return true
				}
				line := fset.Position(call.Pos()).Line - lineOffset
				if line < 1 {
					line = 1
				}
				if !hasAff {
					violations = append(violations, AffordanceViolation{
						Line:    line,
						Token:   tokenName,
						Message: tokenName + " refusal missing next-action affordance",
					})
				} else if isEmp {
					violations = append(violations, AffordanceViolation{
						Line:    line,
						Token:   tokenName,
						Message: tokenName + " refusal has empty next-action affordance",
					})
				}
			}
			return true
		}

		return true
	})

	sort.Slice(violations, func(i, j int) bool {
		if violations[i].Line != violations[j].Line {
			return violations[i].Line < violations[j].Line
		}
		return violations[i].Token < violations[j].Token
	})

	return violations, nil
}

func isVerdictDeny(expr ast.Expr) bool {
	switch e := expr.(type) {
	case *ast.Ident:
		return e.Name == "VerdictDeny"
	case *ast.SelectorExpr:
		return e.Sel.Name == "VerdictDeny"
	}
	return false
}

func getVerdictDenyField(lit *ast.CompositeLit) (bool, ast.Node) {
	for _, elt := range lit.Elts {
		if kv, ok := elt.(*ast.KeyValueExpr); ok {
			if id, ok := kv.Key.(*ast.Ident); ok && id.Name == "Kind" {
				if isVerdictDeny(kv.Value) {
					return true, kv.Value
				}
			}
		}
	}
	if len(lit.Elts) > 0 && isVerdictDeny(lit.Elts[0]) {
		return true, lit.Elts[0]
	}
	return false, nil
}

func isAffordanceName(name string) bool {
	lower := strings.ToLower(strings.ReplaceAll(name, "_", ""))
	switch lower {
	case "nextaction", "affordance", "actionablenextstep", "actionable", "fix", "remedy", "recovery", "next":
		return true
	}
	return false
}

func isExprEmptyString(expr ast.Expr) bool {
	if bl, ok := expr.(*ast.BasicLit); ok && bl.Kind == token.STRING {
		unq, err := strconv.Unquote(bl.Value)
		if err == nil && strings.TrimSpace(unq) == "" {
			return true
		}
	}
	return false
}

func checkAffordanceInLiteral(lit *ast.CompositeLit) (present bool, isEmpty bool) {
	for _, elt := range lit.Elts {
		kv, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			continue
		}
		keyName := ""
		if id, ok := kv.Key.(*ast.Ident); ok {
			keyName = id.Name
		} else if litKey, ok := kv.Key.(*ast.BasicLit); ok && litKey.Kind == token.STRING {
			if unq, err := strconv.Unquote(litKey.Value); err == nil {
				keyName = unq
			}
		}

		if isAffordanceName(keyName) {
			present = true
			if isExprEmptyString(kv.Value) {
				return true, true
			}
			return true, false
		}

		if keyName == "Meta" {
			if metaLit, ok := kv.Value.(*ast.CompositeLit); ok {
				for _, mElt := range metaLit.Elts {
					if mKV, ok := mElt.(*ast.KeyValueExpr); ok {
						mKey := ""
						if bl, ok := mKV.Key.(*ast.BasicLit); ok && bl.Kind == token.STRING {
							if unq, err := strconv.Unquote(bl.Value); err == nil {
								mKey = unq
							}
						} else if id, ok := mKV.Key.(*ast.Ident); ok {
							mKey = id.Name
						}
						if isAffordanceName(mKey) {
							present = true
							if isExprEmptyString(mKV.Value) {
								return true, true
							}
							return true, false
						}
					}
				}
			} else if call, ok := kv.Value.(*ast.CallExpr); ok {
				callName := getCallFuncName(call)
				if isAffordanceName(callName) {
					return true, false
				}
			}
		}
	}
	return false, false
}

func extractRefusalToken(lit *ast.CompositeLit, typeName string) string {
	for _, elt := range lit.Elts {
		if kv, ok := elt.(*ast.KeyValueExpr); ok {
			if id, ok := kv.Key.(*ast.Ident); ok && id.Name == "Kind" {
				if isVerdictDeny(kv.Value) {
					return "VerdictDeny"
				}
			}
		}
	}
	if len(lit.Elts) > 0 && isVerdictDeny(lit.Elts[0]) {
		return "VerdictDeny"
	}

	for _, elt := range lit.Elts {
		if kv, ok := elt.(*ast.KeyValueExpr); ok {
			if id, ok := kv.Key.(*ast.Ident); ok {
				if id.Name == "Code" || id.Name == "Reason" || id.Name == "Token" {
					if bl, ok := kv.Value.(*ast.BasicLit); ok && bl.Kind == token.STRING {
						if unq, err := strconv.Unquote(bl.Value); err == nil && unq != "" {
							return unq
						}
					} else if valId, ok := kv.Value.(*ast.Ident); ok {
						return valId.Name
					} else if sel, ok := kv.Value.(*ast.SelectorExpr); ok {
						return sel.Sel.Name
					}
				}
			}
		}
	}

	if typeName != "" {
		return typeName
	}
	return "refusal"
}

func isRefusalTypeName(name string) bool {
	return strings.HasSuffix(name, "Refusal") || strings.HasSuffix(name, "Denial") ||
		name == "Refusal" || name == "Denial"
}

func getCompositeLitTypeName(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.SelectorExpr:
		return t.Sel.Name
	case *ast.UnaryExpr:
		if t.Op == token.AND {
			return getCompositeLitTypeName(t.X)
		}
	}
	return ""
}

func hasAffordanceField(st *ast.StructType) bool {
	if st == nil || st.Fields == nil {
		return false
	}
	for _, field := range st.Fields.List {
		for _, id := range field.Names {
			if isAffordanceName(id.Name) {
				return true
			}
		}
	}
	return false
}

func getCallFuncName(call *ast.CallExpr) string {
	switch f := call.Fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		return f.Sel.Name
	}
	return ""
}

func checkAffordanceInCall(call *ast.CallExpr) (isRefusalCall bool, hasAff bool, isEmp bool, tokenName string) {
	callName := getCallFuncName(call)

	hasVerdictDeny := false
	for _, arg := range call.Args {
		if isVerdictDeny(arg) {
			hasVerdictDeny = true
			break
		}
	}

	if hasVerdictDeny {
		isRefusalCall = true
		tokenName = "VerdictDeny"
		if strings.Contains(strings.ToLower(callName), "affordance") || strings.Contains(strings.ToLower(callName), "nextaction") {
			return true, true, false, tokenName
		}
		for _, arg := range call.Args {
			if isVerdictDeny(arg) {
				continue
			}
			if bl, ok := arg.(*ast.BasicLit); ok && bl.Kind == token.STRING {
				unq, err := strconv.Unquote(bl.Value)
				if err == nil {
					if strings.TrimSpace(unq) != "" && (strings.Contains(unq, " ") || len(unq) > 15) {
						return true, true, false, tokenName
					}
				}
			}
		}
		return true, false, false, tokenName
	}

	if isRefusalTypeName(callName) || strings.HasPrefix(callName, "NewRefusal") || strings.HasPrefix(callName, "NewDenial") {
		isRefusalCall = true
		tokenName = callName
		if strings.Contains(strings.ToLower(callName), "affordance") || strings.Contains(strings.ToLower(callName), "nextaction") {
			return true, true, false, tokenName
		}
		for _, arg := range call.Args {
			if bl, ok := arg.(*ast.BasicLit); ok && bl.Kind == token.STRING {
				unq, err := strconv.Unquote(bl.Value)
				if err == nil && strings.TrimSpace(unq) != "" && (strings.Contains(unq, " ") || len(unq) > 15) {
					return true, true, false, tokenName
				}
			}
		}
		return true, false, false, tokenName
	}

	return false, false, false, ""
}

func isNolintAffordance(fset *token.FileSet, comments []*ast.CommentGroup, positions ...token.Pos) bool {
	for _, pos := range positions {
		line := fset.Position(pos).Line
		for _, cg := range comments {
			cLine := fset.Position(cg.Pos()).Line
			cEndLine := fset.Position(cg.End()).Line
			if (cLine >= line-1 && cLine <= line) || (cEndLine >= line-1 && cEndLine <= line) {
				for _, c := range cg.List {
					if nolintAffordanceRE.MatchString(c.Text) {
						return true
					}
				}
			}
		}
	}
	return false
}
