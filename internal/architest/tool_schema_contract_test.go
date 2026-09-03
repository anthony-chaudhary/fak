package architest

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// validateToolSchemaContract verifies a tool's JSON Schema against multi-provider
// OpenAPI 3.0 and provider dialect constraints (#10769, #10776):
//  1. Root schema type must be "object".
//  2. Every field in required must be present in properties of that object.
//  3. No partial anyOf/oneOf with required where type != "object" or where
//     required properties are not defined in that branch's properties.
//  4. Whenever type: "array", an items field must be defined.
//  5. No unexpanded $ref references.
//
// Recursively validates nested properties, items, anyOf, oneOf, allOf.
func validateToolSchemaContract(toolName string, schema map[string]any) []string {
	var violations []string

	if schema == nil {
		return []string{fmt.Sprintf("%s: schema is nil", toolName)}
	}

	// 1. Root schema type must be "object"
	rootType, _ := schema["type"].(string)
	if !strings.EqualFold(rootType, "object") {
		violations = append(violations, fmt.Sprintf("%s: root schema type must be 'object' (got %q)", toolName, rootType))
	}

	var validateNode func(path string, node map[string]any)
	validateNode = func(path string, node map[string]any) {
		if node == nil {
			return
		}

		// 5. No unexpanded $ref references
		if ref, ok := node["$ref"]; ok && ref != nil {
			violations = append(violations, fmt.Sprintf("%s: contains unexpanded $ref reference %v", path, ref))
		}

		stype, _ := node["type"].(string)

		// 4. Whenever type: "array", an items field must be defined
		if strings.EqualFold(stype, "array") {
			if items, ok := node["items"]; !ok || items == nil {
				violations = append(violations, fmt.Sprintf("%s: type is 'array' but missing 'items' field", path))
			}
		}

		// 2 & 3. Required properties validation
		var reqList []string
		switch r := node["required"].(type) {
		case []any:
			for _, item := range r {
				if s, ok := item.(string); ok {
					reqList = append(reqList, s)
				}
			}
		case []string:
			reqList = r
		}

		if len(reqList) > 0 {
			if !strings.EqualFold(stype, "object") {
				violations = append(violations, fmt.Sprintf("%s.required: only allowed for type 'object' (got %q)", path, stype))
			}
			props, _ := node["properties"].(map[string]any)
			for _, reqProp := range reqList {
				if props == nil || props[reqProp] == nil {
					violations = append(violations, fmt.Sprintf("%s.required: property %q is not defined in properties", path, reqProp))
				}
			}
		}

		// Recursively validate properties
		if props, ok := node["properties"].(map[string]any); ok {
			keys := make([]string, 0, len(props))
			for k := range props {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, k := range keys {
				if childMap, ok := props[k].(map[string]any); ok {
					validateNode(path+".properties."+k, childMap)
				}
			}
		}

		// Recursively validate items
		if items, ok := node["items"].(map[string]any); ok {
			validateNode(path+".items", items)
		} else if itemsSlice, ok := node["items"].([]any); ok {
			for i, it := range itemsSlice {
				if itMap, ok := it.(map[string]any); ok {
					validateNode(fmt.Sprintf("%s.items[%d]", path, i), itMap)
				}
			}
		}

		// Recursively validate anyOf, oneOf, allOf
		for _, combKey := range []string{"anyOf", "oneOf", "allOf"} {
			if alts, ok := node[combKey].([]any); ok {
				for i, alt := range alts {
					if altMap, ok := alt.(map[string]any); ok {
						validateNode(fmt.Sprintf("%s.%s[%d]", path, combKey, i), altMap)
					}
				}
			}
		}

		// Recursively validate additionalProperties if object
		if addProps, ok := node["additionalProperties"].(map[string]any); ok {
			validateNode(path+".additionalProperties", addProps)
		}
	}

	validateNode(toolName, schema)
	return violations
}

func unwrapExpr(expr ast.Expr) ast.Expr {
	for {
		if paren, ok := expr.(*ast.ParenExpr); ok {
			expr = paren.X
		} else {
			break
		}
	}
	return expr
}

func unquoteStringLit(lit *ast.BasicLit) (string, error) {
	if lit.Kind != token.STRING {
		return "", fmt.Errorf("not a string literal")
	}
	if strings.HasPrefix(lit.Value, "`") && strings.HasSuffix(lit.Value, "`") {
		return lit.Value[1 : len(lit.Value)-1], nil
	}
	return strconv.Unquote(lit.Value)
}

func extractRawMessageString(expr ast.Expr) (string, bool) {
	expr = unwrapExpr(expr)
	call, ok := expr.(*ast.CallExpr)
	if !ok || len(call.Args) != 1 {
		return "", false
	}
	switch fn := call.Fun.(type) {
	case *ast.SelectorExpr:
		if fn.Sel.Name != "RawMessage" {
			return "", false
		}
		if xIdent, ok := fn.X.(*ast.Ident); !ok || xIdent.Name != "json" {
			return "", false
		}
	case *ast.Ident:
		if fn.Name != "RawMessage" {
			return "", false
		}
	default:
		return "", false
	}
	arg := unwrapExpr(call.Args[0])
	lit, ok := arg.(*ast.BasicLit)
	if !ok || lit.Kind != token.STRING {
		return "", false
	}
	str, err := unquoteStringLit(lit)
	if err != nil {
		return "", false
	}
	return str, true
}

// extractMCPSchemas reads internal/gateway/mcp.go (and any sibling variable declarations),
// extracts tool names and their inputSchema raw JSON blocks, and parses them into JSON objects.
func extractMCPSchemas(t *testing.T, mcpGoPath string) map[string]map[string]any {
	t.Helper()

	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, mcpGoPath, nil, 0)
	if err != nil {
		t.Fatalf("extractMCPSchemas: failed to parse %s: %v", mcpGoPath, err)
	}

	rawVarMap := make(map[string]string)

	collectVars := func(fileNode *ast.File) {
		ast.Inspect(fileNode, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.ValueSpec:
				for i, name := range x.Names {
					if i < len(x.Values) {
						if raw, ok := extractRawMessageString(x.Values[i]); ok {
							rawVarMap[name.Name] = raw
						}
					}
				}
			case *ast.AssignStmt:
				for i, lhs := range x.Lhs {
					if ident, ok := lhs.(*ast.Ident); ok && i < len(x.Rhs) {
						if raw, ok := extractRawMessageString(x.Rhs[i]); ok {
							rawVarMap[ident.Name] = raw
						}
					}
				}
			}
			return true
		})
	}

	dir := filepath.Dir(mcpGoPath)
	if entries, err := os.ReadDir(dir); err == nil {
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
				continue
			}
			siblingPath := filepath.Join(dir, entry.Name())
			if siblingPath == mcpGoPath {
				continue
			}
			if sNode, err := parser.ParseFile(fset, siblingPath, nil, 0); err == nil {
				collectVars(sNode)
			}
		}
	}

	collectVars(node)

	schemas := make(map[string]map[string]any)

	ast.Inspect(node, func(n ast.Node) bool {
		compLit, ok := n.(*ast.CompositeLit)
		if !ok {
			return true
		}

		var name string
		var rawSchema string

		for _, elt := range compLit.Elts {
			kv, ok := elt.(*ast.KeyValueExpr)
			if !ok {
				continue
			}
			keyLit, ok := kv.Key.(*ast.BasicLit)
			if !ok || keyLit.Kind != token.STRING {
				continue
			}
			key, err := unquoteStringLit(keyLit)
			if err != nil {
				continue
			}
			switch key {
			case "name":
				if valLit, ok := kv.Value.(*ast.BasicLit); ok && valLit.Kind == token.STRING {
					name, _ = unquoteStringLit(valLit)
				}
			case "inputSchema":
				if raw, ok := extractRawMessageString(kv.Value); ok {
					rawSchema = raw
				} else if ident, ok := kv.Value.(*ast.Ident); ok {
					if raw, ok := rawVarMap[ident.Name]; ok {
						rawSchema = raw
					}
				}
			}
		}

		if name != "" && rawSchema != "" {
			var schemaObj map[string]any
			if err := json.Unmarshal([]byte(rawSchema), &schemaObj); err != nil {
				t.Fatalf("extractMCPSchemas: invalid JSON schema for tool %q: %v", name, err)
			}
			schemas[name] = schemaObj
		}

		return true
	})

	return schemas
}

func locateInternalDir(t *testing.T) string {
	t.Helper()
	dir := internalDir(t)
	if fi, err := os.Stat(dir); err == nil && fi.IsDir() {
		return dir
	}
	if fi, err := os.Stat(".."); err == nil && fi.IsDir() {
		if _, err := os.Stat(filepath.Join("..", "gateway", "mcp.go")); err == nil {
			abs, _ := filepath.Abs("..")
			return abs
		}
	}
	if fi, err := os.Stat("internal"); err == nil && fi.IsDir() {
		abs, _ := filepath.Abs("internal")
		return abs
	}
	wd, _ := os.Getwd()
	cur := wd
	for i := 0; i < 6; i++ {
		if _, err := os.Stat(filepath.Join(cur, "go.mod")); err == nil {
			cand := filepath.Join(cur, "internal")
			if fi, err := os.Stat(cand); err == nil && fi.IsDir() {
				return cand
			}
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			break
		}
		cur = parent
	}
	return dir
}

func TestToolSchemaContract(t *testing.T) {
	mcpPath := filepath.Join(locateInternalDir(t), "gateway", "mcp.go")
	schemas := extractMCPSchemas(t, mcpPath)

	if len(schemas) < 5 {
		t.Fatalf("expected at least 5 tools extracted from %s, got %d", mcpPath, len(schemas))
	}

	expectedNamed := []string{"fak_adjudicate", "fak_syscall", "fak_read", "fak_admit"}
	for _, name := range expectedNamed {
		if _, ok := schemas[name]; !ok {
			t.Errorf("expected tool %q to be extracted from %s", name, mcpPath)
		}
	}

	names := make([]string, 0, len(schemas))
	for name := range schemas {
		names = append(names, name)
	}
	sort.Strings(names)
	t.Logf("extracted %d MCP tools for contract compliance: %v", len(schemas), names)

	for toolName, schema := range schemas {
		violations := validateToolSchemaContract(toolName, schema)
		if len(violations) > 0 {
			t.Errorf("tool %q has %d contract violation(s):\n%s", toolName, len(violations), strings.Join(violations, "\n"))
		}
	}
}

func TestToolSchemaContractMutation(t *testing.T) {
	t.Run("defect_10769_bare_required_anyOf", func(t *testing.T) {
		// The #10769 defect: reintroducing "anyOf": [{"required": ["tool"]}, {"required": ["items"]}]
		// to an object without defining them in branch properties fails the gate.
		broken := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"tool":  map[string]any{"type": "string"},
				"items": map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			},
			"anyOf": []any{
				map[string]any{"required": []any{"tool"}},
				map[string]any{"required": []any{"items"}},
			},
		}
		violations := validateToolSchemaContract("mut_10769", broken)
		if len(violations) == 0 {
			t.Fatal("expected violations for bare required in anyOf branches, got 0")
		}
		hasObjectErr := false
		hasPropErr := false
		for _, v := range violations {
			if strings.Contains(v, "only allowed for type 'object'") {
				hasObjectErr = true
			}
			if strings.Contains(v, "not defined in properties") {
				hasPropErr = true
			}
		}
		if !hasObjectErr || !hasPropErr {
			t.Fatalf("expected both type and property errors, got: %v", violations)
		}
	})

	t.Run("required_property_missing_from_properties", func(t *testing.T) {
		broken := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"existing": map[string]any{"type": "string"},
			},
			"required": []any{"missing_prop"},
		}
		violations := validateToolSchemaContract("mut_req_missing", broken)
		if len(violations) == 0 {
			t.Fatal("expected violation for required property missing from properties, got 0")
		}
		found := false
		for _, v := range violations {
			if strings.Contains(v, "missing_prop") && strings.Contains(v, "not defined in properties") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected violation naming missing_prop, got: %v", violations)
		}
	})

	t.Run("root_schema_type_not_object", func(t *testing.T) {
		broken := map[string]any{
			"type": "string",
		}
		violations := validateToolSchemaContract("mut_root_string", broken)
		if len(violations) == 0 {
			t.Fatal("expected violation for root schema type != object, got 0")
		}
		found := false
		for _, v := range violations {
			if strings.Contains(v, "root schema type must be 'object'") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected root schema type error, got: %v", violations)
		}
	})

	t.Run("array_property_missing_items", func(t *testing.T) {
		broken := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"list": map[string]any{
					"type": "array",
				},
			},
		}
		violations := validateToolSchemaContract("mut_array_missing_items", broken)
		if len(violations) == 0 {
			t.Fatal("expected violation for array missing items, got 0")
		}
		found := false
		for _, v := range violations {
			if strings.Contains(v, "type is 'array' but missing 'items' field") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected array missing items error, got: %v", violations)
		}
	})

	t.Run("unexpanded_ref_fails_gate", func(t *testing.T) {
		broken := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"ref_field": map[string]any{
					"$ref": "#/definitions/foo",
				},
			},
		}
		violations := validateToolSchemaContract("mut_unexpanded_ref", broken)
		if len(violations) == 0 {
			t.Fatal("expected violation for unexpanded $ref, got 0")
		}
		found := false
		for _, v := range violations {
			if strings.Contains(v, "$ref") && strings.Contains(v, "#/definitions/foo") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected unexpanded $ref error, got: %v", violations)
		}
	})
}
