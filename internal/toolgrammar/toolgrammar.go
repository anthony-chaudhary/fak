// Package toolgrammar compiles discriminated union schemas into EBNF grammars
// for constrained tool calling with literal parameter escaping and byte-level space protection.
//
// Clean-room implementation inspired by upstream otheru-ai/ember:src/model/tool_grammar.c:115-148,326-340.
package toolgrammar

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Dialect specifies the markup or syntax dialect for tool call grammar compilation.
type Dialect string

const (
	DialectDSML Dialect = "dsml"
	DialectXML  Dialect = "xml"
	DialectJSON Dialect = "json"
)

// Format is an alias for Dialect to support both naming conventions.
type Format = Dialect

const (
	FormatDSML = DialectDSML
	FormatXML  = DialectXML
	FormatJSON = DialectJSON
)

// VocabType specifies the vocabulary encoding conventions for whitespace rules.
type VocabType int

const (
	// VocabTypeStandard uses standard ASCII whitespace [ \t\n\r]*.
	VocabTypeStandard VocabType = iota
	// VocabTypeByteLevel includes BPE byte-level space marker 'Ġ' (\u0120, bytes 0xC4 0xA0).
	VocabTypeByteLevel
)

func (v VocabType) String() string {
	switch v {
	case VocabTypeByteLevel:
		return "byte_level"
	default:
		return "standard"
	}
}

// BPE space constants for byte-level tokenizers (GPT-2, DeepSeek, Qwen).
const (
	BPESpaceRune rune   = '\u0120'
	BPESpaceChar string = "Ġ"
)

// BPESpaceBytes is the UTF-8 byte sequence for 'Ġ' (0xC4, 0xA0).
var BPESpaceBytes = []byte{0xC4, 0xA0}

// LiteralParameterEscapeRule is the exact EBNF rule used for literal parameter escaping in tag-based dialects.
const LiteralParameterEscapeRule = `strval ::= ([^<] | "<" [^/])* ("<")?`

// GrammarOptions configures tool grammar compilation.
type GrammarOptions struct {
	Dialect        Dialect
	Format         Format
	VocabType      VocabType
	ToolName       string
	RootRule       string
	IncludeToolTag bool
	Whitespace     string
}

// Typed errors for grammar compilation failures.
var (
	ErrInvalidSchema      = errors.New("toolgrammar: invalid JSON schema")
	ErrNoDiscriminator    = errors.New("toolgrammar: could not identify discriminator property or branches")
	ErrUnsupportedDialect = errors.New("toolgrammar: unsupported grammar dialect")
)

// BuildBPEByteMap returns the canonical 256-byte to rune mapping table for BPE tokenizers
// (GPT-2, DeepSeek, Qwen) implementing the standard bytes_to_unicode mapping.
func BuildBPEByteMap() map[byte]rune {
	bs := make([]int, 0, 256)
	for i := int('!'); i <= int('~'); i++ {
		bs = append(bs, i)
	}
	for i := 161; i <= 172; i++ {
		bs = append(bs, i)
	}
	for i := 174; i <= 255; i++ {
		bs = append(bs, i)
	}

	seen := make(map[int]bool, 256)
	for _, b := range bs {
		seen[b] = true
	}

	b2u := make(map[byte]rune, 256)
	for _, b := range bs {
		b2u[byte(b)] = rune(b)
	}

	n := 0
	for b := 0; b < 256; b++ {
		if !seen[b] {
			b2u[byte(b)] = rune(256 + n)
			n++
		}
	}
	return b2u
}

// BuildReverseBPEMap returns the inverse mapping from BPE runes back to raw bytes.
func BuildReverseBPEMap() map[rune]byte {
	b2u := BuildBPEByteMap()
	u2b := make(map[rune]byte, len(b2u))
	for b, r := range b2u {
		u2b[r] = b
	}
	return u2b
}

// ByteToBPE maps a single byte to its GPT-2 / DeepSeek / Qwen byte-level BPE rune.
func ByteToBPE(b byte) rune {
	m := BuildBPEByteMap()
	return m[b]
}

// BPEToByte maps a byte-level BPE rune back to its raw byte.
func BPEToByte(r rune) (byte, bool) {
	m := BuildReverseBPEMap()
	b, ok := m[r]
	return b, ok
}

// EncodeByteLevel maps raw text into byte-level BPE representation (e.g. space 0x20 -> 'Ġ').
func EncodeByteLevel(s string) string {
	b2u := BuildBPEByteMap()
	var sb strings.Builder
	for i := 0; i < len(s); i++ {
		sb.WriteRune(b2u[s[i]])
	}
	return sb.String()
}

// DecodeByteLevel maps byte-level BPE representation back to raw text (e.g. 'Ġ' -> space 0x20).
func DecodeByteLevel(s string) string {
	u2b := BuildReverseBPEMap()
	var buf []byte
	for _, r := range s {
		if b, ok := u2b[r]; ok {
			buf = append(buf, b)
		} else {
			buf = append(buf, []byte(string(r))...)
		}
	}
	return string(buf)
}

// MatchLiteralParameter implements the matching semantics of
// `strval ::= ([^<] | "<" [^/])* ("<")?`.
// It consumes characters from s until encountering a '<' immediately followed by '/',
// preserving all code characters such as relational '<', '<=', '#include <...>',
// and templates 'vector<int>'. It returns the matched content and the remaining text.
func MatchLiteralParameter(s string) (matched, remainder string) {
	i := 0
	for i < len(s) {
		if s[i] == '<' {
			// Closing tag boundary: '<' immediately followed by '/'
			if i+1 < len(s) && s[i+1] == '/' {
				break
			}
			// If '<' is followed by another '<' that starts '</',
			// then s[i] is a trailing '<' matched by ("<")? before the closing tag.
			if i+2 < len(s) && s[i+1] == '<' && s[i+2] == '/' {
				i++
				break
			}
			// '<' followed by non-'/': consume both characters
			if i+1 < len(s) {
				i += 2
				continue
			}
			// Trailing '<' at end of string
			i++
			break
		}
		i++
	}
	return s[:i], s[i:]
}

// ParameterDef represents a single parameter definition within a branch.
type ParameterDef struct {
	Name            string
	Type            string
	Required        bool
	ConstValue      string
	IsDiscriminator bool
}

// BranchDef represents one alternation branch of a discriminated union.
type BranchDef struct {
	Name               string // e.g. "branch_replace"
	DiscriminatorValue string // e.g. "replace"
	Parameters         []ParameterDef
}

type rawSchema struct {
	Name          string                 `json:"name"`
	Title         string                 `json:"title"`
	Description   string                 `json:"description"`
	Type          string                 `json:"type"`
	Properties    map[string]rawProperty `json:"properties"`
	Required      []string               `json:"required"`
	OneOf         []rawSchema            `json:"oneOf"`
	AnyOf         []rawSchema            `json:"anyOf"`
	Discriminator *rawDiscriminator      `json:"discriminator"`
	Parameters    *rawSchema             `json:"parameters"`
	Function      *rawFunction           `json:"function"`
	Schema        *rawSchema             `json:"schema"`
}

type rawFunction struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Parameters  *rawSchema `json:"parameters"`
}

type rawDiscriminator struct {
	PropertyName string `json:"propertyName"`
}

type rawProperty struct {
	Type        string `json:"type"`
	Description string `json:"description"`
	Const       any    `json:"const"`
	Enum        []any  `json:"enum"`
	Default     any    `json:"default"`
}

// CompiledGrammar holds structured metadata and the compiled EBNF output.
type CompiledGrammar struct {
	EBNF          string
	Dialect       Dialect
	VocabType     VocabType
	ToolName      string
	Discriminator string
	Branches      []BranchDef
	RuleCount     int
}

// Compile compiles a JSON Schema into a CompiledGrammar with structured metadata.
func Compile(schemaBytes []byte, opts GrammarOptions) (*CompiledGrammar, error) {
	if len(schemaBytes) == 0 {
		return nil, fmt.Errorf("%w: empty schema bytes", ErrInvalidSchema)
	}

	var root rawSchema
	if err := json.Unmarshal(schemaBytes, &root); err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidSchema, err)
	}

	toolName := opts.ToolName
	if toolName == "" && root.Name != "" {
		toolName = root.Name
	}
	if toolName == "" && root.Title != "" {
		toolName = root.Title
	}

	target := &root
	if target.Function != nil {
		if toolName == "" && target.Function.Name != "" {
			toolName = target.Function.Name
		}
		if target.Function.Parameters != nil {
			target = target.Function.Parameters
		}
	}
	if target.Parameters != nil {
		target = target.Parameters
	}
	if target.Schema != nil {
		target = target.Schema
	}

	if toolName == "" && target.Name != "" {
		toolName = target.Name
	}
	if toolName == "" && target.Title != "" {
		toolName = target.Title
	}
	if toolName == "" {
		toolName = "tool"
	}

	dialect := opts.Dialect
	if dialect == "" {
		dialect = opts.Format
	}
	if dialect == "" {
		dialect = DialectDSML
	}
	switch dialect {
	case DialectDSML, DialectXML, DialectJSON:
	default:
		return nil, fmt.Errorf("%w: %q", ErrUnsupportedDialect, dialect)
	}

	discProp, branches, err := extractBranches(target)
	if err != nil {
		return nil, err
	}

	rootRule := opts.RootRule
	if rootRule == "" {
		rootRule = "root"
	}

	ebnf, ruleCount := generateEBNF(dialect, opts.VocabType, toolName, rootRule, discProp, branches, opts.IncludeToolTag, opts.Whitespace)

	return &CompiledGrammar{
		EBNF:          ebnf,
		Dialect:       dialect,
		VocabType:     opts.VocabType,
		ToolName:      toolName,
		Discriminator: discProp,
		Branches:      branches,
		RuleCount:     ruleCount,
	}, nil
}

// CompileDiscriminatedUnionGrammar compiles a JSON Schema describing a tool
// (such as a file editor with mode-based union parameters) into an EBNF grammar.
//
// It pins the discriminator parameter first in each union alternation branch
// to ensure linear rule scaling O(K * P) rather than combinatorial permutation blowup.
func CompileDiscriminatedUnionGrammar(schemaBytes []byte, opts GrammarOptions) (string, error) {
	cg, err := Compile(schemaBytes, opts)
	if err != nil {
		return "", err
	}
	return cg.EBNF, nil
}

func extractBranches(schema *rawSchema) (string, []BranchDef, error) {
	candidates := schema.OneOf
	if len(candidates) == 0 {
		candidates = schema.AnyOf
	}

	if len(candidates) > 0 {
		discProp := ""
		if schema.Discriminator != nil && schema.Discriminator.PropertyName != "" {
			discProp = schema.Discriminator.PropertyName
		}

		if discProp == "" {
			// Infer discriminator: property present in all branches with single const/enum value
			propCounts := make(map[string]int)
			propUniqueValues := make(map[string]map[string]bool)

			for _, b := range candidates {
				for pName, pDef := range b.Properties {
					val := getConstOrSingleEnum(pDef)
					if val != "" {
						propCounts[pName]++
						if propUniqueValues[pName] == nil {
							propUniqueValues[pName] = make(map[string]bool)
						}
						propUniqueValues[pName][val] = true
					}
				}
			}

			// Look for property matching all branches with distinct values
			for pName, count := range propCounts {
				if count == len(candidates) && len(propUniqueValues[pName]) == len(candidates) {
					discProp = pName
					break
				}
			}

			// Check common discriminator names in top-level properties if still not found
			if discProp == "" {
				for _, commonName := range []string{"mode", "type", "action", "command", "op", "operation"} {
					if _, ok := schema.Properties[commonName]; ok {
						discProp = commonName
						break
					}
				}
			}
		}

		if discProp == "" {
			return "", nil, fmt.Errorf("%w: found oneOf/anyOf branches but no common discriminator property", ErrNoDiscriminator)
		}

		var branches []BranchDef
		for i, b := range candidates {
			discVal := ""
			if pDef, ok := b.Properties[discProp]; ok {
				discVal = getConstOrSingleEnum(pDef)
			}
			if discVal == "" && schema.Properties != nil {
				if pDef, ok := schema.Properties[discProp]; ok {
					if i < len(pDef.Enum) {
						discVal = fmt.Sprintf("%v", pDef.Enum[i])
					}
				}
			}
			if discVal == "" {
				discVal = fmt.Sprintf("variant_%d", i)
			}

			bDef := buildBranchDef(discProp, discVal, b, schema)
			branches = append(branches, bDef)
		}

		return discProp, branches, nil
	}

	// Single object with enum property at top level acting as discriminator
	if len(schema.Properties) > 0 {
		var discProp string
		var enumVals []string

		if schema.Discriminator != nil && schema.Discriminator.PropertyName != "" {
			discProp = schema.Discriminator.PropertyName
			if pDef, ok := schema.Properties[discProp]; ok && len(pDef.Enum) > 0 {
				for _, e := range pDef.Enum {
					enumVals = append(enumVals, fmt.Sprintf("%v", e))
				}
			}
		}

		if discProp == "" {
			for _, candidateName := range []string{"mode", "type", "action", "command", "op", "operation"} {
				if pDef, ok := schema.Properties[candidateName]; ok && len(pDef.Enum) > 0 {
					discProp = candidateName
					for _, e := range pDef.Enum {
						enumVals = append(enumVals, fmt.Sprintf("%v", e))
					}
					break
				}
			}
		}

		if discProp == "" {
			// Find any string property with enum
			for pName, pDef := range schema.Properties {
				if len(pDef.Enum) > 1 {
					discProp = pName
					for _, e := range pDef.Enum {
						enumVals = append(enumVals, fmt.Sprintf("%v", e))
					}
					break
				}
			}
		}

		if discProp != "" && len(enumVals) > 0 {
			var branches []BranchDef
			for _, val := range enumVals {
				bDef := buildBranchDef(discProp, val, *schema, schema)
				branches = append(branches, bDef)
			}
			return discProp, branches, nil
		}

		// Fallback: single default branch
		bDef := buildBranchDef("", "default", *schema, schema)
		return "", []BranchDef{bDef}, nil
	}

	return "", nil, fmt.Errorf("%w: schema contains no properties or union branches", ErrNoDiscriminator)
}

func getConstOrSingleEnum(p rawProperty) string {
	if p.Const != nil {
		return fmt.Sprintf("%v", p.Const)
	}
	if len(p.Enum) == 1 {
		return fmt.Sprintf("%v", p.Enum[0])
	}
	return ""
}

func buildBranchDef(discProp, discVal string, branch rawSchema, topLevel *rawSchema) BranchDef {
	branchName := "branch_" + sanitizeIdent(discVal)
	var params []ParameterDef
	added := make(map[string]bool)

	// PIN DISCRIMINATOR PARAMETER FIRST
	if discProp != "" {
		params = append(params, ParameterDef{
			Name:            discProp,
			Type:            "string",
			Required:        true,
			ConstValue:      discVal,
			IsDiscriminator: true,
		})
		added[discProp] = true
	}

	// Add required parameters in declared order
	for _, reqName := range branch.Required {
		if added[reqName] {
			continue
		}
		pType := getPropertyType(reqName, branch, topLevel)
		params = append(params, ParameterDef{
			Name:     reqName,
			Type:     pType,
			Required: true,
		})
		added[reqName] = true
	}

	// Add any non-required parameters in deterministic alphabetical order
	var optNames []string
	for pName := range branch.Properties {
		if !added[pName] {
			optNames = append(optNames, pName)
		}
	}
	if topLevel != nil && topLevel.Properties != nil {
		for pName := range topLevel.Properties {
			if !added[pName] && branch.Properties != nil {
				if _, ok := branch.Properties[pName]; ok {
					// already collected
				}
			}
		}
	}
	sort.Strings(optNames)

	for _, optName := range optNames {
		pType := getPropertyType(optName, branch, topLevel)
		params = append(params, ParameterDef{
			Name:     optName,
			Type:     pType,
			Required: false,
		})
		added[optName] = true
	}

	return BranchDef{
		Name:               branchName,
		DiscriminatorValue: discVal,
		Parameters:         params,
	}
}

func getPropertyType(name string, branch rawSchema, topLevel *rawSchema) string {
	if p, ok := branch.Properties[name]; ok && p.Type != "" {
		return p.Type
	}
	if topLevel != nil && topLevel.Properties != nil {
		if p, ok := topLevel.Properties[name]; ok && p.Type != "" {
			return p.Type
		}
	}
	return "string"
}

func sanitizeIdent(s string) string {
	var sb strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' {
			sb.WriteRune(r)
		} else {
			sb.WriteRune('_')
		}
	}
	res := sb.String()
	if res == "" {
		return "default"
	}
	return strings.ToLower(res)
}

func generateEBNF(
	dialect Dialect,
	vocabType VocabType,
	toolName string,
	rootRule string,
	discProp string,
	branches []BranchDef,
	includeToolTag bool,
	customWS string,
) (string, int) {
	var lines []string
	ruleCount := 0

	// 1. Root Rule
	var rootExpr string
	var branchNames []string
	for _, b := range branches {
		branchNames = append(branchNames, b.Name)
	}
	unionExpr := strings.Join(branchNames, " | ")
	if len(branchNames) > 1 {
		unionExpr = "(" + unionExpr + ")"
	}

	switch dialect {
	case DialectDSML:
		rootExpr = fmt.Sprintf(`"<function=%s>" ws %s ws "</function>"`, toolName, unionExpr)
	case DialectXML:
		rootExpr = fmt.Sprintf(`"<%s>" ws %s ws "</%s>"`, toolName, unionExpr, toolName)
	case DialectJSON:
		if includeToolTag {
			rootExpr = fmt.Sprintf(`"{" ws "\"name\"" ws ":" ws "\"%s\"" ws "," ws "\"arguments\"" ws ":" ws "{" ws %s ws "}" ws "}"`, toolName, unionExpr)
		} else {
			rootExpr = fmt.Sprintf(`"{" ws %s ws "}"`, unionExpr)
		}
	}
	lines = append(lines, fmt.Sprintf("%s ::= %s", rootRule, rootExpr))
	ruleCount++

	// 2. Branch Rules with PINNED DISCRIMINATOR FIRST
	type paramRule struct {
		name string
		expr string
	}
	paramRulesMap := make(map[string]string)
	needStrval := false
	needJSONStrval := false
	needIntval := false
	needNumval := false
	needBoolval := false

	for _, b := range branches {
		var branchTokens []string

		for _, p := range b.Parameters {
			if p.IsDiscriminator {
				// Pinned discriminator emitted directly in branch definition
				switch dialect {
				case DialectDSML:
					branchTokens = append(branchTokens, fmt.Sprintf(`"<parameter=%s>%s</parameter>"`, p.Name, p.ConstValue))
				case DialectXML:
					branchTokens = append(branchTokens, fmt.Sprintf(`"<%s>%s</%s>"`, p.Name, p.ConstValue, p.Name))
				case DialectJSON:
					branchTokens = append(branchTokens, fmt.Sprintf(`"\"%s\"" ws ":" ws "\"%s\""`, p.Name, p.ConstValue))
				}
				continue
			}

			// Generate parameter rule reference
			ruleName := fmt.Sprintf("param_%s_%s", sanitizeIdent(b.DiscriminatorValue), sanitizeIdent(p.Name))
			var pExpr string

			switch dialect {
			case DialectDSML:
				valRule := "strval"
				switch p.Type {
				case "integer":
					valRule = "intval"
					needIntval = true
				case "number":
					valRule = "numval"
					needNumval = true
				case "boolean":
					valRule = "boolval"
					needBoolval = true
				default:
					needStrval = true
				}
				pExpr = fmt.Sprintf(`"<parameter=%s>" %s "</parameter>"`, p.Name, valRule)

			case DialectXML:
				valRule := "strval"
				switch p.Type {
				case "integer":
					valRule = "intval"
					needIntval = true
				case "number":
					valRule = "numval"
					needNumval = true
				case "boolean":
					valRule = "boolval"
					needBoolval = true
				default:
					needStrval = true
				}
				pExpr = fmt.Sprintf(`"<%s>" %s "</%s>"`, p.Name, valRule, p.Name)

			case DialectJSON:
				valRule := "json_strval"
				switch p.Type {
				case "integer":
					valRule = "intval"
					needIntval = true
				case "number":
					valRule = "numval"
					needNumval = true
				case "boolean":
					valRule = "boolval"
					needBoolval = true
				default:
					needJSONStrval = true
				}
				pExpr = fmt.Sprintf(`"\"%s\"" ws ":" ws %s`, p.Name, valRule)
			}

			paramRulesMap[ruleName] = pExpr

			if dialect == DialectJSON {
				if p.Required {
					branchTokens = append(branchTokens, "ws \",\" ws "+ruleName)
				} else {
					branchTokens = append(branchTokens, fmt.Sprintf("(ws \",\" ws %s)?", ruleName))
				}
			} else {
				if p.Required {
					branchTokens = append(branchTokens, "ws "+ruleName)
				} else {
					branchTokens = append(branchTokens, fmt.Sprintf("(ws %s)?", ruleName))
				}
			}
		}

		branchExpr := strings.Join(branchTokens, " ")
		lines = append(lines, fmt.Sprintf("%s ::= %s", b.Name, branchExpr))
		ruleCount++
	}

	// 3. Parameter Rules (sorted alphabetically for determinism)
	var sortedParamNames []string
	for pName := range paramRulesMap {
		sortedParamNames = append(sortedParamNames, pName)
	}
	sort.Strings(sortedParamNames)
	for _, pName := range sortedParamNames {
		lines = append(lines, fmt.Sprintf("%s ::= %s", pName, paramRulesMap[pName]))
		ruleCount++
	}

	// 4. Value Rules
	if dialect == DialectJSON {
		needJSONStrval = true
	} else {
		needStrval = true
	}

	if needStrval {
		lines = append(lines, LiteralParameterEscapeRule)
		ruleCount++
	}
	if needJSONStrval {
		lines = append(lines, `json_strval ::= "\"" ([^"\\] | "\\" .)* "\""`)
		ruleCount++
	}
	if needIntval {
		lines = append(lines, `intval ::= ("-"? [0-9]+)`)
		ruleCount++
	}
	if needNumval {
		lines = append(lines, `numval ::= ("-"? [0-9]+ ("." [0-9]+)?)`)
		ruleCount++
	}
	if needBoolval {
		lines = append(lines, `boolval ::= ("true" | "false")`)
		ruleCount++
	}

	// 5. Whitespace Rule
	var wsRule string
	if customWS != "" {
		wsRule = fmt.Sprintf("ws ::= %s", customWS)
	} else if vocabType == VocabTypeByteLevel {
		wsRule = `ws ::= ([ \t\n\r] | "Ġ")*`
	} else {
		wsRule = `ws ::= [ \t\n\r]*`
	}
	lines = append(lines, wsRule)
	ruleCount++

	return strings.Join(lines, "\n"), ruleCount
}
