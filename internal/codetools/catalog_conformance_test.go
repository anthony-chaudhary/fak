package codetools

import (
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

type catalogSchema struct {
	Properties map[string]struct {
		Description string `json:"description"`
	} `json:"properties"`
	Required []string `json:"required"`
}

func catalogArgs(name string) any {
	switch name {
	case ToolRead:
		return &ReadArgs{}
	case ToolGrep:
		return &GrepArgs{}
	case ToolGlob:
		return &GlobArgs{}
	case ToolWrite:
		return &WriteArgs{}
	case ToolEdit:
		return &EditArgs{}
	case ToolBash:
		return &BashArgs{}
	case ToolApplyPatch:
		return &PatchArgs{}
	}
	return nil
}

// catalogContractProblems returns diagnostics rather than failing directly so a
// synthetic drift can demonstrate that each contract actually detects omissions.
func catalogContractProblems(defs []ToolDef) []string {
	var problems []string
	schemas := make(map[string]catalogSchema)
	for _, d := range defs {
		var s catalogSchema
		if err := json.Unmarshal(d.Parameters, &s); err != nil {
			problems = append(problems, d.Name+": invalid schema")
			continue
		}
		schemas[d.Name] = s
	}
	snake := regexp.MustCompile(`\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b`)
	requiredWord := regexp.MustCompile(`(?i)\brequired\b`)
	for _, d := range defs {
		s := schemas[d.Name]
		args := catalogArgs(d.Name)
		if args == nil {
			problems = append(problems, d.Name+": no typed decoder fixture")
			continue
		}
		fields := map[string]bool{}
		typ := reflect.TypeOf(args).Elem()
		for i := 0; i < typ.NumField(); i++ {
			fields[strings.Split(typ.Field(i).Tag.Get("json"), ",")[0]] = true
		}
		for name := range fields {
			if _, ok := s.Properties[name]; !ok {
				// apply_patch fuzz is its documented backwards-compatible decoder alias.
				if d.Name != ToolApplyPatch || name != "fuzz" {
					problems = append(problems, d.Name+": undeclared decoder field "+name)
				}
			}
		}
		for name := range s.Properties {
			if !fields[name] {
				problems = append(problems, d.Name+": schema has no decoder field "+name)
			}
		}
		if decodeArgs([]byte(`{"__undeclared_probe":true}`), args) == nil {
			problems = append(problems, d.Name+": decoder accepts an unknown property")
		}
		required := map[string]bool{}
		for _, name := range s.Required {
			required[name] = true
			if !fields[name] {
				problems = append(problems, d.Name+": required field has no decoder "+name)
			}
		}
		descriptions := []string{d.Description}
		for name, property := range s.Properties {
			descriptions = append(descriptions, property.Description)
			if requiredWord.MatchString(property.Description) && !required[name] {
				// #11417 owns apply_patch's expected_version CAS guard gap. Do not
				// reword the promise or silently weaken this explicit exception.
				if d.Name != ToolApplyPatch || name != "expected_version" {
					problems = append(problems, d.Name+": required description missing schema requirement "+name)
				}
			}
		}
		for _, description := range descriptions {
			allowed := map[string]bool{}
			for name := range s.Properties {
				allowed[name] = true
			}
			for _, other := range defs {
				if regexp.MustCompile(`\b` + regexp.QuoteMeta(other.Name) + `\b`).MatchString(description) {
					allowed[other.Name] = true
					for name := range schemas[other.Name].Properties {
						allowed[name] = true
					}
				}
			}
			for _, name := range snake.FindAllString(description, -1) {
				if !allowed[name] {
					problems = append(problems, d.Name+": description names undeclared parameter "+name)
				}
			}
		}
	}
	return problems
}

// fak-test:runtime fast est=10ms lane=default
func TestCatalogDescriptionsMatchSchemas(t *testing.T) {
	defs := Catalog()
	if problems := catalogContractProblems(defs); len(problems) > 0 {
		t.Fatal(strings.Join(problems, "\n"))
	}
	t.Run("undeclared-description", func(t *testing.T) {
		bad := append([]ToolDef(nil), defs...)
		bad[0].Description += " Uses invented_parameter."
		if len(catalogContractProblems(bad)) == 0 {
			t.Fatal("checker missed an undeclared parameter")
		}
	})
	t.Run("required-description", func(t *testing.T) {
		bad := append([]ToolDef(nil), defs...)
		bad[0].Parameters = json.RawMessage(`{"properties":{"file_path":{},"offset":{"description":"required"},"limit":{}},"required":["file_path"]}`)
		if len(catalogContractProblems(bad)) == 0 {
			t.Fatal("checker missed a missing required field")
		}
	})
	t.Run("decoder-schema", func(t *testing.T) {
		bad := append([]ToolDef(nil), defs...)
		bad[0].Parameters = json.RawMessage(`{"properties":{"file_path":{},"offset":{},"limit":{},"extra_field":{}},"required":["file_path"]}`)
		if len(catalogContractProblems(bad)) == 0 {
			t.Fatal("checker missed a decoder/schema mismatch")
		}
	})
}

func minimalCatalogArgs(name string) []byte {
	switch name {
	case ToolRead:
		return []byte(`{"file_path":"a.go"}`)
	case ToolGrep:
		return []byte(`{"pattern":"x"}`)
	case ToolGlob:
		return []byte(`{"pattern":"*.go"}`)
	case ToolWrite:
		return []byte(`{"file_path":"a.go","content":"x","mode":"create"}`)
	case ToolEdit:
		return []byte(`{"file_path":"a.go","old_string":"x","new_string":"y","expected_version":"fv1:observed"}`)
	case ToolBash:
		return []byte(`{"command":"echo x"}`)
	case ToolApplyPatch:
		return []byte(`{"patch":"--- a/a.go\n+++ b/a.go\n@@ -1 +1 @@\n-x\n+y\n"}`)
	}
	return []byte(`{"file_path":"a.go"}`)
}

func mutatingSwitchProblems(ts *Toolset, defs []ToolDef) []string {
	var problems []string
	for _, d := range defs {
		if _, ok := engineFor(d.Name); !ok {
			problems = append(problems, d.Name+": no engine")
		}
		if _, refusal := ts.targetOf(d.Name, minimalCatalogArgs(d.Name)); refusal != nil {
			problems = append(problems, fmt.Sprintf("%s: targetOf: %v", d.Name, refusal))
		}
		var s catalogSchema
		if json.Unmarshal(d.Parameters, &s) != nil {
			problems = append(problems, d.Name+": invalid schema")
			continue
		}
		if _, ok := s.Properties["file_path"]; d.ReadOnly || !ok {
			continue
		}
		body := []byte(`{"file_path":"a.go"}`)
		if _, ok := ts.MutationObservationKey(d.Name, body); !ok {
			problems = append(problems, d.Name+": no mutation observation")
		}
		bound, ok := BindObservedVersion(d.Name, body, "fv1:observed")
		var got struct {
			Version string `json:"expected_version"`
		}
		if !ok || json.Unmarshal(bound, &got) != nil || got.Version != "fv1:observed" {
			problems = append(problems, d.Name+": no version binding")
		}
	}
	return problems
}

// fak-test:runtime fast est=10ms lane=default
func TestMutatingToolNameSwitchesCoverCatalog(t *testing.T) {
	ts, _ := newTestToolset(t)
	if problems := mutatingSwitchProblems(ts, Catalog()); len(problems) > 0 {
		t.Fatal(strings.Join(problems, "\n"))
	}
	fake := ToolDef{Name: "UnknownMutation", Parameters: json.RawMessage(`{"properties":{"file_path":{}}}`)}
	if len(mutatingSwitchProblems(ts, []ToolDef{fake})) == 0 {
		t.Fatal("checker missed unknown mutating tool")
	}
}
