package newmodel

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"
)

// Scaffold holds the parameters for generating a new model family scaffold.
type Scaffold struct {
	Family   string // Family name, lowercase (e.g. "myfamily")
	Topology string // Topology: "prenorm", "postnorm", "parallel", or "identity"
	DryRun   bool   // If true, print the scaffold instead of writing files
}

// Result is the outcome of a scaffolding operation.
type Result struct {
	Family    string   `json:"family"`
	Topology  string   `json:"topology"`
	Edits     []string `json:"edits"`
	NextSteps []string `json:"next_steps"`
}

// Run generates the scaffold for a new model family.
func Run(s Scaffold) (*Result, error) {
	if s.Family == "" {
		return nil, fmt.Errorf("family name is required")
	}
	if s.Topology == "" {
		s.Topology = "identity" // Default to identity spec
	}

	// Validate topology
	validTopologies := map[string]bool{
		"prenorm":  true,
		"postnorm": true,
		"parallel": true,
		"identity": true,
	}
	if !validTopologies[s.Topology] {
		return nil, fmt.Errorf("invalid topology %q; must be one of: prenorm, postnorm, parallel, identity", s.Topology)
	}

	familyKey := strings.ToLower(strings.ReplaceAll(s.Family, "_", ""))

	// Generate the config helper
	configHelper := genConfigHelper(s.Family, familyKey)

	// Generate the resolver spec
	resolverSpec := genResolverSpec(s.Family, s.Topology)

	// Generate the resolver case
	resolverCase := genResolverCase(s.Family, familyKey)

	// Generate the materializer wire
	materializerWire := genMaterializerWire(s.Family)

	// Generate the conformance test stub
	conformanceTest := genConformanceTest(s.Family, s.Topology)

	edits := []string{
		"internal/model/config.go (isMyFamily helper)",
		"internal/model/tensor_resolver.go (resolverSpec + resolveSpecFor case)",
		"internal/model/weights.go (materializer wire in newModel)",
		"internal/model/conformance_test.go (conformance test row)",
	}

	result := &Result{
		Family:   s.Family,
		Topology: s.Topology,
		Edits:    edits,
		NextSteps: []string{
			fmt.Sprintf("1. Add the isMyFamily() helper to internal/model/config.go:\n\n%s\n", configHelper),
			fmt.Sprintf("2. Add the resolverSpec function to internal/model/tensor_resolver.go:\n\n%s\n", resolverSpec),
			fmt.Sprintf("3. Add the case to resolveSpecFor() switch in internal/model/tensor_resolver.go:\n\n%s\n", resolverCase),
			fmt.Sprintf("4. Add the wire to newModel() in internal/model/weights.go:\n\n%s\n", materializerWire),
			fmt.Sprintf("5. Add the conformance test row to internal/model/conformance_test.go:\n\n%s\n", conformanceTest),
			"6. Run: go build ./cmd/fak && fak new-model " + s.Family + " --topology " + s.Topology,
			"7. Implement the family-specific logic (materializer, topology, etc.)",
			"8. Run: ./fak/test.ps1 ./internal/model/ ./internal/architest/",
		},
	}

	return result, nil
}

func genConfigHelper(family, familyKey string) string {
	return fmt.Sprintf(`// is%[1]s reports a %[1]s-family model. The family key lowercases
// model_type + architectures with separators stripped, so "%[2]s" -> "%[3]s".
// Used to gate %[1]s-specific load behavior and tensor resolution.
func (c Config) is%[1]s() bool {
	return strings.Contains(c.archFamilyKey(), "%[3]s")
}
`, strings.Title(family), family, familyKey)
}

func genResolverSpec(family, topology string) string {
	postAttentionLine := `		reqs = append(reqs, tensorReq{canonical: p + "post_attention_layernorm.weight"})
`
	preFFNLine := `		reqs = append(reqs, tensorReq{canonical: p + "pre_feedforward_layernorm.weight"})
`
	postFFNLine := `		reqs = append(reqs, tensorReq{canonical: p + "post_feedforward_layernorm.weight"})
`
	inputLine := `		reqs = append(reqs, tensorReq{canonical: p + "input_layernorm.weight"})
`

	var initLines []string
	var middleLines []string

	switch topology {
	case "prenorm":
		initLines = []string{inputLine}
		middleLines = []string{postAttentionLine, preFFNLine, postFFNLine}
	case "postnorm":
		initLines = []string{}
		middleLines = []string{postAttentionLine}
	case "parallel":
		initLines = []string{inputLine}
		middleLines = []string{postAttentionLine}
	case "identity":
		initLines = []string{}
		middleLines = []string{inputLine}
	}

	initCode := strings.Join(initLines, "")
	middleCode := strings.Join(middleLines, "")

	// Always initialize reqs
	initStmt := "reqs := []tensorReq{}\n" + initCode

	return fmt.Sprintf(`// %sSpec covers %s. %s architecture with %s topology.
func %sSpec(cfg Config) resolverSpec {
	return resolverSpec{
		family:  "%s",
		globals: baseGlobals(),
		perLayer: func(l int) []tensorReq {
			p := layerPrefix(l)
%s
			reqs = append(reqs, stdAttnProjections(p, nil, nil, nil, nil)...)
%s
			reqs = append(reqs, swigluMLP(p)...)
			return reqs
		},
	}
}
`, family, family, strings.Title(family), topology, family, family, initStmt, middleCode)
}

func genResolverCase(family, familyKey string) string {
	return fmt.Sprintf(`	case strings.Contains(fam, "%s"):
		return %sSpec(cfg)
`, familyKey, family)
}

func genMaterializerWire(family string) string {
	return fmt.Sprintf(`	if err := materialize%[1]sTensors(cfg, man, &raw); err != nil {
		return nil, err
	}
`, strings.Title(family))
}

func genConformanceTest(family, topology string) string {
	tmpl := `func Test{{.Family}}SyntheticForward(t *testing.T) {
	cfg := Config{
		HiddenSize:        16,
		NumLayers:         1,
		NumHeads:          2,
		NumKVHeads:        2,
		HeadDim:           6,
		IntermediateSize:  32,
		VocabSize:         40,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		ModelType:         "{{.Family}}",
		Architectures:     []string{"{{.Title}}ForCausalLM"},
		// TODO: set {{.Topology}}-specific axes here
	}
	m := NewSynthetic(cfg)
	if m == nil {
		t.Fatal("NewSynthetic returned nil")
	}
	// Verify the family is recognized
	if !strings.Contains(cfg.archFamilyKey(), "{{.LowerFamily}}") {
		t.Fatalf("family key = %q, want to contain {{.LowerFamily}}", cfg.archFamilyKey())
	}
	// TODO: add forward pass assertions once implemented
	// The scaffold places this test in the conformance suite so the
	// family is visible as UNIMPLEMENTED (red row) until the forward path
	// is wired and assertions are added.
}
`

	data := struct {
		Family      string
		Title       string
		LowerFamily string
		Topology    string
	}{
		Family:      strings.Title(family),
		Title:       strings.Title(family),
		LowerFamily: strings.ToLower(strings.ReplaceAll(family, "_", "")),
		Topology:    topology,
	}

	var buf bytes.Buffer
	t := template.Must(template.New("test").Parse(tmpl))
	t.Execute(&buf, data)
	return buf.String()
}

// Ready reports that the leaf is wired.
func Ready() bool { return true }