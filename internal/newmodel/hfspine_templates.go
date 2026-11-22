package newmodel

import "text/template"

var spineGoTemplate = template.Must(template.New("spine").Parse(`package model

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
)

// {{.C}} ({{.Repo}}@{{.Rev}}, model_type {{printf "%q" .EnvelopeType}}) was
// scaffolded by ` + "`fak new-model --hf`" + ` from its pinned config.json and
// safetensors headers. The pins below are exact; the semantics are not.
// TODO(bring-up): say which of fak's forward paths would mis-run this family
// and why, add the invariants the pins cannot express, and name the OSS routes
// that serve it.
const (
	{{.F}}EnvelopeType = {{printf "%q" .EnvelopeType}}
	{{.F}}TextType     = {{printf "%q" .TextType}}
	{{.F}}Architecture = {{printf "%q" .Architecture}}
	{{.F}}SourcePin    = {{printf "%q" .SourcePin}}
)

// {{.F}}PinnedConfig is every config.json leaf, by dotted path, as compact JSON.
var {{.F}}PinnedConfig = map[string]string{
{{range .Pins}}	{{.}}
{{end}}}

// {{.C}}Config is the pinned identity of a {{.EnvelopeType}} checkpoint plus
// its flattened config leaves.
type {{.C}}Config struct {
	ModelType     string
	TextModelType string
	Architecture  string
	Leaves        map[string]string
}

type {{.F}}Envelope struct {
	Architectures []string ` + "`json:\"architectures\"`" + `
	ModelType     string   ` + "`json:\"model_type\"`" + `
	TextConfig    struct {
		ModelType string ` + "`json:\"model_type\"`" + `
	} ` + "`json:\"text_config\"`" + `
}

// Load{{.C}}Config reads and validates a {{.EnvelopeType}} config.json.
func Load{{.C}}Config(path string) ({{.C}}Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return {{.C}}Config{}, err
	}
	return Parse{{.C}}Config(data)
}

// Parse{{.C}}Config decodes and validates {{.EnvelopeType}} config.json bytes.
func Parse{{.C}}Config(data []byte) ({{.C}}Config, error) {
	var raw {{.F}}Envelope
	if err := json.Unmarshal(data, &raw); err != nil {
		return {{.C}}Config{}, fmt.Errorf("{{.Label}} config: %w", err)
	}
	if len(raw.Architectures) != 1 {
		return {{.C}}Config{}, fmt.Errorf("{{.Label}} config: require exactly one architecture, got %d", len(raw.Architectures))
	}
	leaves, err := {{.F}}FlattenConfig(data)
	if err != nil {
		return {{.C}}Config{}, fmt.Errorf("{{.Label}} config: %w", err)
	}
	cfg := {{.C}}Config{
		ModelType:     raw.ModelType,
		TextModelType: raw.TextConfig.ModelType,
		Architecture:  raw.Architectures[0],
		Leaves:        leaves,
	}
	return cfg, cfg.Validate()
}

// Validate refuses any drift from the released identity or config leaves.
func (c {{.C}}Config) Validate() error {
	if c.ModelType != {{.F}}EnvelopeType || c.TextModelType != {{.F}}TextType || c.Architecture != {{.F}}Architecture {
		return fmt.Errorf("{{.Label}} config: unsupported identity %q/%q/%q", c.ModelType, c.TextModelType, c.Architecture)
	}
	return isExact{{.C}}Config(c.Leaves)
}

// isExact{{.C}}Config refuses any missing, extra or changed config leaf.
func isExact{{.C}}Config(leaves map[string]string) error {
	var drift []string
	for k, want := range {{.F}}PinnedConfig {
		got, ok := leaves[k]
		switch {
		case !ok:
			drift = append(drift, "missing "+k)
		case got != want:
			drift = append(drift, fmt.Sprintf("%s=%s, want %s", k, got, want))
		}
	}
	for k := range leaves {
		if _, ok := {{.F}}PinnedConfig[k]; !ok {
			drift = append(drift, "extra "+k)
		}
	}
	if len(drift) > 0 {
		sort.Strings(drift)
		return fmt.Errorf("{{.Label}} config: drift from the pinned release: %s", strings.Join(drift, "; "))
	}
	return nil
}

// {{.F}}FlattenConfig maps each config leaf to its compact JSON text by dotted
// path, keeping raw number spellings so the pins compare exactly.
func {{.F}}FlattenConfig(data []byte) (map[string]string, error) {
	out := map[string]string{}
	return out, {{.F}}FlattenInto(out, "", data)
}

func {{.F}}FlattenInto(out map[string]string, prefix string, data []byte) error {
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	if len(obj) == 0 && prefix != "" {
		out[strings.TrimSuffix(prefix, ".")] = "{}"
	}
	for k, v := range obj {
		v = bytes.TrimSpace(v)
		if len(v) > 0 && v[0] == '{' {
			if err := {{.F}}FlattenInto(out, prefix+k+".", v); err != nil {
				return err
			}
			continue
		}
		var buf bytes.Buffer
		if err := json.Compact(&buf, v); err != nil {
			return err
		}
		out[prefix+k] = buf.String()
	}
	return nil
}

type {{.C}}Tensor struct {
	Shape []int  ` + "`json:\"shape\"`" + `
	DType string ` + "`json:\"dtype\"`" + `
}

type {{.C}}TensorInventory struct {
	Source  string                   ` + "`json:\"source\"`" + `
	Tensors map[string]{{.C}}Tensor ` + "`json:\"tensors\"`" + `
}

func Load{{.C}}TensorInventory(path string) ({{.C}}TensorInventory, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return {{.C}}TensorInventory{}, err
	}
	var inv {{.C}}TensorInventory
	if err := json.Unmarshal(data, &inv); err != nil {
		return inv, fmt.Errorf("{{.Label}} tensor inventory: %w", err)
	}
	return inv, nil
}

// {{.F}}RequiredTensors holds every non-layer tensor, every tensor of the
// first and last layer, and the first layer where each per-layer tensor takes
// a new shape or dtype.
var {{.F}}RequiredTensors = map[string]{{.C}}Tensor{
{{range .Required}}	{{.}}
{{end}}}

// Validate{{.C}}TensorInventory checks a pinned header inventory against the
// required tensors.
func Validate{{.C}}TensorInventory(cfg {{.C}}Config, inv {{.C}}TensorInventory) error {
	if err := cfg.Validate(); err != nil {
		return err
	}
	if !strings.Contains(inv.Source, {{.F}}SourcePin) {
		return fmt.Errorf("{{.Label}} tensor inventory: unpinned source %q", inv.Source)
	}
	for name, want := range {{.F}}RequiredTensors {
		got, ok := inv.Tensors[name]
		if !ok {
			return fmt.Errorf("{{.Label}} tensor inventory: missing %q", name)
		}
		if !reflect.DeepEqual(got.Shape, want.Shape) || got.DType != want.DType {
			return fmt.Errorf("{{.Label}} tensor inventory: %q = %s%v, want %s%v", name, got.DType, got.Shape, want.DType, want.Shape)
		}
	}
	return nil
}

// Unsupported{{.C}}ExecutionError is the typed refusal for running a
// {{.EnvelopeType}} checkpoint on fak's native forward.
type Unsupported{{.C}}ExecutionError struct {
	Engine       string
	ModelType    string
	Architecture string
}

func (e *Unsupported{{.C}}ExecutionError) Error() string {
	return fmt.Sprintf("%s: unsupported execution for %s/%s: the native forward for this family is not implemented", e.Engine, e.ModelType, e.Architecture)
}

// NativeExecution validates the config and returns the typed refusal.
func (c {{.C}}Config) NativeExecution() error {
	if err := c.Validate(); err != nil {
		return err
	}
	return &Unsupported{{.C}}ExecutionError{Engine: "fak-native", ModelType: c.ModelType, Architecture: c.Architecture}
}

// Is{{.C}} reports whether a generically decoded Config is a {{.EnvelopeType}}
// checkpoint.{{if .TextType}} The wrapper model_type overlays text_config, so
// either spelling identifies the family.{{end}}
func (c Config) Is{{.C}}() bool {
	return c.ModelType == {{.F}}EnvelopeType{{if .TextType}} || c.ModelType == {{.F}}TextType{{end}}
}

func refuse{{.C}}Native(c Config) error {
	if !c.Is{{.C}}() {
		return nil
	}
	return &Unsupported{{.C}}ExecutionError{Engine: "fak-native", ModelType: c.ModelType, Architecture: {{.F}}Architecture}
}
`))

var spineTestTemplate = template.Must(template.New("spinetest").Parse(`package model

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
)

const (
	{{.F}}ConfigPath    = "testdata/{{.F}}/config.json"
	{{.F}}InventoryPath = "testdata/{{.F}}/tensor_inventory.json"
)

// fak-test:runtime fast est=5ms lane=default
func Test{{.C}}PinnedConfigIsExact(t *testing.T) {
	cfg, err := Load{{.C}}Config({{.F}}ConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	var unsupported *Unsupported{{.C}}ExecutionError
	if err := cfg.NativeExecution(); !errors.As(err, &unsupported) {
		t.Fatalf("NativeExecution = %v, want *Unsupported{{.C}}ExecutionError", err)
	}
	for k, v := range cfg.Leaves {
		drifted := map[string]string{}
		for k2, v2 := range cfg.Leaves {
			drifted[k2] = v2
		}
		drifted[k] = v + "0"
		if err := isExact{{.C}}Config(drifted); err == nil {
			t.Fatalf("drifted leaf %s accepted", k)
		}
		break
	}
	extra := map[string]string{"fak_unpinned_probe": "1"}
	for k, v := range cfg.Leaves {
		extra[k] = v
	}
	if err := isExact{{.C}}Config(extra); err == nil {
		t.Fatal("an unpinned extra leaf was accepted")
	}
}

// fak-test:runtime fast est=5ms lane=default
func Test{{.C}}TensorInventoryMatchesPinnedHeaders(t *testing.T) {
	cfg, err := Load{{.C}}Config({{.F}}ConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	inv, err := Load{{.C}}TensorInventory({{.F}}InventoryPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate{{.C}}TensorInventory(cfg, inv); err != nil {
		t.Fatal(err)
	}
	inv.Source = "unpinned"
	if err := Validate{{.C}}TensorInventory(cfg, inv); err == nil {
		t.Fatal("an unpinned inventory source was accepted")
	}
}

// The generic Config decode must classify this family as a typed refusal, not
// a generic forward path. This stays red until the INSERTION.md edits land.
// fak-test:runtime fast est=5ms lane=default
func Test{{.C}}GenericConfigClassifiesAsRefusal(t *testing.T) {
	data, err := os.ReadFile({{.F}}ConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("generic decode: %v", err)
	}
	if !cfg.Is{{.C}}() {
		t.Fatalf("generic decode lost the family (model_type %q)", cfg.ModelType)
	}
	if cfg.IsQwen35Hybrid() {
		t.Fatal("generic decode claims the qwen35 hybrid forward")
	}
	kind, err := ClassifyForwardPath(cfg, nil)
	var unsupported *Unsupported{{.C}}ExecutionError
	if !errors.As(err, &unsupported) {
		t.Fatalf("ClassifyForwardPath = %q, %v; want *Unsupported{{.C}}ExecutionError", kind, err)
	}
}
`))

var insertionTemplate = template.Must(template.New("insertion").Parse(`# {{.C}} spine: hand edits

Emitted by ` + "`fak new-model --hf {{.Repo}}@{{.Rev}} --family {{.F}}`" + ` from
config.json and safetensors headers only. Copy the emitted tree into the repo,
then apply these edits so the generic forward classifier refuses the family.
` + "`Test{{.C}}GenericConfigClassifiesAsRefusal`" + ` is red until they land.
{{range .Insertions}}
## {{.File}} ({{.Mode}})

Anchor:

` + "```go" + `
{{.Anchor}}
` + "```" + `

{{if eq .Mode "after"}}Insert after the anchor:{{else}}Replace the anchor with:{{end}}

` + "```go" + `
{{.Snippet}}
` + "```" + `
{{end}}
## Not derivable from headers

- Semantic invariants the pins cannot express (layer schedules, block
  alignment, precision splits by layer): add them to ` + "`{{.C}}Config.Validate`" + `.
- OSS serving routes and the refusal message's "serve it on" hint.
- Scorecard row prose and ` + "`distinct_from`" + ` in
  ` + "`tools/concept_disambiguation_scorecard.data/rows-model-{{.F}}.json`" + `.

Witness: ` + "`go test ./internal/model -run {{.C}} -count=1`" + `.
`))
