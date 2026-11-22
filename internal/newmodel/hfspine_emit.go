package newmodel

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"go/format"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"text/template"
)

// SpineInsertion is one hand-applied edit to an existing internal/model file.
// Mode "after" inserts Snippet on the line after Anchor; "replace" swaps
// Anchor for Snippet.
type SpineInsertion struct {
	File    string `json:"file"`
	Mode    string `json:"mode"`
	Anchor  string `json:"anchor"`
	Snippet string `json:"snippet"`
}

// HFSpine is a family's emitted bring-up spine, keyed by repo-relative path.
type HFSpine struct {
	Family     string            `json:"family"`
	Files      map[string][]byte `json:"-"`
	Insertions []SpineInsertion  `json:"insertions"`
}

// Paths lists the emitted files in write order.
func (s *HFSpine) Paths() []string { return sortedKeys(s.Files) }

var hfFamilyToken = regexp.MustCompile(`^[a-z][a-z0-9]*$`)

// hfIdentityToken bounds the config identity strings the emitter splices into
// Go source, comments and format strings.
var hfIdentityToken = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)

var hfLayerTensor = regexp.MustCompile(`^(.*?\blayers\.)(\d+)\.(.+)$`)

type spineData struct {
	F, C                                 string
	Repo, Rev, SourcePin                 string
	EnvelopeType, TextType, Architecture string
	Label                                string
	Pins, Required                       []string
}

// EmitHFSpine renders the spine for family from a census. The family token is
// the Go identifier stem, so it must be lowercase alphanumeric.
func EmitHFSpine(c *HFCensus, family string) (*HFSpine, error) {
	if !hfFamilyToken.MatchString(family) {
		return nil, fmt.Errorf("family %q must match %s", family, hfFamilyToken)
	}
	var env struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
		TextConfig    struct {
			ModelType string `json:"model_type"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(c.Config, &env); err != nil {
		return nil, headerRefusal(RefusalHeaderConfigInvalid, hfConfigFile, "%v", err)
	}
	if len(env.Architectures) != 1 || env.ModelType == "" {
		return nil, headerRefusal(RefusalHeaderConfigInvalid, hfConfigFile, "need one architecture and a model_type, got %v / %q", env.Architectures, env.ModelType)
	}
	for _, id := range []struct{ axis, value string }{
		{"model_type", env.ModelType},
		{"architectures[0]", env.Architectures[0]},
		{"text_config.model_type", env.TextConfig.ModelType},
	} {
		if id.value != "" && !hfIdentityToken.MatchString(id.value) {
			return nil, headerRefusal(RefusalHeaderConfigInvalid, hfConfigFile, "%s %q must match %s", id.axis, id.value, hfIdentityToken)
		}
	}
	leaves, err := flattenHFConfig(c.Config)
	if err != nil {
		return nil, headerRefusal(RefusalHeaderConfigInvalid, hfConfigFile, "%v", err)
	}
	d := spineData{
		F: family, C: strings.ToUpper(family[:1]) + family[1:],
		Repo: c.Repo, Rev: c.Revision, SourcePin: c.Repo + "@" + c.Revision[:min(8, len(c.Revision))],
		EnvelopeType: env.ModelType, TextType: env.TextConfig.ModelType, Architecture: env.Architectures[0],
		Label: env.ModelType,
	}
	for _, k := range sortedKeys(leaves) {
		d.Pins = append(d.Pins, strconv.Quote(k)+": "+strconv.Quote(leaves[k])+",")
	}
	for _, name := range requiredHFTensors(c.Tensors) {
		ts := c.Tensors[name]
		dims := make([]string, len(ts.Shape))
		for i, v := range ts.Shape {
			dims[i] = strconv.Itoa(v)
		}
		d.Required = append(d.Required, fmt.Sprintf("%s: {[]int{%s}, %s},", strconv.Quote(name), strings.Join(dims, ", "), strconv.Quote(ts.DType)))
	}

	s := &HFSpine{Family: family, Files: map[string][]byte{}, Insertions: []SpineInsertion{
		// The generic-path check closes the refuse chain, so it stays the anchor as families are added.
		{File: "internal/model/arch_support.go", Mode: "before",
			Anchor:  "\tif cfg.IsQwen35Hybrid() {",
			Snippet: "\tif err := refuse" + d.C + "Native(cfg); err != nil {\n\t\treturn err\n\t}"},
		{File: "internal/model/qwen35.go", Mode: "replace",
			Anchor:  "if c.GLM5Next || c.IsKimiK31() {",
			Snippet: "if c.GLM5Next || c.IsKimiK31() || c.Is" + d.C + "() {"},
	}}
	fixtures := "internal/model/testdata/" + family + "/"
	s.Files[fixtures+"config.json"] = append([]byte(nil), c.Config...)

	var shardNames []string
	for _, sh := range c.Shards {
		shardNames = append(shardNames, sh.File)
	}
	inventory := struct {
		Source  string              `json:"source"`
		Tensors map[string]HFTensor `json:"tensors"`
	}{c.Repo + "@" + c.Revision + " " + strings.Join(shardNames, "+") + " header", c.Tensors}
	if s.Files[fixtures+"tensor_inventory.json"], err = indentJSON(inventory, " "); err != nil {
		return nil, err
	}

	files := []string{hfConfigFile}
	if c.HasIndex {
		files = append(files, hfIndexFile)
	}
	provenance := struct {
		Schema       string          `json:"schema"`
		Repo         string          `json:"repo"`
		Revision     string          `json:"revision"`
		Files        []string        `json:"files"`
		FixtureScope string          `json:"fixture_scope"`
		Shards       []HFShardHeader `json:"shards"`
		// SidecarListing is the HTTP status of a failed file listing: sidecar
		// safetensors outside the index were not censused.
		SidecarListing int `json:"sidecar_listing,omitempty"`
	}{HFCensusSchema, c.Repo, c.Revision, files, "config.json verbatim; safetensors headers only, no tensor payload", c.Shards, c.SidecarListing}
	if s.Files[fixtures+"provenance.json"], err = indentJSON(provenance, "  "); err != nil {
		return nil, err
	}

	for name, tmpl := range map[string]*template.Template{
		"internal/model/" + family + "_spine.go":      spineGoTemplate,
		"internal/model/" + family + "_spine_test.go": spineTestTemplate,
	} {
		src, err := renderGo(tmpl, d)
		if err != nil {
			return nil, fmt.Errorf("render %s: %w", name, err)
		}
		s.Files[name] = src
	}
	if s.Files[fixtures+"INSERTION.md"], err = renderText(insertionTemplate, struct {
		spineData
		Insertions []SpineInsertion
	}{d, s.Insertions}); err != nil {
		return nil, err
	}
	if s.Files["tools/concept_disambiguation_scorecard.data/rows-model-"+family+".json"], err = scorecardRows(d); err != nil {
		return nil, err
	}
	return s, nil
}

// WriteHFSpine writes the spine under outDir. It refuses an existing family
// fixture directory or any existing target file rather than overwrite.
func WriteHFSpine(outDir string, s *HFSpine) error {
	famDir := filepath.Join(outDir, "internal", "model", "testdata", s.Family)
	if _, err := os.Stat(famDir); err == nil {
		return fmt.Errorf("%s already exists; refusing to overwrite a family spine", famDir)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	paths := s.Paths()
	for _, rel := range paths {
		if path.IsAbs(rel) || strings.HasPrefix(path.Clean(rel), "..") {
			return fmt.Errorf("spine path %q escapes the output dir", rel)
		}
		if _, err := os.Stat(filepath.Join(outDir, filepath.FromSlash(rel))); err == nil {
			return fmt.Errorf("%s already exists; refusing to overwrite", rel)
		}
	}
	for _, rel := range paths {
		dst := filepath.Join(outDir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(dst, s.Files[rel], 0o644); err != nil {
			return err
		}
	}
	return nil
}

// flattenHFConfig maps every config leaf to its compact JSON text by dotted
// path. Raw text is kept so number spellings survive exactly. The emitted
// spine carries a copy of this algorithm; the two must stay identical.
func flattenHFConfig(data []byte) (map[string]string, error) {
	out := map[string]string{}
	return out, flattenHFConfigInto(out, "", data)
}

func flattenHFConfigInto(out map[string]string, prefix string, data []byte) error {
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
			if err := flattenHFConfigInto(out, prefix+k+".", v); err != nil {
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

// requiredHFTensors picks every non-layer tensor, every tensor of the first
// and last layer, and the lowest layer at which each per-layer tensor takes a
// new shape or dtype, so mixed-precision layer splits are pinned.
func requiredHFTensors(tensors map[string]HFTensor) []string {
	type layered struct {
		name, pattern string
		layer         int
	}
	var picked, perLayer []string
	var items []layered
	last := -1
	for _, name := range sortedKeys(tensors) {
		m := hfLayerTensor.FindStringSubmatch(name)
		if m == nil {
			picked = append(picked, name)
			continue
		}
		layer, err := strconv.Atoi(m[2])
		if err != nil {
			picked = append(picked, name)
			continue
		}
		ts := tensors[name]
		items = append(items, layered{name, fmt.Sprintf("%sN.%s|%s|%v", m[1], m[3], ts.DType, ts.Shape), layer})
		last = max(last, layer)
	}
	sort.SliceStable(items, func(i, j int) bool { return items[i].layer < items[j].layer })
	seen := map[string]bool{}
	for _, it := range items {
		if it.layer == 0 || it.layer == last || !seen[it.pattern] {
			perLayer = append(perLayer, it.name)
		}
		seen[it.pattern] = true
	}
	out := append(picked, perLayer...)
	sort.Strings(out)
	return out
}

func indentJSON(v any, indent string) ([]byte, error) {
	b, err := json.MarshalIndent(v, "", indent)
	if err != nil {
		return nil, err
	}
	return append(b, '\n'), nil
}

func renderText(t *template.Template, d any) ([]byte, error) {
	var buf bytes.Buffer
	if err := t.Execute(&buf, d); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func renderGo(t *template.Template, d spineData) ([]byte, error) {
	raw, err := renderText(t, d)
	if err != nil {
		return nil, err
	}
	return format.Source(raw)
}

func scorecardRows(d spineData) ([]byte, error) {
	type row struct {
		ID             string   `json:"id"`
		Canonical      string   `json:"canonical"`
		Family         string   `json:"family"`
		Kind           string   `json:"kind"`
		Definition     string   `json:"definition"`
		Distinction    string   `json:"distinction"`
		DistinctFrom   []string `json:"distinct_from"`
		Aliases        []string `json:"aliases"`
		Grounding      string   `json:"grounding"`
		GroundingKind  string   `json:"grounding_kind"`
		GlossaryAnchor string   `json:"glossary_anchor"`
		Parent         string   `json:"parent"`
		Verdict        string   `json:"verdict"`
		Gaps           []string `json:"gaps"`
	}
	grounding := "Unsupported" + d.C + "ExecutionError"
	r := row{
		ID: "model-" + d.F + "-native-refusal", Canonical: strings.ToLower(grounding), Family: "model", Kind: "symbol",
		Definition: fmt.Sprintf("%s (internal/model/%s_spine.go): the typed refusal fak's native forward returns for a %s checkpoint pinned at %s@%s, whose config leaves and safetensors header geometry the spine validates exactly.",
			grounding, d.F, d.EnvelopeType, d.Repo, d.Rev),
		Distinction:  fmt.Sprintf("A load-time refusal for one pinned %s release, not a generic unsupported-architecture error: it names the family and fires before any generic forward path is classified.", d.EnvelopeType),
		DistinctFrom: []string{},
		Aliases:      []string{grounding, d.EnvelopeType},
		Grounding:    grounding, GroundingKind: "symbol", GlossaryAnchor: "docs/fak/concept-glossary.md",
		Verdict: "defined",
		Gaps: []string{
			"definition and distinction prose is generated by fak new-model --hf; replace before landing",
			"distinct_from is empty; name sibling model rows before landing",
		},
	}
	return indentJSON(struct {
		Rows []row `json:"rows"`
	}{[]row{r}}, "  ")
}
