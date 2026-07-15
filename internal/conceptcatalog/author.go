package conceptcatalog

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

type PositionRequest struct {
	ID, Canonical, Family, Definition, Distinction, Kind, Grounding, GroundingKind, Glossary string
	DistinctFrom, Aliases                                                                    []string
	RowFile                                                                                  string
}
type ClassifyRequest struct{ Family, Token, Category, Reason string }
type Change struct {
	Path        string `json:"path"`
	BeforeCount int    `json:"before_count,omitempty"`
	AfterCount  int    `json:"after_count,omitempty"`
	Content     []byte `json:"-"`
}
type Plan struct {
	Mode              string   `json:"mode"`
	Family            string   `json:"family"`
	BeforeFamilyCount int      `json:"before_family_count"`
	AfterFamilyCount  int      `json:"after_family_count"`
	Files             []string `json:"files"`
	Changes           []Change `json:"-"`
}

func PlanPosition(c Catalog, req PositionRequest) (Plan, error) {
	required := map[string]string{"id": req.ID, "canonical": req.Canonical, "family": req.Family, "definition": req.Definition, "distinction": req.Distinction, "kind": req.Kind, "grounding": req.Grounding, "grounding_kind": req.GroundingKind, "glossary": req.Glossary}
	for k, v := range required {
		if strings.TrimSpace(v) == "" {
			return Plan{}, fmt.Errorf("%s is required", k)
		}
	}
	if len(req.DistinctFrom) == 0 {
		return Plan{}, fmt.Errorf("distinct_from requires at least one stable row ID")
	}
	ids := map[string]bool{}
	familyOK := false
	before := 0
	for _, f := range c.Meta.Families {
		if norm(f.ID) == norm(req.Family) {
			familyOK = true
		}
	}
	if !familyOK {
		return Plan{}, fmt.Errorf("unknown family %q", req.Family)
	}
	for _, r := range c.Rows {
		ids[norm(r.ID)] = true
		if norm(r.Family) == norm(req.Family) {
			before++
		}
		if norm(r.ID) == norm(req.ID) {
			return Plan{}, fmt.Errorf("ID %q already exists", req.ID)
		}
		if norm(r.Canonical) == norm(req.Canonical) {
			return Plan{}, fmt.Errorf("canonical %q already exists", req.Canonical)
		}
	}
	for _, ref := range req.DistinctFrom {
		if !ids[norm(ref)] {
			return Plan{}, fmt.Errorf("distinct_from %q is not an existing row ID", ref)
		}
	}
	row := Row{ID: req.ID, Canonical: req.Canonical, Family: req.Family, Kind: req.Kind, Definition: req.Definition, Distinction: req.Distinction, DistinctFrom: req.DistinctFrom, Aliases: req.Aliases, Grounding: req.Grounding, GroundingKind: req.GroundingKind, GlossaryAnchor: req.Glossary, Verdict: "crystal", Gaps: []string{}}
	rowFile := req.RowFile
	if rowFile == "" {
		rowFile = "rows-" + norm(req.Family) + "-authored.json"
	}
	if !strings.HasPrefix(rowFile, "rows-") || !strings.HasSuffix(rowFile, ".json") {
		return Plan{}, fmt.Errorf("row-file must match rows-*.json")
	}
	path := filepath.Join(c.Dir, rowFile)
	content, err := appendRowFile(path, row)
	if err != nil {
		return Plan{}, err
	}
	glossaryPath := req.Glossary
	if !filepath.IsAbs(glossaryPath) {
		glossaryPath = filepath.Join(filepath.Dir(filepath.Dir(c.Dir)), filepath.FromSlash(req.Glossary))
	}
	gb, err := os.ReadFile(glossaryPath)
	if err != nil {
		return Plan{}, err
	}
	heading := "\n\n### " + req.Canonical + "\n\n" + req.Definition + "\n\n**Distinct from:** " + req.Distinction + "\n"
	if bytes.Contains(bytes.ToLower(gb), bytes.ToLower([]byte("### "+req.Canonical))) {
		return Plan{}, fmt.Errorf("glossary already contains %q", req.Canonical)
	}
	gb = append(gb, []byte(heading)...)
	files := []string{filepath.ToSlash(path), filepath.ToSlash(glossaryPath)}
	sort.Strings(files)
	plan := Plan{Mode: "position", Family: req.Family, BeforeFamilyCount: before, AfterFamilyCount: before + 1, Files: files, Changes: []Change{{Path: path, BeforeCount: before, AfterCount: before + 1, Content: content}, {Path: glossaryPath, Content: gb}}}
	return AddGeneratedArtifacts(c, plan)
}

func appendRowFile(path string, row Row) ([]byte, error) {
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		x := struct {
			Rows []Row `json:"rows"`
		}{Rows: []Row{row}}
		out, e := json.MarshalIndent(x, "", "  ")
		if e != nil {
			return nil, e
		}
		return append(out, '\n'), nil
	}
	if err != nil {
		return nil, err
	}
	var check struct {
		Rows []Row `json:"rows"`
	}
	if err = json.Unmarshal(b, &check); err != nil {
		return nil, err
	}
	// Preserve every existing byte. Insert one indented object immediately before
	// the rows array's closing bracket instead of marshal-rewriting the file.
	closeObj := bytes.LastIndex(b, []byte("}"))
	if closeObj < 0 {
		return nil, fmt.Errorf("%s: missing object close", path)
	}
	closeRows := bytes.LastIndex(b[:closeObj], []byte("]"))
	if closeRows < 0 {
		return nil, fmt.Errorf("%s: missing rows close", path)
	}
	rb, e := json.MarshalIndent(row, "    ", "  ")
	if e != nil {
		return nil, e
	}
	prefix := b[:closeRows]
	suffix := b[closeRows:]
	sep := []byte("\n")
	if len(check.Rows) > 0 {
		sep = []byte(",\n")
	}
	out := append([]byte{}, prefix...)
	out = append(out, sep...)
	out = append(out, []byte("    ")...)
	out = append(out, rb...)
	out = append(out, '\n', ' ', ' ')
	out = append(out, suffix...)
	return out, nil
}

// AddGeneratedArtifacts runs the canonical generator against a shadow data
// directory containing the planned mutations. It never alters the workspace.
func AddGeneratedArtifacts(c Catalog, plan Plan) (Plan, error) {
	root := filepath.Dir(filepath.Dir(c.Dir))
	if _, err := os.Stat(filepath.Join(root, "tools", "concept_disambiguation_scorecard.py")); os.IsNotExist(err) {
		return plan, nil
	}
	script := filepath.Join(root, "tools", "concept_disambiguation_scorecard.py")
	if _, err := os.Stat(script); err != nil {
		return Plan{}, fmt.Errorf("canonical generator: %w", err)
	}
	shadow, err := os.MkdirTemp("", "fak-concept-data-*")
	if err != nil {
		return Plan{}, err
	}
	defer os.RemoveAll(shadow)
	entries, err := os.ReadDir(c.Dir)
	if err != nil {
		return Plan{}, err
	}
	proposed := map[string][]byte{}
	for _, ch := range plan.Changes {
		proposed[filepath.Clean(ch.Path)] = ch.Content
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		src := filepath.Join(c.Dir, e.Name())
		b, er := os.ReadFile(src)
		if er != nil {
			return Plan{}, er
		}
		if v, ok := proposed[filepath.Clean(src)]; ok {
			b = v
		}
		if er = os.WriteFile(filepath.Join(shadow, e.Name()), b, 0600); er != nil {
			return Plan{}, er
		}
	}
	outDir := filepath.Join(shadow, "generated")
	cmd := exec.Command("python", script, "--workspace", root, "--data", shadow, "--markdown-dir", outDir)
	cmd.Dir = root
	if b, er := cmd.CombinedOutput(); er != nil {
		return Plan{}, fmt.Errorf("canonical generation failed: %v: %s", er, strings.TrimSpace(string(b)))
	}
	readme, err := os.ReadFile(filepath.Join(outDir, "README.md"))
	if err != nil {
		return Plan{}, err
	}
	readme = append(bytes.TrimRight(readme, "\r\n"), '\n')
	dst := filepath.Join(root, "docs", "concept-disambiguation-scorecard", "README.md")
	plan.Changes = append(plan.Changes, Change{Path: dst, Content: readme})
	plan.Files = append(plan.Files, filepath.ToSlash(dst))
	sort.Strings(plan.Files)
	return plan, nil
}

var categories = map[string]bool{"incidental": true, "false-positive": true, "test-only": true, "build-tag-only": true}

func PlanClassify(c Catalog, req ClassifyRequest) (Plan, error) {
	if strings.TrimSpace(req.Family) == "" || strings.TrimSpace(req.Token) == "" || strings.TrimSpace(req.Reason) == "" {
		return Plan{}, fmt.Errorf("family, token, and reason are required")
	}
	if !categories[req.Category] {
		return Plan{}, fmt.Errorf("category must be incidental, false-positive, test-only, or build-tag-only")
	}
	var meta struct {
		Schema   string           `json:"schema"`
		Glossary string           `json:"glossary"`
		Families []map[string]any `json:"families"`
	}
	p := filepath.Join(c.Dir, "_meta.json")
	b, err := os.ReadFile(p)
	if err != nil {
		return Plan{}, err
	}
	if err = json.Unmarshal(b, &meta); err != nil {
		return Plan{}, err
	}
	before := 0
	found := false
	for _, r := range c.Rows {
		if norm(r.Family) == norm(req.Family) {
			before++
		}
		if norm(r.Family) == norm(req.Family) && token(r.Grounding) == token(req.Token) {
			return Plan{}, fmt.Errorf("token %q grounds positioned concept %s and cannot be classified away", req.Token, r.ID)
		}
	}
	for _, f := range meta.Families {
		if norm(fmt.Sprint(f["id"])) != norm(req.Family) {
			continue
		}
		found = true
		key := "ignore"
		if req.Category == "false-positive" {
			key = "exclude"
		}
		vals := toStrings(f[key])
		for _, v := range vals {
			if token(v) == token(req.Token) {
				return Plan{}, fmt.Errorf("token %q is already classified", req.Token)
			}
		}
		vals = append(vals, req.Token)
		sort.Strings(vals)
		f[key] = vals
		notes, _ := f["classifications"].([]any)
		notes = append(notes, map[string]any{"token": req.Token, "category": req.Category, "reason": req.Reason})
		f["classifications"] = notes
	}
	if !found {
		return Plan{}, fmt.Errorf("unknown family %q", req.Family)
	}
	out, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return Plan{}, err
	}
	out = append(out, '\n')
	plan := Plan{Mode: "classify", Family: req.Family, BeforeFamilyCount: before, AfterFamilyCount: before, Files: []string{filepath.ToSlash(p)}, Changes: []Change{{Path: p, BeforeCount: before, AfterCount: before, Content: out}}}
	return AddGeneratedArtifacts(c, plan)
}
func toStrings(v any) []string {
	var out []string
	switch x := v.(type) {
	case []any:
		for _, v := range x {
			out = append(out, fmt.Sprint(v))
		}
	case []string:
		return append(out, x...)
	}
	return out
}

func Apply(plan Plan) error {
	type prepared struct {
		dst, tmp, backup   string
		existed, installed bool
	}
	var ps []prepared
	cleanup := func() {
		for _, p := range ps {
			_ = os.Remove(p.tmp)
			_ = os.Remove(p.backup)
		}
	}
	rollback := func() {
		for i := len(ps) - 1; i >= 0; i-- {
			p := &ps[i]
			if p.installed {
				_ = os.Remove(p.dst)
			}
			if p.existed {
				_ = os.Rename(p.backup, p.dst)
			}
		}
		cleanup()
	}
	for _, c := range plan.Changes {
		if err := os.MkdirAll(filepath.Dir(c.Path), 0755); err != nil {
			cleanup()
			return err
		}
		f, err := os.CreateTemp(filepath.Dir(c.Path), ".concept-*")
		if err != nil {
			cleanup()
			return err
		}
		tmp := f.Name()
		if _, err = f.Write(c.Content); err != nil {
			f.Close()
			cleanup()
			return err
		}
		if err = f.Close(); err != nil {
			cleanup()
			return err
		}
		_, statErr := os.Stat(c.Path)
		ps = append(ps, prepared{dst: c.Path, tmp: tmp, backup: c.Path + ".concept-backup", existed: statErr == nil})
	}
	for i := range ps {
		p := &ps[i]
		if p.existed {
			_ = os.Remove(p.backup)
			if err := os.Rename(p.dst, p.backup); err != nil {
				rollback()
				return err
			}
		}
		if err := os.Rename(p.tmp, p.dst); err != nil {
			rollback()
			return err
		}
		p.installed = true
	}
	cleanup()
	return nil
}

// ProductionCorpus reports whether a token is grounded outside tests and build-tag-only files.
func ProductionCorpus(root, raw string) (bool, error) {
	want := token(raw)
	if want == "" {
		return false, nil
	}
	skip := map[string]bool{".git": true, "vendor": true, "node_modules": true, "concept_disambiguation_scorecard.data": true}
	found := false
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if skip[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		n := filepath.ToSlash(p)
		if strings.HasSuffix(n, "_test.go") {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(n))
		if ext != ".go" && ext != ".md" && ext != ".json" && ext != ".yml" && ext != ".yaml" && ext != ".toml" {
			return nil
		}
		f, e := os.Open(p)
		if e != nil {
			return nil
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		first := true
		for sc.Scan() {
			line := sc.Text()
			if first && strings.HasPrefix(line, "//go:build") && (strings.Contains(line, "test") || strings.Contains(line, "wip_")) {
				return nil
			}
			first = false
			if strings.Contains(token(line), want) {
				found = true
				return filepath.SkipAll
			}
		}
		return nil
	})
	if err == filepath.SkipAll {
		err = nil
	}
	return found, err
}
