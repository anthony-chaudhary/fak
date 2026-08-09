package conceptcatalog

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
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
	row := Row{ID: req.ID, Canonical: req.Canonical, Family: req.Family, Kind: req.Kind, Definition: req.Definition, Distinction: req.Distinction, DistinctFrom: req.DistinctFrom, Aliases: append([]string{}, req.Aliases...), Grounding: req.Grounding, GroundingKind: req.GroundingKind, GlossaryAnchor: req.Glossary, Verdict: "crystal", Gaps: []string{}}
	rowFile := req.RowFile
	if rowFile == "" {
		rowFile = "rows-" + norm(req.Family) + "-authored.json"
	}
	if !strings.HasPrefix(rowFile, "rows-") || !strings.HasSuffix(rowFile, ".json") {
		return Plan{}, fmt.Errorf("row-file must match rows-*.json")
	}
	path := filepath.Join(c.Dir, rowFile)

	// Draw the OTHER half of every declared boundary. A one-way distinct_from tells
	// the reader who arrives at the new concept how it differs and leaves the reader
	// who arrives at its twin holding both meanings, so the twin's own row gets the
	// back-reference here rather than as a follow-up nobody files.
	edited := map[string][]byte{}
	for _, ref := range req.DistinctFrom {
		twin, ok := rowByID(c, ref)
		if !ok {
			return Plan{}, fmt.Errorf("distinct_from %q is not an existing row ID", ref)
		}
		src, backed := twinRowFile(c, twin)
		if !backed {
			// The twin's row is owned by no data file - an in-memory catalog a caller
			// built rather than one LoadDir read - so there are no bytes to rewrite.
			// The new row still records its own half of the boundary.
			continue
		}
		b, ok := edited[src]
		if !ok {
			var readErr error
			if b, readErr = os.ReadFile(src); readErr != nil {
				return Plan{}, readErr
			}
		}
		nb, changed, err := addBackReference(b, twin.ID, req.ID)
		if err != nil {
			return Plan{}, fmt.Errorf("back-reference %s -> %s: %w", twin.ID, req.ID, err)
		}
		if changed {
			edited[src] = nb
		}
	}

	// The new row can land in a file a back-reference just rewrote; append to those
	// bytes, not to the stale ones on disk.
	existing, ok := edited[path]
	if !ok {
		b, readErr := os.ReadFile(path)
		if readErr != nil && !os.IsNotExist(readErr) {
			return Plan{}, readErr
		}
		existing = b
	}
	delete(edited, path)
	content, err := appendRow(existing, row)
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
	changes := []Change{{Path: path, BeforeCount: before, AfterCount: before + 1, Content: content}, {Path: glossaryPath, Content: gb}}
	for src, b := range edited {
		files = append(files, filepath.ToSlash(src))
		changes = append(changes, Change{Path: src, Content: b})
	}
	sort.Strings(files)
	sort.Slice(changes, func(i, j int) bool { return changes[i].Path < changes[j].Path })
	plan := Plan{Mode: "position", Family: req.Family, BeforeFamilyCount: before, AfterFamilyCount: before + 1, Files: files, Changes: changes}
	plan, snap, err := generateShadow(c, plan)
	if err != nil {
		return Plan{}, err
	}
	// The scorecard discovers confusable pairs from the names themselves, so a new
	// name can collide with a concept the author never thought of. Refuse the landing
	// with the list rather than admitting a name that reads as an existing one.
	if miss := snap.unseparatedFor(req.ID); len(miss) > 0 {
		var b strings.Builder
		fmt.Fprintf(&b, "%q is mistakable for %d concept(s) it does not separate from:", req.Canonical, len(miss))
		for _, p := range miss {
			other, _ := p.Other(req.ID)
			fmt.Fprintf(&b, "\n  %s - %s", other, p.Why)
		}
		b.WriteString("\n\nadd each to distinct_from (the other half of every boundary is written for you)")
		return Plan{}, errors.New(b.String())
	}
	return plan, nil
}

// twinRowFile resolves the data file whose bytes own a row, reporting false when
// no file owns it. LoadDir stamps every row it reads with its shard, so a row
// without one came from a catalog assembled in memory. Re-anchoring an empty
// source on the catalog directory used to yield that directory itself - the
// corpus is a directory of rows-*.json shards, not one file - and reading it
// failed with a path that named the whole corpus.
func twinRowFile(c Catalog, twin Row) (string, bool) {
	src := filepath.FromSlash(strings.TrimSpace(twin.Source))
	if src == "" {
		return "", false
	}
	if st, err := os.Stat(src); err == nil && !st.IsDir() {
		return src, true
	}
	// A catalog loaded through a relative root records a relative source, so the
	// shard's own name is re-anchored on this catalog's directory.
	base := filepath.Base(src)
	if base == "." || base == string(filepath.Separator) {
		return "", false
	}
	return filepath.Join(c.Dir, base), true
}

func rowByID(c Catalog, id string) (Row, bool) {
	for _, r := range c.Rows {
		if norm(r.ID) == norm(id) {
			return r, true
		}
	}
	return Row{}, false
}

// appendRow adds one row to a data file's bytes, or writes a fresh file when the
// existing bytes are empty.
func appendRow(b []byte, row Row) ([]byte, error) {
	if len(b) == 0 {
		x := struct {
			Rows []Row `json:"rows"`
		}{Rows: []Row{row}}
		out, e := json.MarshalIndent(x, "", "  ")
		if e != nil {
			return nil, e
		}
		return append(out, '\n'), nil
	}
	var check struct {
		Rows []Row `json:"rows"`
	}
	if err := json.Unmarshal(b, &check); err != nil {
		return nil, err
	}
	// Preserve every existing byte. Insert one indented object immediately before
	// the rows array's closing bracket instead of marshal-rewriting the file.
	closeObj := bytes.LastIndex(b, []byte("}"))
	if closeObj < 0 {
		return nil, errors.New("data file: missing object close")
	}
	closeRows := bytes.LastIndex(b[:closeObj], []byte("]"))
	if closeRows < 0 {
		return nil, errors.New("data file: missing rows close")
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
	p, _, err := generateShadow(c, plan)
	return p, err
}

// generateShadow also returns the snapshot the planned catalog grades to, so a
// caller can refuse a mutation on what the generator SAW rather than on a second,
// drifting copy of the same rule.
func generateShadow(c Catalog, plan Plan) (Plan, shadowSnapshot, error) {
	var snap shadowSnapshot
	root := filepath.Dir(filepath.Dir(c.Dir))
	if _, err := os.Stat(filepath.Join(root, "tools", "concept_disambiguation_scorecard.py")); os.IsNotExist(err) {
		return plan, snap, nil
	}
	script := filepath.Join(root, "tools", "concept_disambiguation_scorecard.py")
	if _, err := os.Stat(script); err != nil {
		return Plan{}, snap, fmt.Errorf("canonical generator: %w", err)
	}
	shadow, err := os.MkdirTemp("", "fak-concept-data-*")
	if err != nil {
		return Plan{}, snap, err
	}
	defer os.RemoveAll(shadow)
	entries, err := os.ReadDir(c.Dir)
	if err != nil {
		return Plan{}, snap, err
	}
	proposed := map[string][]byte{}
	for _, ch := range plan.Changes {
		proposed[filepath.Clean(ch.Path)] = ch.Content
	}
	written := map[string]bool{}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		src := filepath.Join(c.Dir, e.Name())
		b, er := os.ReadFile(src)
		if er != nil {
			return Plan{}, snap, er
		}
		clean := filepath.Clean(src)
		if v, ok := proposed[clean]; ok {
			b = v
		}
		written[clean] = true
		if er = os.WriteFile(filepath.Join(shadow, e.Name()), b, 0600); er != nil {
			return Plan{}, snap, er
		}
	}
	// A position commonly creates rows-<family>-authored.json. Materialize
	// planned new data files as well as replacements before generation.
	for src, b := range proposed {
		if written[src] || filepath.Dir(src) != filepath.Clean(c.Dir) {
			continue
		}
		if writeErr := os.WriteFile(filepath.Join(shadow, filepath.Base(src)), b, 0600); writeErr != nil {
			return Plan{}, snap, writeErr
		}
	}
	outDir := filepath.Join(shadow, "generated")
	cmd := exec.Command("python", script, "--workspace", root, "--data", shadow, "--markdown-dir", outDir, "--json")
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	// Keep stderr off stdout: stdout is the snapshot JSON this plan is judged on.
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	b, er := cmd.Output()
	if er != nil {
		// The scorecard exits 1 for an honestly generated ACTION snapshot (for
		// example, unrelated families still carry coverage debt). Exit 2 and
		// missing output are generator failures; exit 1 is valid content.
		var exitErr *exec.ExitError
		if !errors.As(er, &exitErr) || exitErr.ExitCode() != 1 {
			return Plan{}, snap, fmt.Errorf("canonical generation failed: %v: %s", er, strings.TrimSpace(stderr.String()))
		}
	}
	if err = json.Unmarshal(b, &snap); err != nil {
		// An exit-1 with EMPTY stdout is a generator crash, not an ACTION snapshot. The
		// branch above deliberately admits exit 1 as valid content, so a Python traceback
		// — which also exits 1 — arrives here as a bare "unexpected end of JSON input"
		// with the only diagnosis, stderr, already discarded. Name the real failure.
		if len(bytes.TrimSpace(b)) == 0 {
			return Plan{}, snap, fmt.Errorf("canonical generator wrote no snapshot (exit ok, empty stdout): %s", strings.TrimSpace(stderr.String()))
		}
		return Plan{}, snap, fmt.Errorf("decode planned snapshot: %w", err)
	}
	// Every generated artifact ages with the catalog: a fresh scorecard beside a
	// stale name index would answer one of the two questions from a retired catalog.
	for _, art := range generatedArtifacts {
		content, readErr := os.ReadFile(filepath.Join(outDir, art.Name))
		if readErr != nil {
			return Plan{}, snap, readErr
		}
		dst := filepath.Join(root, filepath.FromSlash(art.Tracked))
		plan.Changes = append(plan.Changes, Change{Path: dst, Content: content})
		plan.Files = append(plan.Files, filepath.ToSlash(dst))
	}
	sort.Strings(plan.Files)
	return plan, snap, nil
}

var categories = map[string]bool{"incidental": true, "false-positive": true, "test-only": true, "build-tag-only": true}

func PlanClassify(c Catalog, req ClassifyRequest) (Plan, error) {
	if strings.TrimSpace(req.Family) == "" || strings.TrimSpace(req.Token) == "" || strings.TrimSpace(req.Reason) == "" {
		return Plan{}, fmt.Errorf("family, token, and reason are required")
	}
	if !categories[req.Category] {
		return Plan{}, fmt.Errorf("category must be incidental, false-positive, test-only, or build-tag-only")
	}
	// Every top-level key of _meta.json must be declared here. This struct is not a
	// projection for reading — the file is DECODED into it and RE-ENCODED from it, so a
	// key that is absent from the struct is silently deleted from the file a classify
	// writes. "meta" carries {as_of, fak_version}, the dating block the canonical
	// generator REFUSES to render a scorecard without, so dropping it turned every
	// `fak concept classify` into a generator crash. Held verbatim as RawMessage rather
	// than a typed struct: this code has no business reshaping a block it only carries.
	var meta struct {
		Schema   string           `json:"schema"`
		Meta     json.RawMessage  `json:"meta,omitempty"`
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
	// Encode rather than MarshalIndent: Marshal escapes <, > and & into <-style
	// sequences, and this file is full of prose reasons that legitimately contain them
	// ("path helper naming <git-common-dir>/fak/token-cache"). Escaping is valid JSON but
	// it rewrites human-authored text into an unreadable form on every classify, so a
	// two-token edit would arrive as a corpus-wide diff nobody can review.
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err = enc.Encode(meta); err != nil {
		return Plan{}, err
	}
	out := buf.Bytes() // Encode already terminates with a newline.
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
	found, err := ProductionCorpusMany(root, []string{raw})
	return found[token(raw)], err
}

// ProductionCorpusMany resolves many grounding tokens with one bounded tree walk.
// A per-row walk made current-catalog validation scale as rows times corpus size.
func ProductionCorpusMany(root string, raw []string) (map[string]bool, error) {
	wants := map[string]bool{}
	for _, v := range raw {
		if n := token(v); n != "" {
			wants[n] = true
		}
	}
	found := map[string]bool{}
	if len(wants) == 0 {
		return found, nil
	}
	skip := map[string]bool{".git": true, "vendor": true, "node_modules": true, "concept_disambiguation_scorecard.data": true}
	var corpus strings.Builder
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
			corpus.WriteString(token(line))
			corpus.WriteByte('\n')
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	body := corpus.String()
	for want := range wants {
		found[want] = strings.Contains(body, want)
	}
	return found, nil
}
