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
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type PositionRequest struct {
	ID, Canonical, Family, Definition, Distinction, Kind, Grounding, GroundingKind, Glossary string
	DistinctFrom, Aliases                                                                    []string
	RowFile                                                                                  string
}
type ClassifyRequest struct{ Family, Token, Category, Reason string }

type ClassifyResult struct {
	Family   string `json:"family"`
	Token    string `json:"token"`
	Category string `json:"category"`
	Changed  bool   `json:"changed"`
}

type PhaseTimings struct {
	Load     time.Duration `json:"load"`
	Validate time.Duration `json:"validate"`
	Render   time.Duration `json:"render"`
	Total    time.Duration `json:"total"`
}
type Change struct {
	Path        string `json:"path"`
	BeforeCount int    `json:"before_count,omitempty"`
	AfterCount  int    `json:"after_count,omitempty"`
	Content     []byte `json:"-"`
}
type Plan struct {
	Mode              string           `json:"mode"`
	Family            string           `json:"family"`
	BeforeFamilyCount int              `json:"before_family_count"`
	AfterFamilyCount  int              `json:"after_family_count"`
	Files             []string         `json:"files"`
	Changes           []Change         `json:"-"`
	Classifications   []ClassifyResult `json:"classifications,omitempty"`
	Timings           PhaseTimings     `json:"timings,omitempty"`
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
	python, err := ResolvePython()
	if err != nil {
		return Plan{}, snap, fmt.Errorf("canonical generation unsupported: %w", err)
	}
	cmd := exec.Command(python, script, "--workspace", root, "--data", shadow, "--markdown-dir", outDir, "--json")
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
	return PlanClassifyMany(c, []ClassifyRequest{req})
}

// PlanClassifyMany validates and applies a complete classification batch in memory,
// then renders committed derivatives once. No destination is touched until Apply.
func PlanClassifyMany(c Catalog, reqs []ClassifyRequest) (Plan, error) {
	started := time.Now()
	if len(reqs) == 0 {
		return Plan{}, fmt.Errorf("at least one classification row is required")
	}
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
	loaded := time.Now()

	families := make(map[string]map[string]any, len(meta.Families))
	for _, f := range meta.Families {
		families[norm(fmt.Sprint(f["id"]))] = f
	}
	seen := make(map[string]ClassifyRequest, len(reqs))
	results := make([]ClassifyResult, 0, len(reqs))
	for _, req := range reqs {
		req.Family, req.Token, req.Category, req.Reason = strings.TrimSpace(req.Family), strings.TrimSpace(req.Token), strings.TrimSpace(req.Category), strings.TrimSpace(req.Reason)
		if req.Family == "" || req.Token == "" || req.Reason == "" {
			return Plan{}, fmt.Errorf("family, token, and reason are required")
		}
		if !categories[req.Category] {
			return Plan{}, fmt.Errorf("category must be incidental, false-positive, test-only, or build-tag-only")
		}
		f := families[norm(req.Family)]
		if f == nil {
			return Plan{}, fmt.Errorf("unknown family %q", req.Family)
		}
		for _, row := range c.Rows {
			if norm(row.Family) == norm(req.Family) && token(row.Grounding) == token(req.Token) {
				return Plan{}, fmt.Errorf("token %q grounds positioned concept %s and cannot be classified away", req.Token, row.ID)
			}
		}
		key := norm(req.Family) + "\x00" + token(req.Token)
		if prior, ok := seen[key]; ok {
			if prior.Category != req.Category || prior.Reason != req.Reason {
				return Plan{}, fmt.Errorf("conflicting classifications for %s/%s", req.Family, req.Token)
			}
			continue
		}
		seen[key] = req
		bucket := "ignore"
		if req.Category == "false-positive" {
			bucket = "exclude"
		}
		already := false
		for _, v := range toStrings(f[bucket]) {
			if token(v) == token(req.Token) {
				already = true
				break
			}
		}
		if already {
			exact := false
			if notes, ok := f["classifications"].([]any); ok {
				for _, n := range notes {
					if m, ok := n.(map[string]any); ok && token(fmt.Sprint(m["token"])) == token(req.Token) && fmt.Sprint(m["category"]) == req.Category && fmt.Sprint(m["reason"]) == req.Reason {
						exact = true
					}
				}
			}
			if !exact {
				return Plan{}, fmt.Errorf("token %q already has a conflicting classification in family %q", req.Token, req.Family)
			}
			results = append(results, ClassifyResult{Family: req.Family, Token: req.Token, Category: req.Category})
			continue
		}
		vals := append(toStrings(f[bucket]), req.Token)
		sort.Strings(vals)
		f[bucket] = vals
		notes, _ := f["classifications"].([]any)
		f["classifications"] = append(notes, map[string]any{"token": req.Token, "category": req.Category, "reason": req.Reason})
		results = append(results, ClassifyResult{Family: req.Family, Token: req.Token, Category: req.Category, Changed: true})
	}
	validated := time.Now()
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err = enc.Encode(meta); err != nil {
		return Plan{}, err
	}
	plan := Plan{Mode: "classify", Classifications: results}
	if !bytes.Equal(b, buf.Bytes()) {
		plan.Files = []string{filepath.ToSlash(p)}
		plan.Changes = []Change{{Path: p, Content: buf.Bytes()}}
	}
	plan, err = addClassifyGeneratedArtifacts(c, plan)
	if err != nil {
		return Plan{}, err
	}
	// An idempotent rerun emits no writes, including byte-identical derivatives.
	kept := plan.Changes[:0]
	files := plan.Files[:0]
	for _, ch := range plan.Changes {
		old, readErr := os.ReadFile(ch.Path)
		if readErr == nil && bytes.Equal(old, ch.Content) {
			continue
		}
		kept = append(kept, ch)
		files = append(files, filepath.ToSlash(ch.Path))
	}
	plan.Changes, plan.Files = kept, files
	done := time.Now()
	plan.Timings = PhaseTimings{Load: loaded.Sub(started), Validate: validated.Sub(loaded), Render: done.Sub(validated), Total: done.Sub(started)}
	return plan, nil
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

// corpusSkipDir reports whether the grounding walk refuses to descend a directory.
// Excluding run artifacts is a correctness rule before it is a cost one: a token that
// survives only in a stale dispatch log or a build temp tree is not grounded in
// production source, yet the unfiltered walk let 48-day-old scratch output vouch for a
// concept. The "." / "_" prefix is the same convention the go tool uses to ignore a
// directory, and it is where every run-artifact tree in this repo lives
// (.dispatch-runs, _scratch, .scratch-*, .goal-runs).
func corpusSkipDir(name string) bool {
	switch name {
	case ".git", "vendor", "node_modules", "concept_disambiguation_scorecard.data":
		return true
	case ".github", ".claude":
		// Dot-prefixed but tracked production configuration, not run output.
		return false
	}
	return strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_")
}

// ProductionCorpusMany resolves many grounding tokens with one bounded tree walk.
// A per-row walk made current-catalog validation scale as rows times corpus size.
//
// The walk holds one file's tokenized text at a time and stops as soon as every token
// is grounded. Concatenating the whole tree into a single buffer instead made
// `fak concept position` peak at 20.79GB RSS over 112s on this repo, because ~98% of
// the bytes it matched came from in-tree run artifacts rather than source. Matching
// per file is exactly equivalent to matching the concatenation: token() keeps only
// [a-z0-9], so no want can span the "\n" that separates two tokenized lines.
func ProductionCorpusMany(root string, raw []string) (map[string]bool, error) {
	type want struct {
		norm  string
		bytes []byte
	}
	found := map[string]bool{}
	var wants []want
	for _, v := range raw {
		n := token(v)
		if n == "" {
			continue
		}
		if _, dup := found[n]; dup {
			continue
		}
		found[n] = false
		wants = append(wants, want{norm: n, bytes: []byte(n)})
	}
	if len(wants) == 0 {
		return found, nil
	}
	remaining := len(wants)
	var body []byte
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if remaining == 0 {
			return filepath.SkipAll
		}
		if d.IsDir() {
			if p != root && corpusSkipDir(d.Name()) {
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
		body = body[:0]
		sc := bufio.NewScanner(f)
		first := true
		for sc.Scan() {
			line := sc.Text()
			if first && strings.HasPrefix(line, "//go:build") && (strings.Contains(line, "test") || strings.Contains(line, "wip_")) {
				return nil
			}
			first = false
			body = append(body, token(line)...)
			body = append(body, '\n')
		}
		for _, w := range wants {
			if !found[w.norm] && bytes.Contains(body, w.bytes) {
				found[w.norm] = true
				remaining--
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return found, nil
}
