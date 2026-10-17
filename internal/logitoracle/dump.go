package logitoracle

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"time"
)

// DumpSchema names the dump manifest format.
const DumpSchema = "fak.logit-oracle.dump/v1"

const manifestFile = "manifest.json"

// Session is the native model surface a dump drives. Prefill and Step return
// the next-token logits after consuming their tokens.
type Session interface {
	Prefill(ids []int) []float32
	PrefillNoLogits(ids []int)
	Step(id int) []float32
}

// PromptSpec is one tokenized prompt. Positions is the number of logit vectors
// to record: the post-prefill vector plus Positions-1 decode steps.
type PromptSpec struct {
	Name      string `json:"name"`
	IDs       []int  `json:"ids,omitempty"`
	Text      string `json:"text,omitempty"`
	Positions int    `json:"positions,omitempty"`
}

// Top is the reference ranking at one position.
type Top struct {
	IDs    []int     `json:"ids"`
	Logits []float32 `json:"logits"`
	Pieces []string  `json:"pieces,omitempty"`
}

// PromptDump records one prompt's run. Position i holds the logits after the
// prompt plus Fed[0:i]; Fed[i] is the token consumed by decode step i+1.
type PromptDump struct {
	Name         string                      `json:"name"`
	PromptIDs    []int                       `json:"prompt_ids"`
	Positions    int                         `json:"positions"`
	Forced       bool                        `json:"forced"`
	Fed          []int                       `json:"fed"`
	Argmax       []int                       `json:"argmax"`
	Top          []Top                       `json:"top"`
	FedText      string                      `json:"fed_text,omitempty"`
	LogitsFile   string                      `json:"logits_file"`
	LogitsSHA256 string                      `json:"logits_sha256"`
	PrefillNS    int64                       `json:"prefill_ns"`
	DecodeNS     int64                       `json:"decode_ns"`
	Counters     map[string]map[string]int64 `json:"counters,omitempty"`
}

// Manifest describes one dump directory.
type Manifest struct {
	Schema     string            `json:"schema"`
	Label      string            `json:"label,omitempty"`
	CreatedUTC string            `json:"created_utc"`
	Vocab      int               `json:"vocab"`
	Teacher    string            `json:"teacher,omitempty"`
	Identity   map[string]string `json:"identity,omitempty"`
	Env        map[string]string `json:"env,omitempty"`
	Prompts    []PromptDump      `json:"prompts"`
	dir        string
}

// Dir is the directory the manifest was read from or written to.
func (m *Manifest) Dir() string { return m.dir }

// Prompt returns the named prompt record.
func (m *Manifest) Prompt(name string) (*PromptDump, error) {
	for i := range m.Prompts {
		if m.Prompts[i].Name == name {
			return &m.Prompts[i], nil
		}
	}
	return nil, fmt.Errorf("%w: %q", ErrMissingPrompt, name)
}

// DumpOptions configures a dump run.
type DumpOptions struct {
	Dir              string
	Label            string
	PrefillChunk     int
	DefaultPositions int
	TopK             int
	// Teacher, when set, fixes each prompt's ids, position count and fed tokens
	// so the candidate's logits are comparable at every position.
	Teacher    *Manifest
	NewSession func() (Session, func(), error)
	Decode     func(ids []int) string
	// Counters samples cumulative route counters; Dump records per-phase deltas.
	Counters func() map[string]int64
	Identity map[string]string
	Env      map[string]string
	Now      func() time.Time
}

var safeName = regexp.MustCompile(`[^A-Za-z0-9._-]+`)

// Dump runs every prompt in a fresh session and writes the manifest plus one
// little-endian float32 logits file per prompt into opts.Dir.
func Dump(prompts []PromptSpec, opts DumpOptions) (*Manifest, error) {
	if len(prompts) == 0 {
		return nil, ErrNoPrompts
	}
	if opts.NewSession == nil {
		return nil, fmt.Errorf("logitoracle: dump needs a session factory")
	}
	if opts.PrefillChunk <= 0 {
		opts.PrefillChunk = 512
	}
	if opts.DefaultPositions <= 0 {
		opts.DefaultPositions = 64
	}
	if opts.TopK <= 0 {
		opts.TopK = 5
	}
	now := opts.Now
	if now == nil {
		now = time.Now
	}
	if err := os.MkdirAll(opts.Dir, 0o755); err != nil {
		return nil, err
	}
	m := &Manifest{Schema: DumpSchema, Label: opts.Label, CreatedUTC: now().UTC().Format(time.RFC3339),
		Identity: opts.Identity, Env: opts.Env, dir: opts.Dir}
	if opts.Teacher != nil {
		m.Teacher = opts.Teacher.Label
		if m.Teacher == "" {
			m.Teacher = opts.Teacher.Dir()
		}
	}
	seen := map[string]bool{}
	for _, spec := range prompts {
		if spec.Name == "" || seen[spec.Name] {
			return nil, fmt.Errorf("logitoracle: prompt name %q empty or repeated", spec.Name)
		}
		seen[spec.Name] = true
		pd, vocab, err := dumpPrompt(spec, opts, len(m.Prompts))
		if err != nil {
			return nil, fmt.Errorf("prompt %q: %w", spec.Name, err)
		}
		if m.Vocab != 0 && vocab != m.Vocab {
			return nil, fmt.Errorf("%w: %d vs %d", ErrVocabMismatch, vocab, m.Vocab)
		}
		m.Vocab = vocab
		m.Prompts = append(m.Prompts, pd)
	}
	if err := m.Write(); err != nil {
		return nil, err
	}
	return m, nil
}

// Write stores the manifest as manifest.json in its directory.
func (m *Manifest) Write() error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(m.dir, manifestFile), append(data, '\n'), 0o644)
}

func dumpPrompt(spec PromptSpec, opts DumpOptions, index int) (PromptDump, int, error) {
	pd := PromptDump{Name: spec.Name, PromptIDs: append([]int(nil), spec.IDs...), Positions: spec.Positions}
	if pd.Positions <= 0 {
		pd.Positions = opts.DefaultPositions
	}
	var forced []int
	if opts.Teacher != nil {
		tp, err := opts.Teacher.Prompt(spec.Name)
		if err != nil {
			return pd, 0, err
		}
		if !equalInts(tp.PromptIDs, spec.IDs) {
			return pd, 0, ErrTeacherMismatch
		}
		pd.Positions, forced, pd.Forced = tp.Positions, tp.Fed, true
	}
	if len(pd.PromptIDs) == 0 {
		return pd, 0, fmt.Errorf("logitoracle: prompt has no token ids")
	}
	pd.LogitsFile = fmt.Sprintf("%02d-%s.f32", index, safeName.ReplaceAllString(spec.Name, "_"))
	f, err := os.Create(filepath.Join(opts.Dir, pd.LogitsFile))
	if err != nil {
		return pd, 0, err
	}
	defer f.Close()
	h := sha256.New()
	w := bufio.NewWriterSize(io.MultiWriter(f, h), 1<<20)
	sess, closeSession, err := opts.NewSession()
	if err != nil {
		return pd, 0, err
	}
	if closeSession != nil {
		defer closeSession()
	}
	sample := func() map[string]int64 {
		if opts.Counters == nil {
			return nil
		}
		return opts.Counters()
	}
	c0 := sample()
	started := time.Now()
	ids := pd.PromptIDs
	for len(ids) > opts.PrefillChunk {
		sess.PrefillNoLogits(ids[:opts.PrefillChunk])
		ids = ids[opts.PrefillChunk:]
	}
	logits := append([]float32(nil), sess.Prefill(ids)...)
	pd.PrefillNS = time.Since(started).Nanoseconds()
	c1 := sample()
	vocab := len(logits)
	if vocab == 0 {
		return pd, 0, ErrEmptyLogits
	}
	record := func() error {
		if len(logits) != vocab {
			return fmt.Errorf("%w: position %d has %d, want %d", ErrLengthMismatch, len(pd.Argmax), len(logits), vocab)
		}
		top := TopK(logits, opts.TopK)
		t := Top{IDs: top, Logits: make([]float32, len(top))}
		for i, id := range top {
			t.Logits[i] = logits[id]
			if opts.Decode != nil {
				t.Pieces = append(t.Pieces, opts.Decode([]int{id}))
			}
		}
		pd.Top = append(pd.Top, t)
		pd.Argmax = append(pd.Argmax, top[0])
		return binary.Write(w, binary.LittleEndian, logits)
	}
	if err := record(); err != nil {
		return pd, 0, err
	}
	started = time.Now()
	for pos := 1; pos < pd.Positions; pos++ {
		next := pd.Argmax[pos-1]
		if forced != nil {
			if pos-1 >= len(forced) {
				return pd, 0, ErrTeacherMismatch
			}
			next = forced[pos-1]
		}
		pd.Fed = append(pd.Fed, next)
		logits = append(logits[:0], sess.Step(next)...)
		if err := record(); err != nil {
			return pd, 0, err
		}
	}
	pd.DecodeNS = time.Since(started).Nanoseconds()
	c2 := sample()
	if err := w.Flush(); err != nil {
		return pd, 0, err
	}
	pd.LogitsSHA256 = hex.EncodeToString(h.Sum(nil))
	if opts.Counters != nil {
		pd.Counters = map[string]map[string]int64{"prefill": delta(c0, c1), "decode": delta(c1, c2)}
	}
	if opts.Decode != nil && len(pd.Fed) > 0 {
		pd.FedText = opts.Decode(pd.Fed)
	}
	return pd, vocab, f.Close()
}

func delta(a, b map[string]int64) map[string]int64 {
	out := make(map[string]int64, len(b))
	for k, v := range b {
		out[k] = v - a[k]
	}
	return out
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ReadManifest loads a dump manifest from its directory or manifest path.
func ReadManifest(path string) (*Manifest, error) {
	dir := path
	if info, err := os.Stat(path); err == nil && !info.IsDir() {
		dir = filepath.Dir(path)
	} else {
		path = filepath.Join(path, manifestFile)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	if m.Schema != DumpSchema {
		return nil, fmt.Errorf("%w: %q", ErrSchema, m.Schema)
	}
	m.dir = dir
	return &m, nil
}

// logitRows reads fixed-width float32 rows from a dump file.
type logitRows struct {
	f     *os.File
	r     *bufio.Reader
	vocab int
	raw   []byte
}

func openRows(dir string, pd *PromptDump, vocab int) (*logitRows, error) {
	f, err := os.Open(filepath.Join(dir, pd.LogitsFile))
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	if want := int64(pd.Positions) * int64(vocab) * 4; info.Size() != want {
		f.Close()
		return nil, fmt.Errorf("%w: %s holds %d bytes, want %d", ErrLengthMismatch, pd.LogitsFile, info.Size(), want)
	}
	return &logitRows{f: f, r: bufio.NewReaderSize(f, 1<<20), vocab: vocab, raw: make([]byte, vocab*4)}, nil
}

func (l *logitRows) next(dst []float32) error {
	if _, err := io.ReadFull(l.r, l.raw); err != nil {
		return err
	}
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(l.raw[4*i:]))
	}
	return nil
}

func (l *logitRows) close() error { return l.f.Close() }

// CompareManifests compares every reference prompt against the candidate
// prompt of the same name. Positions after the first fed-token difference are
// excluded: their inputs differ, so their logits are not comparable.
func CompareManifests(ref, cand *Manifest, th Thresholds) (Report, error) {
	if err := th.validate(); err != nil {
		return Report{}, err
	}
	if ref.Vocab != cand.Vocab || ref.Vocab <= 0 {
		return Report{}, fmt.Errorf("%w: ref=%d cand=%d", ErrVocabMismatch, ref.Vocab, cand.Vocab)
	}
	if len(ref.Prompts) == 0 {
		return Report{}, ErrNoPrompts
	}
	var prompts []PromptReport
	for i := range ref.Prompts {
		rp := &ref.Prompts[i]
		cp, err := cand.Prompt(rp.Name)
		if err != nil {
			return Report{}, err
		}
		pr, err := comparePrompt(ref, cand, rp, cp, th)
		if err != nil {
			return Report{}, fmt.Errorf("prompt %q: %w", rp.Name, err)
		}
		prompts = append(prompts, pr)
	}
	r := FoldReport(prompts, th)
	r.Reference, r.Candidate = ref.Label, cand.Label
	return r, nil
}

func comparePrompt(ref, cand *Manifest, rp, cp *PromptDump, th Thresholds) (PromptReport, error) {
	if !equalInts(rp.PromptIDs, cp.PromptIDs) {
		return PromptReport{}, ErrPromptMismatch
	}
	positions := min(rp.Positions, cp.Positions)
	comparable := positions
	var firstDiverge *int
	for j := 0; j+1 < positions && j < len(rp.Fed) && j < len(cp.Fed); j++ {
		if rp.Fed[j] != cp.Fed[j] {
			at := j
			firstDiverge = &at
			comparable = j + 1
			break
		}
	}
	rr, err := openRows(ref.dir, rp, ref.Vocab)
	if err != nil {
		return PromptReport{}, err
	}
	defer rr.close()
	cr, err := openRows(cand.dir, cp, cand.Vocab)
	if err != nil {
		return PromptReport{}, err
	}
	defer cr.close()
	acc := NewAccumulator(rp.Name)
	rv, cv := make([]float32, ref.Vocab), make([]float32, cand.Vocab)
	for pos := 0; pos < comparable; pos++ {
		if err := rr.next(rv); err != nil {
			return PromptReport{}, err
		}
		if err := cr.next(cv); err != nil {
			return PromptReport{}, err
		}
		st, err := ComparePosition(rv, cv, th.TopK, th.TieEpsilon)
		if err != nil {
			return PromptReport{}, fmt.Errorf("position %d: %w", pos, err)
		}
		st.Position = pos
		acc.Add(st)
	}
	pr := acc.Fold(max(rp.Positions, cp.Positions), max(rp.Positions, cp.Positions)-comparable, th)
	pr.FirstFedDiverge = firstDiverge
	return pr, nil
}
