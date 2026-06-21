// Package swebench turns SWE-bench Verified into a fak-native benchmark whose
// results are directly comparable to the external "N-Server Cache Benchmarking
// Tool" (the Benchmark repo, "bench") that runs the same task set against an
// SGLang endpoint. The point is to compare on the metrics fak is built to move:
// KV-cache reuse / prefill elimination (the value stack), turns + tokens (the
// turn-tax), in-process adjudication cost, and — where a capable model + the
// official Docker harness are present — resolve-rate + safety.
//
// This file is the dataset spine: the SWE-bench Verified instance type (field
// tags match the official princeton-nlp/SWE-bench_Verified schema so a HF row
// unmarshals directly) plus loaders for both the full dataset and bench's local
// ID-list / difficulty artifacts. Geometry, cost arms, eval, and reporting build
// on top of this — see the package's other files.
package swebench

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

// Instance is one SWE-bench Verified task. The json tags match the official
// princeton-nlp/SWE-bench_Verified row schema, so a line of the HF dataset
// (JSONL) — or a row of the test split exported to JSON — unmarshals straight
// into this struct. Bench's difficulty map supplies Difficulty, which is not a
// HF field; it is overlaid by MergeDifficulty.
type Instance struct {
	InstanceID             string `json:"instance_id"`
	Repo                   string `json:"repo"`                   // "owner/name", e.g. "django/django"
	BaseCommit             string `json:"base_commit"`            // commit the agent starts from
	ProblemStatement       string `json:"problem_statement"`      // the issue text the agent must solve
	Hints                  string `json:"hints_text,omitempty"`   // optional maintainer hints
	Patch                  string `json:"patch,omitempty"`        // gold patch (the fix) — never shown to the agent
	TestPatch              string `json:"test_patch,omitempty"`   // test diff applied at grade time
	Version                string `json:"version,omitempty"`      // repo version (selects the Docker image)
	FailToPass             string `json:"FAIL_TO_PASS,omitempty"` // JSON-encoded []string: tests that must go fail->pass
	PassToPass             string `json:"PASS_TO_PASS,omitempty"` // JSON-encoded []string: tests that must stay passing
	EnvironmentSetupCommit string `json:"environment_setup_commit,omitempty"`
	CreatedAt              string `json:"created_at,omitempty"`

	// Difficulty is bench's official annotation bucket ("<15min", "15min-1hr",
	// "1-4hr", ">4hr"). Populated from data/swebench_verified_difficulty.json,
	// not from the HF row. Empty when unknown.
	Difficulty string `json:"difficulty,omitempty"`
}

// Org, Name, and Number decompose an instance_id of the form
// "<org>__<name>-<number>" (the SWE-bench convention: the repo full name with
// "/" replaced by "__", then "-<issue/PR number>"). Org/name handle hyphenated
// repos ("pylint-dev__pylint-4551" -> pylint-dev / pylint / 4551;
// "scikit-learn__scikit-learn-10297" -> scikit-learn / scikit-learn / 10297).
func (in Instance) Org() string  { org, _, _ := splitInstanceID(in.InstanceID); return org }
func (in Instance) Name() string { _, name, _ := splitInstanceID(in.InstanceID); return name }
func (in Instance) Number() string {
	_, _, num := splitInstanceID(in.InstanceID)
	return num
}

// RepoFull returns the "owner/name" repo. It prefers the explicit Repo field
// (present on full HF rows) and falls back to reconstructing it from the
// instance_id (the only source available for bench's ID-list/difficulty files).
func (in Instance) RepoFull() string {
	if in.Repo != "" {
		return in.Repo
	}
	org, name, _ := splitInstanceID(in.InstanceID)
	if org == "" || name == "" {
		return ""
	}
	return org + "/" + name
}

// splitInstanceID parses "<org>__<name>-<number>". The "<org>__<name>" split is
// on the FIRST "__" (orgs never contain "__"); "<name>-<number>" splits on the
// LAST "-" (the number is the trailing run of digits). Returns empty strings on
// a shape that does not match.
func splitInstanceID(id string) (org, name, number string) {
	i := strings.Index(id, "__")
	if i < 0 {
		return "", "", ""
	}
	org = id[:i]
	rest := id[i+2:]
	j := strings.LastIndex(rest, "-")
	if j < 0 {
		return org, rest, ""
	}
	return org, rest[:j], rest[j+1:]
}

// FailToPassList / PassToPassList decode the JSON-encoded test-name arrays. The
// official dataset stores these as a JSON string ("[\"test_a\", \"test_b\"]");
// a tolerant decoder also accepts an already-decoded []string (some exports do
// this) and a newline/space-separated fallback.
func (in Instance) FailToPassList() []string { return decodeTestList(in.FailToPass) }
func (in Instance) PassToPassList() []string { return decodeTestList(in.PassToPass) }

func decodeTestList(s string) []string {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	if strings.HasPrefix(s, "[") {
		var out []string
		if err := json.Unmarshal([]byte(s), &out); err == nil {
			return out
		}
	}
	return strings.FieldsFunc(s, func(r rune) bool { return r == '\n' || r == ' ' || r == ',' })
}

// Dataset is an ordered, ID-keyed collection of instances. Order is preserved
// from the load source so a run is reproducible; ById gives O(1) overlay.
type Dataset struct {
	Instances []Instance
	byID      map[string]int
}

// NewDataset indexes a slice of instances (last write wins on a duplicate id).
func NewDataset(insts []Instance) *Dataset {
	d := &Dataset{byID: make(map[string]int, len(insts))}
	for _, in := range insts {
		if j, ok := d.byID[in.InstanceID]; ok {
			d.Instances[j] = in
			continue
		}
		d.byID[in.InstanceID] = len(d.Instances)
		d.Instances = append(d.Instances, in)
	}
	return d
}

func (d *Dataset) Len() int { return len(d.Instances) }

// Get returns the instance with id and whether it was present.
func (d *Dataset) Get(id string) (Instance, bool) {
	if j, ok := d.byID[id]; ok {
		return d.Instances[j], true
	}
	return Instance{}, false
}

// LoadDataset reads a full SWE-bench Verified dataset. It accepts both JSONL
// (one instance object per line — the HF `datasets` export format) and a single
// JSON array of instances, auto-detected by the first non-space byte. This is
// the path used on a box where the real dataset is present (a download, or the
// DGX), giving real problem-statement / patch token geometry.
func LoadDataset(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	br := bufio.NewReader(f)
	first, err := peekFirstNonSpace(br)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	var insts []Instance
	if first == '[' {
		if err := json.NewDecoder(br).Decode(&insts); err != nil {
			return nil, fmt.Errorf("%s: JSON array decode: %w", path, err)
		}
	} else {
		sc := bufio.NewScanner(br)
		sc.Buffer(make([]byte, 0, 1<<20), 1<<26) // SWE-bench rows can be large (patches)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			var in Instance
			if err := json.Unmarshal([]byte(line), &in); err != nil {
				return nil, fmt.Errorf("%s: JSONL decode: %w", path, err)
			}
			insts = append(insts, in)
		}
		if err := sc.Err(); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
	}
	return NewDataset(insts), nil
}

// difficultyFile is bench's data/swebench_verified_difficulty.json: a flat map
// of instance_id -> difficulty bucket, plus a "_meta" object that is not an
// instance. It is the richest fully-offline artifact on this box — all 500 ids
// with the official difficulty annotation — so it is the default geometry source
// when the full dataset is absent.
type difficultyMeta struct {
	SourceDataset  string         `json:"_source_dataset"`
	SourceSplit    string         `json:"_source_split"`
	SnapshotDate   string         `json:"_snapshot_date"`
	TotalInstances int            `json:"_total_instances"`
	BucketCounts   map[string]int `json:"_bucket_counts"`
}

// LoadDifficulty reads bench's difficulty map into a Dataset of skeleton
// instances (InstanceID + Repo reconstructed from the id + Difficulty). No
// problem statement or patch is available from this file; geometry derived from
// it is bucket-driven and is flagged as such in any report. Returns the dataset
// and the parsed _meta.
func LoadDifficulty(path string) (*Dataset, difficultyMeta, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, difficultyMeta{}, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, difficultyMeta{}, fmt.Errorf("%s: %w", path, err)
	}
	var meta difficultyMeta
	if m, ok := raw["_meta"]; ok {
		_ = json.Unmarshal(m, &meta)
	}
	ids := make([]string, 0, len(raw))
	for id := range raw {
		if id == "_meta" {
			continue
		}
		ids = append(ids, id)
	}
	sort.Strings(ids) // deterministic order independent of map iteration
	insts := make([]Instance, 0, len(ids))
	for _, id := range ids {
		var diff string
		_ = json.Unmarshal(raw[id], &diff)
		in := Instance{InstanceID: id, Difficulty: diff}
		in.Repo = in.RepoFull()
		insts = append(insts, in)
	}
	return NewDataset(insts), meta, nil
}

// idList is bench's {_meta, instance_ids} smoke-selection format
// (swebench_verified_l3_smoke.json and friends).
type idList struct {
	InstanceIDs []string `json:"instance_ids"`
}

// LoadIDList reads a bench ID-selection file into skeleton instances (id + repo
// only). Use MergeDifficulty / a full-dataset overlay to enrich.
func LoadIDList(path string) (*Dataset, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var l idList
	if err := json.Unmarshal(b, &l); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	insts := make([]Instance, 0, len(l.InstanceIDs))
	for _, id := range l.InstanceIDs {
		in := Instance{InstanceID: id}
		in.Repo = in.RepoFull()
		insts = append(insts, in)
	}
	return NewDataset(insts), nil
}

// MergeDifficulty overlays the Difficulty bucket from src onto every matching
// instance in d (by instance_id). Instances with no match are left unchanged.
// Returns the number of instances annotated.
func (d *Dataset) MergeDifficulty(src *Dataset) int {
	n := 0
	for i := range d.Instances {
		if s, ok := src.Get(d.Instances[i].InstanceID); ok && s.Difficulty != "" {
			d.Instances[i].Difficulty = s.Difficulty
			n++
		}
	}
	return n
}

// Filter returns a new Dataset of instances for which keep returns true,
// preserving order.
func (d *Dataset) Filter(keep func(Instance) bool) *Dataset {
	var out []Instance
	for _, in := range d.Instances {
		if keep(in) {
			out = append(out, in)
		}
	}
	return NewDataset(out)
}

// Limit returns the first n instances (n<=0 means all).
func (d *Dataset) Limit(n int) *Dataset {
	if n <= 0 || n >= len(d.Instances) {
		return d
	}
	return NewDataset(append([]Instance(nil), d.Instances[:n]...))
}

func peekFirstNonSpace(br *bufio.Reader) (byte, error) {
	for {
		b, err := br.ReadByte()
		if err != nil {
			return 0, err
		}
		switch b {
		case ' ', '\t', '\r', '\n':
			continue
		default:
			return b, br.UnreadByte()
		}
	}
}
