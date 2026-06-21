package swebench

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSplitInstanceID(t *testing.T) {
	cases := []struct{ id, org, name, num string }{
		{"django__django-12345", "django", "django", "12345"},
		{"pylint-dev__pylint-4551", "pylint-dev", "pylint", "4551"},
		{"scikit-learn__scikit-learn-10297", "scikit-learn", "scikit-learn", "10297"},
		{"astropy__astropy-12907", "astropy", "astropy", "12907"},
		{"mwaskom__seaborn-3069", "mwaskom", "seaborn", "3069"},
		{"sphinx-doc__sphinx-7440", "sphinx-doc", "sphinx", "7440"},
		{"garbage", "", "", ""},
	}
	for _, c := range cases {
		org, name, num := splitInstanceID(c.id)
		if org != c.org || name != c.name || num != c.num {
			t.Errorf("%s: got (%q,%q,%q) want (%q,%q,%q)", c.id, org, name, num, c.org, c.name, c.num)
		}
	}
}

func TestRepoFull(t *testing.T) {
	// reconstructed from id
	in := Instance{InstanceID: "scikit-learn__scikit-learn-10297"}
	if got := in.RepoFull(); got != "scikit-learn/scikit-learn" {
		t.Errorf("RepoFull from id = %q", got)
	}
	// explicit Repo wins
	in2 := Instance{InstanceID: "x__y-1", Repo: "real/repo"}
	if got := in2.RepoFull(); got != "real/repo" {
		t.Errorf("RepoFull explicit = %q", got)
	}
}

func TestDecodeTestList(t *testing.T) {
	in := Instance{
		FailToPass: `["tests/test_a.py::test_one", "tests/test_b.py::test_two"]`,
		PassToPass: "",
	}
	got := in.FailToPassList()
	if len(got) != 2 || got[0] != "tests/test_a.py::test_one" {
		t.Errorf("FailToPassList = %v", got)
	}
	if in.PassToPassList() != nil {
		t.Errorf("empty PassToPass should decode to nil")
	}
}

func TestLoadDatasetJSONLAndArray(t *testing.T) {
	dir := t.TempDir()
	jsonl := filepath.Join(dir, "d.jsonl")
	os.WriteFile(jsonl, []byte(
		`{"instance_id":"django__django-1","repo":"django/django","problem_statement":"fix it","FAIL_TO_PASS":"[\"t::a\"]"}`+"\n"+
			`{"instance_id":"django__django-2","repo":"django/django","problem_statement":"again"}`+"\n"), 0o644)
	d, err := LoadDataset(jsonl)
	if err != nil {
		t.Fatalf("LoadDataset jsonl: %v", err)
	}
	if d.Len() != 2 {
		t.Fatalf("jsonl len = %d", d.Len())
	}
	one, ok := d.Get("django__django-1")
	if !ok || one.ProblemStatement != "fix it" || len(one.FailToPassList()) != 1 {
		t.Errorf("jsonl instance wrong: %+v", one)
	}

	arr := filepath.Join(dir, "d.json")
	os.WriteFile(arr, []byte(`[{"instance_id":"a__b-1","repo":"a/b"}]`), 0o644)
	d2, err := LoadDataset(arr)
	if err != nil {
		t.Fatalf("LoadDataset array: %v", err)
	}
	if d2.Len() != 1 {
		t.Fatalf("array len = %d", d2.Len())
	}
}

// TestLoadDifficultyLocal exercises the real bench difficulty file when its
// path is supplied via FAK_SWEBENCH_DIFFICULTY; it skips cleanly everywhere else
// (CI / a clean checkout / any other machine) so the test stays hermetic and no
// developer-home path is baked into tracked source (issue #180).
func TestLoadDifficultyLocal(t *testing.T) {
	path := os.Getenv("FAK_SWEBENCH_DIFFICULTY")
	if path == "" {
		t.Skip("set FAK_SWEBENCH_DIFFICULTY to the SWE-bench Verified difficulty file to run this test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("FAK_SWEBENCH_DIFFICULTY=%s not present: %v", path, err)
	}
	d, meta, err := LoadDifficulty(path)
	if err != nil {
		t.Fatalf("LoadDifficulty: %v", err)
	}
	if d.Len() != 500 {
		t.Errorf("expected 500 SWE-bench Verified instances, got %d", d.Len())
	}
	if meta.TotalInstances != 500 {
		t.Errorf("_meta total = %d", meta.TotalInstances)
	}
	// spot-check a known instance got a difficulty + reconstructed repo
	in, ok := d.Get("astropy__astropy-12907")
	if !ok || in.Difficulty == "" || in.Repo != "astropy/astropy" {
		t.Errorf("astropy instance wrong: %+v", in)
	}
}

func TestMergeDifficulty(t *testing.T) {
	full := NewDataset([]Instance{{InstanceID: "a__b-1", ProblemStatement: "x"}})
	diff := NewDataset([]Instance{{InstanceID: "a__b-1", Difficulty: "1-4hr"}})
	if n := full.MergeDifficulty(diff); n != 1 {
		t.Fatalf("merged %d", n)
	}
	in, _ := full.Get("a__b-1")
	if in.Difficulty != "1-4hr" || in.ProblemStatement != "x" {
		t.Errorf("merge wrong: %+v", in)
	}
}
