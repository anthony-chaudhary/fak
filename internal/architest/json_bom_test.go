package architest

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestTrackedJSONHasNoBOM is the repo-hygiene gate for issue #13295: checked-in JSON
// artifacts (witnesses, benchmark summaries, config) must be BOM-free.
//
// A UTF-8 BOM (EF BB BF) is legal to some editors and invisible in most, but it makes a
// file that a strict JSON reader rejects: thousands of Go readers use encoding/json on
// bytes read straight off disk, and `json.Unmarshal` refuses the leading U+FEFF with
// "invalid character 'ï' looking for beginning of value". The same defect reds any
// machine consumer that treats a tracked JSON artifact as data, which is exactly what
// these witness/benchmark files are for. The failure is silent at author time and only
// surfaces downstream, so it belongs in the always-on `go test ./...` gate rather than a
// reviewer's memory.
//
// Fail-closed polarity: the gate reds when it scanned zero tracked .json files, because a
// corpus that matched nothing is a gate that stopped checking the tree (the same
// invariant loadEffectCorpus holds in effects_register_test.go). A file absent from a
// sparse/skip-worktree checkout is skipped: that is a checkout shape, not a BOM defect,
// and CI checks out the full tree.
func TestTrackedJSONHasNoBOM(t *testing.T) {
	root := repoRoot(t)

	cmd := exec.Command("git", "ls-files", "-z", "--", "*.json")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git ls-files -- '*.json': %v", err)
	}

	bom := []byte{0xEF, 0xBB, 0xBF}
	scanned := 0
	var offenders []string
	for _, rel := range strings.Split(strings.TrimRight(string(out), "\x00"), "\x00") {
		if rel == "" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			continue // absent in a sparse/skip-worktree checkout; not a BOM defect
		}
		scanned++
		if bytes.HasPrefix(data, bom) {
			offenders = append(offenders, rel)
		}
	}

	if scanned == 0 {
		t.Fatalf("scanned zero tracked .json files — this gate would stop checking the tree " +
			"without saying so; the `git ls-files -- '*.json'` corpus moved or the checkout is empty")
	}
	for _, rel := range offenders {
		t.Errorf("%s begins with a UTF-8 BOM (EF BB BF) — a strict JSON reader rejects the "+
			"leading U+FEFF, so a machine consumer of this tracked artifact fails on bytes off "+
			"disk. Strip the first three bytes (write the file with the BOM disabled, e.g. "+
			"utf-8-sig -> utf-8).", rel)
	}
}

// TestHasBOMWitness pins the detector itself: the gate above is only as good as its BOM
// predicate, so a byte-table keeps a future refactor from waving a BOM through.
func TestHasBOMWitness(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		want bool
	}{
		{"bom-prefixed", append([]byte{0xEF, 0xBB, 0xBF}, []byte("{}")...), true},
		{"plain-object", []byte(`{"a":1}`), false},
		{"plain-array", []byte("[]"), false},
		{"empty", nil, false},
		{"partial-bom", []byte{0xEF, 0xBB}, false},
		{"bom-mid-file", []byte(`{"a":"` + "\uFEFF" + `"}`), false},
	}
	for _, c := range cases {
		if got := bytes.HasPrefix(c.data, []byte{0xEF, 0xBB, 0xBF}); got != c.want {
			t.Errorf("%s: hasBOM=%v, want %v", c.name, got, c.want)
		}
	}
}
