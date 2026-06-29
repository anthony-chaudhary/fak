package benchlineagegate

import (
	"os"
	"path/filepath"
	"testing"
)

// TestClassify verifies the pure gate core on synthetic sources (verify the
// verifier): an unstamped emitter is the offense, a stamped one is clean, an
// input-decoder is not an emitter at all, and the exempt marker opts out.
func TestClassify(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want Verdict
	}{
		{
			name: "unstamped emitter",
			src:  `b, _ := json.MarshalIndent(report, "", "  "); os.WriteFile(out, b, 0o644)`,
			want: Unstamped,
		},
		{
			name: "stamped via MarshalReport",
			src:  `b, _ := benchcli.MarshalReport(report); os.WriteFile(out, b, 0o644)`,
			want: Stamped,
		},
		{
			name: "stamped via WriteReport",
			src:  `benchcli.WriteReport(out, rep.JSON())`,
			want: Stamped,
		},
		{
			name: "decoder only is not an emitter",
			src:  `var c Config; json.Unmarshal(raw, &c)`,
			want: NotEmitter,
		},
		{
			name: "marshal to stdout only is not an emitter",
			src:  `b, _ := json.MarshalIndent(r, "", "  "); fmt.Println(string(b))`,
			want: NotEmitter,
		},
		{
			name: "exempt marker opts a JSONL writer out",
			src:  "// lineage:exempt: JSONL fixture stream\nb,_ := json.Marshal(t); outFile.Write(b)\nos.WriteFile(p, b, 0o644)",
			want: Exempt,
		},
		{
			name: "signal hidden in a doc comment does not fabricate an emitter",
			src:  "// this command does not call os.WriteFile or json.Marshal itself\nrender(r)",
			want: NotEmitter,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(tc.src); got != tc.want {
				t.Fatalf("Classify = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestOffenseString locks the human-facing fix line + reason code.
func TestOffenseString(t *testing.T) {
	got := Offense{Path: "cmd/foobench/main.go"}.String()
	if want := "cmd/foobench/main.go writes a benchmark report without lineage; emit it through " +
		"benchcli.WriteReport / benchcli.MarshalReport (or mark \"lineage:exempt\" with a reason) " +
		"(BENCH_EMITTER_UNSTAMPED)"; got != want {
		t.Fatalf("offense string =\n %q\nwant\n %q", got, want)
	}
}

// TestEveryBenchEmitterStampsLineage is the live trunk guard (#9): scanning the real
// tracked bench sources must yield ZERO unstamped emitters. The day a new
// cmd/*bench* writes a report without routing through benchcli's lineage stamper,
// this reds the trunk with its path.
func TestEveryBenchEmitterStampsLineage(t *testing.T) {
	root := repoRoot(t)
	offenses, err := ScanTree(root)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(offenses) > 0 {
		t.Errorf("%d benchmark emitter(s) write a report without stamping lineage:", len(offenses))
		for _, o := range offenses {
			t.Errorf("  %s", o)
		}
	}
}

// repoRoot walks up from the test's working directory to the module root (go.mod).
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
