package hooks

import "testing"

// gate_priorart_test.go — focused unit tests for the PRIOR_ART advisory gate. Built on the
// in-package diffOf helper (hooks_test.go), the same fixture style the other gate tests use.

func TestMatchKernelGlob(t *testing.T) {
	cases := []struct {
		path, glob string
		want       bool
	}{
		{"internal/metalgemm/foo.go", "internal/metalgemm/*", true},     // trailing /* prefix form
		{"internal/model/awq_cuda.go", "internal/model/awq*.go", true},  // single-segment wildcard
		{"internal/model/q6k.metal", "internal/model/*.metal", true},    // extension wildcard
		{"internal/compute/cuda.go", "internal/compute/cuda.go", true},  // exact match
		{`internal\model\awq_cuda.go`, "internal/model/awq*.go", true},  // backslashes normalized
		{"internal/model/parallel.go", "internal/model/awq*.go", false}, // non-match
		{"README.md", "internal/model/awq*.go", false},                  // non-kernel
	}
	for _, c := range cases {
		if got := matchKernelGlob(c.path, c.glob); got != c.want {
			t.Errorf("matchKernelGlob(%q, %q) = %v, want %v", c.path, c.glob, got, c.want)
		}
	}
}

// (a) touching a kernel file yields exactly one PRIOR_ART finding naming the SOTA reference.
func TestPriorArt_kernelFileEmitsAdvisory(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"internal/model/awq_cuda.go": {"package model", "// new kernel"},
	})
	f, err := gatePriorArt(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 1 {
		t.Fatalf("expected exactly one PRIOR_ART finding, got %d: %+v", len(f), f)
	}
	if f[0].Gate != "PRIOR_ART" {
		t.Errorf("gate name = %q, want PRIOR_ART", f[0].Gate)
	}
	if f[0].File != "internal/model/awq_cuda.go" {
		t.Errorf("finding File = %q, want the touched kernel path", f[0].File)
	}
	// The advisory must carry the AWQ row's SOTA reference (Marlin / AutoAWQ), read from
	// sotamatrix — not hard-coded here.
	if !hasFindingFor(f, "PRIOR_ART", "Marlin") {
		t.Errorf("advisory should name the Marlin reference; got %+v", f)
	}
	if !hasFindingFor(f, "PRIOR_ART", "AutoAWQ") {
		t.Errorf("advisory should name the AutoAWQ reference; got %+v", f)
	}
	if !hasFindingFor(f, "PRIOR_ART", `Prior-art:`) {
		t.Errorf("advisory should suggest a Prior-art: trailer; got %+v", f)
	}
}

// (b) touching a non-kernel file yields zero findings.
func TestPriorArt_nonKernelFileQuiet(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"README.md":     {"some prose"},
		"docs/notes.md": {"a note"},
	})
	f, err := gatePriorArt(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 0 {
		t.Fatalf("non-kernel files must yield no PRIOR_ART finding, got %+v", f)
	}
}

// (c) an added line carrying a "Prior-art:" token suppresses the gate entirely.
func TestPriorArt_priorArtTrailerSuppresses(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"internal/model/awq_cuda.go": {
			"package model",
			"// Prior-art: Marlin mixed-precision INT4 kernel, studied before this rewrite.",
		},
	})
	f, err := gatePriorArt(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 0 {
		t.Fatalf("a staged 'Prior-art:' line must silence the gate, got %+v", f)
	}
}

// (d) touching TWO files of the SAME op yields ONE finding (dedupe by op slug). Both paths match
// the awq-int4-gemm row's "internal/model/awq*.go" glob.
func TestPriorArt_dedupeByOp(t *testing.T) {
	d := diffOf("/r", map[string][]string{
		"internal/model/awq.go":      {"package model"},
		"internal/model/awq_cuda.go": {"package model"},
	})
	f, err := gatePriorArt(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(f) != 1 {
		t.Fatalf("two files of one op must dedupe to a single finding, got %d: %+v", len(f), f)
	}
}
