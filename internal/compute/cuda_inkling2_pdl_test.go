package compute

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// ink2PDLKernelRe matches the kernel symbol of a __global__ definition whose name marks it
// PDL-aware for ink2_pdl_rewrite (cuda/inkling2_pdl.cuh). k_ink2p_ (batched prefill, also
// captured inside the spec verify and commit graphs) is outside the prefix set, so edges
// into it stay fully serialized.
var ink2PDLKernelRe = regexp.MustCompile(`__global__[^;{]*?\b(k_ink2[vd]?_[A-Za-z0-9_]*)\s*\(`)

// ink2PDLKernelsMissingWait returns the PDL-aware kernels in src whose body does not open
// with ink2_pdl_enter(). The rewrite turns every edge into such a kernel programmatic, so a
// kernel that reads or writes before waiting would race its predecessor.
func ink2PDLKernelsMissingWait(src string) (kernels, missing []string) {
	for _, m := range ink2PDLKernelRe.FindAllStringSubmatchIndex(src, -1) {
		name := src[m[2]:m[3]]
		rest := src[m[1]:]
		brace := strings.IndexAny(rest, "{;")
		if brace < 0 || rest[brace] == ';' {
			continue
		}
		kernels = append(kernels, name)
		if !strings.HasPrefix(strings.TrimSpace(rest[brace+1:]), "ink2_pdl_enter();") {
			missing = append(missing, name)
		}
	}
	return kernels, missing
}

func TestInkling2PDLKernelsWaitFirst(t *testing.T) {
	files, err := filepath.Glob("cuda/*.cuh")
	if err != nil {
		t.Fatal(err)
	}
	cu, err := filepath.Glob("*.cu")
	if err != nil {
		t.Fatal(err)
	}
	total := 0
	for _, f := range append(files, cu...) {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		kernels, missing := ink2PDLKernelsMissingWait(string(b))
		total += len(kernels)
		for _, k := range missing {
			t.Errorf("%s: kernel %s must open with ink2_pdl_enter(); (programmatic edges into it would race)", f, k)
		}
	}
	if total < 40 {
		t.Fatalf("found only %d PDL-aware kernels; the scan no longer sees the Inkling-2 kernel set", total)
	}
}

func TestInkling2PDLKernelScanCatchesLateWait(t *testing.T) {
	src := "__global__ void __launch_bounds__(256)\n    k_ink2_ok(int *a) {\n  ink2_pdl_enter();\n  a[0] = 1;\n}\n" +
		"__global__ void k_ink2v_bad(int *a) {\n  if (a[0]) return;\n  ink2_pdl_enter();\n}\n" +
		"__global__ void k_ink2p_prefill(int *a) {\n  a[0] = 2;\n}\n" +
		"template <int R> __global__ void k_ink2d_decl(int *a);\n"
	kernels, missing := ink2PDLKernelsMissingWait(src)
	if strings.Join(kernels, ",") != "k_ink2_ok,k_ink2v_bad" || strings.Join(missing, ",") != "k_ink2v_bad" {
		t.Fatalf("kernels=%v missing=%v", kernels, missing)
	}
}
