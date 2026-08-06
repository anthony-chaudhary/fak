package tools_test

// Contract tests for the EP_PREFLIGHT gate in tools/glm52_ep_witness.sh and for the self-check
// witness that describes it (issue #4952).
//
// The witness JSON these tests guard was, for three weeks, a FALSE witness: it named
// "tools/glm52_ep_witness.sh (EP_PREFLIGHT gate)" and "internal/compute.RequiredFreeBytes +
// TestRequiredFreeBytesIsExactFitBoundary" as its code_under_test, and NEITHER existed anywhere in
// the tree. A published self-check whose subject does not exist is worse than no self-check: it
// reads as evidence, it is shaped like evidence, and it certifies nothing. The witness has since
// been made true by writing the code it names.
//
// TestGLM52EPPreflightWitnessNamesCodeThatExists is what keeps it true. It is deliberately a
// RESOLVER, not a restatement: it reads code_under_test out of the JSON and insists each entry
// resolves to a real file, symbol, or test in this tree. Delete RequiredFreeBytes and this test
// goes red rather than the witness quietly going back to lying.

import (
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const glm52EPWitnessJSON = "../experiments/glm-gpu-witness/glm52-ep-preflight-guard-selfcheck-2026-07-15.json"

type glm52EPSelfcheck struct {
	RequiredFreeDerivation struct {
		Formula  string  `json:"formula"`
		Headroom float64 `json:"headroom"`
		Ranks8   struct {
			PlanTotalGiB    float64 `json:"plan_total_gib"`
			RequiredFreeGiB float64 `json:"required_free_gib"`
		} `json:"ranks_8"`
		Ranks7 struct {
			PlanTotalGiB    float64 `json:"plan_total_gib"`
			RequiredFreeGiB float64 `json:"required_free_gib"`
		} `json:"ranks_7"`
	} `json:"required_free_derivation"`
	CodeUnderTest []string `json:"code_under_test"`
}

func loadGLM52EPSelfcheck(t *testing.T) glm52EPSelfcheck {
	t.Helper()
	raw, err := os.ReadFile(glm52EPWitnessJSON)
	if err != nil {
		t.Fatalf("read witness: %v", err)
	}
	var w glm52EPSelfcheck
	if err := json.Unmarshal(raw, &w); err != nil {
		t.Fatalf("parse witness: %v", err)
	}
	return w
}

// Every code_under_test entry must resolve to something real. This is the test that would have
// caught the original false witness on the day it was published.
func TestGLM52EPPreflightWitnessNamesCodeThatExists(t *testing.T) {
	w := loadGLM52EPSelfcheck(t)
	if len(w.CodeUnderTest) == 0 {
		t.Fatal("witness declares no code_under_test — a self-check with no subject certifies nothing")
	}

	// A resolver per named entry. A new entry with no resolver fails loudly rather than passing
	// by default, which is the whole failure mode this test exists to prevent.
	resolvers := map[string]func() error{
		"tools/glm52_ep_witness.sh (EP_PREFLIGHT gate)": func() error {
			return mustContain("../tools/glm52_ep_witness.sh",
				"# --- pre-flight per-GPU free-VRAM gate",
				"EP_PREFLIGHT_REFUSE",
				"EP_PREFLIGHT_OK",
				"EP_PREFLIGHT_SKIP",
				"REQUIRE_FREE_GIB")
		},
		"internal/compute.RequiredFreeBytes + TestRequiredFreeBytesIsExactFitBoundary": func() error {
			if err := mustContain("../internal/compute/requiredfree.go", "func RequiredFreeBytes(wantBytes int64, headroom float64) int64"); err != nil {
				return err
			}
			return mustContain("../internal/compute/requiredfree_test.go", "func TestRequiredFreeBytesIsExactFitBoundary(")
		},
	}

	for _, entry := range w.CodeUnderTest {
		resolve, ok := resolvers[entry]
		if !ok {
			t.Errorf("code_under_test %q has no resolver in this test — add one, so the witness cannot name code nobody checks", entry)
			continue
		}
		if err := resolve(); err != nil {
			t.Errorf("code_under_test %q does not resolve: %v", entry, err)
		}
	}
}

func mustContain(path string, needles ...string) error {
	raw, err := os.ReadFile(filepath.FromSlash(path))
	if err != nil {
		return err
	}
	body := string(raw)
	var missing []string
	for _, n := range needles {
		if !strings.Contains(body, n) {
			missing = append(missing, strconv.Quote(n))
		}
	}
	if len(missing) > 0 {
		return &missingError{path: path, missing: missing}
	}
	return nil
}

type missingError struct {
	path    string
	missing []string
}

func (e *missingError) Error() string {
	return e.path + " is missing " + strings.Join(e.missing, ", ")
}

// The shell gate's threshold and the in-process fit check must be the same number. If they drift,
// the gate either admits a run the load then refuses (which is how #4952 got misfiled as a code
// regression) or refuses one that would have loaded fine.
func TestGLM52EPPreflightThresholdMatchesRequiredFreeBytes(t *testing.T) {
	w := loadGLM52EPSelfcheck(t)
	headroom, table := parseEPPreflightGateNumbers(t)

	if headroom != w.RequiredFreeDerivation.Headroom {
		t.Fatalf("gate headroom %g != witness headroom %g", headroom, w.RequiredFreeDerivation.Headroom)
	}

	for _, tc := range []struct {
		ranks    int
		planGiB  float64
		wantFree float64
	}{
		{8, w.RequiredFreeDerivation.Ranks8.PlanTotalGiB, w.RequiredFreeDerivation.Ranks8.RequiredFreeGiB},
		{7, w.RequiredFreeDerivation.Ranks7.PlanTotalGiB, w.RequiredFreeDerivation.Ranks7.RequiredFreeGiB},
	} {
		gotPlan, ok := table[tc.ranks]
		if !ok {
			t.Errorf("gate has no PLAN_GIB for RANKS=%d, but the witness publishes one (%.2f GiB)", tc.ranks, tc.planGiB)
			continue
		}
		if gotPlan != tc.planGiB {
			t.Errorf("RANKS=%d: gate PLAN_GIB=%.2f, witness plan_total_gib=%.2f", tc.ranks, gotPlan, tc.planGiB)
			continue
		}

		// The gate thresholds in MiB (nvidia-smi's unit), so it must be the byte requirement
		// rounded UP to the next whole MiB — never down, which would under-require.
		const mib = int64(1) << 20
		wantBytes := int64(gotPlan * float64(mib) * 1024)
		req := compute.RequiredFreeBytes(wantBytes, headroom)
		gateMiB := int64(math.Ceil(gotPlan * 1024 / (1 - headroom)))

		if gateMiB*mib < req {
			t.Errorf("RANKS=%d: gate demands %d MiB but the fit check needs %d bytes (%d MiB) — the gate would ADMIT a run the load then refuses",
				tc.ranks, gateMiB, req, (req+mib-1)/mib)
		}
		if gateMiB*mib >= req+mib {
			t.Errorf("RANKS=%d: gate demands %d MiB, more than a MiB above the fit requirement %d bytes — it would refuse loadable runs",
				tc.ranks, gateMiB, req)
		}

		// And the number the witness publishes, to the precision it publishes it at.
		if got := math.Round(float64(gateMiB)/1024*10) / 10; got != tc.wantFree {
			t.Errorf("RANKS=%d: gate requires %.1f GiB, witness publishes %.1f GiB", tc.ranks, got, tc.wantFree)
		}
	}
}

var (
	epHeadroomRe = regexp.MustCompile(`EP_HEADROOM="\$\{EP_HEADROOM:-([0-9.]+)\}"`)
	epPlanRowRe  = regexp.MustCompile(`(?m)^\s*([0-9]+)\)\s*PLAN_GIB=([0-9.]+)\s*;;`)
)

func parseEPPreflightGateNumbers(t *testing.T) (headroom float64, planGiB map[int]float64) {
	t.Helper()
	raw, err := os.ReadFile(filepath.FromSlash("../tools/glm52_ep_witness.sh"))
	if err != nil {
		t.Fatalf("read witness script: %v", err)
	}
	m := epHeadroomRe.FindStringSubmatch(string(raw))
	if m == nil {
		t.Fatal("no EP_HEADROOM default in tools/glm52_ep_witness.sh — the gate's headroom must be readable to be checkable")
	}
	if headroom, err = strconv.ParseFloat(m[1], 64); err != nil {
		t.Fatalf("EP_HEADROOM %q: %v", m[1], err)
	}
	planGiB = map[int]float64{}
	for _, row := range epPlanRowRe.FindAllStringSubmatch(string(raw), -1) {
		ranks, err := strconv.Atoi(row[1])
		if err != nil {
			t.Fatalf("PLAN_GIB rank %q: %v", row[1], err)
		}
		if planGiB[ranks], err = strconv.ParseFloat(row[2], 64); err != nil {
			t.Fatalf("PLAN_GIB %q: %v", row[2], err)
		}
	}
	if len(planGiB) == 0 {
		t.Fatal("no PLAN_GIB table in tools/glm52_ep_witness.sh — the gate would never arm")
	}
	return headroom, planGiB
}

// The five scenarios the witness publishes, run for real against the extracted gate block with a
// mock nvidia-smi. No GPU needed; skipped where bash is unavailable.
func TestGLM52EPPreflightSelfcheckScenariosPass(t *testing.T) {
	bash, err := exec.LookPath("bash")
	if err != nil {
		t.Skipf("no bash on PATH: %v", err)
	}
	out, err := exec.Command(bash, filepath.FromSlash("glm52_ep_preflight_selfcheck.sh"), "..").CombinedOutput()
	if err != nil || !strings.Contains(string(out), "EP_PREFLIGHT_SELFCHECK_PASS") {
		t.Fatalf("EP_PREFLIGHT self-check did not pass (%v):\n%s", err, out)
	}
}
