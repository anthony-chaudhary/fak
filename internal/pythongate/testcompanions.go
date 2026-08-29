package pythongate

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// TestCompanion is a reviewed, test-only Python file paired with an exact
// grandfathered module. These rows are deliberately separate from grandfathered:
// they do not make a new Python tool permanent, and they do not admit future
// *_test.py files automatically.
type TestCompanion struct {
	Path         string
	Module       string
	IntroducedBy string
}

const toolingQualityDebtCohort = "3f6ec55214412c206f8a10534430e3956d853d29"

// testCompanions records the individually reviewed #9233 cohort. Every row must
// remain an exact sibling test of a tracked, grandfathered module and import that
// module in-process; TestTestCompanionProvenance enforces all three bindings.
var testCompanions = []TestCompanion{
	{"tools/agent_test_harness_test.py", "tools/agent_test_harness.py", toolingQualityDebtCohort},
	{"tools/bench_endpoint_server_test.py", "tools/bench_endpoint_server.py", toolingQualityDebtCohort},
	{"tools/bench_migrate_all_test.py", "tools/bench_migrate_all.py", toolingQualityDebtCohort},
	{"tools/bench_migrate_comprehensive_test.py", "tools/bench_migrate_comprehensive.py", toolingQualityDebtCohort},
	{"tools/compact_guard_combined_dogfood_test.py", "tools/compact_guard_combined_dogfood.py", toolingQualityDebtCohort},
	{"tools/compact_history_dogfood_test.py", "tools/compact_history_dogfood.py", toolingQualityDebtCohort},
	{"tools/dispatch_account_topup_test.py", "tools/dispatch_account_topup.py", toolingQualityDebtCohort},
	{"tools/dispatch_glm_docs_test.py", "tools/dispatch_glm_docs.py", toolingQualityDebtCohort},
	{"tools/dogfood_getting_started_test.py", "tools/dogfood_getting_started.py", toolingQualityDebtCohort},
	{"tools/fanout_plot_test.py", "tools/fanout_plot.py", toolingQualityDebtCohort},
	{"tools/fleet_fit_test.py", "tools/fleet_fit.py", toolingQualityDebtCohort},
	{"tools/fleet_heatmap_test.py", "tools/fleet_heatmap.py", toolingQualityDebtCohort},
	{"tools/gcp_gpu_probe_test.py", "tools/gcp_gpu_probe.py", toolingQualityDebtCohort},
	{"tools/gcp_quota_request_test.py", "tools/gcp_quota_request.py", toolingQualityDebtCohort},
	{"tools/gen_notebooks_test.py", "tools/gen_notebooks.py", toolingQualityDebtCohort},
	{"tools/gen_session_effectiveness_svg_test.py", "tools/gen_session_effectiveness_svg.py", toolingQualityDebtCohort},
	{"tools/gen_social_preview_test.py", "tools/gen_social_preview.py", toolingQualityDebtCohort},
	{"tools/gen_structured_data_test.py", "tools/gen_structured_data.py", toolingQualityDebtCohort},
	{"tools/grafana/gen_dashboard_test.py", "tools/grafana/gen_dashboard.py", toolingQualityDebtCohort},
	{"tools/hero_video_gen_test.py", "tools/hero_video_gen.py", toolingQualityDebtCohort},
	{"tools/info_overlay_gen_test.py", "tools/info_overlay_gen.py", toolingQualityDebtCohort},
	{"tools/init_private_repo_test.py", "tools/init_private_repo.py", toolingQualityDebtCohort},
	{"tools/plot_turn_agent_mac_vs_baseline_test.py", "tools/plot_turn_agent_mac_vs_baseline.py", toolingQualityDebtCohort},
	{"tools/run_sweep_test.py", "tools/run_sweep.py", toolingQualityDebtCohort},
	{"tools/run_turn_agent_realistic_sweep_test.py", "tools/run_turn_agent_realistic_sweep.py", toolingQualityDebtCohort},
	{"tools/turntax_plot_test.py", "tools/turntax_plot.py", toolingQualityDebtCohort},
}

func admittedTestCompanions(repoRoot string, tracked, baseline map[string]bool) map[string]bool {
	admitted := make(map[string]bool, len(testCompanions))
	for _, companion := range testCompanions {
		if !tracked[companion.Path] || !tracked[companion.Module] || !baseline[companion.Module] {
			continue
		}
		if companion.Path != strings.TrimSuffix(companion.Module, ".py")+"_test.py" {
			continue
		}
		source, err := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(companion.Path)))
		if err != nil || !importsModule(source, companion.Module) {
			continue
		}
		admitted[companion.Path] = true
	}
	return admitted
}

func importsModule(source []byte, modulePath string) bool {
	name := strings.TrimSuffix(filepath.Base(modulePath), ".py")
	const inspectImports = `import ast, sys
target = sys.argv[1]
try:
    tree = ast.parse(sys.stdin.read())
except SyntaxError:
    raise SystemExit(2)
for node in ast.walk(tree):
    if isinstance(node, ast.Import) and any(alias.name == target for alias in node.names):
        raise SystemExit(0)
    if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module == target:
        raise SystemExit(0)
raise SystemExit(1)
`
	cmd := exec.Command("python3", "-I", "-S", "-c", inspectImports, name)
	cmd.Stdin = bytes.NewReader(source)
	return cmd.Run() == nil
}
