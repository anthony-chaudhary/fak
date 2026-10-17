package compute

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
// Source witness only: bind both native default-on sites, where exactly "0"
// turns off only its own family. Native compilation and device execution need
// the tagged fixture.
func TestV41TokenPanelNativeOptInSource(t *testing.T) {
	t.Parallel()
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	// Preserve string literals while removing comments, so commented-out code
	// cannot satisfy the contract.
	comments := regexp.MustCompile(`"(?:\\.|[^"\\])*"|//[^\n]*|/\*(?s:.*?)\*/`)
	source := comments.ReplaceAllStringFunc(string(raw), func(s string) string {
		if strings.HasPrefix(s, "/") {
			return " "
		}
		return s
	})
	// The opt-in sites must be unconditional across build configurations. Ignore
	// every conditional-preprocessor branch rather than accepting an #if 0 decoy.
	var unconditional []string
	depth := 0
	for _, line := range strings.Split(source, "\n") {
		if directive, ok := strings.CutPrefix(strings.TrimSpace(line), "#"); ok {
			fields := strings.Fields(directive)
			if len(fields) > 0 {
				switch fields[0] {
				case "if", "ifdef", "ifndef":
					depth++
				case "endif":
					depth--
					if depth < 0 {
						t.Fatal("unbalanced native preprocessor condition")
					}
				}
			}
			continue
		}
		if depth == 0 {
			unconditional = append(unconditional, line)
		}
	}
	if depth != 0 {
		t.Fatal("unterminated native preprocessor condition")
	}
	source = strings.Join(unconditional, "\n")
	normalize := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	body := func(name string) string {
		t.Helper()
		pattern := regexp.MustCompile(`(?ms)^(?:int|static bool) ` + regexp.QuoteMeta(name) + `\([^)]*\) \{.*?^\}`)
		matches := pattern.FindAllString(source, -1)
		if len(matches) != 1 {
			t.Fatalf("expected one active %s definition, got %d", name, len(matches))
		}
		return normalize(matches[0])
	}
	policy := `static bool v41TokenPanelEnvEnabled(const char* name) {
		const char* value = std::getenv(name);
		return !(value && value[0] == '0' && value[1] == '\0');
	}`
	if body("v41TokenPanelEnvEnabled") != normalize(policy) {
		t.Error("native policy must disable a family only on its exact \"0\" value")
	}
	if strings.Count(normalize(source), `std::getenv("FAK_V41_Q2K_TOKEN_PANEL")`)+strings.Count(normalize(source), `std::getenv("FAK_V41_Q3K_TOKEN_PANEL")`) != 0 {
		t.Error("token-panel variables must be read only through the shared default-on policy")
	}
	init := body("fvk_init")
	startup := `
	const bool q2TokenEnabled = v41TokenPanelEnvEnabled("FAK_V41_Q2K_TOKEN_PANEL");
	const bool q3TokenEnabled = v41TokenPanelEnvEnabled("FAK_V41_Q3K_TOKEN_PANEL");
	if (q2TokenEnabled && g_v41IndexerFloatControls) {
		buildKernel(g_kern[K_V41_Q2K_TOKEN_PANEL], P("v41_q2k_token_panel.spv"), 3, 12);
	}
	if (q3TokenEnabled && g_v41IndexerFloatControls) {
		buildKernel(g_kern[K_V41_Q3K_TOKEN_PANEL], P("v41_q3k_token_panel.spv"), 3, 12);
	}`
	if !strings.Contains(init, normalize(startup)) {
		t.Error("startup must build each panel family unless its own variable is exactly 0, with float controls")
	}
	for _, kernel := range []string{"K_V41_Q2K_TOKEN_PANEL", "K_V41_Q3K_TOKEN_PANEL"} {
		if strings.Count(normalize(source), "buildKernel(g_kern["+kernel+"]") != 1 {
			t.Errorf("%s must have exactly one guarded pipeline creation", kernel)
		}
	}
	for name, want := range map[string]string{
		"fvk_have_v41_q2k_token_panel": `int fvk_have_v41_q2k_token_panel(void) {
			return v41TokenPanelEnvEnabled("FAK_V41_Q2K_TOKEN_PANEL") && g_ready && g_v41IndexerFloatControls &&
				g_kern[K_V41_Q2K_TOKEN_PANEL].pipe != VK_NULL_HANDLE && g_kern[K_V41_Q2K_TOKEN_PANEL].nbuf == 3;
		}`,
		"fvk_have_v41_q3k_token_panel": `int fvk_have_v41_q3k_token_panel(void) {
			return v41TokenPanelEnvEnabled("FAK_V41_Q3K_TOKEN_PANEL") && g_ready && g_v41IndexerFloatControls &&
				g_kern[K_V41_Q3K_TOKEN_PANEL].pipe != VK_NULL_HANDLE && g_kern[K_V41_Q3K_TOKEN_PANEL].nbuf == 3;
		}`,
	} {
		if body(name) != normalize(want) {
			t.Errorf("%s changed default-on policy, readiness, float-control, or live-pipeline guards", name)
		}
	}
}
