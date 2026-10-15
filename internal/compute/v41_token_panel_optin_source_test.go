package compute

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
// Source witness only: bind both native opt-in sites and keep Q3-only from
// creating Q2. Native compilation and device execution need the tagged fixture.
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
		pattern := regexp.MustCompile(`(?ms)^int ` + regexp.QuoteMeta(name) + `\([^)]*\) \{.*?^\}`)
		matches := pattern.FindAllString(source, -1)
		if len(matches) != 1 {
			t.Fatalf("expected one active %s definition, got %d", name, len(matches))
		}
		return normalize(matches[0])
	}
	init := body("fvk_init")
	startup := `
	const char* q2TokenEnv = std::getenv("FAK_V41_Q2K_TOKEN_PANEL");
	const char* q3TokenEnv = std::getenv("FAK_V41_Q3K_TOKEN_PANEL");
	const bool q2TokenOptedIn = q2TokenEnv && q2TokenEnv[0] == '1' && q2TokenEnv[1] == '\0';
	const bool q3TokenOptedIn = q3TokenEnv && q3TokenEnv[0] == '1' && q3TokenEnv[1] == '\0';
	if (q2TokenOptedIn && g_v41IndexerFloatControls) {
		buildKernel(g_kern[K_V41_Q2K_TOKEN_PANEL], P("v41_q2k_token_panel.spv"), 3, 12);
	}
	if ((q2TokenOptedIn || q3TokenOptedIn) && g_v41IndexerFloatControls) {
		buildKernel(g_kern[K_V41_Q3K_TOKEN_PANEL], P("v41_q3k_token_panel.spv"), 3, 12);
	}`
	if !strings.Contains(init, normalize(startup)) {
		t.Error("startup must build Q2 only on exact Q2=1, and Q3 on exact Q2=1 or Q3=1, with float controls")
	}
	for _, kernel := range []string{"K_V41_Q2K_TOKEN_PANEL", "K_V41_Q3K_TOKEN_PANEL"} {
		if strings.Count(normalize(source), "buildKernel(g_kern["+kernel+"]") != 1 {
			t.Errorf("%s must have exactly one guarded pipeline creation", kernel)
		}
	}
	for name, want := range map[string]string{
		"fvk_have_v41_q2k_token_panel": `int fvk_have_v41_q2k_token_panel(void) {
			const char* enabled = std::getenv("FAK_V41_Q2K_TOKEN_PANEL");
			return enabled && enabled[0] == '1' && enabled[1] == '\0' && g_ready && g_v41IndexerFloatControls &&
				g_kern[K_V41_Q2K_TOKEN_PANEL].pipe != VK_NULL_HANDLE && g_kern[K_V41_Q2K_TOKEN_PANEL].nbuf == 3;
		}`,
		"fvk_have_v41_q3k_token_panel": `int fvk_have_v41_q3k_token_panel(void) {
			const char* q2Enabled = std::getenv("FAK_V41_Q2K_TOKEN_PANEL");
			const char* q3Enabled = std::getenv("FAK_V41_Q3K_TOKEN_PANEL");
			return ((q2Enabled && q2Enabled[0] == '1' && q2Enabled[1] == '\0') ||
				(q3Enabled && q3Enabled[0] == '1' && q3Enabled[1] == '\0')) &&
				g_ready && g_v41IndexerFloatControls &&
				g_kern[K_V41_Q3K_TOKEN_PANEL].pipe != VK_NULL_HANDLE && g_kern[K_V41_Q3K_TOKEN_PANEL].nbuf == 3;
		}`,
	} {
		if body(name) != normalize(want) {
			t.Errorf("%s changed exact opt-in, readiness, float-control, or live-pipeline guards", name)
		}
	}
}
