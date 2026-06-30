package policy

import (
	"encoding/json"
	"html"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestPolicyEditorPageShipsRequiredUX(t *testing.T) {
	pagePath := filepath.Join("..", "..", "docs", "policy-editor.html")
	b, err := os.ReadFile(pagePath)
	if err != nil {
		t.Fatalf("read policy editor page: %v", err)
	}
	page := string(b)

	required := map[string]string{
		`id="import-file"`:     "JSON import",
		`id="load-json"`:       "JSON load/apply",
		`id="policy-json"`:     "JSON export text area",
		`id="copy-json"`:       "copy export",
		`id="download-json"`:   "download export",
		`id="validation-list"`: "real-time validation panel",
		`id="add-allow"`:       "visual allow-list builder",
		`id="add-deny"`:        "visual deny-list builder",
		`id="add-arg-rule"`:    "argument-rule builder",
	}
	for needle, label := range required {
		if !strings.Contains(page, needle) {
			t.Fatalf("policy editor missing %s control (%s)", label, needle)
		}
	}

	re := regexp.MustCompile(`(?s)<script type="application/json" id="starter-policy">\s*(.*?)\s*</script>`)
	match := re.FindStringSubmatch(page)
	if match == nil {
		t.Fatal("policy editor missing starter policy JSON")
	}
	starter := []byte(html.UnescapeString(match[1]))
	if _, err := ParseRuntime(starter); err != nil {
		t.Fatalf("starter policy must pass policy.ParseRuntime: %v", err)
	}

	var manifest Manifest
	if err := json.Unmarshal(starter, &manifest); err != nil {
		t.Fatalf("starter policy JSON unmarshal: %v", err)
	}
	if len(manifest.Allow) == 0 || len(manifest.Deny) == 0 || len(manifest.ArgRules) == 0 {
		t.Fatalf("starter policy should exercise allow, deny, and arg_rules; got allow=%d deny=%d arg_rules=%d",
			len(manifest.Allow), len(manifest.Deny), len(manifest.ArgRules))
	}
}
