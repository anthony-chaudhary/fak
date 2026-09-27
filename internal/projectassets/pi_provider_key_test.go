package projectassets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

const (
	fragileHiveKeyCommand   = `!powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-Content -LiteralPath 'C:\Users\USER\.fak\keys\hive.key' -Raw).Trim()"`
	testHiveAPIBaseURL      = "https://api-cdn.thehive.ai/api/v3"
	testHiveAPIKeyReference = "${HIVE_API_KEY}"
)

func TestPiProviderKeyCommandIsFragile(t *testing.T) {
	cases := []struct {
		name   string
		apiKey string
		want   bool
	}{
		{"powershell get-content", fragileHiveKeyCommand, true},
		{"pwsh get-content", `!pwsh -NoProfile -Command "(Get-Content -LiteralPath 'C:\k.key' -Raw).Trim()"`, true},
		{"cat command", `!cat 'C:/Users/USER/.fak/keys/hive.key'`, false},
		{"environment reference", `${HIVE_API_KEY}`, false},
		{"literal", "fak", false},
		{"unrelated command", "!op read op://vault/hive/key", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := PiProviderKeyCommandIsFragile(tc.apiKey); got != tc.want {
				t.Fatalf("PiProviderKeyCommandIsFragile(%q) = %v, want %v", tc.apiKey, got, tc.want)
			}
		})
	}
}

func TestNormalizePiProviderKeyCommandScopesLegacyPath(t *testing.T) {
	cases := []struct {
		name        string
		apiKey      string
		want        string
		wantChanged bool
	}{
		{name: "known Hive path", apiKey: fragileHiveKeyCommand, want: `${HIVE_API_KEY}`, wantChanged: true},
		{
			name:   "custom path",
			apiKey: `!powershell -Command "(Get-Content -LiteralPath 'D:\operator\custom.key' -Raw).Trim()"`,
			want:   `!powershell -Command "(Get-Content -LiteralPath 'D:\operator\custom.key' -Raw).Trim()"`,
		},
		{name: "environment reference", apiKey: `${HIVE_API_KEY}`, want: `${HIVE_API_KEY}`},
		{name: "literal", apiKey: "operator-literal", want: "operator-literal"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, changed := NormalizePiProviderKeyCommand(tc.apiKey)
			if got != tc.want || changed != tc.wantChanged {
				t.Fatalf("NormalizePiProviderKeyCommand(%q) = (%q, %v), want (%q, %v)", tc.apiKey, got, changed, tc.want, tc.wantChanged)
			}
		})
	}
}

func TestRepairPiProviderKeyCommandScopesManagedHiveProvider(t *testing.T) {
	const userFailure = `!cat 'C:/Users/USER/.fak/keys/hive.key'`
	const customPath = `!cat 'D:/operator/custom.key'`
	cases := []struct {
		name        string
		baseURL     string
		apiKey      string
		want        string
		wantChanged bool
	}{
		{"exact user failure", testHiveAPIBaseURL, userFailure, testHiveAPIKeyReference, true},
		{"managed endpoint trailing slash", testHiveAPIBaseURL + "/", fragileHiveKeyCommand, testHiveAPIKeyReference, true},
		{"custom path on managed Hive endpoint", testHiveAPIBaseURL, customPath, customPath, false},
		{"known path on custom endpoint", "https://operator.example/v1", userFailure, userFailure, false},
		{"literal on managed Hive endpoint", testHiveAPIBaseURL, "operator-literal", "operator-literal", false},
		{"already bound", testHiveAPIBaseURL, testHiveAPIKeyReference, testHiveAPIKeyReference, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			provider := map[string]interface{}{"baseUrl": tc.baseURL, "apiKey": tc.apiKey}
			changed := repairPiProviderKeyCommand(provider)
			if got := provider["apiKey"]; got != tc.want || changed != tc.wantChanged {
				t.Fatalf("repair = (%q, %v), want (%q, %v)", got, changed, tc.want, tc.wantChanged)
			}
		})
	}
}

// TestEnsurePiProviderConfigRepairsExactUserFailure witnesses the observed
// `!cat 'C:/Users/USER/.fak/keys/hive.key'` failure without reading that file.
func TestEnsurePiProviderConfigRepairsExactUserFailure(t *testing.T) {
	target := filepath.Join(t.TempDir(), "models.json")
	existing := map[string]interface{}{
		"rootMarker": "preserve",
		"providers": map[string]interface{}{
			"hive-ai": map[string]interface{}{
				"api": "openai-completions", "apiKey": `!cat 'C:/Users/USER/.fak/keys/hive.key'`,
				"baseUrl": testHiveAPIBaseURL, "customField": "preserve",
				"models": []interface{}{map[string]interface{}{"id": "deepseek-ai/DeepSeek-V4.1-Flash"}},
			},
			"custom-hive-path": map[string]interface{}{
				"api": "openai-completions", "apiKey": `!cat 'D:/operator/custom.key'`,
				"baseUrl": testHiveAPIBaseURL, "models": []interface{}{map[string]interface{}{"id": "custom-path-model"}},
			},
			"custom-endpoint": map[string]interface{}{
				"api": "openai-completions", "apiKey": `!cat 'C:/Users/USER/.fak/keys/hive.key'`,
				"baseUrl": "https://operator.example/v1", "models": []interface{}{map[string]interface{}{"id": "custom-endpoint-model"}},
			},
			"literal": map[string]interface{}{
				"api": "openai-completions", "apiKey": "operator-literal",
				"baseUrl": testHiveAPIBaseURL, "models": []interface{}{map[string]interface{}{"id": "literal-model"}},
			},
		},
	}
	raw, err := json.Marshal(existing)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, raw, 0o644); err != nil {
		t.Fatal(err)
	}

	_, modified, err := EnsurePiProviderConfig(target, "http://127.0.0.1:18101/v1", "deepseek-ai/DeepSeek-V4.1-Flash")
	if err != nil {
		t.Fatalf("EnsurePiProviderConfig: %v", err)
	}
	if !modified {
		t.Fatal("expected legacy managed Hive binding to be repaired")
	}

	var got map[string]interface{}
	repaired, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(repaired, &got); err != nil {
		t.Fatal(err)
	}
	providers := got["providers"].(map[string]interface{})
	if key := providers["hive-ai"].(map[string]interface{})["apiKey"]; key != testHiveAPIKeyReference {
		t.Fatalf("managed Hive apiKey = %q, want %q", key, testHiveAPIKeyReference)
	}
	if got["rootMarker"] != "preserve" || providers["hive-ai"].(map[string]interface{})["customField"] != "preserve" {
		t.Fatal("repair dropped sibling fields")
	}
	for _, id := range []string{"custom-hive-path", "custom-endpoint", "literal"} {
		if !reflect.DeepEqual(providers[id], existing["providers"].(map[string]interface{})[id]) {
			t.Errorf("provider %q changed:\n got %#v\nwant %#v", id, providers[id], existing["providers"].(map[string]interface{})[id])
		}
	}

	_, modifiedAgain, err := EnsurePiProviderConfig(target, "http://127.0.0.1:18101/v1", "deepseek-ai/DeepSeek-V4.1-Flash")
	if err != nil {
		t.Fatalf("second EnsurePiProviderConfig: %v", err)
	}
	if modifiedAgain {
		t.Fatal("second repair changed an already-normalized config")
	}
}
