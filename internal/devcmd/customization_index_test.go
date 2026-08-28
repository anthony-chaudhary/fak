package devcmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCustomizationIndexCommandTextAndJSON(t *testing.T) {
	path := writeCustomizationFixture(t)
	for _, args := range [][]string{{"--index", path, "--as-of", "2026-08-17"}, {"--index", path, "--as-of", "2026-08-17", "--json"}} {
		var stdout, stderr bytes.Buffer
		if code := RunCustomizationIndex(&stdout, &stderr, args); code != 0 {
			t.Fatalf("code=%d stderr=%s", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), `"axes": 1`) && !strings.Contains(stdout.String(), "axes=1") {
			t.Fatalf("stdout=%s", stdout.String())
		}
	}
}

func TestCustomizationIndexCommandReturnsInvalid(t *testing.T) {
	path := writeCustomizationFixture(t)
	data, _ := os.ReadFile(path)
	os.WriteFile(path, bytes.Replace(data, []byte(`"fak_status":"present"`), []byte(`"fak_status":"maybe"`), 1), 0o600)
	var stdout, stderr bytes.Buffer
	if code := RunCustomizationIndex(&stdout, &stderr, []string{"--index", path, "--as-of", "2026-08-17"}); code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "invalid status") {
		t.Fatalf("stdout=%s", stdout.String())
	}
}

func writeCustomizationFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "index.json")
	data := `{"schema":"fak-agent-customization-index/1","updated_at":"2026-08-17","maintenance":{"review_interval_days":30,"status_values":["present"],"disposition_values":["default"]},"layers":[{"id":"authoring"}],"sources":[{"id":"source","observed_at":"2026-08-01","checked_revision":"abc"}],"axes":[{"axis_id":"instructions","layer":"authoring","user_need":"configure","evidence":["source"],"fak_status":"present","disposition":"default"}]}`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
