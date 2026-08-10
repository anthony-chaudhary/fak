package devcmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestProjectCompletionToyBringupRender(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "issues.json")
	body := func(standard string, points int) string {
		return "## Parent context\nParent: #36\n\n## Work estimate\nEstimate: " + projectPoints(points) + " points\n\n## Overall completion contribution\nContribution: " + projectPoints(points) + "/20 points\n\n## Completion standard\n" + standard + " complete\n"
	}
	data := `[{"number":1,"title":"toy","state":"closed","body":` + quote(body("Demo", 2)) + `},{"number":2,"title":"production","state":"open","body":` + quote(body("Production", 18)) + `}]`
	if err := os.WriteFile(path, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := RunProjectCompletion(&out, &errOut, []string{"--from-issues", path}); code != 0 {
		t.Fatalf("code=%d err=%s out=%s", code, errOut.String(), out.String())
	}
	got := out.String()
	if !strings.Contains(got, "production complete: 0.00/20.00 points (0.0%)") || !strings.Contains(got, "closed demo") {
		t.Fatalf("misleading render:\n%s", got)
	}
}
func projectPoints(n int) string {
	if n == 2 {
		return "2"
	}
	return "18"
}
func quote(s string) string {
	var b bytes.Buffer
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '\\', '"':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\n':
			b.WriteString(`\n`)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}
