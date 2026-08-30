package devindex

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildRemainingExtractionReportEdgeAndAdversarialInputs(t *testing.T) {
	oversized := "package main\n" + "/*" + strings.Repeat("x", 1<<20) + "*/\n" +
		`import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig() } }
func cmdConfig(){}
`
	tests := []struct {
		name       string
		files      map[string]string
		nodes      []ImportNode
		wantErr    string
		wantReport bool
	}{
		{
			name:    "EdgeEmptyPackage",
			files:   map[string]string{},
			wantErr: "parsed only 0 non-test file(s)",
		},
		{
			name: "AdversarialMalformedSource",
			files: map[string]string{
				"main.go": "package main\nfunc main( {\n",
			},
			wantErr: "cmd/fak/main.go:",
		},
		{
			name: "EdgeOversizedSource",
			files: map[string]string{
				"main.go": oversized,
			},
			wantReport: true,
		},
		{
			name: "AdversarialHostileImportGraph",
			files: map[string]string{
				"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "config": cmdConfig() } }
func cmdConfig(){}
`,
			},
			nodes: []ImportNode{
				{ImportPath: "github.com/anthony-chaudhary/fak/cmd/fak", Imports: []string{"cycle", "cycle", "missing"}},
				{ImportPath: "cycle", Imports: []string{"github.com/anthony-chaudhary/fak/cmd/fak", "cycle"}},
				{ImportPath: "cycle", Imports: []string{"duplicate-overwrite"}},
			},
			wantReport: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.wantReport {
				for i := len(tt.files); i < vsFileFloor; i++ {
					tt.files[fmt.Sprintf("padding_%03d.go", i)] = "package main\n"
				}
			}
			root := writeEdgeExtractionFixture(t, tt.files)
			report, err := BuildRemainingExtractionReport(root, tt.nodes)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want substring %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if tt.wantReport && (report.Schema != "fak-dev-extraction-report/1" || report.Counts.Commands != 1) {
				t.Fatalf("report = %+v, want one complete candidate", report)
			}
		})
	}
}

func TestBuildRemainingExtractionReportEdgeErrorPaths(t *testing.T) {
	t.Run("EdgeMalformedGlobRoot", func(t *testing.T) {
		_, err := BuildRemainingExtractionReport("[", nil)
		if err == nil || !strings.Contains(err.Error(), "syntax error in pattern") {
			t.Fatalf("error = %v, want malformed glob error", err)
		}
	})

	t.Run("EdgeUnreadableGoEntry", func(t *testing.T) {
		root := writeEdgeExtractionFixture(t, nil)
		if err := os.Mkdir(filepath.Join(root, "cmd", "fak", "unreadable.go"), 0o755); err != nil {
			t.Fatal(err)
		}
		_, err := BuildRemainingExtractionReport(root, nil)
		if err == nil || !strings.Contains(err.Error(), "unreadable.go") {
			t.Fatalf("error = %v, want read failure naming unreadable.go", err)
		}
	})

	t.Run("EdgeNoTierDevHandlers", func(t *testing.T) {
		pkg := loadExtractionFixture(t, map[string]string{
			"main.go": `package main
import "os"
func main(){ switch os.Args[1] { case "help": cmdHelp() } }
func cmdHelp(){}
`,
		})
		_, _, err := buildRemainingExtractionReport(pkg)
		if err == nil || err.Error() != "no remaining TierDev runtime handlers found" {
			t.Fatalf("error = %v, want no TierDev handlers error", err)
		}
	})
}

func writeEdgeExtractionFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "cmd", "fak")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, src := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}
