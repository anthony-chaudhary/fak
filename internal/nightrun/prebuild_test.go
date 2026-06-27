package nightrun

import "testing"

func TestParseGoRun(t *testing.T) {
	cases := []struct {
		run      string
		wantPkg  string
		wantArgs string
		wantOK   bool
	}{
		{"go run ./cmd/radixbench", "./cmd/radixbench", "", true},
		{"go run ./cmd/modelbench -dir internal/model/.cache/x", "./cmd/modelbench", "-dir internal/model/.cache/x", true},
		{"fak resume validate -corpus ~/.claude/projects -json", "", "", false}, // not a go run
		{"go test ./internal/compute", "", "", false},                           // go test, not run
		{"go run ./internal/foo", "", "", false},                                // not ./cmd/
		{"go run ./cmd/...", "", "", false},                                     // wildcard package — left to go run
		{"FAK_X=1 go run ./cmd/y", "", "", false},                               // env prefix — not a bare go run
	}
	for _, c := range cases {
		pkg, args, ok := parseGoRun(c.run)
		if ok != c.wantOK || pkg != c.wantPkg || args != c.wantArgs {
			t.Errorf("parseGoRun(%q) = (%q,%q,%v), want (%q,%q,%v)", c.run, pkg, args, ok, c.wantPkg, c.wantArgs, c.wantOK)
		}
	}
}

func TestQuoteIfNeeded(t *testing.T) {
	if got := quoteIfNeeded("/tmp/bin/radixbench"); got != "/tmp/bin/radixbench" {
		t.Errorf("space-free path must be unquoted, got %q", got)
	}
	if got := quoteIfNeeded(`C:\Program Files\bin\x.exe`); got != `"C:\Program Files\bin\x.exe"` {
		t.Errorf("spaced path must be quoted, got %q", got)
	}
}

// maybePrebuildRun must return a non-go-run command unchanged even with no toolchain.
func TestMaybePrebuildRunPassthrough(t *testing.T) {
	run := "fak resume validate -corpus ~/.claude/projects -json"
	if got := maybePrebuildRun(nil, "", run); got != run {
		t.Errorf("a non-go-run command must pass through unchanged; got %q", got)
	}
}
