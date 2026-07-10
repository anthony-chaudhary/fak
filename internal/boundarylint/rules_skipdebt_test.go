package boundarylint

import "testing"

// TestSkipDebtRule verifies the classifier (verify the verifier): a bare, unconditional
// skip is flagged as SKIP_DEBT; a skip guarded by a documented platform / short-mode /
// environment condition is an honest conditional and is NOT flagged.
func TestSkipDebtRule(t *testing.T) {
	cases := []struct {
		name string
		src  string
		want int
	}{
		{"bare Skip flagged", `package p
import "testing"
func TestX(t *testing.T) { t.Skip("flaky, fix later") }`, 1},
		{"bare SkipNow flagged", `package p
import "testing"
func TestX(t *testing.T) { t.SkipNow() }`, 1},
		{"bare Skipf flagged", `package p
import "testing"
func TestX(t *testing.T) { t.Skipf("broken since %d", 42) }`, 1},
		{"platform-guarded skip clean", `package p
import ("runtime"; "testing")
func TestX(t *testing.T) {
	if runtime.GOOS == "windows" { t.Skip("POSIX-only") }
	// real assertions follow
}`, 0},
		{"short-mode-guarded skip clean", `package p
import "testing"
func TestX(t *testing.T) {
	if testing.Short() { t.Skip("slow; run without -short") }
}`, 0},
		{"env-guarded skip clean", `package p
import ("os"; "testing")
func TestX(t *testing.T) {
	if os.Getenv("CI") == "" { t.Skip("needs CI credentials") }
}`, 0},
		{"GOARCH-guarded skip clean", `package p
import ("runtime"; "testing")
func TestX(t *testing.T) {
	if runtime.GOARCH != "amd64" { t.SkipNow() }
}`, 0},
		{"no skip clean", `package p
import "testing"
func TestX(t *testing.T) { if 1 != 1 { t.Fatal("x") } }`, 0},
		{"documented and bare together: only the bare flagged", `package p
import ("runtime"; "testing")
func TestX(t *testing.T) {
	if runtime.GOOS == "plan9" { t.Skip("unsupported OS") } // documented — clean
	t.Skip("TODO: rewrite")                                 // bare — debt
}`, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := checkSrc(t, SkipDebt{}, tc.src)
			if len(got) != tc.want {
				t.Fatalf("got %d findings %v, want %d", len(got), got, tc.want)
			}
			for _, f := range got {
				if f.Code != "SKIP_DEBT" {
					t.Errorf("finding code = %q, want SKIP_DEBT", f.Code)
				}
			}
		})
	}
}

// TestSkipDebtFlagsTheBareSkipLine pins that in a mixed fixture the finding points at
// the bare skip, not the documented one — the work-list must name the site to fix.
func TestSkipDebtFlagsTheBareSkipLine(t *testing.T) {
	src := `package p
import ("runtime"; "testing")
func TestX(t *testing.T) {
	if runtime.GOOS == "plan9" { t.Skip("unsupported OS") }
	t.Skip("TODO: rewrite")
}`
	got := checkSrc(t, SkipDebt{}, src)
	if len(got) != 1 {
		t.Fatalf("got %d findings %v, want exactly 1 (the bare skip)", len(got), got)
	}
	// The documented skip is on line 4; the bare skip is on line 5.
	if got[0].Line != 5 {
		t.Errorf("flagged line = %d, want 5 (the bare t.Skip, not the guarded one on line 4)", got[0].Line)
	}
}
