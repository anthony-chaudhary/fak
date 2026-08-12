package hooks

import "testing"

func popupDiff(files map[string]string, added ...string) *StagedDiff {
	cache := make(map[string]fileEntry, len(files))
	paths := make([]string, 0, len(files))
	for rel, src := range files {
		paths = append(paths, rel)
		cache[rel] = fileEntry{data: []byte(src), exists: true}
	}
	return &StagedDiff{StagedPaths: paths, AddedPaths: added, fileCache: cache}
}

func TestDesktopPopupRejectsNewUnsuppressedGoHelperAtPreCommit(t *testing.T) {
	src := `package helper
import "os/exec"
func run() error {
	cmd := exec.Command("go", "test", "./...")
	return cmd.Run()
}`
	got, err := CheckDesktopPopup(popupDiff(map[string]string{"internal/newhelper/run.go": src}, "internal/newhelper/run.go"))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Gate != "DESKTOP_POPUP_REGRESSION" || got[0].File != "internal/newhelper/run.go" {
		t.Fatalf("new unsuppressed helper findings = %#v, want one pre-commit refusal", got)
	}
}

func TestDesktopPopupAcceptsNewWindowgateHelper(t *testing.T) {
	src := `package helper
import "github.com/anthony-chaudhary/fak/internal/windowgate"
func run() error {
	return windowgate.Command("go", "test", "./...").Run()
}`
	got, err := CheckDesktopPopup(popupDiff(map[string]string{"internal/newhelper/run.go": src}, "internal/newhelper/run.go"))
	if err != nil || len(got) != 0 {
		t.Fatalf("windowgate helper findings = %#v, err=%v", got, err)
	}
}

func TestDesktopPopupRejectsNewSimilarPythonAndPowerShellHelpers(t *testing.T) {
	files := map[string]string{
		"tools/new_helper.py":  "import subprocess\nsubprocess.run(['go', 'test', './...'])\n",
		"tools/new_helper.ps1": "Start-Process go -ArgumentList 'test ./...'\n",
	}
	got, err := CheckDesktopPopup(popupDiff(files, "tools/new_helper.py", "tools/new_helper.ps1"))
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, f := range got {
		seen[f.File] = true
	}
	for rel := range files {
		if !seen[rel] {
			t.Errorf("new unsuppressed helper %s was admitted; findings=%#v", rel, got)
		}
	}
}

func TestDesktopPopupReadsCandidateIndexNotPeerWorktree(t *testing.T) {
	staged := `package helper
import "github.com/anthony-chaudhary/fak/internal/windowgate"
func run() error { return windowgate.Command("git", "status").Run() }`
	d := popupDiff(map[string]string{"cmd/fak/dispatch_new.go": staged}, "cmd/fak/dispatch_new.go")
	got, err := CheckDesktopPopup(d)
	if err != nil || len(got) != 0 {
		t.Fatalf("candidate-index-safe source findings = %#v, err=%v", got, err)
	}
}

func TestDesktopPopupGateIsRegisteredFailClosed(t *testing.T) {
	for _, g := range PreCommitGates() {
		if g.Name == "DESKTOP_POPUP_REGRESSION" {
			if g.DefaultMode != "" || g.ModeEnv != "" || g.EscapeEnv != "FLEET_ALLOW_POPUP" {
				t.Fatalf("popup gate policy = %#v, want block-by-default with documented override", g)
			}
			return
		}
	}
	t.Fatal("DESKTOP_POPUP_REGRESSION is not registered at pre-commit")
}
