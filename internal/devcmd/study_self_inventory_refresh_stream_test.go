package devcmd

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

// Refresh must hash committed blobs as a stream: it never materializes the tree, and
// the manifest it writes must still verify against an extracted committed tree.
func TestStudyInventoryRefreshStreamsWithoutExtraction(t *testing.T) {
	repo, git := seedStudySelfRepo(t)
	writeStudySelfFile(t, repo, "README.md", "one\n")
	writeStudySelfFile(t, repo, "cmd/tool/main.go", "package main\n")
	gitStudySelf(t, git, "add", "README.md", "cmd/tool/main.go")
	gitStudySelf(t, git, "commit", "-qm", "seed")

	extracts := installStudySelfInventoryExtractCounter(t)
	var out, errOut bytes.Buffer
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--refresh", "--root", repo}); code != 0 || extracts() != 0 {
		t.Fatalf("refresh code=%d extracts=%d stderr=%s", code, extracts(), errOut.String())
	}

	gitStudySelf(t, git, "add", studymonitor.DefaultSelfInventoryPath)
	gitStudySelf(t, git, "commit", "-qm", "manifest")
	out.Reset()
	errOut.Reset()
	if code := RunStudyInventory(&out, &errOut, []string{"--self", "--verify", "--root", repo}); code != 0 || extracts() != 1 {
		t.Fatalf("verify after streamed refresh code=%d extracts=%d stdout=%s stderr=%s", code, extracts(), out.String(), errOut.String())
	}
}
