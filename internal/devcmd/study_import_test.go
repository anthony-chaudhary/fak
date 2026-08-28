package devcmd

import (
	"bytes"
	"testing"
)

func TestStudyImportCommandRequiresStoreForLiveImport(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runStudyImport(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}
