package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/cmd/internal/democapture"
	"github.com/anthony-chaudhary/fak/internal/harnessweb"
)

func TestCapturedSelfcheckReceipt(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := harnessweb.Run(context.Background(), &stdout, &stderr, []string{"--selfcheck"}); code != 0 {
		t.Fatalf("selfcheck exit=%d stderr=%s", code, stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("selfcheck wrote stderr: %s", stderr.String())
	}
	for _, want := range []string{
		"HARNESS_WEB_SELFCHECK ok",
		"protocol=fak.harness.run/v1",
		"normal=8 resumed=2 approval=4 failure=3",
		"skins=2 runs=3 goals=1 dashboards=8",
		"html_sha256=4955e5b3de7a2c3b1e97aa09e3fcf18e6c173551783dfcc5d9941dc007861ecb",
	} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("receipt missing %q: %s", want, stdout.String())
		}
	}
	if err := democapture.MatchMarkdown("EXAMPLE-OUTPUT.md", stdout.Bytes()); err != nil {
		t.Fatal(err)
	}
}
