// fak-test:runtime fast est=1s lane=default
package trajectory

import (
	"strings"
	"testing"
)

func TestDetectDegenerateCycleSpaceless64Char(t *testing.T) {
	t.Parallel()
	spaceless := strings.Repeat("abcd1234", 8) // 64 non-whitespace chars
	text := strings.Repeat(spaceless, 4)
	got, ok := DetectDegenerateCycle(0, "content", text)
	if !ok {
		t.Fatal("expected 64-char spaceless degenerate cycle to be detected")
	}
	if got.RepeatCount != 4 || got.SpanTokens < 16 {
		t.Fatalf("unexpected spaceless cycle: %+v", got)
	}
}
