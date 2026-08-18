package linefmt

import (
	"strings"
	"testing"
)

func TestWriterTerminatesLines(t *testing.T) {
	var out strings.Builder
	write := Writer(&out)
	write("value=%d", 3)
	if got := out.String(); got != "value=3\n" {
		t.Fatalf("output = %q", got)
	}
}
