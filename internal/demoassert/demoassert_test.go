package demoassert

import (
	"bytes"
	"strings"
	"testing"
)

func TestRecorder(t *testing.T) {
	var out bytes.Buffer
	r := Recorder{Writer: &out}
	r.Fail("got %d", 2)
	if !r.Failed || !strings.Contains(out.String(), "FAIL: got 2") {
		t.Fatalf("recorder = %+v %q", r, out.String())
	}
}
