package main

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
)

func TestSelfcheckCapturesStablePrefixAndDynamicTurn(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run(context.Background(), &stdout, &stderr, []string{"-selfcheck"}); code != 0 {
		t.Fatalf("run=%d stderr=%s", code, stderr.String())
	}
	var got witness
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Verdict != "PASS" || !got.StablePrefixUnchanged || !got.FullDigestChanged || len(got.Turns) != 2 {
		t.Fatalf("bad witness: %#v", got)
	}
	if got.Turns[0].Snapshot.Fragments[0].Content == got.Turns[1].Snapshot.Fragments[0].Content {
		t.Fatal("turn-scoped content did not change")
	}
}
