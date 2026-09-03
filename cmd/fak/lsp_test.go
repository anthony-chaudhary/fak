package main

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

func TestRunLSPCleanExit(t *testing.T) {
	var in, out, errb bytes.Buffer
	initMsg := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}`
	shutdownMsg := `{"jsonrpc":"2.0","id":2,"method":"shutdown"}`
	exitMsg := `{"jsonrpc":"2.0","method":"exit"}`
	for _, m := range []string{initMsg, shutdownMsg, exitMsg} {
		in.WriteString(fmt.Sprintf("Content-Length: %d\r\n\r\n%s", len(m), m))
	}
	code := runLSP(&in, &out, &errb, nil)
	if code != 0 {
		t.Fatalf("runLSP exited %d, stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "fak-native-lsp") {
		t.Fatalf("runLSP output missing serverInfo: %s", out.String())
	}
}
