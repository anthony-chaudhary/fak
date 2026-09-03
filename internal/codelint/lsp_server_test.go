package codelint

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"
)

func TestFindingToLSPDiagnostic(t *testing.T) {
	f := Finding{
		Pack:     "go",
		Code:     "GO_PARSE",
		File:     "main.go",
		Line:     10,
		Col:      5,
		Severity: Error,
		Detail:   "expected ';', found 'EOF'",
	}
	d := FindingToLSPDiagnostic(f)
	if d.Severity != LSPSeverityError {
		t.Errorf("got severity %d, want %d", d.Severity, LSPSeverityError)
	}
	if d.Range.Start.Line != 9 || d.Range.Start.Character != 4 {
		t.Errorf("got 0-based pos line %d, char %d; want 9, 4", d.Range.Start.Line, d.Range.Start.Character)
	}
	if d.Code != "GO_PARSE" || d.Source != "fak/go" || d.Message != "expected ';', found 'EOF'" {
		t.Errorf("unexpected fields in diagnostic: %+v", d)
	}
}

func TestLSPServerLifecycleAndDiagnostics(t *testing.T) {
	var clientIn bytes.Buffer  // server reads from clientIn
	var serverOut bytes.Buffer // server writes to serverOut

	// Helper to send frame from client
	sendFrame := func(payload string) {
		clientIn.WriteString(fmt.Sprintf("Content-Length: %d\r\n\r\n%s", len(payload), payload))
	}

	// 1. Send initialize
	sendFrame(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"processId":123,"rootUri":"file:///test"}}`)

	// 2. Send initialized notification
	sendFrame(`{"jsonrpc":"2.0","method":"initialized","params":{}}`)

	// 3. Open broken Go file
	brokenSrc := "package main\n\nfunc broken( {\n"
	brokenOpen := map[string]any{
		"jsonrpc": "2.0",
		"method":  "textDocument/didOpen",
		"params": map[string]any{
			"textDocument": map[string]any{
				"uri":        "file:///test/broken.go",
				"languageId": "go",
				"version":    1,
				"text":       brokenSrc,
			},
		},
	}
	bOpenJSON, _ := json.Marshal(brokenOpen)
	sendFrame(string(bOpenJSON))

	// 4. Request document symbols
	docSymbolReq := `{"jsonrpc":"2.0","id":2,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":"file:///test/valid.go"}}}`
	sendFrame(docSymbolReq)

	// 5. Change broken file to valid
	validSrc := "package main\n\ntype Foo struct {\n\tBar int\n}\n\nfunc valid() {}\n"
	changeMsg := map[string]any{
		"jsonrpc": "2.0",
		"method":  "textDocument/didChange",
		"params": map[string]any{
			"textDocument": map[string]any{
				"uri":     "file:///test/broken.go",
				"version": 2,
			},
			"contentChanges": []map[string]any{
				{"text": validSrc},
			},
		},
	}
	changeJSON, _ := json.Marshal(changeMsg)
	sendFrame(string(changeJSON))

	// 6. Request symbols for now-valid file
	docSymbolReq2 := `{"jsonrpc":"2.0","id":3,"method":"textDocument/documentSymbol","params":{"textDocument":{"uri":"file:///test/broken.go"}}}`
	sendFrame(docSymbolReq2)

	// 7. Close document
	closeMsg := `{"jsonrpc":"2.0","method":"textDocument/didClose","params":{"textDocument":{"uri":"file:///test/broken.go"}}}`
	sendFrame(closeMsg)

	// 8. Shutdown and exit
	sendFrame(`{"jsonrpc":"2.0","id":4,"method":"shutdown"}`)
	sendFrame(`{"jsonrpc":"2.0","method":"exit"}`)

	server := NewLSPServer(&clientIn, &serverOut, nil)
	if err := server.Run(context.Background()); err != nil {
		t.Fatalf("server run error: %v", err)
	}

	// Now parse messages written by server to serverOut
	messages := readAllFrames(t, &serverOut)
	if len(messages) < 6 {
		t.Fatalf("expected at least 6 server messages, got %d", len(messages))
	}

	// Msg 0: Initialize response
	var initResp struct {
		ID     int `json:"id"`
		Result struct {
			Capabilities struct {
				DocumentSymbolProvider bool `json:"documentSymbolProvider"`
			} `json:"capabilities"`
		} `json:"result"`
	}
	if err := json.Unmarshal(messages[0], &initResp); err != nil {
		t.Fatalf("unmarshal init response: %v", err)
	}
	if initResp.ID != 1 || !initResp.Result.Capabilities.DocumentSymbolProvider {
		t.Errorf("unexpected init response: %s", string(messages[0]))
	}

	// Msg 1: PublishDiagnostics for broken.go (should have parse error)
	var diag1 struct {
		Method string `json:"method"`
		Params struct {
			URI         string          `json:"uri"`
			Diagnostics []LSPDiagnostic `json:"diagnostics"`
		} `json:"params"`
	}
	if err := json.Unmarshal(messages[1], &diag1); err != nil {
		t.Fatalf("unmarshal diag1: %v", err)
	}
	if diag1.Method != "textDocument/publishDiagnostics" || len(diag1.Params.Diagnostics) == 0 {
		t.Fatalf("expected diagnostics for broken code, got: %s", string(messages[1]))
	}
	if diag1.Params.Diagnostics[0].Code != "GO_PARSE" {
		t.Errorf("expected GO_PARSE error, got %v", diag1.Params.Diagnostics[0].Code)
	}

	// Find the symbol response for broken.go after fix (ID=3)
	var foundSymbols bool
	for _, m := range messages {
		var symResp struct {
			ID     int                 `json:"id"`
			Result []LSPDocumentSymbol `json:"result"`
		}
		if err := json.Unmarshal(m, &symResp); err == nil && symResp.ID == 3 {
			if len(symResp.Result) < 2 {
				t.Errorf("expected at least 2 symbols (Foo, valid), got %d: %s", len(symResp.Result), string(m))
			} else {
				names := []string{symResp.Result[0].Name, symResp.Result[1].Name}
				if !strings.Contains(strings.Join(names, ","), "Foo") || !strings.Contains(strings.Join(names, ","), "valid") {
					t.Errorf("expected symbols Foo and valid, got %v", names)
				}
			}
			foundSymbols = true
			break
		}
	}
	if !foundSymbols {
		t.Errorf("did not find documentSymbol response for ID=3")
	}

	// Find close diagnostics (should be empty array)
	var foundCloseDiag bool
	for _, m := range messages {
		var d struct {
			Method string `json:"method"`
			Params struct {
				URI         string          `json:"uri"`
				Diagnostics []LSPDiagnostic `json:"diagnostics"`
			} `json:"params"`
		}
		if err := json.Unmarshal(m, &d); err == nil && d.Method == "textDocument/publishDiagnostics" && len(d.Params.Diagnostics) == 0 {
			foundCloseDiag = true
			break
		}
	}
	if !foundCloseDiag {
		t.Errorf("did not find empty diagnostics after didClose")
	}
}

func readAllFrames(t *testing.T, r io.Reader) [][]byte {
	t.Helper()
	br := bufioReader(r)
	var frames [][]byte
	for {
		frame, err := readLSPFrame(br)
		if err != nil {
			break
		}
		frames = append(frames, frame)
	}
	return frames
}

func bufioReader(r io.Reader) *bufio.Reader {
	if br, ok := r.(*bufio.Reader); ok {
		return br
	}
	return bufio.NewReader(r)
}
