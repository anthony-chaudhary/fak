package codelint

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
)

// LSPDocumentSymbol represents a programming construct (function, type, etc.) in a file.
type LSPDocumentSymbol struct {
	Name           string              `json:"name"`
	Detail         string              `json:"detail,omitempty"`
	Kind           int                 `json:"kind"`
	Range          LSPRange            `json:"range"`
	SelectionRange LSPRange            `json:"selectionRange"`
	Children       []LSPDocumentSymbol `json:"children,omitempty"`
}

// LSP symbol kinds matching the Language Server Protocol 3.17 specification.
const (
	SymbolKindStruct    = 23
	SymbolKindInterface = 11
	SymbolKindFunction  = 12
	SymbolKindVariable  = 13
	SymbolKindConstant  = 14
	SymbolKindMethod    = 6
)

// FindingToLSPDiagnostic converts a codelint Finding (1-based line/col) to an LSPDiagnostic (0-based).
func FindingToLSPDiagnostic(f Finding) LSPDiagnostic {
	line := f.Line - 1
	if line < 0 {
		line = 0
	}
	col := f.Col - 1
	if col < 0 {
		col = 0
	}
	sev := LSPSeverityError
	if f.Severity == Warning {
		sev = LSPSeverityWarning
	}
	return LSPDiagnostic{
		Range: LSPRange{
			Start: LSPPosition{Line: line, Character: col},
			End:   LSPPosition{Line: line, Character: col + 1},
		},
		Severity: sev,
		Code:     LSPDiagnosticCode(f.Code),
		Source:   "fak/" + f.Pack,
		Message:  f.Detail,
	}
}

type lspInboundMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type lspOutboundMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  any             `json:"params,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   *lspError       `json:"error,omitempty"`
}

type lspError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// LSPServer is an in-process Language Server Protocol server backed by codelint.
// It connects over stdio or any io.Reader/Writer pair, publishing diagnostics
// and document symbols at microsecond speeds with zero external toolchain dependencies.
type LSPServer struct {
	reader   *bufio.Reader
	writer   io.Writer
	mu       sync.Mutex
	registry *Registry
	docs     map[string]string
	docsMu   sync.RWMutex
}

// NewLSPServer constructs an LSPServer reading from r and writing to w.
func NewLSPServer(r io.Reader, w io.Writer, reg *Registry) *LSPServer {
	if reg == nil {
		reg = DefaultRegistry()
	}
	return &LSPServer{
		reader:   bufio.NewReader(r),
		writer:   w,
		registry: reg,
		docs:     make(map[string]string),
	}
}

// Run processes incoming LSP frames until EOF, context cancellation, or client exit.
func (s *LSPServer) Run(ctx context.Context) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		frame, err := readLSPFrame(s.reader)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		var msg lspInboundMessage
		if err := json.Unmarshal(frame, &msg); err != nil {
			continue
		}
		if s.dispatch(ctx, msg) {
			return nil
		}
	}
}

func (s *LSPServer) dispatch(ctx context.Context, msg lspInboundMessage) (exit bool) {
	switch msg.Method {
	case "initialize":
		s.reply(msg.ID, map[string]any{
			"capabilities": map[string]any{
				"textDocumentSync": map[string]any{
					"openClose": true,
					"change":    1, // Full sync
					"save":      true,
				},
				"documentSymbolProvider": true,
			},
			"serverInfo": map[string]any{
				"name":    "fak-native-lsp",
				"version": "1.0.0",
			},
		})
	case "initialized":
		// notification, no-op
	case "textDocument/didOpen":
		var p struct {
			TextDocument struct {
				URI  string `json:"uri"`
				Text string `json:"text"`
			} `json:"textDocument"`
		}
		if err := json.Unmarshal(msg.Params, &p); err == nil {
			s.setDoc(p.TextDocument.URI, p.TextDocument.Text)
			s.lintAndPublish(ctx, p.TextDocument.URI, []byte(p.TextDocument.Text))
		}
	case "textDocument/didChange":
		var p struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
			ContentChanges []struct {
				Text string `json:"text"`
			} `json:"contentChanges"`
		}
		if err := json.Unmarshal(msg.Params, &p); err == nil && len(p.ContentChanges) > 0 {
			text := p.ContentChanges[len(p.ContentChanges)-1].Text
			s.setDoc(p.TextDocument.URI, text)
			s.lintAndPublish(ctx, p.TextDocument.URI, []byte(text))
		}
	case "textDocument/didSave":
		var p struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
		}
		if err := json.Unmarshal(msg.Params, &p); err == nil {
			content := s.getDoc(p.TextDocument.URI)
			if content != "" {
				s.lintAndPublish(ctx, p.TextDocument.URI, []byte(content))
			} else {
				path := URIToPath(p.TextDocument.URI)
				if bytes, err := os.ReadFile(path); err == nil {
					s.lintAndPublish(ctx, p.TextDocument.URI, bytes)
				}
			}
		}
	case "textDocument/didClose":
		var p struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
		}
		if err := json.Unmarshal(msg.Params, &p); err == nil {
			s.deleteDoc(p.TextDocument.URI)
			s.notify("textDocument/publishDiagnostics", map[string]any{
				"uri":         p.TextDocument.URI,
				"diagnostics": []LSPDiagnostic{},
			})
		}
	case "textDocument/documentSymbol":
		var p struct {
			TextDocument struct {
				URI string `json:"uri"`
			} `json:"textDocument"`
		}
		if err := json.Unmarshal(msg.Params, &p); err == nil {
			s.handleDocumentSymbol(msg.ID, p.TextDocument.URI)
		} else {
			s.reply(msg.ID, []LSPDocumentSymbol{})
		}
	case "shutdown":
		s.reply(msg.ID, nil)
	case "exit":
		return true
	default:
		if len(msg.ID) > 0 {
			s.reply(msg.ID, nil)
		}
	}
	return false
}

func (s *LSPServer) handleDocumentSymbol(id json.RawMessage, uri string) {
	path := URIToPath(uri)
	if !strings.EqualFold(filepath.Ext(path), ".go") {
		s.reply(id, []LSPDocumentSymbol{})
		return
	}
	src := s.getDoc(uri)
	var srcBytes []byte
	if src != "" {
		srcBytes = []byte(src)
	} else {
		data, err := os.ReadFile(path)
		if err != nil {
			s.reply(id, []LSPDocumentSymbol{})
			return
		}
		srcBytes = data
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, srcBytes, 0)
	if err != nil {
		s.reply(id, []LSPDocumentSymbol{})
		return
	}
	symbols := documentSymbolsForGo(fset, f)
	s.reply(id, symbols)
}

func (s *LSPServer) lintAndPublish(ctx context.Context, uri string, content []byte) {
	path := URIToPath(uri)
	findings, _ := s.registry.LintBytes(ctx, path, content)
	diags := make([]LSPDiagnostic, 0, len(findings))
	for _, f := range findings {
		diags = append(diags, FindingToLSPDiagnostic(f))
	}
	s.notify("textDocument/publishDiagnostics", map[string]any{
		"uri":         uri,
		"diagnostics": diags,
	})
}

func (s *LSPServer) setDoc(uri, text string) {
	s.docsMu.Lock()
	defer s.docsMu.Unlock()
	s.docs[uri] = text
}

func (s *LSPServer) getDoc(uri string) string {
	s.docsMu.RLock()
	defer s.docsMu.RUnlock()
	return s.docs[uri]
}

func (s *LSPServer) deleteDoc(uri string) {
	s.docsMu.Lock()
	defer s.docsMu.Unlock()
	delete(s.docs, uri)
}

func (s *LSPServer) send(msg lspOutboundMessage) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return writeLSPFrame(s.writer, data)
}

func (s *LSPServer) reply(id json.RawMessage, result any) {
	_ = s.send(lspOutboundMessage{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	})
}

func (s *LSPServer) notify(method string, params any) {
	_ = s.send(lspOutboundMessage{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	})
}

// URIToPath extracts the filesystem path from a file:// URI.
func URIToPath(uri string) string {
	u, err := url.Parse(uri)
	if err != nil || u.Scheme != "file" {
		return uri
	}
	p := u.Path
	if runtime.GOOS == "windows" {
		p = strings.TrimPrefix(p, "/")
		p = filepath.FromSlash(p)
	}
	return p
}

func readLSPFrame(r *bufio.Reader) ([]byte, error) {
	contentLength := -1
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 && strings.EqualFold(strings.TrimSpace(parts[0]), "Content-Length") {
			n, err := strconv.Atoi(strings.TrimSpace(parts[1]))
			if err == nil {
				contentLength = n
			}
		}
	}
	if contentLength < 0 {
		return nil, fmt.Errorf("lsp: missing or invalid Content-Length")
	}
	buf := make([]byte, contentLength)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

func writeLSPFrame(w io.Writer, payload []byte) error {
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(payload))
	if _, err := io.WriteString(w, header); err != nil {
		return err
	}
	_, err := w.Write(payload)
	return err
}

func documentSymbolsForGo(fset *token.FileSet, file *ast.File) []LSPDocumentSymbol {
	var symbols []LSPDocumentSymbol
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			kind := SymbolKindFunction
			if d.Recv != nil && len(d.Recv.List) > 0 {
				kind = SymbolKindMethod
			}
			start := fset.Position(d.Pos())
			end := fset.Position(d.End())
			selStart := fset.Position(d.Name.Pos())
			selEnd := fset.Position(d.Name.End())
			symbols = append(symbols, LSPDocumentSymbol{
				Name: d.Name.Name,
				Kind: kind,
				Range: LSPRange{
					Start: LSPPosition{Line: start.Line - 1, Character: start.Column - 1},
					End:   LSPPosition{Line: end.Line - 1, Character: end.Column - 1},
				},
				SelectionRange: LSPRange{
					Start: LSPPosition{Line: selStart.Line - 1, Character: selStart.Column - 1},
					End:   LSPPosition{Line: selEnd.Line - 1, Character: selEnd.Column - 1},
				},
			})
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					kind := SymbolKindStruct
					if _, ok := s.Type.(*ast.InterfaceType); ok {
						kind = SymbolKindInterface
					}
					start := fset.Position(s.Pos())
					end := fset.Position(s.End())
					selStart := fset.Position(s.Name.Pos())
					selEnd := fset.Position(s.Name.End())
					symbols = append(symbols, LSPDocumentSymbol{
						Name: s.Name.Name,
						Kind: kind,
						Range: LSPRange{
							Start: LSPPosition{Line: start.Line - 1, Character: start.Column - 1},
							End:   LSPPosition{Line: end.Line - 1, Character: end.Column - 1},
						},
						SelectionRange: LSPRange{
							Start: LSPPosition{Line: selStart.Line - 1, Character: selStart.Column - 1},
							End:   LSPPosition{Line: selEnd.Line - 1, Character: selEnd.Column - 1},
						},
					})
				case *ast.ValueSpec:
					kind := SymbolKindVariable
					if d.Tok == token.CONST {
						kind = SymbolKindConstant
					}
					for _, name := range s.Names {
						start := fset.Position(name.Pos())
						end := fset.Position(name.End())
						symbols = append(symbols, LSPDocumentSymbol{
							Name: name.Name,
							Kind: kind,
							Range: LSPRange{
								Start: LSPPosition{Line: start.Line - 1, Character: start.Column - 1},
								End:   LSPPosition{Line: end.Line - 1, Character: end.Column - 1},
							},
							SelectionRange: LSPRange{
								Start: LSPPosition{Line: start.Line - 1, Character: start.Column - 1},
								End:   LSPPosition{Line: end.Line - 1, Character: end.Column - 1},
							},
						})
					}
				}
			}
		}
	}
	return symbols
}
