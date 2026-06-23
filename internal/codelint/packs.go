package codelint

// packs.go holds the built-in language-server packs and the off-the-hot-path runner
// that shells out to an external checker. Adding a language is one entry in
// DefaultPacks (an in-process Check, or a cmdPack that names a binary + a parser).

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"go/parser"
	"go/scanner"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// DefaultPacks returns the kernel's built-in packs: Go and JSON parse in-process
// (the stdlib already knows those grammars, so they need no external tool and always
// have an opinion); Python and CUDA shell out to their toolchains and degrade to "no
// opinion" where the toolchain is absent (the common case for CUDA off a GPU host).
func DefaultPacks() []Pack {
	return []Pack{
		goPack(),
		jsonPack(),
		pythonPack(),
		cudaPack(),
	}
}

// ---- in-process packs ------------------------------------------------------

func goPack() Pack { return Pack{Lang: "go", Exts: []string{".go"}, Check: goCheck} }

// goCheck parses the file with the stdlib Go parser and reports every syntax error.
// This is the most valuable, zero-false-positive tier: a parse error is code no Go
// toolchain would accept, decided in-process with no gofmt/gopls dependency. It does
// NOT do type/semantic checking — that needs the whole package's context, which a
// single agent-written file in isolation does not have, and would emit false
// "undefined: X" findings on perfectly good code.
func goCheck(_ context.Context, path string) ([]Finding, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	if _, perr := parser.ParseFile(fset, path, src, parser.AllErrors); perr != nil {
		var el scanner.ErrorList
		if errors.As(perr, &el) {
			out := make([]Finding, 0, len(el))
			for _, e := range el {
				out = append(out, Finding{
					Pack: "go", Code: "GO_PARSE", File: path,
					Line: e.Pos.Line, Col: e.Pos.Column, Severity: Error, Detail: e.Msg,
				})
			}
			return out, nil
		}
		return []Finding{{Pack: "go", Code: "GO_PARSE", File: path, Severity: Error, Detail: perr.Error()}}, nil
	}
	return nil, nil
}

func jsonPack() Pack { return Pack{Lang: "json", Exts: []string{".json"}, Check: jsonCheck} }

// jsonCheck validates the file is well-formed JSON, reporting the first syntax error
// with its line:col (the stdlib decoder surfaces a byte offset, which we convert).
func jsonCheck(_ context.Context, path string) ([]Finding, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var v any
	if uerr := json.Unmarshal(src, &v); uerr != nil {
		line, col := 0, 0
		var se *json.SyntaxError
		if errors.As(uerr, &se) {
			line, col = offsetToLineCol(src, int(se.Offset))
		}
		return []Finding{{
			Pack: "json", Code: "JSON_PARSE", File: path,
			Line: line, Col: col, Severity: Error, Detail: uerr.Error(),
		}}, nil
	}
	return nil, nil
}

// offsetToLineCol maps a 0-based byte offset into 1-based line/column.
func offsetToLineCol(src []byte, off int) (line, col int) {
	if off < 0 {
		return 0, 0
	}
	if off > len(src) {
		off = len(src)
	}
	line, col = 1, 1
	for i := 0; i < off; i++ {
		if src[i] == '\n' {
			line++
			col = 1
		} else {
			col++
		}
	}
	return line, col
}

// ---- external (shell-out) packs --------------------------------------------

// pyCheckSrc compiles the target with the stdlib compiler and prints one normalized
// "<file>:<line>:<col>: error: <Type>: <msg>" line per syntax error to stderr, then
// exits non-zero. Controlling the output format (rather than scraping a traceback)
// keeps the parser robust across Python versions. Python is the canonical tensor
// language (PyTorch / JAX / NumPy), so this is the "lint code like tensors" pack.
const pyCheckSrc = `import sys
try:
    compile(open(sys.argv[1], 'rb').read(), sys.argv[1], 'exec')
except SyntaxError as e:
    sys.stderr.write("%s:%d:%d: error: %s: %s\n" % (
        sys.argv[1], e.lineno or 0, e.offset or 0, type(e).__name__, e.msg))
    sys.exit(1)
`

func pythonPack() Pack {
	return cmdPack(cmdSpec{
		Lang: "python",
		Exts: []string{".py", ".pyi"},
		Bins: []string{"python3", "python"},
		Args: func(file string) []string { return []string{"-I", "-c", pyCheckSrc, file} },
		Code: "PYTHON_SYNTAX",
	})
}

// cudaPack lints CUDA kernels (the tensor code that runs ON the GPU) with nvcc in
// syntax-only mode. nvcc is absent on every host except a CUDA box, so this pack is
// a no-op there and lints for real on the DGX/GPU host — the same graceful-degrade
// the rest of the GPU surface uses.
func cudaPack() Pack {
	return cmdPack(cmdSpec{
		Lang: "cuda",
		Exts: []string{".cu", ".cuh"},
		Bins: []string{"nvcc"},
		Args: func(file string) []string { return []string{"-fsyntax-only", "-x", "cu", file} },
		Code: "CUDA_SYNTAX",
	})
}

// cmdSpec declares an external-checker pack: the binaries to try (first found on
// PATH wins), the argv builder, and the closed Code its error findings carry.
type cmdSpec struct {
	Lang string
	Exts []string
	Bins []string
	Args func(file string) []string
	Code string // finding Code for an error from this checker
}

// cmdPack wraps a cmdSpec into a Pack whose Check shells out off the hot path,
// bounded by DefaultTimeout, parses the checker's GCC/MSVC-style diagnostics, and
// degrades to (nil, nil) when no binary is found or the run could not start.
func cmdPack(s cmdSpec) Pack {
	return Pack{
		Lang: s.Lang,
		Exts: s.Exts,
		Check: func(ctx context.Context, path string) ([]Finding, error) {
			bin := firstOnPath(s.Bins)
			if bin == "" {
				return nil, nil // checker absent: no opinion
			}
			stdout, stderr, ran := runChecker(ctx, bin, s.Args(path))
			if !ran {
				return nil, nil // could not start / timed out: no opinion
			}
			base := filepath.Base(path)
			var out []Finding
			for _, f := range parseDiagnostics(s.Lang, s.Code, stdout, stderr) {
				// Keep only diagnostics about the file we asked to lint (a checker may
				// also mention headers it pulled in); re-address them to path so the
				// caller sees its own filename, not a temp path.
				if f.File == "" || filepath.Base(f.File) == base {
					f.File = path
					out = append(out, f)
				}
			}
			return out, nil
		},
	}
}

func firstOnPath(bins []string) string {
	for _, b := range bins {
		if p, err := exec.LookPath(b); err == nil {
			return p
		}
	}
	return ""
}

// runChecker runs bin off the hot path under DefaultTimeout. ran=false means the
// process could not be started or was killed by the timeout (treated as "no
// opinion"); a non-zero exit with output is normal — that is the checker reporting
// problems — so it returns ran=true.
func runChecker(ctx context.Context, bin string, args []string) (stdout, stderr string, ran bool) {
	cctx, cancel := context.WithTimeout(ctx, DefaultTimeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, bin, args...)
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	err := cmd.Run()
	if cctx.Err() != nil {
		return "", "", false // timed out / cancelled
	}
	if err != nil {
		var ee *exec.ExitError
		if !errors.As(err, &ee) {
			return "", "", false // failed to start, not a normal non-zero exit
		}
	}
	return out.String(), errb.String(), true
}

var (
	// GCC / Clang / nvcc (POSIX): path:line[:col]: error|warning: message
	reGCC = regexp.MustCompile(`^(.*?):(\d+):(?:(\d+):)?\s*(error|fatal error|warning):\s*(.*)$`)
	// MSVC / nvcc (Windows host): path(line[,col]): error|warning C####: message
	reMSVC = regexp.MustCompile(`^(.*?)\((\d+)(?:,(\d+))?\):\s*(error|fatal error|warning)[^:]*:\s*(.*)$`)
)

// parseDiagnostics turns a checker's combined output into findings, tolerating both
// the POSIX (path:line:col:) and Windows (path(line):) diagnostic formats. Lines
// that match neither are ignored, so banner/usage noise never becomes a false
// finding. errorCode is the Code stamped on error-severity findings.
func parseDiagnostics(pack, errorCode, stdout, stderr string) []Finding {
	var out []Finding
	for _, raw := range strings.Split(stdout+"\n"+stderr, "\n") {
		line := strings.TrimRight(raw, "\r")
		m := reGCC.FindStringSubmatch(line)
		if m == nil {
			m = reMSVC.FindStringSubmatch(line)
		}
		if m == nil {
			continue
		}
		ln, _ := strconv.Atoi(m[2])
		col, _ := strconv.Atoi(m[3])
		sev := Warning
		code := strings.ToUpper(pack) + "_WARNING"
		if strings.Contains(m[4], "error") {
			sev = Error
			code = errorCode
		}
		out = append(out, Finding{
			Pack: pack, Code: code, File: m[1], Line: ln, Col: col, Severity: sev, Detail: m[5],
		})
	}
	return out
}
