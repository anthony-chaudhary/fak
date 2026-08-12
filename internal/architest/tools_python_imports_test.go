package architest

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// TestToolsPythonImportsResolve is the sibling of TestWorkflowCommandsResolve, for the other
// half of the same port-then-delete failure.
//
// That test catches a retired tools/X.py that a .github/workflows/*.yml still invokes. It cannot
// catch a retired tools/X.py that another tools/*.py still IMPORTS, because the reference is a
// module name rather than a path — and that is exactly the gap the fleet_version retirement fell
// through. 6b45eab293 ("refactor(fleetversion): port and retire Python reporter (#6333)") deleted
// tools/fleet_version.py after porting its logic to internal/appversion, but 31 committed tools
// still did `import fleet_version`, and Go is not importable from Python. Nothing failed at commit
// time; the module simply stopped existing, and `python3 tools/fleet_sessions.py` died with
// ModuleNotFoundError on a clean checkout of the trunk tip.
//
// This walks tools/**/*.py and asserts every non-external module it imports resolves to a real
// file. A retirement now has to migrate its importers in the same commit, or fail HERE — in the
// always-on `go test ./...` gate, naming the missing module and who still imports it — instead of
// silently breaking every consumer until someone runs one by hand. Pure-stdlib and hermetic
// (reads tracked files, no network, no subprocess), matching the guard it sits beside.
func TestToolsPythonImportsResolve(t *testing.T) {
	root := filepath.Dir(internalDir(t)) // repo root = parent of internal/
	toolsDir := filepath.Join(root, "tools")

	// importer path -> module names it imports that do not resolve.
	dangling := map[string][]string{}
	scanned := 0

	err := filepath.WalkDir(toolsDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.HasSuffix(path, ".py") {
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		scanned++
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			rel = path
		}
		rel = filepath.ToSlash(rel)
		for _, mod := range pythonImports(string(data)) {
			if externalPythonModules[mod] {
				continue
			}
			if pythonModuleExists(filepath.Dir(path), toolsDir, mod) {
				continue
			}
			dangling[rel] = append(dangling[rel], mod)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walk tools dir %s: %v", toolsDir, err)
	}
	if scanned == 0 {
		t.Fatalf("scanned no .py files under %s — the guard would pass vacuously", toolsDir)
	}

	if len(dangling) == 0 {
		return
	}

	// Report by MODULE, not by file: one retirement breaks many importers, and the module name
	// is the thing to re-add or migrate.
	byModule := map[string][]string{}
	for importer, mods := range dangling {
		for _, mod := range mods {
			byModule[mod] = append(byModule[mod], importer)
		}
	}
	mods := make([]string, 0, len(byModule))
	for mod := range byModule {
		mods = append(mods, mod)
	}
	sort.Strings(mods)

	for _, mod := range mods {
		importers := byModule[mod]
		sort.Strings(importers)
		shown := importers
		if len(shown) > 8 {
			shown = shown[:8]
		}
		t.Errorf("tools/%s.py does not exist, but %d tool(s) import %q: %s%s\n"+
			"\tEither re-add the module, migrate every importer off it, or — if %q is a new "+
			"stdlib/third-party dependency rather than an in-repo module — add it to "+
			"externalPythonModules in this file.",
			mod, len(importers), mod, strings.Join(shown, ", "),
			map[bool]string{true: ", ...", false: ""}[len(importers) > len(shown)], mod)
	}
}

var (
	rePyImport = regexp.MustCompile(`^import\s+(.+)$`)
	rePyFrom   = regexp.MustCompile(`^from\s+([.\w]+)\s+import\s+\S`)
	// Triple-quoted blocks are stripped before scanning so that prose in a docstring — this
	// repo's tools are heavily documented, and lines like "import the roster first" occur —
	// cannot be mistaken for an import statement. Go's regexp has no backreferences, so the
	// two quote styles are stripped in separate passes.
	reTripleDouble = regexp.MustCompile(`(?s)"""(.*?)"""`)
	reTripleSingle = regexp.MustCompile(`(?s)'''(.*?)'''`)
)

// pythonImports returns the distinct top-level module names imported by one Python source.
//
// It is deliberately conservative: it reads only unambiguous, full-line `import x` / `from x
// import y` statements (leading whitespace allowed, since several tools import inside a
// try/except or a function), and takes the first dotted segment, which is the module that has to
// exist on sys.path. Relative imports are skipped — `from . import x` resolves inside a package
// and names no top-level module.
func pythonImports(src string) []string {
	src = reTripleDouble.ReplaceAllString(src, "")
	src = reTripleSingle.ReplaceAllString(src, "")

	seen := map[string]bool{}
	var out []string
	add := func(name string) {
		name = strings.TrimSpace(name)
		if idx := strings.Index(name, "."); idx >= 0 {
			name = name[:idx] // `import os.path` needs `os`
		}
		if name == "" || seen[name] || !isPythonIdent(name) {
			return
		}
		seen[name] = true
		out = append(out, name)
	}

	for _, line := range strings.Split(src, "\n") {
		line = strings.TrimSpace(strings.TrimSuffix(line, "\r"))
		if m := rePyFrom.FindStringSubmatch(line); m != nil {
			if strings.HasPrefix(m[1], ".") {
				continue // relative import
			}
			add(m[1])
			continue
		}
		m := rePyImport.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		rest := m[1]
		if idx := strings.Index(rest, "#"); idx >= 0 {
			rest = rest[:idx] // `import fleet_version  # noqa: E402`
		}
		// `import a, b as c` — each comma-separated clause names one module.
		for _, clause := range strings.Split(rest, ",") {
			if idx := strings.Index(clause, " as "); idx >= 0 {
				clause = clause[:idx]
			}
			add(clause)
		}
	}
	return out
}

func isPythonIdent(s string) bool {
	for i, r := range s {
		switch {
		case r == '_':
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return s != ""
}

// pythonModuleExists reports whether a module name resolves to an in-repo file. A tool run as
// `python3 tools/x.py` gets its OWN directory on sys.path, so a nested tool resolves siblings in
// that directory; tools/ is checked as well because that is where the shared helpers live and
// where the nested tools insert on sys.path.
func pythonModuleExists(importerDir, toolsDir, mod string) bool {
	for _, dir := range []string{importerDir, toolsDir} {
		if _, err := os.Stat(filepath.Join(dir, mod+".py")); err == nil {
			return true
		}
		if _, err := os.Stat(filepath.Join(dir, mod, "__init__.py")); err == nil {
			return true
		}
	}
	return false
}

// externalPythonModules is every module a tool may import WITHOUT a corresponding tools/*.py:
// the Python standard library, plus the third-party and optional packages this repo's tools
// actually use. A name here can never be an in-repo module, so listing a stdlib superset is
// safe and keeps the guard from tripping on an ordinary new import.
var externalPythonModules = map[string]bool{
	// Standard library.
	"__future__": true, "abc": true, "argparse": true, "array": true, "ast": true,
	"asyncio": true, "atexit": true, "base64": true, "binascii": true, "bisect": true,
	"builtins": true, "bz2": true, "calendar": true, "cmath": true, "codecs": true,
	"collections": true, "colorsys": true, "concurrent": true, "configparser": true,
	"contextlib": true, "copy": true, "csv": true, "ctypes": true, "dataclasses": true,
	"datetime": true, "decimal": true, "difflib": true, "dis": true, "email": true,
	"enum": true, "errno": true, "faulthandler": true, "filecmp": true, "fileinput": true,
	"fnmatch": true, "fractions": true, "ftplib": true, "functools": true, "gc": true,
	"getopt": true, "getpass": true, "glob": true, "gzip": true, "hashlib": true,
	"heapq": true, "hmac": true, "html": true, "http": true, "imaplib": true,
	"importlib": true, "inspect": true, "io": true, "ipaddress": true, "itertools": true,
	"json": true, "keyword": true, "linecache": true, "locale": true, "logging": true,
	"lzma": true, "mailbox": true, "marshal": true, "math": true, "mimetypes": true,
	"mmap": true, "multiprocessing": true, "netrc": true, "numbers": true, "operator": true,
	"os": true, "pathlib": true, "pickle": true, "pkgutil": true, "platform": true,
	"plistlib": true, "posixpath": true, "pprint": true, "profile": true, "pstats": true,
	"pty": true, "pwd": true, "py_compile": true, "queue": true, "quopri": true,
	"random": true, "re": true, "readline": true, "reprlib": true, "resource": true,
	"runpy": true, "sched": true, "secrets": true, "select": true, "selectors": true,
	"shelve": true, "shlex": true, "shutil": true, "signal": true, "site": true,
	"smtplib": true, "socket": true, "socketserver": true, "sqlite3": true, "ssl": true,
	"stat": true, "statistics": true, "string": true, "stringprep": true, "struct": true,
	"subprocess": true, "symtable": true, "sys": true, "sysconfig": true, "tarfile": true,
	"tempfile": true, "termios": true, "textwrap": true, "threading": true, "time": true,
	"timeit": true, "token": true, "tokenize": true, "tomllib": true, "trace": true,
	"traceback": true, "tracemalloc": true, "tty": true, "types": true, "typing": true,
	"unicodedata": true, "unittest": true, "urllib": true, "uuid": true, "venv": true,
	"warnings": true, "wave": true, "weakref": true, "webbrowser": true, "xml": true,
	"xmlrpc": true, "zipfile": true, "zlib": true, "zoneinfo": true,

	// Third-party, and optional integrations imported inside try/except.
	"huggingface_hub": true, "imageio_ffmpeg": true, "matplotlib": true, "numpy": true,
	"openai": true, "PIL": true, "psutil": true, "pytest": true, "requests": true,
	"scipy": true, "yaml": true,

	// `dos` is the DOS substrate's own Python package, not an in-repo tools module:
	// tools/claims_salience_register.py imports dos.salience inside a try/except and
	// degrades to (None, "") when it is absent.
	"dos": true,
}
