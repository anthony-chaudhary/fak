package pyexec

import (
	"errors"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

type fakeFS struct {
	files map[string]bool
	works map[string]bool
	path  map[string]string
}

func (f fakeFS) host(env map[string]string) host {
	return host{
		goos:   "windows",
		getenv: func(k string) string { return env[k] },
		glob: func(pattern string) ([]string, error) {
			var out []string
			for p := range f.files {
				if ok, _ := filepath.Match(strings.ReplaceAll(pattern, `\`, "/"), strings.ReplaceAll(p, `\`, "/")); ok {
					out = append(out, p)
				}
			}
			return out, nil
		},
		isFile: func(p string) bool { return f.files[p] },
		lookPath: func(name string) (string, error) {
			if p, ok := f.path[name]; ok {
				return p, nil
			}
			return "", exec.ErrNotFound
		},
		probe: func(i Interpreter) bool { return f.works[i.Path] },
	}
}

const (
	storeStub = `C:\Users\u\AppData\Local\Microsoft\WindowsApps\python.exe`
	pf312     = `C:\Program Files\Python312\python.exe`
	pf39      = `C:\Program Files\Python39\python.exe`
	conda     = `C:\Users\u\miniconda3\python.exe`
	launcher  = `C:\Windows\py.exe`
)

var winEnv = map[string]string{
	"ProgramFiles": `C:\Program Files`,
	"LOCALAPPDATA": `C:\Users\u\AppData\Local`,
	"USERPROFILE":  `C:\Users\u`,
	"SystemRoot":   `C:\Windows`,
}

// fak-test:runtime fast est=10ms
func TestResolveOffPathUsesAbsoluteInstall(t *testing.T) {
	cases := []struct {
		name string
		fs   fakeFS
		want Interpreter
		err  error
	}{
		{"store_stub_on_path_rejected", fakeFS{
			files: map[string]bool{pf312: true},
			works: map[string]bool{storeStub: true, pf312: true},
			path:  map[string]string{"python": storeStub},
		}, Interpreter{Path: pf312}, nil},
		{"newest_program_files_install", fakeFS{
			files: map[string]bool{pf39: true, pf312: true},
			works: map[string]bool{pf39: true, pf312: true},
		}, Interpreter{Path: pf312}, nil},
		{"broken_install_skipped_for_conda", fakeFS{
			files: map[string]bool{pf312: true, conda: true},
			works: map[string]bool{conda: true},
		}, Interpreter{Path: conda}, nil},
		{"py_launcher_last", fakeFS{
			files: map[string]bool{launcher: true},
			works: map[string]bool{launcher: true},
		}, Interpreter{Path: launcher, Args: []string{"-3"}}, nil},
		{"nothing", fakeFS{path: map[string]string{"python": storeStub}, works: map[string]bool{storeStub: true}}, Interpreter{}, ErrUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.fs.host(winEnv).resolve()
			if !errors.Is(err, tc.err) || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("resolve() = %+v, %v; want %+v, %v", got, err, tc.want, tc.err)
			}
		})
	}
}

// fak-test:runtime fast est=10ms
func TestResolvePrefersWorkingPathEntry(t *testing.T) {
	fs := fakeFS{
		files: map[string]bool{pf312: true},
		works: map[string]bool{`D:\py\python.exe`: true, pf312: true},
		path:  map[string]string{"python": `D:\py\python.exe`},
	}
	got, err := fs.host(winEnv).resolve()
	if err != nil || got.Path != `D:\py\python.exe` {
		t.Fatalf("resolve() = %+v, %v", got, err)
	}
}

// fak-test:runtime fast est=10ms
func TestResolveNonWindowsIsPathOnly(t *testing.T) {
	fs := fakeFS{files: map[string]bool{pf312: true}, works: map[string]bool{pf312: true}}
	h := fs.host(winEnv)
	h.goos = "linux"
	if _, err := h.resolve(); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("resolve() err = %v, want ErrUnavailable", err)
	}
}

// fak-test:runtime fast est=10ms
func TestCommandLeavesNonPythonArgv(t *testing.T) {
	argv := []string{"go", "version"}
	if got := Command(argv); !reflect.DeepEqual(got, argv) {
		t.Fatalf("Command(%v) = %v", argv, got)
	}
}
