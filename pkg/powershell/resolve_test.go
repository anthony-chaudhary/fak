package powershell

import (
	"errors"
	"os/exec"
	"testing"
)

func fakeHost(env map[string]string, files ...string) host {
	present := map[string]bool{}
	for _, f := range files {
		present[f] = true
	}
	return host{
		getenv:   func(k string) string { return env[k] },
		isFile:   func(p string) bool { return present[p] },
		lookPath: func(string) (string, error) { return "", exec.ErrNotFound },
	}
}

// fak-test:runtime fast est=10ms
func TestResolveOrder(t *testing.T) {
	winPS := `D:\Win\System32\WindowsPowerShell\v1.0\powershell.exe`
	pwsh := `C:\Program Files\PowerShell\7\pwsh.exe`
	cases := []struct {
		name  string
		host  host
		want  string
		error error
	}{
		{"systemroot_without_path", fakeHost(map[string]string{"SystemRoot": `D:\Win`}, winPS, pwsh), winPS, nil},
		{"pwsh_fallback", fakeHost(map[string]string{"SystemRoot": `D:\Win`}, pwsh), pwsh, nil},
		{"none", fakeHost(nil), "", ErrUnavailable},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.host.resolve()
			if !errors.Is(err, tc.error) || got != tc.want {
				t.Fatalf("resolve() = %q, %v; want %q, %v", got, err, tc.want, tc.error)
			}
		})
	}
}

// fak-test:runtime fast est=10ms
func TestResolvePathLookupFallback(t *testing.T) {
	h := fakeHost(nil)
	h.lookPath = func(name string) (string, error) {
		if name == "pwsh.exe" {
			return `E:\bin\pwsh.exe`, nil
		}
		return "", exec.ErrNotFound
	}
	if got, err := h.resolve(); err != nil || got != `E:\bin\pwsh.exe` {
		t.Fatalf("resolve() = %q, %v", got, err)
	}
}
