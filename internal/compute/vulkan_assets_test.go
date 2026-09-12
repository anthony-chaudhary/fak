package compute

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveVulkanSPIRVDir(t *testing.T) {
	t.Run("explicit override wins unchanged", func(t *testing.T) {
		executableCalled := false
		evalSymlinksCalled := false
		statCalled := false
		want := "  ../custom shaders  "
		got := resolveVulkanSPIRVDir(
			want,
			func() (string, error) {
				executableCalled = true
				return "", errors.New("must not be called")
			},
			func(string) (string, error) {
				evalSymlinksCalled = true
				return "", errors.New("must not be called")
			},
			func(string) (fs.FileInfo, error) {
				statCalled = true
				return nil, errors.New("must not be called")
			},
		)
		if got != want {
			t.Fatalf("resolveVulkanSPIRVDir(explicit) = %q, want byte-for-byte %q", got, want)
		}
		if executableCalled || evalSymlinksCalled || statCalled {
			t.Fatalf("explicit override invoked callbacks: executable=%v evalSymlinks=%v stat=%v", executableCalled, evalSymlinksCalled, statCalled)
		}
	})

	t.Run("unavailable executable", func(t *testing.T) {
		statCalled := false
		for _, tc := range []struct {
			name string
			path string
			err  error
		}{
			{name: "error", err: errors.New("executable unavailable")},
			{name: "empty path"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got := resolveVulkanSPIRVDir("", func() (string, error) {
					return tc.path, tc.err
				}, func(path string) (string, error) {
					return path, nil
				}, func(string) (fs.FileInfo, error) {
					statCalled = true
					return nil, nil
				})
				if got != "" {
					t.Fatalf("resolveVulkanSPIRVDir() = %q, want empty", got)
				}
			})
		}
		if statCalled {
			t.Fatal("unavailable executable invoked stat callback")
		}
	})

	t.Run("unavailable resolved executable", func(t *testing.T) {
		statCalled := false
		for _, tc := range []struct {
			name string
			path string
			err  error
		}{
			{name: "error", err: errors.New("symlink evaluation failed")},
			{name: "empty path"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got := resolveVulkanSPIRVDir("", func() (string, error) {
					return filepath.Join(t.TempDir(), "fak"), nil
				}, func(string) (string, error) {
					return tc.path, tc.err
				}, func(string) (fs.FileInfo, error) {
					statCalled = true
					return nil, nil
				})
				if got != "" {
					t.Fatalf("resolveVulkanSPIRVDir() = %q, want empty", got)
				}
			})
		}
		if statCalled {
			t.Fatal("unavailable resolved executable invoked stat callback")
		}
	})

	t.Run("candidate validation", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			prepare func(*testing.T, string)
			stat    func(string) (fs.FileInfo, error)
			wantDir bool
		}{
			{
				name: "stat error",
				stat: func(string) (fs.FileInfo, error) {
					return nil, errors.New("stat failed")
				},
			},
			{name: "missing", stat: os.Stat},
			{
				name: "regular file",
				prepare: func(t *testing.T, candidate string) {
					t.Helper()
					if err := os.WriteFile(candidate, []byte("not a directory"), 0o600); err != nil {
						t.Fatal(err)
					}
				},
				stat: os.Stat,
			},
			{
				name: "directory",
				prepare: func(t *testing.T, candidate string) {
					t.Helper()
					if err := os.Mkdir(candidate, 0o700); err != nil {
						t.Fatal(err)
					}
				},
				stat:    os.Stat,
				wantDir: true,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				binDir := t.TempDir()
				executable := filepath.Join(binDir, "fak")
				candidate := filepath.Join(binDir, "spirv")
				if tc.prepare != nil {
					tc.prepare(t, candidate)
				}
				got := resolveVulkanSPIRVDir("", func() (string, error) {
					return executable, nil
				}, func(path string) (string, error) {
					return path, nil
				}, tc.stat)
				want := ""
				if tc.wantDir {
					want = candidate
				}
				if got != want {
					t.Fatalf("resolveVulkanSPIRVDir() = %q, want %q", got, want)
				}
			})
		}
	})

	t.Run("relocation ignores working directory", func(t *testing.T) {
		installDir := t.TempDir()
		candidate := filepath.Join(installDir, "spirv")
		if err := os.Mkdir(candidate, 0o700); err != nil {
			t.Fatal(err)
		}

		workingDir := t.TempDir()
		if err := os.Mkdir(filepath.Join(workingDir, "spirv"), 0o700); err != nil {
			t.Fatal(err)
		}
		t.Chdir(workingDir)

		got := resolveVulkanSPIRVDir("", func() (string, error) {
			return filepath.Join(workingDir, "shim", "fak"), nil
		}, func(string) (string, error) {
			return filepath.Join(installDir, "fak"), nil
		}, os.Stat)
		if got != candidate {
			t.Fatalf("resolveVulkanSPIRVDir() = %q, want executable-relative %q", got, candidate)
		}
	})
}
