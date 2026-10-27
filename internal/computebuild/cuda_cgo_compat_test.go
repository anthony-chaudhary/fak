package computebuild

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestCUDACgoLoaderCompatibility(t *testing.T) {
	root := t.TempDir()
	toolkit, driver, stub, existing := filepath.Join(root, "toolkit"), filepath.Join(root, "driver"), filepath.Join(root, "stubs"), filepath.Join(root, "existing")
	for _, path := range []string{toolkit, driver, stub, existing} {
		if err := os.MkdirAll(path, 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(driver, "libcuda.so.1"), []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LD_LIBRARY_PATH", strings.Join([]string{stub, existing, toolkit}, string(os.PathListSeparator)))
	env := SynthesizeCUDACgoEnv(&Toolchain{CUDALib: []string{driver, toolkit, stub}}, root, false, nil)
	if strings.Contains(env["LD_LIBRARY_PATH"], stub) || !strings.Contains(env["LD_LIBRARY_PATH"], existing) || !strings.Contains(env["LD_LIBRARY_PATH"], toolkit) {
		t.Fatalf("loader=%q", env["LD_LIBRARY_PATH"])
	}
	if strings.Index(env["LD_LIBRARY_PATH"], driver) > strings.Index(env["LD_LIBRARY_PATH"], toolkit) {
		t.Fatal("driver must precede toolkit")
	}
	if !strings.Contains(env["CGO_LDFLAGS"], "-Wl,-rpath,"+QuoteArg(toolkit)) || strings.Contains(env["CGO_LDFLAGS"], "-Wl,-rpath,"+QuoteArg(driver)) || strings.Contains(env["CGO_LDFLAGS"], "-Wl,-rpath,"+QuoteArg(stub)) {
		t.Fatalf("linker=%q", env["CGO_LDFLAGS"])
	}
	win := SynthesizeCUDACgoEnv(&Toolchain{IsWindows: true, CUDALib: []string{toolkit}}, root, false, nil)
	if _, ok := win["LD_LIBRARY_PATH"]; ok {
		t.Fatal("Windows loader changed")
	}
}

// fak-test:runtime fast est=20ms
func TestCUDANCCLVersionedLibraryCompatibility(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Linux NCCL soname compatibility")
	}
	root := t.TempDir()
	home, pkg := filepath.Join(root, "nccl"), filepath.Join(root, "compute")
	for _, path := range []string{filepath.Join(home, "usr/include"), filepath.Join(home, "usr/lib"), pkg} {
		if err := os.MkdirAll(path, 0755); err != nil {
			t.Fatal(err)
		}
	}
	library := filepath.Join(home, "usr/lib/libnccl.so.2")
	if err := os.WriteFile(library, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	tc := &Toolchain{}
	for i := 0; i < 2; i++ {
		prepared, err := PrepareCUDANCCLToolchain(tc, home, root)
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(filepath.Join(root, ".fak/cuda-nccl-lib/libnccl.so"))
		if err != nil || string(data) != "fixture" {
			t.Fatalf("link data=%q err=%v", data, err)
		}
		if len(prepared.CUDAInc) != 1 || len(prepared.CUDALib) != 2 || len(tc.CUDALib) != 0 {
			t.Fatal("NCCL paths or caller toolchain changed")
		}
	}
	link := filepath.Join(root, ".fak/cuda-nccl-lib/libnccl.so")
	if err := os.Remove(link); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(link, []byte("owned regular file"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := PrepareCUDANCCLToolchain(tc, home, root); err == nil {
		t.Fatal("regular linker name must not be overwritten")
	}
}
