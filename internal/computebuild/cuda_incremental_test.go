package computebuild

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// Native helper processes execute before testing parses compiler arguments.
func init() {
	if os.Getenv("FAK_CUDA_INCREMENTAL_HELPER") != "1" {
		return
	}
	a := os.Args[1:]
	if len(a) > 0 && a[0] == "build" {
		if err := os.WriteFile("go-link-flags", []byte(os.Getenv("CGO_LDFLAGS")), 0600); err != nil {
			os.Exit(26)
		}
		os.Exit(0)
	}
	for _, arg := range a {
		if arg == "--version" || arg == "/Bv" || arg == "/?" {
			fmt.Println("synthetic CUDA host compiler 1")
			os.Exit(0)
		}
		if strings.HasPrefix(arg, "-print-prog-name=") {
			fmt.Print(os.Args[0])
			os.Exit(0)
		}
	}
	if len(a) > 0 && (a[0] == "--list-elf" || a[0] == "--list-ptx") {
		if a[0] == "--list-elf" {
			fmt.Println("ELF file 1: cuda_kernels.sm_120.cubin")
		}
		os.Exit(0)
	}
	if len(a) > 2 && a[0] == "rcs" {
		if os.Getenv("FAK_CUDA_INCREMENTAL_ARCHIVE_FAIL") == "1" {
			os.WriteFile(a[1], []byte("partial"), 0600)
			os.Exit(19)
		}
		// Model ar's retain-on-update behavior, including its stale-member bug.
		members := map[string]bool{}
		if raw, err := os.ReadFile(a[1]); err == nil {
			_ = json.Unmarshal(raw, &members)
		}
		for _, obj := range a[2:] {
			members[filepath.Base(obj)] = true
		}
		raw, _ := json.Marshal(members)
		if os.WriteFile(a[1], raw, 0600) != nil {
			os.Exit(21)
		}
		os.Exit(0)
	}
	var source, output string
	dependency := false
	for i, arg := range a {
		if arg == "-M" {
			dependency = true
		}
		if strings.HasSuffix(arg, ".cu") {
			source = arg
		}
		if arg == "-o" && i+1 < len(a) {
			output = a[i+1]
		}
	}
	if dependency {
		if _, err := os.Stat("bad-dependencies"); err == nil {
			fmt.Println("invalid dependency data")
			os.Exit(0)
		}
		fmt.Printf("fak_cuda_dependency_target : %s \\\n", source)
		for _, path := range []string{"local.h", "transitive header.h", filepath.Join(os.Getenv("FAK_CUDA_INCREMENTAL_SDK"), "sdk.h")} {
			fmt.Printf(" %s", strings.ReplaceAll(path, " ", "\\ "))
		}
		fmt.Println()
		os.Exit(0)
	}
	if source == "" || output == "" {
		os.Exit(22)
	}
	f, err := os.OpenFile("compiles", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		os.Exit(23)
	}
	fmt.Fprintln(f, source)
	f.Close()
	if _, err := os.Stat("fail-compile"); err == nil {
		os.WriteFile(output, []byte("partial"), 0600)
		os.Exit(18)
	}
	if _, err := os.Stat("delay-compile"); err == nil {
		time.Sleep(70 * time.Millisecond)
	}
	var content []byte
	for _, path := range []string{source, "local.h", "transitive header.h", filepath.Join(os.Getenv("FAK_CUDA_INCREMENTAL_SDK"), "sdk.h")} {
		raw, err := os.ReadFile(path)
		if err != nil {
			os.Exit(24)
		}
		content = append(content, raw...)
	}
	if os.WriteFile(output, content, 0600) != nil {
		os.Exit(25)
	}
	os.Exit(0)
}

func cudaIncrementalFixture(t *testing.T) (string, *Toolchain) {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "internal", "compute")
	sdk := filepath.Join(root, "toolkit")
	for _, path := range []string{dir, filepath.Join(sdk, "bin"), filepath.Join(sdk, "nvvm", "libdevice")} {
		if err := os.MkdirAll(path, 0755); err != nil {
			t.Fatal(err)
		}
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(self)
	if err != nil {
		t.Fatal(err)
	}
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	tool := filepath.Join(sdk, "bin", "cuobjdump"+suffix)
	if err := os.WriteFile(tool, raw, 0755); err != nil {
		t.Fatal(err)
	}
	for path, body := range map[string]string{
		filepath.Join(dir, "cuda_arch.txt"):                     "sm_120\nsm_89\n",
		filepath.Join(dir, "cuda_kernels.cu"):                   "kernel\n",
		filepath.Join(dir, "cuda_nccl.cu"):                      "nccl\n",
		filepath.Join(dir, "cuda_nccl_pg.cu"):                   "nccl pg\n",
		filepath.Join(dir, "local.h"):                           "local\n",
		filepath.Join(dir, "transitive header.h"):               "transitive\n",
		filepath.Join(sdk, "sdk.h"):                             "SDK header\n",
		filepath.Join(sdk, "nvvm", "libdevice", "libdevice.bc"): "device bitcode\n",
	} {
		if err := os.WriteFile(path, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("FAK_CUDA_INCREMENTAL_HELPER", "1")
	t.Setenv("FAK_CUDA_INCREMENTAL_SDK", sdk)
	return root, &Toolchain{NVCC: tool, CXX: tool, AR: tool, CUDAHome: sdk}
}

func cudaCompileCount(t *testing.T, root string) int {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(root, "internal", "compute", "compiles"))
	if errors.Is(err, os.ErrNotExist) {
		return 0
	}
	if err != nil {
		t.Fatal(err)
	}
	return len(strings.Fields(string(raw)))
}

func cudaReadNativeReceipt(t *testing.T, root string) CUDANativeBuildReceipt {
	t.Helper()
	var r CUDANativeBuildReceipt
	raw, err := os.ReadFile(filepath.Join(root, DefaultCUDANativeBuildReceiptPath))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime slow est=25s
func TestCUDAIncrementalContentAndToolchainInvalidation(t *testing.T) {
	root, tc := cudaIncrementalFixture(t)
	compile := func(arch string, nccl bool) {
		t.Helper()
		if _, err := CompileCUDAKernels(context.Background(), tc, root, arch, nccl, io.Discard); err != nil {
			t.Fatal(err)
		}
	}
	compile("sm_120", false)
	if got := cudaCompileCount(t, root); got != 1 {
		t.Fatalf("cold compiles=%d", got)
	}
	compile("sm_120", false)
	if got := cudaCompileCount(t, root); got != 1 {
		t.Fatalf("unchanged compiles=%d; want 1 total", got)
	}
	r := cudaReadNativeReceipt(t, root)
	if r.Mode != "warm" || r.Executed != 0 || r.Reused != 1 {
		t.Fatalf("warm receipt: %+v", r)
	}
	for _, path := range []string{
		filepath.Join(root, "internal", "compute", "cuda_kernels.cu"),
		filepath.Join(root, "internal", "compute", "transitive header.h"),
		filepath.Join(tc.CUDAHome, "sdk.h"),
		filepath.Join(tc.CUDAHome, "nvvm", "libdevice", "libdevice.bc"),
	} {
		info, err := os.Stat(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("changed bytes"), 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(path, info.ModTime(), info.ModTime()); err != nil {
			t.Fatal(err)
		}
		before := cudaCompileCount(t, root)
		compile("sm_120", false)
		if got := cudaCompileCount(t, root); got != before+1 {
			t.Fatalf("content change %s compiled %d -> %d", path, before, got)
		}
	}
	// A host compiler replacement must invalidate reuse even with identical
	// reported version and timestamps. Appended executable data remains runnable.
	beforeHost := cudaCompileCount(t, root)
	info, err := os.Stat(tc.CXX)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(tc.CXX, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte{0}); err != nil {
		f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(tc.CXX, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	compile("sm_120", false)
	if got := cudaCompileCount(t, root); got != beforeHost+1 {
		t.Fatal("compiler content change reused object")
	}
	before := cudaCompileCount(t, root)
	compile("sm_89", false)
	if got := cudaCompileCount(t, root); got != before+1 {
		t.Fatalf("arch change compiles %d -> %d", before, got)
	}
	before = cudaCompileCount(t, root)
	compile("sm_89", true)
	if got := cudaCompileCount(t, root); got != before+3 {
		t.Fatalf("NCCL change compiles %d -> %d", before, got)
	}
	compile("sm_89", true)
	if got := cudaCompileCount(t, root); got != before+3 {
		t.Fatalf("warm NCCL compiled %d", got)
	}
	t.Setenv("FAK_CUDA_INCREMENTAL_EFFECTIVE_FLAG", "changed")
	before = cudaCompileCount(t, root)
	compile("sm_89", true)
	if got := cudaCompileCount(t, root); got != before+3 {
		t.Fatalf("environment change compiles %d -> %d", before, got)
	}
	tc.IsWindows = true
	t.Setenv("NVCC_CCBIN", tc.CXX)
	subtool := filepath.Join(filepath.Dir(tc.CXX), "c2.dll")
	if err := os.WriteFile(subtool, []byte("MSVC subtool"), 0600); err != nil {
		t.Fatal(err)
	}
	compile("sm_89", false)
	before = cudaCompileCount(t, root)
	compile("sm_89", false)
	if cudaCompileCount(t, root) != before {
		t.Fatal("unchanged Windows compiler closure missed cache")
	}
	if err := os.WriteFile(subtool, []byte("changed MSVC subtool"), 0600); err != nil {
		t.Fatal(err)
	}
	compile("sm_89", false)
	if cudaCompileCount(t, root) != before+1 {
		t.Fatal("MSVC subtool replacement reused object")
	}
	if _, err := cudaToolchainDigest(context.Background(), tc, append(os.Environ(), "Path=ambiguous")); err == nil {
		t.Fatal("ambiguous Windows environment accepted for reuse")
	}
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime slow est=20s
func TestCUDAIncrementalCorruptionFailureAndConcurrency(t *testing.T) {
	root, tc := cudaIncrementalFixture(t)
	compile := func() error {
		_, err := CompileCUDAKernels(context.Background(), tc, root, "sm_120", false, io.Discard)
		return err
	}
	if err := compile(); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "internal", "compute")
	obj := filepath.Join(dir, "cuda_kernels.o")
	for _, path := range []string{obj, cudaManifestPath(obj)} {
		if err := os.WriteFile(path, []byte("corrupt"), 0600); err != nil {
			t.Fatal(err)
		}
		before := cudaCompileCount(t, root)
		if err := compile(); err != nil {
			t.Fatal(err)
		}
		if got := cudaCompileCount(t, root); got != before+1 {
			t.Fatalf("corrupt %s did not compile", path)
		}
	}
	if err := os.Remove(obj); err != nil {
		t.Fatal(err)
	}
	beforeMissing := cudaCompileCount(t, root)
	if err := compile(); err != nil {
		t.Fatal(err)
	}
	if cudaCompileCount(t, root) != beforeMissing+1 {
		t.Fatal("missing object reused manifest")
	}
	header := filepath.Join(dir, "local.h")
	headerBytes, err := os.ReadFile(header)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(header); err != nil {
		t.Fatal(err)
	}
	if err := compile(); err == nil {
		t.Fatal("missing dependency accepted")
	}
	if _, err := os.Stat(cudaManifestPath(obj)); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("missing dependency retained reusable manifest")
	}
	if err := os.WriteFile(header, headerBytes, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bad-dependencies"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		before := cudaCompileCount(t, root)
		if err := compile(); err != nil {
			t.Fatal(err)
		}
		if cudaCompileCount(t, root) != before+1 {
			t.Fatal("invalid dependency reused object")
		}
	}
	if _, err := os.Stat(cudaManifestPath(obj)); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("uncacheable compile published manifest")
	}
	if err := os.Remove(filepath.Join(dir, "bad-dependencies")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "fail-compile"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	old, err := os.ReadFile(obj)
	if err != nil {
		t.Fatal(err)
	}
	if err := compile(); err == nil {
		t.Fatal("failed compiler accepted")
	}
	current, _ := os.ReadFile(obj)
	if string(current) != string(old) {
		t.Fatal("failed compile replaced prior object")
	}
	if _, err := os.Stat(cudaManifestPath(obj)); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("failed compile published manifest")
	}
	if err := os.Remove(filepath.Join(dir, "fail-compile")); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "delay-compile"), nil, 0600); err != nil {
		t.Fatal(err)
	}
	before := cudaCompileCount(t, root)
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for range 2 {
		wg.Add(1)
		go func() { defer wg.Done(); errs <- compile() }()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if got := cudaCompileCount(t, root); got != before+1 {
		t.Fatalf("overlapping builders compiled %d -> %d; want one", before, got)
	}
	if err := os.Remove(filepath.Join(tc.CUDAHome, "nvvm", "libdevice", "libdevice.bc")); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		before := cudaCompileCount(t, root)
		if err := compile(); err != nil {
			t.Fatal(err)
		}
		if cudaCompileCount(t, root) != before+1 {
			t.Fatal("incomplete SDK identity reused object")
		}
	}
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime fast est=10ms
func TestCUDADependencyParser(t *testing.T) {
	// NVCC 13.4.3 -M output starts with this exact target and continuation form.
	const nvcc134 = "fak_cuda_dependency_target : cuda_kernels.cu \\\n    /usr/include/stdc-predef.h \\\n    /usr/local/cuda/include/cuda_runtime.h \\\n    /usr/local/cuda/include/crt/host_config.h\n"
	want := []string{"cuda_kernels.cu", "/usr/include/stdc-predef.h", "/usr/local/cuda/include/cuda_runtime.h", "/usr/local/cuda/include/crt/host_config.h"}
	for _, raw := range []string{nvcc134, strings.ReplaceAll(nvcc134, "\n", "\r\n")} {
		paths, err := parseCUDADependencies(raw)
		if err != nil || !slices.Equal(paths, want) {
			t.Fatalf("NVCC 13.4.3 dependencies=%q err=%v", paths, err)
		}
	}
	paths, err := parseCUDADependencies("fak_cuda_dependency_target: source.cu \\\n /usr/include/a.h path\\ with\\ spaces.h C:\\SDK\\header.h\n")
	if err != nil || len(paths) != 4 || paths[2] != "path with spaces.h" || paths[3] != `C:\SDK\header.h` {
		t.Fatalf("dependencies=%q err=%v", paths, err)
	}
	for _, raw := range []string{"", "target: source.cu", "fak_cuda_dependency_target:", "fak_cuda_dependency_target: a\x00", "other fak_cuda_dependency_target : source.cu", "prefix:fak_cuda_dependency_target: source.cu", "notice\nfak_cuda_dependency_target : source.cu"} {
		if _, err := parseCUDADependencies(raw); err == nil {
			t.Fatalf("accepted invalid depfile %q", raw)
		}
	}
}

// fak-test:justify why=regression when=changed:internal/computebuild/**
// fak-test:runtime medium est=5s
func TestCUDADirectAPIsPrepareNCCL(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Linux NCCL versioned-library staging")
	}
	root, tc := cudaIncrementalFixture(t)
	home := filepath.Join(root, "nccl")
	include := filepath.Join(home, "usr", "include")
	library := filepath.Join(home, "usr", "lib")
	for _, dir := range []string{include, library} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(library, "libnccl.so.2"), []byte("versioned NCCL"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.test/cuda\n\ngo 1.26\n"), 0600); err != nil {
		t.Fatal(err)
	}
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(self)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go"), raw, 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("NCCL_HOME", home)
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	if _, err := BuildCUDALib(context.Background(), tc, root, "sm_120", true, io.Discard); err != nil {
		t.Fatal(err)
	}
	staged := filepath.Join(root, ".fak", "cuda-nccl-lib")
	link := filepath.Join(staged, "libnccl.so")
	if data, err := os.ReadFile(link); err != nil || string(data) != "versioned NCCL" {
		t.Fatalf("staged NCCL data=%q err=%v", data, err)
	}
	checkFlags := func() {
		t.Helper()
		r := cudaReadNativeReceipt(t, root)
		if len(r.Objects) == 0 {
			t.Fatal("native API retained no object evidence")
		}
		for _, object := range r.Objects {
			if !slices.Contains(object.EffectiveFlags, "-I"+include) {
				t.Fatalf("NCCL include missing from %s: %q", object.Source, object.EffectiveFlags)
			}
		}
	}
	checkFlags()
	cfg := &CUDAConfig{Command: "build", RepoRoot: root, Arch: "sm_120", EnableNCCL: true, Toolchain: tc, Stdout: io.Discard, Stderr: io.Discard}
	if err := RunCUDA(context.Background(), cfg); err != nil {
		t.Fatal(err)
	}
	checkFlags()
	if cfg.Toolchain == tc || !slices.Contains(cfg.Toolchain.CUDAInc, include) || !slices.Contains(cfg.Toolchain.CUDALib, staged) || len(tc.CUDAInc) != 0 || len(tc.CUDALib) != 0 {
		t.Fatal("direct API lost prepared NCCL paths or mutated caller toolchain")
	}
	flags, err := os.ReadFile(filepath.Join(root, "go-link-flags"))
	if err != nil || !strings.Contains(string(flags), "-L"+QuoteArg(staged)) || !strings.Contains(string(flags), "-lnccl") {
		t.Fatalf("Go link flags=%q err=%v", flags, err)
	}
	if err := os.Remove(link); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(link, []byte("owned regular file"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := BuildCUDALib(context.Background(), tc, root, "sm_120", true, io.Discard); err == nil {
		t.Fatal("NCCL preparation overwrote a regular linker name")
	}
	r := cudaReadNativeReceipt(t, root)
	if r.Outcome != "failed" || r.Executed != 0 || r.Reused != 0 || len(r.Objects) != 0 {
		t.Fatal("preparation failure retained previous native object counts")
	}
	cfg.Toolchain = tc
	if err := RunCUDA(context.Background(), cfg); err == nil {
		t.Fatal("RunCUDA accepted failed NCCL preparation")
	}
	if len(cfg.Receipt.Phases) != 1 || cfg.Receipt.Phases[0].Name != "prepare_nccl" || cfg.Receipt.Phases[0].Outcome != "failed" {
		t.Fatalf("NCCL preparation failure unrecorded: %+v", cfg.Receipt.Phases)
	}
}
