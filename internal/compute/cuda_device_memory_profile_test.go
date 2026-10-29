package compute

import (
	"context"
	"crypto/sha256"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestCUDADeviceMemoryProfileFacts(t *testing.T) {
	for _, tc := range []struct {
		name  string
		facts cudaDeviceMemoryFacts
		cache int64
		warp  int
		known bool
	}{
		{"observed-not-halo", cudaDeviceMemoryFacts{6 << 20, 32, true, true}, 6 << 20, 32, true},
		{"cache-only", cudaDeviceMemoryFacts{40 << 20, 32, true, false}, 40 << 20, 0, true},
		{"warp-only", cudaDeviceMemoryFacts{40 << 20, 64, false, true}, 0, 64, true},
		{"failed-both", cudaDeviceMemoryFacts{40 << 20, 32, false, false}, 0, 0, false},
		{"zero", cudaDeviceMemoryFacts{0, 0, true, true}, 0, 0, false},
		{"negative", cudaDeviceMemoryFacts{-1, -32, true, true}, 0, 0, false},
		{"overflow", cudaDeviceMemoryFacts{1 << 32, 1 << 32, true, true}, 0, 0, false},
		{"invalid-cache-partial", cudaDeviceMemoryFacts{-1, 32, true, true}, 0, 32, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, known := cudaDeviceMemoryProfileFromFacts("sm_80", tc.facts)
			want := DeviceMemoryProfile{Architecture: "sm_80", CacheCapacityBytes: tc.cache, SubgroupWidth: tc.warp,
				CacheCapacityKnown: tc.cache > 0, SubgroupWidthKnown: tc.warp > 0}
			if got != want || known != tc.known || got.CachePlanningAvailable() {
				t.Fatalf("facts = (%+v, %t), want (%+v, %t), never complete planning", got, known, want, tc.known)
			}
		})
	}
}

// fak-test:runtime slow est=180s lane=optin
// This is the three child-process deadlines, not a measured latency. It executes
// the exact adapter method with a query spy, NOT CUDA property getters or a GPU.
func TestCUDADeviceMemoryProfileIdentityContract(t *testing.T) {
	if os.Getenv("FAK_TEST_CUDA_MEMORY_PROFILE_HOST") != "1" {
		t.Skip("set FAK_TEST_CUDA_MEMORY_PROFILE_HOST=1 for the exact-adapter host witness")
	}
	// The pre-provider corpus must still COMPILE and exercise actual resolver
	// behavior. Missing optional source is not itself the RED: without a provider,
	// the real resolver accepts the conflicting architecture's complete profile.
	var method, facts, mapper, query string
	if _, err := os.Stat("cuda_device_memory_profile.go"); err == nil {
		method = cudaMemoryProfileDeclaration(t, "cuda_device_memory_profile.go", "DeviceMemoryProfile", "cudaBackend")
		facts = cudaMemoryProfileDeclaration(t, "cuda_device_memory_profile_contract.go", "cudaDeviceMemoryFacts", "")
		mapper = cudaMemoryProfileDeclaration(t, "cuda_device_memory_profile_contract.go", "cudaDeviceMemoryProfileFromFacts", "")
		query = cudaMemoryProfileQuerySpy
	} else if !os.IsNotExist(err) {
		t.Fatal(err)
	}
	profile := cudaMemoryProfileDeclaration(t, "device_memory_profile.go", "DeviceMemoryProfile", "")
	planning := cudaMemoryProfileDeclaration(t, "device_memory_profile.go", "CachePlanningAvailable", "DeviceMemoryProfile")
	provider := cudaMemoryProfileDeclaration(t, "device_memory_profile.go", "DeviceMemoryProfileProvider", "")
	resolver := cudaMemoryProfileDeclaration(t, "device_memory_profile.go", "ResolveDeviceMemoryProfile", "")
	lookup := cudaMemoryProfileDeclaration(t, "device_memory_profile.go", "LookupDeviceMemoryProfile", "")
	normalize := cudaMemoryProfileDeclaration(t, "rocm_arch.go", "normalizeGFX", "")
	name := cudaMemoryProfileDeclaration(t, "cuda_backend_state.go", "Name", "cudaBackend")
	tier := cudaMemoryProfileDeclaration(t, "cuda_backend_state.go", "Tier", "cudaBackend")
	rank := cudaMemoryProfileDeclaration(t, "compute.go", "RankUploader", "")
	group := cudaMemoryProfileDeclaration(t, "compute.go", "ProcessGroupBackend", "")
	t.Logf("adapter present=%t exact adapter SHA256=%x mapper SHA256=%x resolver SHA256=%x", method != "", sha256.Sum256([]byte(method)), sha256.Sum256([]byte(mapper)), sha256.Sum256([]byte(resolver)))
	// Bind the identity premise to ALL current production Go source, not just
	// CPU-selected files. This is a source contract, not CUDA execution evidence.
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	assignments := 0
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), ".go") || strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		f, err := parser.ParseFile(token.NewFileSet(), entry.Name(), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(f, func(n ast.Node) bool {
			if a, ok := n.(*ast.AssignStmt); ok {
				for _, lhs := range a.Lhs {
					if id, ok := lhs.(*ast.Ident); ok && id.Name == "cudaDev" {
						assignments++
					}
				}
			}
			return true
		})
	}
	if assignments != 1 {
		t.Fatalf("cudaDev assignments=%d, re-audit ordinary-backend identity", assignments)
	}
	initSource, err := os.ReadFile("cuda.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(initSource), "var cudaMu sync.Mutex") ||
		!strings.Contains(string(initSource), "if C.fcuda_init(&name[0], 256, &sm, &total) != 0 {\n\t\treturn") {
		t.Fatal("ordinary initialization/plain-mutex source contract changed")
	}
	closure := cudaSourceClosure(t, "cuda_kernels.cu")
	if !strings.Contains(closure, "if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 1;") ||
		!strings.Contains(closure, "if (cudaSetDevice(0) != cudaSuccess) return 2;") ||
		!strings.Contains(closure, "if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 3;") {
		t.Fatal("native initializer no longer establishes ordinary device zero")
	}
	goExe := ""
	if root := runtime.GOROOT(); root != "" {
		goExe = filepath.Join(root, "bin", "go")
		if runtime.GOOS == "windows" {
			goExe += ".exe"
		}
	} else {
		goExe, err = exec.LookPath("go")
		if err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("host witness Go executable: %s", goExe)
	for _, variant := range []struct {
		name, methods string
		refused       bool
	}{
		{"ordinary", "", false},
		{"rank", "func (*cudaBackend) UploadRank(Tensor, Dtype, int) (Tensor, error) { panic(\"must not call rank method\") }", true},
		{"process-group", `func (*cudaBackend) ProcessGroupUniqueID() ([]byte,error) { panic("must not call group") }
func (*cudaBackend) InitProcessGroup([]byte,int,int,int) error { panic("must not call group") }
func (*cudaBackend) AllReduceSumPG(Tensor) (Tensor,error) { panic("must not call group") }
func (*cudaBackend) DestroyProcessGroup() { panic("must not call group") }`, true},
	} {
		t.Run(variant.name, func(t *testing.T) {
			dir := t.TempDir()
			refused := "false"
			if variant.refused {
				refused = "true"
			}
			body := cudaMemoryProfileHostShim + "\nconst refuseCapability = " + refused + "\n" +
				strings.Join([]string{profile, planning, provider, resolver, lookup, normalize, name, tier, facts, mapper, query, rank, group, method, variant.methods}, "\n")
			probe := filepath.Join(dir, "memory_profile_test.go")
			if err := os.WriteFile(probe, []byte(body), 0600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, goExe, "test", "-p=1", "-trimpath", "-buildvcs=false", "-count=1", "-v", "-timeout=45s", probe)
			cmd.Dir = dir
			for _, entry := range os.Environ() {
				key, _, _ := strings.Cut(entry, "=")
				switch key {
				case "GOWORK", "GOENV", "GOTOOLCHAIN", "CGO_ENABLED", "GOMAXPROCS", "GOFLAGS":
					continue
				}
				cmd.Env = append(cmd.Env, entry)
			}
			cmd.Env = append(cmd.Env, "GOWORK=off", "GOENV=off", "GOTOOLCHAIN=local", "CGO_ENABLED=0", "GOMAXPROCS=1", "GOFLAGS=")
			out, err := cmd.CombinedOutput()
			t.Logf("exact-adapter host witness:\n%s", out)
			if ctx.Err() != nil || err != nil {
				t.Fatalf("host witness: %v deadline=%v", err, ctx.Err())
			}
		})
	}
}

func cudaMemoryProfileDeclaration(t *testing.T, path, name, receiver string) string {
	t.Helper()
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	fs := token.NewFileSet()
	f, err := parser.ParseFile(fs, path, src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var found string
	for _, decl := range f.Decls {
		if g, ok := decl.(*ast.GenDecl); ok && receiver == "" && g.Tok == token.TYPE {
			for _, spec := range g.Specs {
				if typ, ok := spec.(*ast.TypeSpec); ok && typ.Name.Name == name {
					if found != "" {
						t.Fatalf("duplicate %s", name)
					}
					found = "type " + string(src[fs.Position(typ.Pos()).Offset:fs.Position(typ.End()).Offset])
				}
			}
		}
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Name.Name != name {
			continue
		}
		gotReceiver := ""
		if fn.Recv != nil && len(fn.Recv.List) == 1 {
			typ := fn.Recv.List[0].Type
			if ptr, ok := typ.(*ast.StarExpr); ok {
				typ = ptr.X
			}
			if id, ok := typ.(*ast.Ident); ok {
				gotReceiver = id.Name
			}
		}
		if gotReceiver != receiver {
			continue
		}
		if found != "" {
			t.Fatalf("duplicate %s", name)
		}
		found = string(src[fs.Position(fn.Pos()).Offset:fs.Position(fn.End()).Offset])
	}
	if found == "" {
		t.Fatalf("missing production declaration %s in %s", name, path)
	}
	return found
}

const cudaMemoryProfileHostShim = `package compute
import ("testing"; "strings")
type Tensor struct{}
type Dtype uint8
// Only methods actually used by the unmodified resolver are needed by this
// host dependency layer. The production CUDA receiver satisfies full Backend.
type Backend interface { Name() string; Tier() string }
type cudaBackend struct { name, tier string }
func DeviceMemoryProfileForVulkanTier(string) (DeviceMemoryProfile,bool) { panic("unexpected Vulkan route") }
var cudaDev *cudaBackend
type trackedMutex struct { locked bool; locks, unlocks int }
func (m *trackedMutex) Lock() { if m.locked { panic("double lock") }; m.locked=true; m.locks++ }
func (m *trackedMutex) Unlock() { if !m.locked { panic("unlock without lock") }; m.locked=false; m.unlocks++ }
var cudaMu trackedMutex
var queries int
func TestIdentity(t *testing.T) {
 for _, name := range []string{"nil","uninitialized","foreign","ordinary"} {
  t.Run(name,func(t *testing.T) {
   cudaMu=trackedMutex{}; queries=0
   c:=&cudaBackend{name:"cuda",tier:"sm_80"};cudaDev=c
   switch name {case "nil":c=nil;case "uninitialized":cudaDev=nil;case "foreign":cudaDev=&cudaBackend{name:"cuda",tier:"sm_80"}}
   var got DeviceMemoryProfile; var known bool; var failure any
   func(){ defer func(){failure=recover()}(); got,known=ResolveDeviceMemoryProfile(c,"gfx1151") }()
   if failure!=nil { t.Errorf("resolver panicked instead of refusing CUDA identity: %v",failure); return }
   if name!="ordinary" || refuseCapability {
    if known || got!=(DeviceMemoryProfile{}) || queries!=0 || cudaMu.locks!=0 || cudaMu.unlocks!=0 { t.Fatalf("refusal had effects: %+v %t queries=%d mutex=%+v",got,known,queries,cudaMu) }
   } else {
    want:=DeviceMemoryProfile{Architecture:"sm_80",CacheCapacityBytes:6<<20,SubgroupWidth:32,CacheCapacityKnown:true,SubgroupWidthKnown:true}
    if !known || got!=want || got.CachePlanningAvailable() || queries!=1 || cudaMu.locked || cudaMu.locks!=1 || cudaMu.unlocks!=1 { t.Fatalf("ordinary partial facts/locking: %+v %t queries=%d mutex=%+v",got,known,queries,cudaMu) }
   }
  })
 }
}
`

const cudaMemoryProfileQuerySpy = `func cudaQueryDeviceMemoryFacts(device int) cudaDeviceMemoryFacts {
 if device != 0 || !cudaMu.locked { panic("wrong identity or query outside mutex") }
 queries++
 return cudaDeviceMemoryFacts{6<<20,32,true,true}
}`
