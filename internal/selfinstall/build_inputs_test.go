package selfinstall

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestBuildInputIdentityTracksExecutableGraphAndEnvelope(t *testing.T) {
	dir := t.TempDir()
	writeBuildInputFixture(t, dir, "go.mod", "module example.test/buildinputs\n\ngo 1.26\n")
	writeBuildInputFixture(t, dir, "main.go", "package main\n\nimport (\n _ \"embed\"\n \"fmt\"\n \"example.test/buildinputs/lib\"\n)\n\n//go:embed asset.txt\nvar asset string\n\nfunc main() { fmt.Print(asset, lib.Value) }\n")
	if err := os.MkdirAll(filepath.Join(dir, "lib"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeBuildInputFixture(t, dir, "lib/lib.go", "package lib\nconst Value = \"one\"\n")
	writeBuildInputFixture(t, dir, "asset.txt", "one\n")
	writeBuildInputFixture(t, dir, "README.md", "first\n")
	writeBuildInputFixture(t, dir, "main_test.go", "package main\n\nimport \"testing\"\nfunc TestFixture(t *testing.T) {}\n")

	ctx := context.Background()
	baseline := mustBuildInputIdentity(t, ctx, dir, buildInputOptions{})
	if baseline.Digest == "" || baseline.Envelope["GOVERSION"] == "" {
		t.Fatalf("incomplete identity: %#v", baseline)
	}

	writeBuildInputFixture(t, dir, "README.md", "second\n")
	assertBuildInputDigest(t, baseline.Digest, mustBuildInputIdentity(t, ctx, dir, buildInputOptions{}).Digest, true, "documentation")
	writeBuildInputFixture(t, dir, "main_test.go", "package main\n\nimport \"testing\"\nfunc TestChangedFixture(t *testing.T) {}\n")
	assertBuildInputDigest(t, baseline.Digest, mustBuildInputIdentity(t, ctx, dir, buildInputOptions{}).Digest, true, "test source")
	writeBuildInputFixture(t, dir, "lib/lib.go", "package lib\nconst Value = \"two\"\n")
	assertBuildInputDigest(t, baseline.Digest, mustBuildInputIdentity(t, ctx, dir, buildInputOptions{}).Digest, false, "transitive runtime source")
	writeBuildInputFixture(t, dir, "lib/lib.go", "package lib\nconst Value = \"one\"\n")

	writeBuildInputFixture(t, dir, "asset.txt", "two\n")
	assetChanged := mustBuildInputIdentity(t, ctx, dir, buildInputOptions{})
	assertBuildInputDigest(t, baseline.Digest, assetChanged.Digest, false, "embedded asset")
	writeBuildInputFixture(t, dir, "asset.txt", "one\n")
	writeBuildInputFixture(t, dir, "main.go", "package main\n\nimport (\n _ \"embed\"\n \"fmt\"\n \"example.test/buildinputs/lib\"\n)\n\n//go:embed asset.txt\nvar asset string\n\nfunc main() { fmt.Print(\"runtime:\", asset, lib.Value) }\n")
	assertBuildInputDigest(t, baseline.Digest, mustBuildInputIdentity(t, ctx, dir, buildInputOptions{}).Digest, false, "runtime source")

	writeBuildInputFixture(t, dir, "main.go", "package main\n\nimport (\n _ \"embed\"\n \"fmt\"\n \"example.test/buildinputs/lib\"\n)\n\n//go:embed asset.txt\nvar asset string\n\nfunc main() { fmt.Print(asset, lib.Value) }\n")
	if err := os.MkdirAll(filepath.Join(dir, "lib"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeBuildInputFixture(t, dir, "lib/lib.go", "package lib\nconst Value = \"one\"\n")
	assertBuildInputDigest(t, baseline.Digest, mustBuildInputIdentity(t, ctx, dir, buildInputOptions{Tags: []string{"alternate"}}).Digest, false, "build tags")
	assertBuildInputDigest(t, baseline.Digest, mustBuildInputIdentity(t, ctx, dir, buildInputOptions{BuildFlags: []string{"-mod=mod"}}).Digest, false, "build flags")
	override := "CGO_ENABLED=0"
	if baseline.Envelope["CGO_ENABLED"] == "0" {
		override = "CGO_ENABLED=1"
	}
	assertBuildInputDigest(t, baseline.Digest, mustBuildInputIdentity(t, ctx, dir, buildInputOptions{Env: []string{override}}).Digest, false, "environment override")
}

func TestBuildInputIdentityRejectsTraversalAndInvalidEnvironment(t *testing.T) {
	if _, err := secureBuildInputPath(t.TempDir(), filepath.Join("..", "escape.go")); err == nil {
		t.Fatal("secureBuildInputPath accepted traversal")
	}
	dir := t.TempDir()
	writeBuildInputFixture(t, dir, "go.mod", "module example.test/badenv\n\ngo 1.26\n")
	writeBuildInputFixture(t, dir, "main.go", "package main\nfunc main() {}\n")
	if _, err := deriveBuildInputIdentity(context.Background(), dir, ".", buildInputOptions{Env: []string{"MISSING_EQUALS"}}); err == nil {
		t.Fatal("deriveBuildInputIdentity accepted malformed environment override")
	}
}

func mustBuildInputIdentity(t *testing.T, ctx context.Context, dir string, opts buildInputOptions) buildInputIdentity {
	t.Helper()
	identity, err := deriveBuildInputIdentity(ctx, dir, ".", opts)
	if err != nil {
		t.Fatalf("derive build input identity (%s/%s): %v", runtime.GOOS, runtime.GOARCH, err)
	}
	return identity
}

func assertBuildInputDigest(t *testing.T, left, right string, equal bool, change string) {
	t.Helper()
	if (left == right) != equal {
		t.Fatalf("%s digest equality = %v, want %v\nleft: %s\nright: %s", change, left == right, equal, left, right)
	}
}

func writeBuildInputFixture(t *testing.T, dir, name, body string) {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}
