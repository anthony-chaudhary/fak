package main

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s
func TestServeLaunchEvidenceBindsProcessAndPreservesExisting(t *testing.T) {
	_, sf := newServeFlagSet()
	path := filepath.Join(t.TempDir(), "launch.json")
	*sf.launchEvidencePath = path
	sf.startupConfigSHA256 = serveLaunchDigest([]byte("startup bytes"))
	*sf.nativeQwenQ4KPrefillChunk = 2048
	*sf.baseURL = "https://secret-upstream.invalid/credential"
	t.Setenv("INVOCATION_ID", "0123456789abcdef0123456789abcdef")
	rt := &serveRuntime{}
	if err := writeServeLaunchEvidence(sf, rt); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var receipt serveLaunchEvidence
	if err = json.Unmarshal(data, &receipt); err != nil {
		t.Fatal(err)
	}
	executablePath, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	executable, err := os.ReadFile(executablePath)
	if err != nil {
		t.Fatal(err)
	}
	wantExecutableSHA256 := fmt.Sprintf("%x", sha256.Sum256(executable))
	if receipt.PID != os.Getpid() || receipt.InvocationID != os.Getenv("INVOCATION_ID") || receipt.ExecutableSHA256 != wantExecutableSHA256 || receipt.PrefillChunkCeiling != 2048 || receipt.ConfigSHA256 != sf.startupConfigSHA256 || receipt.Backend != "unresolved" {
		t.Fatalf("invalid startup binding: %+v", receipt)
	}
	if strings.Contains(string(data), "secret-upstream") {
		t.Fatal("non-allowlisted value leaked")
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !info.Mode().IsRegular() {
		t.Fatalf("evidence mode = %v, want regular file", info.Mode())
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0600 {
		t.Fatal("evidence must be private")
	}
	if err = writeServeLaunchEvidence(sf, rt); !errors.Is(err, os.ErrExist) {
		t.Fatalf("existing receipt must refuse: %v", err)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(data) {
		t.Fatal("existing receipt changed")
	}
}

// fak-test:runtime fast est=1s
func TestServeLaunchEvidenceRefusesSymlinkAndRelativePath(t *testing.T) {
	_, sf := newServeFlagSet()
	*sf.launchEvidencePath = "relative.json"
	if err := writeServeLaunchEvidence(sf, &serveRuntime{}); err == nil {
		t.Fatal("relative path accepted")
	}
	root := t.TempDir()
	target := filepath.Join(root, "target")
	if err := os.WriteFile(target, []byte("preserve"), 0600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(root, "link")
	if err := os.Symlink(target, link); err != nil {
		t.Skip(err)
	}
	*sf.launchEvidencePath = link
	if err := writeServeLaunchEvidence(sf, &serveRuntime{}); !errors.Is(err, os.ErrExist) {
		t.Fatalf("symlink accepted: %v", err)
	}
}
