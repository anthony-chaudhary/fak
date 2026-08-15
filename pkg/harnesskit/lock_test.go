package harnesskit

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
)

func TestProductLockVerifiesAndProjectsLaunchReceipt(t *testing.T) {
	lock := ProductLock{Schema: ProductLockSchema, Environment: LockEnvironment{OS: "linux", Arch: "amd64", Contract: "v1"}, Components: []LockedComponent{{ID: "legal-pack", Version: "1.0.0", Digest: "sha256:legal", Source: "registry/legal"}}, Assets: []LockedAsset{{Kind: "policy", ID: "tools", Source: "company", Locked: true}, {Kind: "instruction", ID: "citations", Value: "cite-primary", Source: "legal"}}}
	lock.ID = lockDigest(t, lock)
	raw, _ := json.Marshal(lock)
	got, err := ParseProductLock(raw)
	if err != nil {
		t.Fatal(err)
	}
	if got.Profile().ID != "legal" || got.InstructionText() != "cite-primary" {
		t.Fatalf("lock=%#v", got)
	}
	receipt := got.LaunchReceipt("product")
	if receipt.LockID != lock.ID || receipt.Profile != "legal" || len(receipt.Assets) != 2 {
		t.Fatalf("receipt=%#v", receipt)
	}
}

func TestProductLockRejectsTamperAndInlineSecret(t *testing.T) {
	lock := ProductLock{Schema: ProductLockSchema, Components: []LockedComponent{{ID: "p", Version: "1.0.0", Digest: "sha256:p", Source: "r"}}, Assets: []LockedAsset{{Kind: "instruction", ID: "i", Value: "one", Source: "legal"}}}
	lock.ID = lockDigest(t, lock)
	raw, _ := json.Marshal(lock)
	raw = []byte(strings.Replace(string(raw), `"value":"one"`, `"value":"two"`, 1))
	if _, err := ParseProductLock(raw); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("err=%v", err)
	}
	lock.Assets = []LockedAsset{{Kind: "secret", ID: "db", Source: "company"}}
	lock.ID = lockDigest(t, lock)
	raw, _ = json.Marshal(lock)
	if _, err := ParseProductLock(raw); err == nil || !strings.Contains(err.Error(), "no opaque reference") {
		t.Fatalf("err=%v", err)
	}
}

func lockDigest(t *testing.T, lock ProductLock) string {
	t.Helper()
	lock.ID = ""
	raw, err := json.Marshal(lock)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256Sum(raw)
	return "sha256:" + sum
}
func sha256Sum(raw []byte) string {
	h := sha256.New()
	h.Write(raw)
	return hex.EncodeToString(h.Sum(nil))
}
