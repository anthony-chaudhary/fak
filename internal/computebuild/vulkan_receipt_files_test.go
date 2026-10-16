package computebuild

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type receiptPassReader struct {
	*bytes.Reader
	second           []byte
	readErr, seekErr error
	consumed         int
}

func (r *receiptPassReader) Read(p []byte) (int, error) {
	if r.readErr != nil {
		return 0, r.readErr
	}
	// Exercise legal short reads rather than assuming one read fills a buffer.
	if len(p) > 2 {
		p = p[:2]
	}
	n, err := r.Reader.Read(p)
	r.consumed += n
	return n, err
}

func (r *receiptPassReader) Seek(offset int64, whence int) (int64, error) {
	if r.seekErr != nil {
		return 0, r.seekErr
	}
	if r.second != nil {
		r.Reader = bytes.NewReader(r.second)
	}
	return r.Reader.Seek(offset, whence)
}

// fak-test:runtime fast est=20ms
// Small pass budgets witness allocation admission; sparse oversize files below
// exercise the real receipt entry without materializing a large byte buffer.
func TestVulkanReceiptReadBudget(t *testing.T) {
	const limit int64 = 8
	for _, tc := range []struct {
		name, first, second, wantError string
		wantConsumed                   int
	}{
		{"exact limit", "12345678", "", "", 16},
		{"short reads", "abc", "", "", 6},
		{"empty", "", "", "", 0},
		{"first pass excess", "123456789MORE", "", "receipt read limit of 8 bytes", 9},
		{"growth on reread", "abc", "123456789MORE", "receipt read limit of 8 bytes", 12},
		{"same size mutation", "abc", "abd", "changed during read", 6},
		{"shrink on reread", "abcd", "abc", "changed during read", 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := &receiptPassReader{Reader: bytes.NewReader([]byte(tc.first))}
			if tc.second != "" {
				r.second = []byte(tc.second)
			}
			got, err := readVulkanReceiptTwice(r, "test receipt", limit)
			if tc.wantError == "" {
				if err != nil || string(got) != tc.first {
					t.Fatalf("read = %q, %v; want complete %q", got, err, tc.first)
				}
			} else if err == nil || !strings.Contains(err.Error(), tc.wantError) || got != nil {
				t.Fatalf("read = %q, %v; want refusal %q", got, err, tc.wantError)
			}
			if r.consumed != tc.wantConsumed {
				t.Fatalf("consumed %d bytes, want %d across admitted passes", r.consumed, tc.wantConsumed)
			}
		})
	}
	t.Run("read and rewind errors", func(t *testing.T) {
		cause := errors.New("receipt IO witness")
		for _, r := range []*receiptPassReader{
			{Reader: bytes.NewReader([]byte("abc")), readErr: cause},
			{Reader: bytes.NewReader([]byte("abc")), seekErr: cause},
		} {
			got, err := readVulkanReceiptTwice(r, "test receipt", limit)
			if got != nil || !errors.Is(err, cause) {
				t.Fatalf("lost IO refusal: %q, %v", got, err)
			}
		}
	})
	t.Run("real regular file and oversize refusal", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "receipt.json")
		body := []byte(`{"schema":"fixture"}`)
		if err := os.WriteFile(path, body, 0600); err != nil {
			t.Fatal(err)
		}
		got, err := readStableVulkanReceipt(path, "Vulkan receipt")
		if err != nil || !bytes.Equal(got, body) {
			t.Fatalf("stable receipt: %q, %v", got, err)
		}
		// Truncate creates a sparse file: the refusal occurs at stat, before reads.
		if err := os.Truncate(path, vulkanReceiptReadLimit+1); err != nil {
			t.Fatal(err)
		}
		got, err = readStableVulkanReceipt(path, "Vulkan receipt")
		if got != nil || err == nil || !strings.Contains(err.Error(), "Vulkan receipt exceeds receipt read limit of 16777216 bytes") {
			t.Fatalf("oversize receipt: %q, %v", got, err)
		}
	})
	t.Run("path admission", func(t *testing.T) {
		if _, err := readStableVulkanReceipt("relative.json", "Vulkan receipt"); err == nil {
			t.Fatal("relative path accepted")
		}
		if _, err := readStableVulkanReceipt(t.TempDir(), "Vulkan receipt"); err == nil {
			t.Fatal("directory accepted")
		}
	})
	t.Run("verifier entrypoints refuse before evidence", func(t *testing.T) {
		dir := t.TempDir()
		large, small := filepath.Join(dir, "large.json"), filepath.Join(dir, "small.json")
		if err := os.WriteFile(large, nil, 0600); err != nil {
			t.Fatal(err)
		}
		if err := os.Truncate(large, vulkanReceiptReadLimit+1); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(small, []byte(`{}`), 0600); err != nil {
			t.Fatal(err)
		}
		for _, verify := range []struct {
			name string
			call func(context.Context, VulkanBinaryReceiptEvidence, vulkanGitRunner, vulkanGitRunner) (*VulkanBinaryReceiptIdentityVerification, error)
		}{
			{"V2", verifyVulkanBinaryReceiptIdentityWithRunners},
			{"V3", verifyVulkanBinaryReceiptV3IdentityWithRunners},
			{"V4", verifyVulkanBinaryReceiptV4IdentityWithRunners},
			{"current successor", vulkanRegistryV23.verifyIdentityWithRunners},
		} {
			t.Run(verify.name, func(t *testing.T) {
				for _, comparison := range []bool{false, true} {
					evidence := VulkanBinaryReceiptEvidence{ReceiptPath: large}
					label := "Vulkan receipt"
					if comparison {
						evidence.ReceiptPath = small
						evidence.Comparison = &VulkanBinaryReceiptEvidence{ReceiptPath: large}
						label = "comparison Vulkan receipt"
					}
					calls := 0
					runner := func(context.Context, string, ...string) ([]byte, error) {
						calls++
						return nil, errors.New("evidence must not run")
					}
					got, err := verify.call(context.Background(), evidence, runner, runner)
					if got != nil || err == nil || !strings.Contains(err.Error(), label+" exceeds receipt read limit of 16777216 bytes") || calls != 0 {
						t.Fatalf("comparison=%v: result=%v err=%v evidence calls=%d", comparison, got, err, calls)
					}
				}
			})
		}
	})
}

var _ io.ReadSeeker = (*receiptPassReader)(nil)
