package ggufload

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestOpenWeightsAuthenticatedRetirementEvidence(t *testing.T) {
	dir := t.TempDir()
	var shards []ShardIdentity
	for i := range 2 {
		data := writeSplitShard(t, uint32(i), 2, 2, i == 0, []splitTensor{{
			name: fmt.Sprintf("shard%d.weight", i), dims: []uint64{2}, typ: TensorF32,
			data: f32Payload(float32(i+1), float32(i+2)),
		}})
		path := filepath.Join(dir, fmt.Sprintf("tiny-%05d-of-00002.gguf", i+1))
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		shards = append(shards, ShardIdentity{Path: path, Size: int64(len(data)), RawFileSHA256: hex.EncodeToString(sum[:])})
	}
	raw, err := json.Marshal(shards)
	if err != nil {
		t.Fatal(err)
	}
	setSum := sha256.Sum256(raw)
	digest := hex.EncodeToString(setSum[:])
	retirementFailure := errors.New("authenticated reader retirement sentinel")
	const primaryText = "gguf: parse authenticated shard 2: context canceled"
	// Keep the healthy control first and use one root: an expected-red receipt
	// must witness only the lost evidence after every ownership check passed.
	for _, injected := range []error{nil, retirementFailure} {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		var readers []*authenticatedRetirementReader
		opened := 0
		ws, cause := openWeightsAuthenticated(ctx, shards, digest, false,
			func(path string, mapped bool) (io.ReaderAt, int64, io.Closer, []byte, error) {
				if mapped {
					t.Fatal("authenticated retirement witness requested mmap")
				}
				r, size, closer, data, err := openShardReader(path, false)
				if err != nil {
					return r, size, closer, data, err
				}
				opened++
				tracked := &authenticatedRetirementReader{inner: closer}
				if opened == 1 {
					tracked.injected = injected
				}
				readers = append(readers, tracked)
				t.Cleanup(func() {
					if tracked.attempts == 0 {
						_ = tracked.inner.Close()
					}
				})
				// Both real readers have been retained. The second shard's
				// authentication observes cancellation before its first read.
				if opened == 2 {
					cancel()
				}
				return r, size, tracked, data, nil
			})
		cancel()
		if ws != nil || opened != 2 || len(readers) != 2 || !errors.Is(cause, context.Canceled) {
			t.Fatalf("authenticated cancellation changed: source=%v error=%v opens=%d readers=%d", ws, cause, opened, len(readers))
		}
		if cause.Error() != primaryText && !strings.HasPrefix(cause.Error(), primaryText+"\n") {
			t.Fatalf("primary authenticated refusal changed: %v", cause)
		}
		for i, reader := range readers {
			if reader.attempts != 1 || reader.completed != 1 || reader.innerErr != nil {
				t.Fatalf("reader %d: attempts=%d completions=%d actual close error=%v", i, reader.attempts, reader.completed, reader.innerErr)
			}
		}
		if injected == nil {
			if cause.Error() != primaryText {
				t.Fatalf("healthy retirement changed primary error: %v", cause)
			}
			t.Log("AUTH_READER_HEALTHY_CANCELED_RETIREMENT_CONFIRMED")
			continue
		}
		// The wrapper closed the actual file before returning its sentinel.
		// This witnesses lost error evidence, not a leaked OS descriptor.
		if !errors.Is(cause, retirementFailure) {
			t.Fatalf("AUTH_READER_RETIREMENT_ERROR_EVIDENCE_LOST: primary cancellation returned without the observed reader Close error: %v", cause)
		}
	}
}

type authenticatedRetirementReader struct {
	inner               io.Closer
	injected, innerErr  error
	attempts, completed int
}

func (r *authenticatedRetirementReader) Close() error {
	r.attempts++
	r.innerErr = r.inner.Close()
	r.completed++
	return errors.Join(r.innerErr, r.injected)
}

// fak-test:runtime fast est=100ms lane=default
func TestOpenWeightsAuthenticatedRetirementResult(t *testing.T) {
	for _, tc := range []struct {
		name                         string
		preflight, invalid, opaque   bool
		cancel, closeErrors, success bool
		known, completed             bool
		shards                       int
	}{
		{name: "preflight-cancel", preflight: true, known: true, completed: true, shards: 2},
		{name: "manifest-refusal", invalid: true, known: true, completed: true, shards: 2},
		{name: "auth-cancel-clean", cancel: true, known: true, completed: true, shards: 2},
		{name: "auth-cancel-close-error", cancel: true, closeErrors: true, known: true, shards: 2},
		{name: "opaque-open-clean-retirement", opaque: true, shards: 2},
		{name: "opaque-open-observed-close-error", opaque: true, closeErrors: true, known: true, shards: 2},
		{name: "transfer-is-not-retirement", success: true, shards: 2},
		{name: "bounded-error-overflow", cancel: true, closeErrors: true, known: true, shards: 10},
	} {
		t.Run(tc.name, func(t *testing.T) {
			shards, digest := authenticatedRetirementShards(t, tc.shards)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			if tc.preflight {
				cancel()
			}
			if tc.invalid {
				digest = strings.Repeat("0", 64)
			}
			openFailure := errors.New("authenticated opaque opener sentinel")
			var readers []*authenticatedRetirementReader
			var closeFailures []error
			calls := 0
			got := openWeightsAuthenticatedWithResult(ctx, shards, digest, false,
				func(path string, mapped bool) (io.ReaderAt, int64, io.Closer, []byte, error) {
					calls++
					if mapped {
						t.Fatal("result control requested mmap")
					}
					if tc.opaque && calls == 2 {
						return nil, 0, nil, nil, openFailure
					}
					r, size, closer, data, err := openShardReader(path, false)
					if err != nil {
						return r, size, closer, data, err
					}
					tracked := &authenticatedRetirementReader{inner: closer}
					if tc.closeErrors && (calls == 1 || tc.shards == 10) {
						tracked.injected = fmt.Errorf("authenticated Close sentinel %d", calls)
						closeFailures = append(closeFailures, tracked.injected)
					}
					readers = append(readers, tracked)
					t.Cleanup(func() {
						if tracked.attempts == 0 {
							_ = tracked.inner.Close()
						}
					})
					if tc.cancel && calls == len(shards) {
						cancel()
					}
					return r, size, tracked, data, nil
				})
			cancel()
			if got.OwnershipTransferred != tc.success || (got.Source != nil) != tc.success ||
				got.Retirement.Available != tc.known || got.Retirement.Completed != tc.completed {
				t.Fatalf("ownership or retirement result: %+v", got)
			}
			wantCalls, wantReaders := tc.shards, tc.shards
			if tc.preflight || tc.invalid {
				wantCalls, wantReaders = 0, 0
			} else if tc.opaque {
				wantReaders = 1
			}
			if calls != wantCalls || len(readers) != wantReaders {
				t.Fatalf("opens=%d retained=%d, want %d and %d", calls, len(readers), wantCalls, wantReaders)
			}
			if tc.success {
				if got.Err != nil || got.Retirement.Err != nil {
					t.Fatalf("successful transfer errors: %+v", got)
				}
				for _, reader := range readers {
					if reader.attempts != 0 {
						t.Fatal("transferred source was closed by its constructor")
					}
				}
				if err := got.Source.Close(); err != nil {
					t.Fatal(err)
				}
			} else {
				if got.Err == nil {
					t.Fatal("failed construction lost its primary error")
				}
				primary := got.Err
				if tc.closeErrors {
					joined, ok := got.Err.(interface{ Unwrap() []error })
					if !ok || len(joined.Unwrap()) != 2 || joined.Unwrap()[1] != got.Retirement.Err {
						t.Fatalf("primary-first result changed: %+v", got)
					}
					primary = joined.Unwrap()[0]
				} else if got.Retirement.Err != nil {
					t.Fatalf("clean retirement fabricated an error: %v", got.Retirement.Err)
				}
				switch {
				case tc.preflight:
					if primary != context.Canceled {
						t.Fatalf("preflight error identity changed: %v", primary)
					}
				case tc.invalid:
					if primary.Error() != "gguf: authenticated shard manifest SHA-256 mismatch" {
						t.Fatalf("manifest refusal changed: %v", primary)
					}
				case tc.opaque:
					if primary.Error() != "gguf: open authenticated shard 2: "+openFailure.Error() || errors.Unwrap(primary) != openFailure {
						t.Fatalf("opaque opener primary changed: %v", primary)
					}
				default:
					want := fmt.Sprintf("gguf: parse authenticated shard %d: context canceled", tc.shards)
					if primary.Error() != want || errors.Unwrap(primary) != context.Canceled {
						t.Fatalf("authentication cancellation primary changed: %v", primary)
					}
				}
				if errors.Is(got.Retirement.Err, context.Canceled) || errors.Is(got.Retirement.Err, openFailure) {
					t.Fatal("retirement error contains the construction cause")
				}
			}
			for i, reader := range readers {
				if reader.attempts != 1 || reader.completed != 1 || reader.innerErr != nil {
					t.Fatalf("reader %d: attempts=%d completions=%d actual close error=%v", i, reader.attempts, reader.completed, reader.innerErr)
				}
			}
			for i, failure := range closeFailures {
				want := i < authenticatedWeightCloseErrorLimit
				if errors.Is(got.Err, failure) != want || errors.Is(got.Retirement.Err, failure) != want {
					t.Fatalf("bounded Close error identity %d changed: %+v", i, got)
				}
			}
			if tc.shards == 10 {
				if !errors.Is(got.Retirement.Err, errAuthenticatedWeightCloseOverflow) ||
					strings.Count(got.Retirement.Err.Error(), errAuthenticatedWeightCloseOverflow.Error()) != 1 {
					t.Fatalf("missing or repeated overflow failure marker: %v", got.Retirement.Err)
				}
			} else if errors.Is(got.Err, errAuthenticatedWeightCloseOverflow) {
				t.Fatal("bounded control fabricated overflow")
			}
		})
	}
}

func authenticatedRetirementShards(t *testing.T, count int) ([]ShardIdentity, string) {
	t.Helper()
	dir := t.TempDir()
	var shards []ShardIdentity
	for i := range count {
		data := writeSplitShard(t, uint32(i), uint32(count), uint32(count), i == 0, []splitTensor{{
			name: fmt.Sprintf("shard%d.weight", i), dims: []uint64{2}, typ: TensorF32,
			data: f32Payload(float32(i+1), float32(i+2)),
		}})
		path := filepath.Join(dir, fmt.Sprintf("tiny-%05d-of-%05d.gguf", i+1, count))
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		shards = append(shards, ShardIdentity{Path: path, Size: int64(len(data)), RawFileSHA256: hex.EncodeToString(sum[:])})
	}
	raw, err := json.Marshal(shards)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	return shards, hex.EncodeToString(sum[:])
}
