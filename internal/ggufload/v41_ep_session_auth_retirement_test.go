package ggufload

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41ExpertParallelSessionAuthenticatedRetirementEvidence(t *testing.T) {
	t.Setenv("FAK_GGUF_MMAP", "0")
	shards, digest := ds41AuthenticatedEPShards(t)
	if len(shards) != 2 {
		t.Fatalf("bounded fixture changed: %d shards", len(shards))
	}
	backend := newDS41EPRecorder(&model.Model{})
	req := ds41RetirementRequest(t, shards, digest, backend)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	preflight := LoadV41ExpertParallelSessionWithResult(ctx, req)
	if preflight.Err != context.Canceled || preflight.Panicked || preflight.OwnershipTransferred ||
		!preflight.Retirement.Available || !preflight.Retirement.Completed || preflight.Retirement.Err != nil {
		t.Fatalf("preflight retirement control changed: %+v", preflight)
	}
	t.Log("AUTH_OUTER_PREFLIGHT_RETIREMENT_CONFIRMED")
	// Keep the manifest internally valid, but make its authenticated identity
	// disagree with the first real tiny shard. The auth reader owns and retires
	// that reader before it returns the ordinary hash-mismatch refusal.
	req.Shards[0].RawFileSHA256 = strings.Repeat("0", 64)
	raw, err := json.Marshal(req.Shards)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	req.ExpectedShardSetSHA256 = hex.EncodeToString(sum[:])
	auth := openWeightsAuthenticatedWithResult(t.Context(), req.Shards, req.ExpectedShardSetSHA256, false, openShardReader)
	if auth.Err == nil || auth.Err.Error() != "gguf: shard 1 raw file SHA-256 mismatch" || auth.Source != nil ||
		auth.OwnershipTransferred || !auth.Retirement.Available || !auth.Retirement.Completed || auth.Retirement.Err != nil {
		t.Fatalf("qualified authentication-retirement premise changed: %+v", auth)
	}
	t.Log("AUTH_READER_HASH_REFUSAL_RETIREMENT_CONFIRMED")
	got := LoadV41ExpertParallelSessionWithResult(t.Context(), req)
	if got.Err == nil || got.Err.Error() != "gguf: shard 1 raw file SHA-256 mismatch" || got.Panicked ||
		got.OwnershipTransferred || got.Session != nil || got.Coordinator != nil || got.Retirement.Err != nil {
		t.Fatalf("authenticated refusal changed: %+v", got)
	}
	if len(backend.live) != 0 || len(backend.order) != 0 || backend.routed != 0 {
		t.Fatal("authentication refusal performed materialization or forward work")
	}
	if !got.Retirement.Available || !got.Retirement.Completed {
		t.Fatalf("AUTH_OUTER_RETIREMENT_EVIDENCE_NOT_CONSUMED: ordinary authenticated refusal returned without proven reader retirement: %+v", got)
	}
}

// fak-test:runtime medium est=2s lane=default
func TestV41ExpertParallelSessionAuthenticatedRetirementResult(t *testing.T) {
	t.Setenv("FAK_GGUF_MMAP", "0")
	t.Setenv("FAK_PAGED_KV", "0")
	t.Setenv("FAK_NUMA_INTERLEAVE", "off")
	t.Setenv("FAK_NUMA_REPLICAS", "off")
	shards, digest := ds41AuthenticatedEPShards(t)
	if len(shards) != 2 {
		t.Fatalf("bounded fixture changed: %d shards", len(shards))
	}
	openFailure := errors.New("typed auth opaque open sentinel")
	closeFailure := errors.New("typed auth ordinary Close sentinel")
	closePanic := errors.New("typed auth last-reader Close panic")
	for _, tc := range []struct {
		name                                     string
		opaque, closeError, closePanic, transfer bool
		known, complete                          bool
	}{
		{name: "auth-cancel-clean", known: true, complete: true},
		{name: "auth-cancel-close-error", closeError: true, known: true},
		{name: "opaque-open-clean-stays-unknown", opaque: true},
		{name: "opaque-open-close-error-is-known-false", opaque: true, closeError: true, known: true},
		{name: "auth-close-panic-stays-unknown", closePanic: true},
		{name: "typed-transfer-is-not-retirement", transfer: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			req := ds41RetirementRequest(t, shards, digest, newDS41EPRecorder(&model.Model{}))
			calls := 0
			var readers []*ds41RetirementReader
			var auth authenticatedWeightOpenResult
			got := loadV41ExpertParallelSessionWithAuthenticatedResult(ctx, req,
				func(ctx context.Context, manifest []ShardIdentity, set string) authenticatedWeightOpenResult {
					auth = openWeightsAuthenticatedWithResult(ctx, manifest, set, false,
						func(path string, mapped bool) (io.ReaderAt, int64, io.Closer, []byte, error) {
							calls++
							if mapped {
								t.Fatal("typed auth control requested mmap")
							}
							if tc.opaque && calls == 2 {
								return nil, 0, nil, nil, openFailure
							}
							r, size, closer, data, err := openShardReader(path, false)
							if err != nil {
								return r, size, closer, data, err
							}
							reader := &ds41RetirementReader{inner: closer}
							if tc.closeError && calls == 1 {
								reader.injected = closeFailure
							}
							if tc.closePanic && calls == len(shards) {
								// Panic after the last actual reader closes; no fixture
								// descriptor is left behind by this legacy unwind control.
								reader.panicOnClose, reader.panicValue = true, closePanic
							}
							readers = append(readers, reader)
							t.Cleanup(func() {
								if reader.attempts == 0 {
									_ = reader.inner.Close()
								}
							})
							if !tc.transfer && calls == len(shards) {
								cancel()
							}
							return r, size, reader, data, nil
						})
					return auth
				})
			ownedSession := got.Session
			var ownedModel *model.Model
			if ownedSession != nil {
				ownedModel = ownedSession.M
			}
			defer func() {
				if ownedModel != nil {
					_ = ownedModel.CloseWeights()
				}
			}()
			defer func() {
				if ownedSession != nil {
					ownedSession.Close()
				}
			}()
			wantReaders := len(shards)
			if tc.opaque {
				wantReaders--
			}
			if calls != 2 || len(readers) != wantReaders || got.Panicked != tc.closePanic ||
				got.OwnershipTransferred != tc.transfer || got.Coordinator != nil ||
				got.Retirement.Available != tc.known || got.Retirement.Completed != tc.complete {
				t.Fatalf("typed auth state: opens=%d readers=%d result=%+v", calls, len(readers), got)
			}
			if tc.transfer {
				if got.Err != nil || got.Session == nil || got.Retirement.Err != nil ||
					!auth.OwnershipTransferred || auth.Retirement.Available {
					t.Fatalf("typed transfer: auth=%+v outer=%+v", auth, got)
				}
				for _, reader := range readers {
					if reader.attempts != 0 {
						t.Fatal("successful transfer retired its readers before caller cleanup")
					}
				}
				session := ownedSession
				ownedSession = nil
				session.Close()
				m := ownedModel
				ownedModel = nil
				if err := m.CloseWeights(); err != nil {
					t.Fatal(err)
				}
			} else if tc.closePanic {
				var carrier *V41ExpertParallelLoadPanicError
				if got.Session != nil || got.PrimaryPanic != closePanic || !errors.As(got.Err, &carrier) ||
					carrier.PanicValue() != closePanic || carrier.Operation != "construction" || got.Retirement.Err != nil ||
					errors.Is(got.Err, context.Canceled) {
					t.Fatalf("actual auth unwind panic changed: %+v", got)
				}
				// The unchanged auth unwind replaced its earlier cancellation with
				// this closer panic. Only the actually propagated payload is promised.
				_, payload, panicked := v41EPLoadInvoke("legacy typed witness", func() error {
					_, _, err := v41EPLoadLegacyResult(got)
					return err
				})
				if !panicked || payload != closePanic {
					t.Fatal("legacy wrapper changed the propagated auth panic")
				}
			} else {
				if got.Session != nil || got.Err == nil || got.Err != auth.Err || got.Retirement.Err != auth.Retirement.Err {
					t.Fatalf("typed auth identities changed: auth=%+v outer=%+v", auth, got)
				}
				primary := got.Err
				if tc.closeError {
					joined, ok := got.Err.(interface{ Unwrap() []error })
					if !ok || len(joined.Unwrap()) != 2 || joined.Unwrap()[1] != got.Retirement.Err ||
						!errors.Is(got.Retirement.Err, closeFailure) {
						t.Fatalf("auth cleanup duplicated or lost: %+v", got)
					}
					primary = joined.Unwrap()[0]
				}
				wantPrimary := error(context.Canceled)
				if tc.opaque {
					wantPrimary = openFailure
				}
				if errors.Unwrap(primary) != wantPrimary || errors.Is(got.Retirement.Err, wantPrimary) {
					t.Fatalf("auth primary/retirement separation changed: %+v", got)
				}
				_, _, legacyErr := v41EPLoadLegacyResult(got)
				if legacyErr != got.Err {
					t.Fatal("legacy wrapper changed ordinary error identity")
				}
			}
			for i, reader := range readers {
				if reader.attempts != 1 || reader.completed != 1 || reader.innerErr != nil {
					t.Fatalf("reader %d retirement: %+v", i, reader)
				}
			}
		})
	}
	t.Run("incoherent-evidence-stays-unknown", func(t *testing.T) {
		primary := errors.New("typed evidence primary")
		for _, tc := range []struct {
			name   string
			result authenticatedWeightOpenResult
		}{
			{"unavailable", authenticatedWeightOpenResult{Err: primary}},
			{"unavailable-with-error", authenticatedWeightOpenResult{Err: primary, Retirement: authenticatedWeightRetirement{Err: closeFailure}}},
			{"complete-with-error", authenticatedWeightOpenResult{Err: primary, Retirement: authenticatedWeightRetirement{Available: true, Completed: true, Err: closeFailure}}},
			{"incomplete-without-error", authenticatedWeightOpenResult{Err: primary, Retirement: authenticatedWeightRetirement{Available: true}}},
			{"transfer-claimed", authenticatedWeightOpenResult{Err: primary, OwnershipTransferred: true, Retirement: authenticatedWeightRetirement{Available: true, Completed: true}}},
			{"partial-source", authenticatedWeightOpenResult{Source: &WeightSource{}, Err: primary, Retirement: authenticatedWeightRetirement{Available: true, Completed: true}}},
			{"no-primary", authenticatedWeightOpenResult{Retirement: authenticatedWeightRetirement{Available: true, Completed: true}}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				req := ds41RetirementRequest(t, shards, digest, newDS41EPRecorder(&model.Model{}))
				got := loadV41ExpertParallelSessionWithAuthenticatedResult(t.Context(), req,
					func(context.Context, []ShardIdentity, string) authenticatedWeightOpenResult { return tc.result })
				if got.Err == nil || got.Panicked || got.OwnershipTransferred || got.Session != nil ||
					got.Retirement.Available || got.Retirement.Completed || got.Retirement.Err != nil ||
					(tc.result.Err != nil && got.Err != primary) {
					t.Fatalf("malformed evidence was promoted: %+v", got)
				}
			})
		}
	})
}
