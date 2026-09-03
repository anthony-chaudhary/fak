package agentopt

import (
	"math"
	"sync"
	"testing"
)

func buildSampleTestDocs() []Document {
	return []Document{
		{
			Path:     "internal/gateway/mux.go",
			Basename: "mux.go",
			Symbols:  []string{"MuxRouter", "NewMuxRouter", "RouteHandler", "ServeHTTP", "PathPrefix"},
			Content:  "package gateway\n\n// MuxRouter handles HTTP request dispatching, endpoint multiplexing, and URL path matching.\nfunc NewMuxRouter() *MuxRouter { return &MuxRouter{} }\nfunc (m *MuxRouter) RouteHandler(pattern string) {}\n",
		},
		{
			Path:     "internal/auth/token_validator.go",
			Basename: "token_validator.go",
			Symbols:  []string{"TokenValidator", "NewTokenValidator", "ValidateToken", "JWTClaims", "VerifySignature"},
			Content:  "package auth\n\n// TokenValidator verifies signed JWT authentication tokens, expiration timestamps, and issuer claims.\nfunc ValidateToken(raw string) (*JWTClaims, error) { return nil, nil }\nfunc VerifySignature(token string, secret []byte) bool { return true }\n",
		},
		{
			Path:     "internal/ratelimit/token_bucket.go",
			Basename: "token_bucket.go",
			Symbols:  []string{"TokenBucketLimiter", "NewTokenBucketLimiter", "TakeToken", "Allow", "RefillRate"},
			Content:  "package ratelimit\n\n// TokenBucketLimiter implements distributed rate limiting and token bucket throttling with burst capacity.\nfunc NewTokenBucketLimiter(rate float64, burst int) *TokenBucketLimiter { return &TokenBucketLimiter{} }\nfunc (tb *TokenBucketLimiter) TakeToken() bool { return true }\nfunc (tb *TokenBucketLimiter) Allow() bool { return true }\n",
		},
		{
			Path:     "internal/storage/kv_store.go",
			Basename: "kv_store.go",
			Symbols:  []string{"KVStore", "NewKVStore", "GetRecord", "PutRecord", "DeleteRecord", "FlushSnapshot"},
			Content:  "package storage\n\n// KVStore provides persistent key value storage, partitioned records, and disk snapshot persistence.\nfunc NewKVStore(path string) *KVStore { return &KVStore{} }\nfunc (s *KVStore) GetRecord(k string) ([]byte, error) { return nil, nil }\nfunc (s *KVStore) PutRecord(k string, v []byte) error { return nil }\n",
		},
		{
			Path:     "internal/telemetry/span_tracer.go",
			Basename: "span_tracer.go",
			Symbols:  []string{"SpanTracer", "StartSpan", "EndSpan", "TraceContext", "InjectHeaders"},
			Content:  "package telemetry\n\n// SpanTracer records distributed tracing spans, latency timings, and export traces across network hops.\nfunc StartSpan(name string) *TraceContext { return &TraceContext{} }\nfunc EndSpan(ctx *TraceContext) {}\n",
		},
	}
}

func TestHybridReRanking(t *testing.T) {
	t.Run("ExactSymbolQueries", func(t *testing.T) {
		reranker := NewHybridReranker()
		reranker.IndexDocuments(buildSampleTestDocs())

		testCases := []struct {
			query        string
			denseScores  map[string]float64
			expectedPath string
		}{
			{
				query: "TokenValidator",
				// Even if dense score slightly favors another file with token in its text,
				// the exact symbol boost guarantees TokenValidator tops the ranking.
				denseScores: map[string]float64{
					"internal/ratelimit/token_bucket.go": 0.88,
					"internal/auth/token_validator.go":   0.65,
				},
				expectedPath: "internal/auth/token_validator.go",
			},
			{
				query: "TakeToken",
				denseScores: map[string]float64{
					"internal/auth/token_validator.go": 0.75,
				},
				expectedPath: "internal/ratelimit/token_bucket.go",
			},
			{
				query: "MuxRouter",
				denseScores: map[string]float64{
					"internal/gateway/mux.go": 0.50,
				},
				expectedPath: "internal/gateway/mux.go",
			},
			{
				query:        "KVStore",
				denseScores:  nil,
				expectedPath: "internal/storage/kv_store.go",
			},
			{
				query:        "SpanTracer",
				denseScores:  nil,
				expectedPath: "internal/telemetry/span_tracer.go",
			},
		}

		for _, tc := range testCases {
			ranked := reranker.Rerank(tc.query, tc.denseScores)
			if len(ranked) == 0 {
				t.Fatalf("query %q returned no items", tc.query)
			}
			topItem := ranked[0]
			if topItem.Path != tc.expectedPath {
				t.Errorf("query %q: expected top-1 path %q, got %q (score: %f)",
					tc.query, tc.expectedPath, topItem.Path, topItem.Score)
			}
			if !topItem.Match.ExactSymbolHit {
				t.Errorf("query %q: expected ExactSymbolHit to be true for %q", tc.query, topItem.Path)
			}
			if topItem.Rank != 0 {
				t.Errorf("query %q: expected Rank=0, got %d", tc.query, topItem.Rank)
			}
		}
	})

	t.Run("ExactFilePathAndBasenameQueries", func(t *testing.T) {
		reranker := NewHybridReranker()
		reranker.IndexDocuments(buildSampleTestDocs())

		testCases := []struct {
			query        string
			expectedPath string
		}{
			{
				query:        "internal/gateway/mux.go",
				expectedPath: "internal/gateway/mux.go",
			},
			{
				query:        "token_bucket.go",
				expectedPath: "internal/ratelimit/token_bucket.go",
			},
			{
				query:        "token_validator",
				expectedPath: "internal/auth/token_validator.go",
			},
			{
				query:        "kv_store.go",
				expectedPath: "internal/storage/kv_store.go",
			},
		}

		for _, tc := range testCases {
			ranked := reranker.Rerank(tc.query, nil)
			if len(ranked) == 0 {
				t.Fatalf("query %q returned no items", tc.query)
			}
			topItem := ranked[0]
			if topItem.Path != tc.expectedPath {
				t.Errorf("query %q: expected top-1 path %q, got %q (score: %f)",
					tc.query, tc.expectedPath, topItem.Path, topItem.Score)
			}
		}
	})

	t.Run("ConceptualQuestions", func(t *testing.T) {
		reranker := NewHybridReranker()
		reranker.IndexDocuments(buildSampleTestDocs())

		testCases := []struct {
			query        string
			denseScores  map[string]float64
			expectedPath string
		}{
			{
				query: "how do I handle distributed rate limiting and token bucket throttling",
				denseScores: map[string]float64{
					"internal/ratelimit/token_bucket.go": 0.95,
					"internal/gateway/mux.go":            0.50,
					"internal/auth/token_validator.go":   0.45,
					"internal/storage/kv_store.go":       0.20,
					"internal/telemetry/span_tracer.go":  0.10,
				},
				expectedPath: "internal/ratelimit/token_bucket.go",
			},
			{
				query: "verifying signed tokens and checking issuer claims",
				denseScores: map[string]float64{
					"internal/auth/token_validator.go":   0.94,
					"internal/ratelimit/token_bucket.go": 0.40,
					"internal/gateway/mux.go":            0.30,
					"internal/storage/kv_store.go":       0.20,
					"internal/telemetry/span_tracer.go":  0.15,
				},
				expectedPath: "internal/auth/token_validator.go",
			},
			{
				query: "persisting key value data and flushing disk snapshot records",
				denseScores: map[string]float64{
					"internal/storage/kv_store.go":       0.93,
					"internal/ratelimit/token_bucket.go": 0.35,
					"internal/gateway/mux.go":            0.25,
					"internal/auth/token_validator.go":   0.20,
					"internal/telemetry/span_tracer.go":  0.10,
				},
				expectedPath: "internal/storage/kv_store.go",
			},
			{
				query: "monitoring distributed tracing latency timings across network hops",
				denseScores: map[string]float64{
					"internal/telemetry/span_tracer.go":  0.96,
					"internal/gateway/mux.go":            0.40,
					"internal/ratelimit/token_bucket.go": 0.20,
					"internal/auth/token_validator.go":   0.15,
					"internal/storage/kv_store.go":       0.10,
				},
				expectedPath: "internal/telemetry/span_tracer.go",
			},
		}

		for _, tc := range testCases {
			ranked := reranker.Rerank(tc.query, tc.denseScores)
			if len(ranked) == 0 {
				t.Fatalf("query %q returned no items", tc.query)
			}
			topItem := ranked[0]
			if topItem.Path != tc.expectedPath {
				t.Errorf("query %q: expected top-1 path %q, got %q (score: %f)",
					tc.query, tc.expectedPath, topItem.Path, topItem.Score)
			}
			// Conceptual questions should not trigger exact symbol hits.
			if topItem.Match.ExactSymbolHit {
				t.Errorf("query %q: expected ExactSymbolHit to be false", tc.query)
			}
			// Both BM25 and dense should contribute to RRF.
			if topItem.BM25Rank < 0 || topItem.DenseRank < 0 {
				t.Errorf("query %q: expected positive BM25Rank and DenseRank, got BM25=%d, Dense=%d",
					tc.query, topItem.BM25Rank, topItem.DenseRank)
			}
		}
	})

	t.Run("DenseScoreRescuesVocabularyMismatch", func(t *testing.T) {
		reranker := NewHybridReranker()
		reranker.IndexDocuments(buildSampleTestDocs())

		// Query with terms not directly in document text: semantic similarity lifts correct item.
		query := "traffic congestion management through leaky reservoir algorithms"
		denseScores := map[string]float64{
			"internal/ratelimit/token_bucket.go": 0.95,
			"internal/gateway/mux.go":            0.15,
			"internal/auth/token_validator.go":   0.10,
			"internal/storage/kv_store.go":       0.05,
			"internal/telemetry/span_tracer.go":  0.05,
		}

		ranked := reranker.Rerank(query, denseScores)
		if len(ranked) == 0 {
			t.Fatal("expected non-empty ranking")
		}
		if ranked[0].Path != "internal/ratelimit/token_bucket.go" {
			t.Errorf("expected dense arm to rescue vocabulary mismatch for token_bucket, got %q",
				ranked[0].Path)
		}
	})

	t.Run("RRFArithmeticVerification", func(t *testing.T) {
		reranker := NewHybridReranker(HybridRerankerConfig{
			RRFK: 60.0,
		})
		reranker.IndexDocuments(buildSampleTestDocs())

		query := "rate limiting"
		denseScores := map[string]float64{
			"internal/ratelimit/token_bucket.go": 0.99,
		}

		ranked := reranker.Rerank(query, denseScores)
		topItem := ranked[0]

		if topItem.BM25Rank == 0 && topItem.DenseRank == 0 {
			expectedRRF := (1.0 / 61.0) + (1.0 / 61.0)
			diff := math.Abs(topItem.Match.RRFScore - expectedRRF)
			if diff > 1e-6 {
				t.Errorf("expected RRFScore %f, got %f (diff: %e)", expectedRRF, topItem.Match.RRFScore, diff)
			}
		}
	})

	t.Run("EdgeCasesAndRobustness", func(t *testing.T) {
		emptyReranker := NewHybridReranker()
		if res := emptyReranker.Rerank("anything", nil); res != nil {
			t.Errorf("expected nil for empty reranker, got %v", res)
		}

		reranker := NewHybridReranker()
		reranker.IndexDocuments(buildSampleTestDocs())
		if reranker.DocumentCount() != 5 {
			t.Errorf("expected 5 documents, got %d", reranker.DocumentCount())
		}

		doc, ok := reranker.GetDocument("internal/gateway/mux.go")
		if !ok || doc.Basename != "mux.go" {
			t.Errorf("failed to retrieve indexed document: %+v, ok=%v", doc, ok)
		}

		// Auto symbol extraction test
		autoDoc := Document{
			Path:    "pkg/calc/math.go",
			Content: "package calc\n\nfunc Add(a, b int) int { return a + b }\ntype Point struct { X, Y int }\n",
		}
		reranker.IndexDocument(autoDoc)
		storedAuto, ok := reranker.GetDocument("pkg/calc/math.go")
		if !ok {
			t.Fatal("failed to find auto-indexed document")
		}
		if len(storedAuto.Symbols) < 2 {
			t.Errorf("expected auto extracted symbols >= 2, got %v", storedAuto.Symbols)
		}

		// Clear test
		reranker.Clear()
		if reranker.DocumentCount() != 0 {
			t.Errorf("expected 0 documents after Clear, got %d", reranker.DocumentCount())
		}
	})

	t.Run("ConcurrentReadWriteSafety", func(t *testing.T) {
		reranker := NewHybridReranker()
		reranker.IndexDocuments(buildSampleTestDocs())

		var wg sync.WaitGroup
		for i := 0; i < 20; i++ {
			wg.Add(2)
			go func() {
				defer wg.Done()
				_ = reranker.Rerank("distributed rate limiting", map[string]float64{
					"internal/ratelimit/token_bucket.go": 0.9,
				})
			}()
			go func(idx int) {
				defer wg.Done()
				if idx%5 == 0 {
					reranker.IndexDocument(Document{
						Path:    "dynamic/file.go",
						Content: "package dynamic\nfunc DynamicSymbol() {}\n",
					})
				}
			}(i)
		}
		wg.Wait()
	})
}
