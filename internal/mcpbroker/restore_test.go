package mcpbroker

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestExactOriginalRestore(t *testing.T) {
	pretty := "{\n" + strings.Repeat(" ", 100) + `"n":9007199254740993,"n":1e+09,"s":"\u0061  b"` + "\n}"
	compact := `{"n":9007199254740993,"n":1e+09,"s":"\u0061  b"}`
	quote := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	rawContent := []byte(`[{ "type" : "text", "annotations": { "x":1e+09, "x":2 }, "text" : ` + quote(pretty) + `, "_meta" : { "a" : true } }]`)
	rawResult := []byte(`{"content":` + string(rawContent) + `,"structuredContent":` + compact + `}`)

	t.Run("ByteExactRoundTrip", func(t *testing.T) {
		sessionID := "session-alpha"
		out, receipt := CompactStructuredContentWithReceipt(
			rawResult,
			rawContent,
			WithExactOriginalRetention(true),
			WithSessionID(sessionID),
		)

		if receipt == nil {
			t.Fatalf("expected non-nil CompressionReceipt")
		}
		if receipt.Reason != ReasonSaved {
			t.Fatalf("expected ReasonSaved, got %q", receipt.Reason)
		}
		if receipt.RestoreHandle == "" {
			t.Fatalf("expected non-empty RestoreHandle on receipt")
		}
		if receipt.Metadata == nil || receipt.Metadata["restore_handle"] != receipt.RestoreHandle {
			t.Fatalf("expected receipt.Metadata[restore_handle] to match %q", receipt.RestoreHandle)
		}
		if !strings.HasPrefix(receipt.RestoreHandle, "sha256:") {
			t.Fatalf("expected handle prefix 'sha256:', got %q", receipt.RestoreHandle)
		}
		if bytes.Equal(out, rawContent) {
			t.Fatalf("expected compacted output to differ from raw original content")
		}

		// Compute expected SHA256 of original content
		expectedSum := sha256.Sum256(rawContent)
		expectedHash := hex.EncodeToString(expectedSum[:])
		expectedHandle := "sha256:" + expectedHash
		if receipt.RestoreHandle != expectedHandle {
			t.Fatalf("handle mismatch: got %q, want %q", receipt.RestoreHandle, expectedHandle)
		}

		// Restore via package-level RestoreOriginal
		restored, err := RestoreOriginal(sessionID, receipt.RestoreHandle)
		if err != nil {
			t.Fatalf("RestoreOriginal failed: %v", err)
		}
		if !bytes.Equal(restored, rawContent) {
			t.Fatalf("restored bytes do not match original bytes:\ngot:  %s\nwant: %s", string(restored), string(rawContent))
		}

		// Also verify restore using bare hex hash without "sha256:" prefix
		restoredBare, err := RestoreOriginal(sessionID, expectedHash)
		if err != nil {
			t.Fatalf("RestoreOriginal with bare hash failed: %v", err)
		}
		if !bytes.Equal(restoredBare, rawContent) {
			t.Fatalf("restored bytes with bare hash do not match original bytes")
		}

		// Verify broker.RestoreOriginal
		broker := NewBroker()
		restoredBroker, err := broker.RestoreOriginal(sessionID, receipt.RestoreHandle)
		if err != nil {
			t.Fatalf("broker.RestoreOriginal failed: %v", err)
		}
		if !bytes.Equal(restoredBroker, rawContent) {
			t.Fatalf("broker.RestoreOriginal bytes mismatch")
		}
	})

	t.Run("RejectCrossSessionAccess", func(t *testing.T) {
		ownerSession := "session-producer"
		otherSession := "session-intruder"

		_, receipt := CompactStructuredContentWithReceipt(
			rawResult,
			rawContent,
			WithExactOriginalRetention(true),
			WithSessionID(ownerSession),
		)
		if receipt.RestoreHandle == "" {
			t.Fatalf("expected valid handle")
		}

		// Cross-session access attempt must be rejected (fail-closed, unauthorized error)
		restored, err := RestoreOriginal(otherSession, receipt.RestoreHandle)
		if err == nil {
			t.Fatalf("expected error on cross-session access, got nil with %d bytes", len(restored))
		}
		if !errors.Is(err, ErrRestoreUnauthorized) {
			t.Fatalf("expected ErrRestoreUnauthorized, got %v", err)
		}
		if restored != nil {
			t.Fatalf("expected nil bytes on unauthorized error, got %s", string(restored))
		}

		// Empty session ID must also be rejected as unauthorized
		_, err = RestoreOriginal("", receipt.RestoreHandle)
		if !errors.Is(err, ErrRestoreUnauthorized) {
			t.Fatalf("expected ErrRestoreUnauthorized for empty session, got %v", err)
		}

		// Non-existent handle must return ErrRestoreNotFound
		_, err = RestoreOriginal(ownerSession, "sha256:0000000000000000000000000000000000000000000000000000000000000000")
		if !errors.Is(err, ErrRestoreNotFound) {
			t.Fatalf("expected ErrRestoreNotFound for missing handle, got %v", err)
		}
	})

	t.Run("PreservationFailureLeavesIdentity", func(t *testing.T) {
		t.Run("InjectedStoreFailure", func(t *testing.T) {
			store := NewRestoreStore()
			injectedErr := errors.New("simulated store disk failure")
			store.SetInjectedError(injectedErr)

			out, receipt := CompactStructuredContentWithReceipt(
				rawResult,
				rawContent,
				WithExactOriginalRetention(true),
				WithSessionID("session-test"),
				WithRestoreStore(store),
			)

			// Must emit identity output safely
			if !bytes.Equal(out, rawContent) {
				t.Fatalf("expected identity output on preservation failure, got transformed bytes")
			}
			if receipt.Reason != ReasonPreserveFailed {
				t.Fatalf("expected ReasonPreserveFailed, got %q", receipt.Reason)
			}
			if receipt.BytesSaved != 0 {
				t.Fatalf("expected 0 bytes saved on preservation failure, got %d", receipt.BytesSaved)
			}
			if receipt.OutputBytes != len(rawContent) {
				t.Fatalf("expected output bytes %d, got %d", len(rawContent), receipt.OutputBytes)
			}
			if receipt.RestoreHandle != "" {
				t.Fatalf("expected empty RestoreHandle on failure, got %q", receipt.RestoreHandle)
			}
		})

		t.Run("SizeCapExceeded", func(t *testing.T) {
			// Configure store with tiny size cap smaller than rawContent
			store := NewRestoreStore(WithStoreMaxBytes(10))

			out, receipt := CompactStructuredContentWithReceipt(
				rawResult,
				rawContent,
				WithExactOriginalRetention(true),
				WithSessionID("session-test"),
				WithRestoreStore(store),
			)

			if !bytes.Equal(out, rawContent) {
				t.Fatalf("expected identity output when size cap exceeded")
			}
			if receipt.Reason != ReasonPreserveFailed {
				t.Fatalf("expected ReasonPreserveFailed, got %q", receipt.Reason)
			}
			if receipt.BytesSaved != 0 {
				t.Fatalf("expected 0 bytes saved, got %d", receipt.BytesSaved)
			}
		})

		t.Run("MissingSessionIDFailClosed", func(t *testing.T) {
			// When exact-original is requested without a session/trace ID, it must fail safely
			out, receipt := CompactStructuredContentWithReceipt(
				rawResult,
				rawContent,
				WithExactOriginalRetention(true),
				WithSessionID(""), // empty session
			)

			if !bytes.Equal(out, rawContent) {
				t.Fatalf("expected identity output when session ID is missing")
			}
			if receipt.Reason != ReasonPreserveFailed {
				t.Fatalf("expected ReasonPreserveFailed, got %q", receipt.Reason)
			}
		})
	})

	t.Run("OperatorNoopNoneAuthoritative", func(t *testing.T) {
		for _, envVal := range []string{"noop", "none"} {
			t.Run("FAK_COMPRESSOR="+envVal, func(t *testing.T) {
				t.Setenv("FAK_COMPRESSOR", envVal)

				out, receipt := CompactStructuredContentWithReceipt(
					rawResult,
					rawContent,
					WithExactOriginalRetention(true),
					WithSessionID("session-test"),
				)

				if !bytes.Equal(out, rawContent) {
					t.Fatalf("expected identity output when FAK_COMPRESSOR=%s", envVal)
				}
				if receipt.Reason != ReasonOptOut {
					t.Fatalf("expected ReasonOptOut, got %q", receipt.Reason)
				}
				if receipt.RestoreHandle != "" {
					t.Fatalf("expected empty RestoreHandle when disabled")
				}
			})
		}
	})

	t.Run("MetadataAndContextConfiguration", func(t *testing.T) {
		// Enabled via Context
		ctx := WithExactOriginalContext(context.Background(), true)
		ctx = WithSessionContext(ctx, "session-from-ctx")

		out, receipt := CompactStructuredContentWithReceipt(
			rawResult,
			rawContent,
			WithCompressionContext(ctx),
		)
		if receipt.Reason != ReasonSaved {
			t.Fatalf("expected ReasonSaved via context, got %q", receipt.Reason)
		}
		if bytes.Equal(out, rawContent) {
			t.Fatalf("expected compression via context")
		}
		if receipt.RestoreHandle == "" {
			t.Fatalf("expected non-empty RestoreHandle via context")
		}

		restored, err := RestoreOriginal("session-from-ctx", receipt.RestoreHandle)
		if err != nil {
			t.Fatalf("restore failed: %v", err)
		}
		if !bytes.Equal(restored, rawContent) {
			t.Fatalf("restored bytes mismatch")
		}

		// Enabled via Metadata
		mdCtx := WithCallMetadata(context.Background(), map[string]string{
			"exact_original": "true",
			"session_id":     "session-from-md",
		})
		outMD, receiptMD := CompactStructuredContentWithReceipt(
			rawResult,
			rawContent,
			WithCompressionContext(mdCtx),
		)
		if receiptMD.Reason != ReasonSaved {
			t.Fatalf("expected ReasonSaved via metadata, got %q", receiptMD.Reason)
		}
		if receiptMD.RestoreHandle == "" {
			t.Fatalf("expected non-empty RestoreHandle via metadata")
		}
		if bytes.Equal(outMD, rawContent) {
			t.Fatalf("expected compression")
		}

		restoredMD, err := RestoreOriginal("session-from-md", receiptMD.RestoreHandle)
		if err != nil {
			t.Fatalf("restore via metadata session failed: %v", err)
		}
		if !bytes.Equal(restoredMD, rawContent) {
			t.Fatalf("restored bytes mismatch via metadata")
		}
	})

	t.Run("BrokerRouteCallIntegration", func(t *testing.T) {
		broker := NewBroker()
		toolName := "fetch_structured"

		err := broker.RegisterTool(ToolRegistration{
			Name: toolName,
			Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
				compressed, receipt := CompactStructuredContentWithReceipt(
					rawResult,
					rawContent,
					WithCompressionContext(ctx),
				)
				resp := &CallResponse{
					Tool:               toolName,
					Content:            compressed,
					CompressionReceipt: receipt,
				}
				if receipt != nil && receipt.RestoreHandle != "" {
					resp.Metadata = map[string]string{
						"restore_handle": receipt.RestoreHandle,
					}
				}
				return resp, nil
			},
		})
		if err != nil {
			t.Fatalf("RegisterTool failed: %v", err)
		}

		sessionID := "session-broker-integration"
		callResp, err := broker.RouteCall(context.Background(), CallRequest{
			Tool:      toolName,
			SessionID: sessionID,
			Metadata: map[string]string{
				"exact_original": "true",
				"session_id":     sessionID,
			},
		})
		if err != nil {
			t.Fatalf("RouteCall failed: %v", err)
		}

		handle := callResp.CompressionReceipt.RestoreHandle
		if handle == "" {
			t.Fatalf("expected RestoreHandle on broker call response")
		}

		// Verify broker restore succeeds for owning session
		restored, err := broker.RestoreOriginal(sessionID, handle)
		if err != nil {
			t.Fatalf("broker.RestoreOriginal failed: %v", err)
		}
		if !bytes.Equal(restored, rawContent) {
			t.Fatalf("broker restored content mismatch")
		}

		// Verify cross-session access rejected on broker
		_, err = broker.RestoreOriginal("other-session", handle)
		if !errors.Is(err, ErrRestoreUnauthorized) {
			t.Fatalf("expected ErrRestoreUnauthorized from broker.RestoreOriginal, got %v", err)
		}
	})

	t.Run("BrokerSessionLevelPolicy", func(t *testing.T) {
		broker := NewBroker()
		toolName := "session_policy_tool"

		_ = broker.RegisterTool(ToolRegistration{
			Name: toolName,
			Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
				compressed, receipt := CompactStructuredContentWithReceipt(
					rawResult,
					rawContent,
					WithCompressionContext(ctx),
				)
				return &CallResponse{
					Tool:               toolName,
					Content:            compressed,
					CompressionReceipt: receipt,
				}, nil
			},
		})

		sessionID := "session-opt-in"
		broker.SetSessionExactOriginal(sessionID, true)
		if !broker.GetSessionExactOriginal(sessionID) {
			t.Fatalf("expected session exact original to be true")
		}

		// Call without explicit "exact_original" in metadata, only session ID
		ctx := WithSessionContext(context.Background(), sessionID)
		callResp, err := broker.RouteCall(ctx, CallRequest{
			Tool:      toolName,
			SessionID: sessionID,
		})
		if err != nil {
			t.Fatalf("RouteCall failed: %v", err)
		}

		if callResp.CompressionReceipt == nil || callResp.CompressionReceipt.RestoreHandle == "" {
			t.Fatalf("expected RestoreHandle from session-level policy")
		}

		restored, err := broker.RestoreOriginal(sessionID, callResp.CompressionReceipt.RestoreHandle)
		if err != nil {
			t.Fatalf("restore failed: %v", err)
		}
		if !bytes.Equal(restored, rawContent) {
			t.Fatalf("restored bytes mismatch")
		}
	})
}
