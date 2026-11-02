package cachemeta

import (
	"errors"
	"testing"
)

// TestTransferChannelPairedVsOneSidedContract pins the borrowed LMCache channel
// contract (ADAPT of lmcache/v1/transfer_channel/abstract.py:21-285@1b7dff2):
// a paired channel admits exactly BatchedSend/BatchedRecv and refuses the
// one-sided operations with ErrOneSidedCallOnPaired; a one-sided channel admits
// exactly BatchedWrite/BatchedRead and refuses the paired operations with
// ErrPairedCallOnOneSided. The in-memory fake records which calls were accepted,
// so the allowed-call sets are pinned by observation, not by absence of panic.
func TestTransferChannelPairedVsOneSidedContract(t *testing.T) {
	// Compile-time proof that the fake satisfies the seam.
	var _ TransferChannel = (*MemoryTransferChannel)(nil)

	batch := func(digests ...string) []TransferSpanRequest {
		reqs := make([]TransferSpanRequest, len(digests))
		for i, d := range digests {
			reqs[i] = TransferSpanRequest{
				SpanDigest: d,
				FromTier:   TierHBM,
				ToTier:     TierDRAM,
				Owner:      "engine",
				Bytes:      128,
			}
		}
		return reqs
	}

	t.Run("paired channel admits only the paired topology", func(t *testing.T) {
		ch := NewMemoryTransferChannel(TransferModePaired)
		if got := ch.Mode(); got != TransferModePaired {
			t.Fatalf("Mode() = %q, want %q", got, TransferModePaired)
		}

		res, err := ch.BatchedSend(batch("span-a"))
		if err != nil {
			t.Fatalf("BatchedSend on paired channel: unexpected error %v", err)
		}
		if len(res) != 1 || res[0].SpanDigest != "span-a" || res[0].Outcome != KVTransferOK {
			t.Fatalf("BatchedSend results = %+v, want one OK result for span-a", res)
		}

		if _, err := ch.BatchedRecv(batch("span-b")); err != nil {
			t.Fatalf("BatchedRecv on paired channel: unexpected error %v", err)
		}

		if _, err := ch.BatchedWrite(batch("span-c")); !errors.Is(err, ErrOneSidedCallOnPaired) {
			t.Fatalf("BatchedWrite on paired channel: error = %v, want errors.Is ErrOneSidedCallOnPaired", err)
		}
		if _, err := ch.BatchedRead(batch("span-d")); !errors.Is(err, ErrOneSidedCallOnPaired) {
			t.Fatalf("BatchedRead on paired channel: error = %v, want errors.Is ErrOneSidedCallOnPaired", err)
		}

		wantCalls := []TransferOperation{OpBatchedSend, OpBatchedRecv}
		if len(ch.Calls) != len(wantCalls) {
			t.Fatalf("accepted calls = %v, want exactly %v (refusals must not record)", ch.Calls, wantCalls)
		}
		for i, op := range wantCalls {
			if ch.Calls[i] != op {
				t.Fatalf("accepted calls = %v, want exactly %v", ch.Calls, wantCalls)
			}
		}
	})

	t.Run("one-sided channel admits only the one-sided topology", func(t *testing.T) {
		ch := NewMemoryTransferChannel(TransferModeOneSided)
		if got := ch.Mode(); got != TransferModeOneSided {
			t.Fatalf("Mode() = %q, want %q", got, TransferModeOneSided)
		}

		res, err := ch.BatchedWrite(batch("span-w"))
		if err != nil {
			t.Fatalf("BatchedWrite on one-sided channel: unexpected error %v", err)
		}
		if len(res) != 1 || res[0].SpanDigest != "span-w" || res[0].Outcome != KVTransferOK {
			t.Fatalf("BatchedWrite results = %+v, want one OK result for span-w", res)
		}

		if _, err := ch.BatchedRead(batch("span-r")); err != nil {
			t.Fatalf("BatchedRead on one-sided channel: unexpected error %v", err)
		}

		if _, err := ch.BatchedSend(batch("span-s")); !errors.Is(err, ErrPairedCallOnOneSided) {
			t.Fatalf("BatchedSend on one-sided channel: error = %v, want errors.Is ErrPairedCallOnOneSided", err)
		}
		if _, err := ch.BatchedRecv(batch("span-v")); !errors.Is(err, ErrPairedCallOnOneSided) {
			t.Fatalf("BatchedRecv on one-sided channel: error = %v, want errors.Is ErrPairedCallOnOneSided", err)
		}

		wantCalls := []TransferOperation{OpBatchedWrite, OpBatchedRead}
		if len(ch.Calls) != len(wantCalls) {
			t.Fatalf("accepted calls = %v, want exactly %v (refusals must not record)", ch.Calls, wantCalls)
		}
		for i, op := range wantCalls {
			if ch.Calls[i] != op {
				t.Fatalf("accepted calls = %v, want exactly %v", ch.Calls, wantCalls)
			}
		}
	})

	t.Run("guard is fail-closed over the closed operation set", func(t *testing.T) {
		cases := []struct {
			mode    TransferMode
			op      TransferOperation
			wantErr error
		}{
			{TransferModePaired, OpBatchedSend, nil},
			{TransferModePaired, OpBatchedRecv, nil},
			{TransferModePaired, OpBatchedWrite, ErrOneSidedCallOnPaired},
			{TransferModePaired, OpBatchedRead, ErrOneSidedCallOnPaired},
			{TransferModeOneSided, OpBatchedSend, ErrPairedCallOnOneSided},
			{TransferModeOneSided, OpBatchedRecv, ErrPairedCallOnOneSided},
			{TransferModeOneSided, OpBatchedWrite, nil},
			{TransferModeOneSided, OpBatchedRead, nil},
			// An unrecognized mode is never silently treated as either topology.
			{"", OpBatchedSend, ErrUnknownTransferMode},
			{"", OpBatchedWrite, ErrUnknownTransferMode},
			{"future-mode", OpBatchedRecv, ErrUnknownTransferMode},
		}
		for _, tc := range cases {
			err := CheckTransferCall(tc.mode, tc.op)
			if tc.wantErr == nil {
				if err != nil {
					t.Errorf("CheckTransferCall(%q, %q) = %v, want nil", tc.mode, tc.op, err)
				}
				continue
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("CheckTransferCall(%q, %q) = %v, want errors.Is %v", tc.mode, tc.op, err, tc.wantErr)
			}
		}
	})
}
