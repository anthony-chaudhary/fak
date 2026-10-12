//go:build darwin && arm64 && cgo

package model

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"hash"
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func qwen35MetalMTPVerifyPanelReceipt(tokens, base int, graphReceipt metalgemm.GraphReceipt, owner *metalQwen35MTPCheckpoint, operationReceipts ...*qwen35MetalMTPOperations) Qwen35MetalMTPVerifyPanelReceipt {
	operations := &qwen35MetalMTPOperations{}
	if len(operationReceipts) > 0 && operationReceipts[0] != nil {
		operations = operationReceipts[0]
	}
	layers, lineageIdentities, checkpointBinding := qwen35MetalMTPCheckpointBinding(base, owner)
	receipt := Qwen35MetalMTPVerifyPanelReceipt{
		Schema: Qwen35MetalMTPVerifyPanelSchema, Path: Qwen35MetalMTPVerifyPanelPath,
		StateDigestDomain: Qwen35MetalMTPStateDigestDomain, Tokens: tokens, Base: base,
		GDNCheckpointBindingSHA256: checkpointBinding, GDNCheckpointLayers: layers, GDNCheckpointLineageSHA256: lineageIdentities,
		CommandBuffers: 1, Encoders: graphReceipt.Encoders, CheckpointLayers: len(owner.checkpoints),
		IntermediateWaits: graphReceipt.IntermediateWaits, IntermediateReadbacks: graphReceipt.IntermediateReadbacks,
		TerminalWaits: qwen35MetalMTPObservedCount(graphReceipt.CompletedWait), TerminalReadbacks: graphReceipt.HostReadbacks,
		HostUploadBytes: graphReceipt.HostUploadBytes, HostReadbackBytes: graphReceipt.HostReadbackBytes,
		Committed: graphReceipt.Committed, CompletedWait: graphReceipt.CompletedWait, TimingAvailable: graphReceipt.TimingAvailable,
		GPUMilliseconds: graphReceipt.GPUMilliseconds, WaitMilliseconds: graphReceipt.WaitMilliseconds,
		Q6KDownProjectionOperations: operations.q6Down, Q6KHeadOperations: operations.q6Head,
	}
	if graphReceipt.Committed && graphReceipt.CompletedWait {
		for _, checkpoint := range owner.checkpoints {
			cr := checkpoint.Receipt()
			receipt.DeviceCheckpointCopies += cr.DeviceCopies
			receipt.CheckpointBufferSwaps += cr.BufferSwaps
			receipt.HostStateUploads += cr.HostStateUploads
			receipt.HostStateReadbacks += cr.HostStateReadbacks
		}
	}
	return receipt
}

// qwen35MetalMTPCheckpointBinding aggregates checkpoint lineage: native owner,
// mutation version, checkpoint generation, and covered layer/copy accounting.
// Recurrent and convolution buffer byte contents are intentionally not hashed.
func qwen35MetalMTPCheckpointBinding(base int, owner *metalQwen35MTPCheckpoint) ([]int, []string, string) {
	layers := make([]int, 0, len(owner.backupLayer))
	for layer := range owner.backupLayer {
		layers = append(layers, layer)
	}
	sort.Ints(layers)
	h := sha256.New()
	_, _ = h.Write([]byte("fak.qwen35-metal-mtp-p4-gdn-checkpoint/v1\x00"))
	qwen35MetalMTPDigestUint64(h, uint64(base))
	qwen35MetalMTPDigestUint64(h, uint64(len(layers)))
	lineageIdentities := make([]string, 0, len(layers))
	for i, layer := range layers {
		qwen35MetalMTPDigestUint64(h, uint64(layer))
		if i < len(owner.checkpoints) {
			receipt := owner.checkpoints[i].Receipt()
			lineageIdentity, identityErr := owner.checkpoints[i].StateIdentity()
			if identityErr != nil {
				lineageIdentity = ""
			}
			lineageIdentities = append(lineageIdentities, lineageIdentity)
			_, _ = h.Write([]byte(lineageIdentity))
			qwen35MetalMTPDigestUint64(h, uint64(receipt.DeviceCopies))
			qwen35MetalMTPDigestUint64(h, uint64(receipt.BufferSwaps))
			qwen35MetalMTPDigestUint64(h, uint64(receipt.HostStateUploads))
			qwen35MetalMTPDigestUint64(h, uint64(receipt.HostStateReadbacks))
		}
	}
	return layers, lineageIdentities, hex.EncodeToString(h.Sum(nil))
}

func qwen35MetalMTPTransactionDigest(state, checkpoint string) string {
	// state binds token, hidden, and KV content. checkpoint binds only GDN
	// checkpoint lineage; it makes no recurrent/conv byte-content claim.
	h := sha256.New()
	_, _ = h.Write([]byte("fak.qwen35-metal-mtp-p4-transaction/v1\x00"))
	_, _ = h.Write([]byte(state))
	_, _ = h.Write([]byte(checkpoint))
	return hex.EncodeToString(h.Sum(nil))
}

func qwen35MetalMTPObservedCount(observed bool) int {
	if observed {
		return 1
	}
	return 0
}

func qwen35MetalMTPStateDigest(base int, ids []int, rawHidden []float32, kv []Qwen35MetalMTPKVPanel) string {
	h := sha256.New()
	_, _ = h.Write([]byte(Qwen35MetalMTPStateDigestDomain))
	qwen35MetalMTPDigestUint64(h, uint64(base))
	qwen35MetalMTPDigestUint64(h, uint64(len(ids)))
	for _, id := range ids {
		qwen35MetalMTPDigestUint64(h, uint64(id))
	}
	qwen35MetalMTPDigestFloats(h, rawHidden)
	for _, panel := range kv {
		qwen35MetalMTPDigestUint64(h, uint64(panel.Layer))
		qwen35MetalMTPDigestFloats(h, panel.KRaw)
		qwen35MetalMTPDigestFloats(h, panel.KPost)
		qwen35MetalMTPDigestFloats(h, panel.V)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func qwen35MetalMTPDigestUint64(h hash.Hash, value uint64) {
	var bits [8]byte
	binary.LittleEndian.PutUint64(bits[:], value)
	_, _ = h.Write(bits[:])
}

func qwen35MetalMTPDigestFloats(h hash.Hash, values []float32) {
	qwen35MetalMTPDigestUint64(h, uint64(len(values)))
	for _, value := range values {
		qwen35MetalMTPDigestUint64(h, uint64(math.Float32bits(value)))
	}
}
