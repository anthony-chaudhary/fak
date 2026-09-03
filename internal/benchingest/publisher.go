package benchingest

import (
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/benchsnapshot"
)

// SnapshotPublisher publishes ingested benchmark evidence and correlation receipts
// into a sequence-addressed, watcher-safe evidence directory (#10302).
type SnapshotPublisher struct {
	writer *benchsnapshot.SnapshotWriter
}

// NewSnapshotPublisher constructs a publisher backed by a snapshot writer.
func NewSnapshotPublisher(writer *benchsnapshot.SnapshotWriter) *SnapshotPublisher {
	return &SnapshotPublisher{writer: writer}
}

// PublishCorrelationReceipt encodes receipt as payload and appends a sequence-addressed
// snapshot to the writer.
func (p *SnapshotPublisher) PublishCorrelationReceipt(
	receipt *CorrelationReceipt,
) (benchsnapshot.SnapshotReceipt, error) {
	if receipt == nil {
		return benchsnapshot.SnapshotReceipt{}, fmt.Errorf("correlation receipt cannot be nil")
	}
	payload, err := json.Marshal(receipt)
	if err != nil {
		return benchsnapshot.SnapshotReceipt{}, fmt.Errorf("marshal correlation receipt: %w", err)
	}
	if p.writer == nil {
		return benchsnapshot.SnapshotReceipt{}, fmt.Errorf("snapshot writer cannot be nil")
	}
	return p.writer.WriteSnapshot(payload)
}
