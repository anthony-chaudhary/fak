package sessionledger

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	KindInterruptedAssistant = "assistant_interrupted"
	KindTurnTerminal         = "turn_terminal"

	TurnStatusInterrupted = "interrupted"
	interruptedTurnSchema = "fak-interrupted-turn/1"
)

// TerminalReason is the bounded, client-safe classification of an abnormal
// model turn. Cause is a closed token owned by the caller; Evidence contains no
// raw provider body.
type TerminalReason struct {
	Cause    string `json:"cause"`
	Evidence string `json:"evidence"`
}

// InterruptedTurn is the durable input for an abnormal streamed turn. Chunks
// are the exact content deltas successfully delivered to the caller's sink.
type InterruptedTurn struct {
	Turn   int
	Chunks []string
	Reason TerminalReason
}

// InterruptedAssistant is the reconstructed assistant record. The append-only
// ledger holds a bounded receipt while Chunks live in its content-addressed
// object store, so a long visible prefix is never replaced by an elision stub.
type InterruptedAssistant struct {
	Role   string         `json:"role"`
	Turn   int            `json:"turn"`
	Chunks []string       `json:"chunks"`
	Reason TerminalReason `json:"reason"`
}

// InterruptedAssistantReceipt is the bounded ledger row for one interrupted
// assistant prefix.
type InterruptedAssistantReceipt struct {
	Schema       string                 `json:"schema"`
	Role         string                 `json:"role"`
	Turn         int                    `json:"turn"`
	Chunks       ModelRequestContentRef `json:"chunks"`
	ChunkCount   int                    `json:"chunk_count"`
	ContentBytes int                    `json:"content_bytes"`
	Reason       TerminalReason         `json:"reason"`
	LedgerEntry  Hash                   `json:"-"`
}

// TurnTerminal balances one admitted model request with an explicit abnormal
// outcome, including the empty-prefix case where no assistant record exists.
type TurnTerminal struct {
	Schema string         `json:"schema"`
	Turn   int            `json:"turn"`
	Status string         `json:"status"`
	Reason TerminalReason `json:"reason"`
}

// InterruptedTurnReceipt names both durable effects. Assistant is nil when the
// stream delivered no non-empty content.
type InterruptedTurnReceipt struct {
	Assistant *InterruptedAssistantReceipt
	Terminal  TurnTerminal
}

// AppendInterruptedTurn preserves a non-empty assistant prefix and always
// appends its terminal reason. It appends no assistant record for an empty
// prefix, keeping pre-stream cancellation cheap while still closing the turn.
func (l *Ledger) AppendInterruptedTurn(trace string, interrupted InterruptedTurn) (InterruptedTurnReceipt, error) {
	if strings.TrimSpace(trace) == "" {
		return InterruptedTurnReceipt{}, errors.New("sessionledger: interrupted turn trace is required")
	}
	if interrupted.Turn <= 0 {
		return InterruptedTurnReceipt{}, errors.New("sessionledger: interrupted turn must be positive")
	}
	if strings.TrimSpace(interrupted.Reason.Cause) == "" {
		return InterruptedTurnReceipt{}, errors.New("sessionledger: interrupted turn reason is required")
	}

	receipt := InterruptedTurnReceipt{}
	contentBytes := 0
	for _, chunk := range interrupted.Chunks {
		contentBytes += len(chunk)
	}
	if contentBytes > 0 {
		chunks, err := json.Marshal(interrupted.Chunks)
		if err != nil {
			return InterruptedTurnReceipt{}, fmt.Errorf("sessionledger: marshal interrupted chunks: %w", err)
		}
		ref, err := l.putModelRequestObject(chunks)
		if err != nil {
			return InterruptedTurnReceipt{}, fmt.Errorf("sessionledger: store interrupted chunks: %w", err)
		}
		assistant := InterruptedAssistantReceipt{
			Schema: interruptedTurnSchema, Role: "assistant", Turn: interrupted.Turn,
			Chunks: ref, ChunkCount: len(interrupted.Chunks), ContentBytes: contentBytes,
			Reason: interrupted.Reason,
		}
		raw, err := json.Marshal(assistant)
		if err != nil {
			return InterruptedTurnReceipt{}, fmt.Errorf("sessionledger: marshal interrupted assistant receipt: %w", err)
		}
		entry, err := l.Append(trace, KindInterruptedAssistant, raw)
		if err != nil {
			return InterruptedTurnReceipt{}, err
		}
		assistant.LedgerEntry = entry.Hash
		receipt.Assistant = &assistant
	}

	terminal := TurnTerminal{
		Schema: interruptedTurnSchema, Turn: interrupted.Turn,
		Status: TurnStatusInterrupted, Reason: interrupted.Reason,
	}
	raw, err := json.Marshal(terminal)
	if err != nil {
		return InterruptedTurnReceipt{}, fmt.Errorf("sessionledger: marshal interrupted turn terminal: %w", err)
	}
	if _, err := l.Append(trace, KindTurnTerminal, raw); err != nil {
		return InterruptedTurnReceipt{}, err
	}
	receipt.Terminal = terminal
	return receipt, nil
}

// ReconstructInterruptedAssistant materializes and verifies the newest
// interrupted assistant record for turn. A zero turn selects the newest record.
func (l *Ledger) ReconstructInterruptedAssistant(trace string, turn int) (InterruptedAssistant, InterruptedAssistantReceipt, error) {
	entries, err := l.Chain(trace)
	if err != nil {
		return InterruptedAssistant{}, InterruptedAssistantReceipt{}, err
	}
	for i := len(entries) - 1; i >= 0; i-- {
		entry := entries[i]
		if entry.Kind != KindInterruptedAssistant {
			continue
		}
		var receipt InterruptedAssistantReceipt
		if err := json.Unmarshal(entry.Content, &receipt); err != nil {
			return InterruptedAssistant{}, InterruptedAssistantReceipt{}, fmt.Errorf("sessionledger: decode interrupted assistant receipt: %w", err)
		}
		if turn != 0 && receipt.Turn != turn {
			continue
		}
		if receipt.Schema != interruptedTurnSchema || receipt.Role != "assistant" || receipt.Turn <= 0 {
			return InterruptedAssistant{}, InterruptedAssistantReceipt{}, errors.New("sessionledger: invalid interrupted assistant receipt")
		}
		chunksRaw, err := l.resolveModelRequestObject(receipt.Chunks)
		if err != nil {
			return InterruptedAssistant{}, InterruptedAssistantReceipt{}, fmt.Errorf("sessionledger: resolve interrupted chunks: %w", err)
		}
		var chunks []string
		if err := json.Unmarshal(chunksRaw, &chunks); err != nil {
			return InterruptedAssistant{}, InterruptedAssistantReceipt{}, fmt.Errorf("sessionledger: decode interrupted chunks: %w", err)
		}
		contentBytes := 0
		for _, chunk := range chunks {
			contentBytes += len(chunk)
		}
		if len(chunks) != receipt.ChunkCount || contentBytes != receipt.ContentBytes {
			return InterruptedAssistant{}, InterruptedAssistantReceipt{}, errors.New("sessionledger: interrupted assistant receipt does not match stored chunks")
		}
		receipt.LedgerEntry = entry.Hash
		return InterruptedAssistant{
			Role: receipt.Role, Turn: receipt.Turn, Chunks: chunks, Reason: receipt.Reason,
		}, receipt, nil
	}
	return InterruptedAssistant{}, InterruptedAssistantReceipt{}, fmt.Errorf("sessionledger: interrupted assistant for trace %q turn %d not found", trace, turn)
}
