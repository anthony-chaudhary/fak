package fabricmap

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

const HelperSchema = "fak.fabricmap-helper.v1"

type HelperRequest struct {
	Schema   string   `json:"schema"`
	Link     Link     `json:"link"`
	Access   Access   `json:"access"`
	Transfer Transfer `json:"transfer"`
}
type HelperResponse struct {
	Schema   string            `json:"schema"`
	Receipt  HopReceipt        `json:"receipt"`
	CPUPath  string            `json:"cpu_path"`
	Evidence map[string]string `json:"evidence,omitempty"`
}

// ProcessAdapter invokes a node-local hardware helper with one JSON request on
// stdin and one JSON response on stdout. Command is an argv vector: no shell is
// involved. Driver/vendor dependencies remain outside the Go module.
type ProcessAdapter struct {
	Command          []string
	Timeout          time.Duration
	MaxResponseBytes int
}

func (a ProcessAdapter) Transfer(ctx context.Context, link Link, access Access, transfer Transfer) (HopReceipt, error) {
	if len(a.Command) == 0 || strings.TrimSpace(a.Command[0]) == "" {
		return HopReceipt{}, errors.New("hardware helper command is required")
	}
	if transfer.Source != link.From || transfer.Destination != link.To {
		return HopReceipt{}, errors.New("helper transfer direction does not match link")
	}
	if transfer.Bytes == 0 {
		return HopReceipt{}, errors.New("helper transfer bytes must be positive")
	}
	timeout := a.Timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	maxBytes := a.MaxResponseBytes
	if maxBytes <= 0 {
		maxBytes = 1 << 20
	}
	request := HelperRequest{Schema: HelperSchema, Link: cloneLink(link), Access: cloneAccess(access), Transfer: transfer}
	input, err := json.Marshal(request)
	if err != nil {
		return HopReceipt{}, err
	}
	callCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	command := exec.CommandContext(callCtx, a.Command[0], a.Command[1:]...)
	command.Stdin = bytes.NewReader(append(input, '\n'))
	var stdout, stderr limitedBuffer
	stdout.max = maxBytes
	stderr.max = 32 << 10
	command.Stdout = &stdout
	command.Stderr = &stderr
	err = command.Run()
	if callCtx.Err() != nil {
		return HopReceipt{}, fmt.Errorf("hardware helper timeout: %w", callCtx.Err())
	}
	if err != nil {
		return HopReceipt{}, fmt.Errorf("hardware helper failed: %w: %s", err, safeHelperError(stderr.String()))
	}
	if stdout.overflow {
		return HopReceipt{}, fmt.Errorf("hardware helper response exceeds %d bytes", maxBytes)
	}
	decoder := json.NewDecoder(bytes.NewReader(stdout.Bytes()))
	decoder.DisallowUnknownFields()
	var response HelperResponse
	if err := decoder.Decode(&response); err != nil {
		return HopReceipt{}, fmt.Errorf("decode hardware helper response: %w", err)
	}
	if response.Schema != HelperSchema {
		return HopReceipt{}, fmt.Errorf("hardware helper schema %q, want %q", response.Schema, HelperSchema)
	}
	if strings.TrimSpace(response.CPUPath) == "" {
		return HopReceipt{}, errors.New("hardware helper CPU-path evidence is required")
	}
	receipt := response.Receipt
	if receipt.LinkID != link.ID || receipt.From != link.From || receipt.To != link.To || receipt.Transport != link.Transport || receipt.Bytes != transfer.Bytes {
		return HopReceipt{}, errors.New("hardware helper receipt does not match directed transfer")
	}
	if receipt.State != HopComplete || receipt.StartedAt.IsZero() || receipt.CompletedAt.IsZero() || receipt.CompletedAt.Before(receipt.StartedAt) || receipt.Integrity.Algorithm == "" || receipt.Integrity.Digest == "" {
		return HopReceipt{}, errors.New("hardware helper receipt lacks completion evidence")
	}
	return receipt, nil
}

type limitedBuffer struct {
	bytes.Buffer
	max      int
	overflow bool
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	original := len(p)
	remaining := b.max - b.Len()
	if remaining <= 0 {
		b.overflow = true
		return original, nil
	}
	if len(p) > remaining {
		b.overflow = true
		p = p[:remaining]
	}
	_, _ = b.Buffer.Write(p)
	return original, nil
}
func safeHelperError(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "no public diagnostic"
	}
	if len(value) > 512 {
		value = value[:512]
	}
	return value
}
