// Package remotemaintresult binds a source-validated maintenance receipt to one
// result-admission call. Its scan probe never replaces the admitted payload.
package remotemaintresult

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"io"
	"net"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

const maintenanceTool = "fak_remote_maintenance"

type bindingKey struct{}

type receiptIdentity struct {
	schema, grantDigest, profile, operationDigest, context, target, hostKey, operation, boot string
}

type binding struct {
	call     *abi.ToolCall
	trace    string
	seq      uint64
	tool     string
	bodyHash [sha256.Size]byte
	identity receiptIdentity
}

// Bind records trusted source authority after Prepared has validated the bound
// receipt. Callers must pass the actual canonical call used for result admission;
// serialized metadata never grants this authority. Invalid input leaves ctx alone.
func Bind(ctx context.Context, c *abi.ToolCall, grant remotemaint.Grant, receiptBytes []byte) context.Context {
	if ctx == nil || c == nil || c.Tool != maintenanceTool || !bindableGrant(grant) {
		return ctx
	}
	receipt, ok := canonicalReceipt(receiptBytes)
	identity := receiptIdentity{
		schema: remotemaint.ReceiptSchema, grantDigest: remotemaint.GrantDigest(grant),
		profile: grant.ProfileID, operationDigest: grant.OperationDigest, context: grant.ContextID,
		target: grant.Target, hostKey: grant.HostKey, operation: grant.Operation, boot: grant.BootID,
	}
	if !ok || identityOf(receipt) != identity {
		return ctx
	}
	return context.WithValue(ctx, bindingKey{}, binding{
		call: c, trace: c.TraceID, seq: c.SeqNo, tool: c.Tool,
		bodyHash: sha256.Sum256(receiptBytes), identity: identity,
	})
}

// PIIProbe returns a separate canonical receipt with only the source-bound SSH
// separator neutralized; the account and host remain visible to the scanner.
// The caller must retain secret, injection, and non-email PII
// checks on the original body, then scan the probe for every remaining threat.
// It must never use the probe as result content.
func PIIProbe(ctx context.Context, c *abi.ToolCall, receiptBytes []byte) ([]byte, bool) {
	receipt, ok := boundReceipt(ctx, c, receiptBytes)
	if !ok || !sshTarget(receipt.Target) {
		return nil, false
	}
	receipt.Target = strings.Replace(receipt.Target, "@", " ", 1)
	probe, err := json.Marshal(receipt)
	return probe, err == nil
}

// TrustedControlReceipt recognizes only exact source-validated terminal control
// bytes after the result detectors have run. It grants no operation authority,
// never lowers existing trace taint, and excludes all readback observations.
func TrustedControlReceipt(ctx context.Context, c *abi.ToolCall, receiptBytes []byte) bool {
	receipt, ok := boundReceipt(ctx, c, receiptBytes)
	if !ok || !receipt.Verified || receipt.Readback != nil {
		return false
	}
	switch receipt.Operation {
	case remotemaint.OperationMaintainLeaseAcquire, remotemaint.OperationMaintainLeaseRelease:
		return receipt.Outcome == remotemaint.OutcomeApplied
	case remotemaint.OperationSystemdProfileInstallReload:
		return len(receipt.ReservationLeases) == 0 && (receipt.Outcome == remotemaint.OutcomeApplied ||
			receipt.Outcome == remotemaint.OutcomeUnchanged || receipt.Outcome == remotemaint.OutcomeRolledBack)
	default:
		return false
	}
}

func bindableGrant(grant remotemaint.Grant) bool {
	switch grant.Operation {
	case remotemaint.OperationMaintainLeaseAcquire, remotemaint.OperationMaintainLeaseRelease:
		return grant.Reservation != nil && grant.Target == "" && grant.HostKey == "" && grant.BootID == ""
	case remotemaint.OperationSystemdProfileInstallReload, remotemaint.OperationSystemdProfileReadback:
		return grant.Reservation == nil && sshTarget(grant.Target)
	default:
		return false
	}
}

func boundReceipt(ctx context.Context, c *abi.ToolCall, receiptBytes []byte) (remotemaint.Receipt, bool) {
	if ctx == nil || c == nil {
		return remotemaint.Receipt{}, false
	}
	b, ok := ctx.Value(bindingKey{}).(binding)
	if !ok || c != b.call || c.TraceID != b.trace || c.SeqNo != b.seq || c.Tool != b.tool ||
		c.Tool != maintenanceTool || sha256.Sum256(receiptBytes) != b.bodyHash {
		return remotemaint.Receipt{}, false
	}
	receipt, ok := canonicalReceipt(receiptBytes)
	if !ok || identityOf(receipt) != b.identity {
		return remotemaint.Receipt{}, false
	}
	return receipt, true
}

func identityOf(r remotemaint.Receipt) receiptIdentity {
	return receiptIdentity{
		schema: r.Schema, grantDigest: r.GrantDigest, profile: r.ProfileID,
		operationDigest: r.OperationDigest, context: r.ContextID, target: r.Target,
		hostKey: r.HostKey, operation: r.Operation, boot: r.BootID,
	}
}

func canonicalReceipt(raw []byte) (remotemaint.Receipt, bool) {
	var r remotemaint.Receipt
	if len(raw) == 0 || len(raw) > 16<<10 {
		return r, false
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if dec.Decode(&r) != nil || dec.Decode(&struct{}{}) != io.EOF {
		return r, false
	}
	canonical, err := json.Marshal(r)
	// The trusted source marshals Receipt directly. Exact canonical equality also
	// refuses duplicate keys, case-folded aliases, and alternative JSON spellings.
	return r, err == nil && bytes.Equal(raw, canonical)
}

func sshTarget(target string) bool {
	if len(target) > 512 || strings.Count(target, "@") != 1 {
		return false
	}
	user, host, _ := strings.Cut(target, "@")
	if len(user) == 0 || len(user) > 64 || len(host) == 0 {
		return false
	}
	for i, c := range []byte(user) {
		if !(asciiLetter(c) || c == '_' || i > 0 && (c >= '0' && c <= '9' || c == '.' || c == '-')) {
			return false
		}
	}
	if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
		return strings.Contains(host, ":") && net.ParseIP(host[1:len(host)-1]) != nil
	}
	if net.ParseIP(host) != nil {
		return true
	}
	if len(host) > 253 {
		return false
	}
	for _, label := range strings.Split(host, ".") {
		if len(label) == 0 || len(label) > 63 || label[0] == '-' || label[len(label)-1] == '-' {
			return false
		}
		for _, c := range []byte(label) {
			if !(asciiLetter(c) || c >= '0' && c <= '9' || c == '-') {
				return false
			}
		}
	}
	return true
}

func asciiLetter(c byte) bool { return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' }
