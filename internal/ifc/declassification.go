package ifc

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func declassPreimage(prevHash, trace string, turn int, from, to abi.TaintLabel, rationale, witness string, ts int64) string {
	return fmt.Sprintf("%d:%s|%d:%s|%d|%d|%d|%d:%s|%d:%s|%d",
		len(prevHash), prevHash,
		len(trace), trace,
		turn,
		from,
		to,
		len(rationale), rationale,
		len(witness), witness,
		ts)
}

// Declassify lowers the taint level for trace (or turn) to Trusted, recording an
// auditable, hash-chained receipt in the ledger.
func (l *Ledger) Declassify(trace string, rationale string, witness any) (*DeclassificationReceipt, error) {
	trimmed := strings.TrimSpace(rationale)
	if trimmed == "" {
		return nil, errors.New("ifc: declassification requires non-empty rationale")
	}

	witnessStr := ""
	if witness != nil {
		witnessStr = fmt.Sprint(witness)
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	l.ensureLocked()

	cur := l.levelLocked(trace)
	now := time.Now().UnixNano()

	base, turn, isTurn := ParseTurnTrace(trace)
	if isTurn {
		if l.turnTaint[base] == nil {
			l.turnTaint[base] = map[int]abi.TaintLabel{}
		}
		l.turnTaint[base][turn] = abi.TaintTrusted
		l.mark[trace] = abi.TaintTrusted
	} else {
		turn = 0
		l.mark[trace] = abi.TaintTrusted
		if l.turnTaint[trace] != nil {
			for tNum := range l.turnTaint[trace] {
				l.turnTaint[trace][tNum] = abi.TaintTrusted
			}
		}
		for m := range l.mark {
			if b, _, ok := ParseTurnTrace(m); ok && b == trace {
				l.mark[m] = abi.TaintTrusted
			}
		}
	}
	l.touchLocked(trace)

	preimage := declassPreimage(l.lastReceiptHash, trace, turn, cur, abi.TaintTrusted, trimmed, witnessStr, now)
	sum := sha256.Sum256([]byte(preimage))
	receiptHash := hex.EncodeToString(sum[:])

	rcpt := DeclassificationReceipt{
		Trace:       trace,
		Turn:        turn,
		FromLevel:   cur,
		ToLevel:     abi.TaintTrusted,
		Rationale:   trimmed,
		Witness:     witnessStr,
		Timestamp:   now,
		PrevHash:    l.lastReceiptHash,
		ReceiptHash: receiptHash,
	}

	l.lastReceiptHash = receiptHash
	l.declass = append(l.declass, rcpt)
	return &rcpt, nil
}

// Declassifications returns a copy of all recorded declassification receipts.
func (l *Ledger) Declassifications() []DeclassificationReceipt {
	l.mu.RLock()
	defer l.mu.RUnlock()
	if len(l.declass) == 0 {
		return nil
	}
	out := make([]DeclassificationReceipt, len(l.declass))
	copy(out, l.declass)
	return out
}

// VerifyDeclassifications verifies the cryptographic integrity and hash chaining
// of all declassification receipts recorded in the ledger.
func (l *Ledger) VerifyDeclassifications() error {
	l.mu.RLock()
	defer l.mu.RUnlock()
	prevHash := ""
	for i, r := range l.declass {
		if r.PrevHash != prevHash {
			return fmt.Errorf("ifc: declassification chain broken at receipt %d: prev_hash %q != expected %q", i, r.PrevHash, prevHash)
		}
		expectedPreimage := declassPreimage(r.PrevHash, r.Trace, r.Turn, r.FromLevel, r.ToLevel, r.Rationale, r.Witness, r.Timestamp)
		sum := sha256.Sum256([]byte(expectedPreimage))
		expectedHash := hex.EncodeToString(sum[:])
		if r.ReceiptHash != expectedHash {
			return fmt.Errorf("ifc: declassification hash mismatch at receipt %d: got %s, want %s", i, r.ReceiptHash, expectedHash)
		}
		prevHash = r.ReceiptHash
	}
	return nil
}
