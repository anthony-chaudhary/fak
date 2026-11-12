package agent

import (
	"log"
	"os"
	"strings"
	"time"
)

// RadixAttribution is the env-gated debug arm for the radix-on cold-prefill tax
// (ticket prefill-radix-cold-prefill-tax-20261108). FAK_INKERNEL_RADIX_DEBUG=1 arms
// per-seam phase timing on the planner; FAK_INKERNEL_RADIX_DEBUG_LOG=1 adds one line
// per observed event. Pure observation: no behavior change, and every hot-path read
// is a single mutex-guarded bool check when disarmed.
//
// The phases attribute the three candidate cost classes the ticket names:
//   - lookups_ns:  the step-1 persistent-prefix walk under the planner lock;
//   - capture_ns:  the O(prefix-bytes) KV/snapshot clone that makes the tree own an
//     independent copy — the insert-time work on the prefill critical path;
//   - inserts_ns:  the locked tree section (boundary walk + attach + budget/eviction
//     bookkeeping); evictions_seen reports how much of it was budget-pressure eviction.
type RadixAttribution struct {
	Turns      int64 // demand turns whose step-1 lookup was observed
	Admissions int64 // admission events (prompt / continuation / checkpoint / scoped)

	LookupsNs      int64
	CaptureNs      int64
	InsertsNs      int64
	BytesCloned    int64 // Σ payload bytes cloned into the tree
	LogitsBytes    int64 // Σ cached-logit copies
	OffLockClones  int64 // admissions whose payload clone ran outside the planner lock
	SyncAdmissions int64 // admissions that kept the historical fully-locked section
	EvictionsSeen  int64 // tree LRU evictions observed across admissions

	// Last-event breakdown (zero when disarmed or before the first event):
	LastPhase         string
	LastPromptTokens  int
	LastMatchedTokens int
	LastLookupNs      int64
	LastCaptureNs     int64
	LastInsertNs      int64
	LastOffLock       bool
}

// EnableRadixAttribution arms the debug arm on a bare-test planner (constructors arm
// it from the environment). logLines additionally emits the per-event line.
func (p *InKernelPlanner) EnableRadixAttribution(logLines bool) {
	p.radixAttrMu.Lock()
	p.radixDebug = true
	p.radixDebugLog = logLines
	p.radixAttrMu.Unlock()
}

func radixFlagArmed(key, def string) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(key)))
	if v == "" {
		return def == "on"
	}
	switch v {
	case "1", "on", "true", "yes":
		return true
	default:
		return false
	}
}

// radixAttrBegin reports whether the arm is live and stamps now when it is.
func (p *InKernelPlanner) radixAttrBegin() (time.Time, bool) {
	p.radixAttrMu.Lock()
	armed := p.radixDebug
	p.radixAttrMu.Unlock()
	if !armed {
		return time.Time{}, false
	}
	return time.Now(), true
}

// radixAttrRecordTurn folds one step-1 lookup observation into the ledger.
func (p *InKernelPlanner) radixAttrRecordTurn(tier string, lookupNs int64, promptTok, matched int) {
	p.radixAttrMu.Lock()
	defer p.radixAttrMu.Unlock()
	a := &p.radixAttr
	a.Turns++
	a.LookupsNs += lookupNs
	a.LastPhase = "lookup"
	a.LastPromptTokens = promptTok
	a.LastMatchedTokens = matched
	a.LastLookupNs = lookupNs
	a.LastCaptureNs = 0
	a.LastInsertNs = 0
	a.LastOffLock = false
	if p.radixDebugLog {
		log.Printf("inkernel_radix_attr phase=lookup tier=%s prompt=%dtok matched=%dtok lookup_ns=%d",
			tier, promptTok, matched, lookupNs)
	}
}

// radixAttrRecord folds one admission event's phase timings into the ledger. It is
// called only when the arm is enabled, so its mutex never sits on the default serve
// path. evictionsDelta must be measured while holding p.mu on both sides.
func (p *InKernelPlanner) radixAttrRecord(phase string, promptTok, matched int, tier string, lookupNs, captureNs, insertNs int64, bytesCloned, logitsBytes int64, offLock bool, evictionsDelta int) {
	p.radixAttrMu.Lock()
	defer p.radixAttrMu.Unlock()
	a := &p.radixAttr
	a.Admissions++
	a.LookupsNs += lookupNs
	a.CaptureNs += captureNs
	a.InsertsNs += insertNs
	a.BytesCloned += bytesCloned
	a.LogitsBytes += logitsBytes
	if offLock {
		a.OffLockClones++
	} else {
		a.SyncAdmissions++
	}
	a.EvictionsSeen += int64(evictionsDelta)
	a.LastPhase = phase
	a.LastPromptTokens = promptTok
	a.LastMatchedTokens = matched
	a.LastLookupNs = lookupNs
	a.LastCaptureNs = captureNs
	a.LastInsertNs = insertNs
	a.LastOffLock = offLock
	if p.radixDebugLog {
		log.Printf("inkernel_radix_attr phase=%s tier=%s prompt=%dtok matched=%dtok lookup_ns=%d capture_ns=%d insert_ns=%d bytes_cloned=%d logits_bytes=%d off_lock=%v evictions_delta=%d",
			phase, tier, promptTok, matched, lookupNs, captureNs, insertNs, bytesCloned, logitsBytes, offLock, evictionsDelta)
	}
}

// RadixAttribution returns a snapshot of the debug-arm ledger. Disarmed planners
// report the zero value.
func (p *InKernelPlanner) RadixAttribution() RadixAttribution {
	p.radixAttrMu.Lock()
	defer p.radixAttrMu.Unlock()
	return p.radixAttr
}

// nsSince converts a stamped start into nanoseconds, reporting zero when the stamp
// was not taken (arm disarmed at that seam).
func nsSince(start time.Time, armed bool) int64 {
	if !armed {
		return 0
	}
	return time.Since(start).Nanoseconds()
}

// radixLogitsBytes prices a cached-logit copy the way the tree does (4 bytes per
// float).
func radixLogitsBytes(logits []float32) int64 {
	return int64(len(logits)) * 4
}

// radixEvictions reads the tree's LRU-eviction counter, taking p.mu itself. It is a
// debug-arm probe: the default serve path never calls it.
func (p *InKernelPlanner) radixEvictions() int {
	if p.tree == nil {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.tree.Evictions()
}
