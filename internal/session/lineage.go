package session

import (
	"encoding/binary"
	"encoding/hex"
	"strings"
)

// lineage.go — the session continuation lineage expressed as a uint64 EPOCH, the
// shared id space with internal/abi's SpeculationContext{Epoch,ParentEpoch} (#914,
// epic #912). When a budget-exhausted session re-continues, it mints a continuation
// id (continuationID = "win-" + the first 8 bytes of sha256(trace+rev), in hex).
// Those 16 hex chars ARE a uint64, so a continuation id and an abi epoch are two
// encodings of ONE lineage point. This file owns that bijection so the lineage
// bridge (internal/epochbridge) can map a session generation onto an abi epoch
// without re-spelling the id format. It adds no field and changes no behavior — the
// mint is still continuationID; these are read-only projections of it.

// continuationPrefix marks every minted continuation id. It must match the prefix
// continuationID writes (usage.go); TestContinuationEpochRoundTrip binds the two, so
// a drift in the mint format breaks the test rather than silently desyncing epochs.
const continuationPrefix = "win-"

// continuationHexLen is the hex width continuationID emits after the prefix: 16 hex
// chars = the first 8 bytes of the sha256 = exactly one uint64 epoch.
const continuationHexLen = 16

// ContinuationID is the exported form of the fresh-window handoff id a budget-
// exhausted session hands its next generation — the same value the internal mint
// writes to State.ContinuationID. Exported so the lineage bridge can derive the
// epoch a continuation from (trace, rev) would carry.
func ContinuationID(trace string, rev uint64) string {
	return continuationID(trace, rev)
}

// ContinuationEpoch decodes a continuation id to its uint64 epoch — its point in the
// shared lineage id space (abi.SpeculationContext.Epoch). The bool is false for any
// string that is not a well-formed continuation id (notably an ORIGINAL trace that
// never came from a re-continuation), which a caller reads as "generation 0 / epoch
// 0" — never a guessed non-zero epoch.
func ContinuationEpoch(id string) (uint64, bool) {
	if !strings.HasPrefix(id, continuationPrefix) {
		return 0, false
	}
	h := id[len(continuationPrefix):]
	if len(h) != continuationHexLen {
		return 0, false
	}
	b, err := hex.DecodeString(h)
	if err != nil {
		return 0, false
	}
	return binary.BigEndian.Uint64(b), true
}

// ContinuationIDForEpoch is the inverse of ContinuationEpoch over the id's hex tail:
// it rebuilds the continuation id a given epoch encodes. It round-trips every id
// ContinuationID produces — ContinuationIDForEpoch(epoch) == the original id — so the
// id and the epoch are two faces of one lineage value, not two id spaces.
func ContinuationIDForEpoch(epoch uint64) string {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], epoch)
	return continuationPrefix + hex.EncodeToString(b[:])
}
