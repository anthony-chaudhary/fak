package model

import (
	"os"
	"strconv"
	"strings"
)

// parShardCount resolves how many node-shards the parallel chunk cursor splits into when
// the box spans a multi-node topology. FAK_PAR_SHARDS pins it — the deliberate knob a
// multi-node launcher sets — and the resolved count obeys the two invariants the sharded
// cursor depends on:
//
//   - it never exceeds `participants` (the number of cursors that can actually drain a
//     shard): a shard with no participant would never drain; and
//   - it clamps to 1 — the original single-cursor behaviour — whenever the host reports
//     no multi-node topology, i.e. FAK_PAR_SHARDS is unset, empty, or not a positive
//     integer.
//
// Pure over its input and the environment: same participants + same env, same count.
func parShardCount(participants int) int {
	if participants < 1 {
		return 1
	}
	shards := 1
	if v := strings.TrimSpace(os.Getenv("FAK_PAR_SHARDS")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			shards = n
		}
	}
	if shards > participants {
		shards = participants
	}
	if shards < 1 {
		shards = 1
	}
	return shards
}
