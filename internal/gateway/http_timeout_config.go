package gateway

import (
	"os"
	"strconv"
	"time"
)

// serveWriteTimeoutDefault picks the WriteTimeout default for the backend the gateway is
// serving, so a non-streaming turn never trips the deadline DURING a long synchronous decode
// (#1015). A LOCAL model — the in-kernel fused model — answers a single turn in seconds to
// MINUTES (a cpu-offload GLM-5.2 decode is multi-minute), and the whole response is written
// only after that turn finishes, so any finite write deadline measured from the request
// headers races the decode: 0 (no timeout) is the only correct default there. A "proxy" to a
// hosted API is fast and is the network-exposed surface a slow-loris cares about, so it keeps
// the conservative 90s. The mock/unknown backends are instant; the conservative default is
// harmless for them. FAK_HTTP_WRITE_TIMEOUT_S overrides this in every case (durEnv).
func serveWriteTimeoutDefault(kind string) time.Duration {
	if kind == "inkernel" {
		return 0 // a local model turn can legitimately run for minutes — no whole-handler deadline
	}
	return 90 * time.Second
}

// durEnv reads an integer-seconds timeout override from the environment, returning
// def when the var is unset or unparseable. A non-negative integer wins: 0 selects
// Go's "no timeout" semantics (an explicit, documented opt-out for a long-running
// local backend); a negative value is rejected and def is kept. This is the seam
// that lets a slow CPU-served model finish a turn without tripping WriteTimeout.
func durEnv(name string, def time.Duration) time.Duration {
	v := os.Getenv(name)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return def
	}
	return time.Duration(n) * time.Second
}
