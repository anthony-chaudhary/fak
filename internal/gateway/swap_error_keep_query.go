package gateway

import (
	"os"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/negframe"
)

const swapErrorKeepQueryEnv = "FAK_SWAP_ERROR_KEEP_QUERY"

var swapErrorKeepQueryTotal atomic.Uint64

// ErrorQueryTurn is the explicit boundary between a failed result and the
// originating-task pin it must not replace.
type ErrorQueryTurn struct {
	Content     string `json:"content"`
	Correction  string `json:"correction,omitempty"`
	RestoreID   string `json:"restore_id"`
	Failed      bool   `json:"failed"`
	Replacement bool   `json:"replacement"`
}

// SwapErrorKeepQuery replaces a failed error turn with an affordance-first
// instruction while carrying the originating-task restore handle byte-for-byte.
// The environment lever is intentionally default-off during observation.
func SwapErrorKeepQuery(in ErrorQueryTurn) ErrorQueryTurn {
	if !in.Failed || !swapErrorKeepQueryEnabled() {
		return in
	}
	candidate := strings.TrimSpace(in.Correction)
	if candidate == "" {
		candidate = strings.TrimSpace(in.Content)
	}
	if candidate == "" {
		return in
	}
	out := in
	out.Content = negframe.Reframe(candidate)
	out.Correction = ""
	out.Failed = false
	out.Replacement = true
	swapErrorKeepQueryTotal.Add(1)
	return out
}

func swapErrorKeepQueryEnabled() bool {
	on, err := strconv.ParseBool(strings.TrimSpace(os.Getenv(swapErrorKeepQueryEnv)))
	return err == nil && on
}

func SwapErrorKeepQueryTotal() uint64 { return swapErrorKeepQueryTotal.Load() }

func SwapErrorKeepQueryPrometheus() string {
	return "# HELP fak_swap_error_keep_query_total Failed result turns replaced while preserving the query pin.\n" +
		"# TYPE fak_swap_error_keep_query_total counter\n" +
		"fak_swap_error_keep_query_total " + strconv.FormatUint(SwapErrorKeepQueryTotal(), 10) + "\n"
}
