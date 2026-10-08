package gateway

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// HeaderCtxPaging is both the per-request operator override (on|off|auto) and the
// response header reporting the oversize-paging decision as
// "<auto-on|on|off>;reason=<reason>".
const HeaderCtxPaging = "X-Fak-Ctx-Paging"

// envCtxPaging is the process default (auto|on|off); the request header wins over it.
const envCtxPaging = "FAK_CTX_PAGING"

type ctxPagingDecision struct {
	// suppressReason is the ctxmmu.PagingSuppressed* reason, "" when paging is enabled.
	suppressReason string
	header         string
}

// resolveChatCtxPaging decides whether oversize-but-benign tool results may be paged
// out to {"_paged":true} restore stubs for this request. In auto mode it pages only
// when the client can restore a stub: it declares a restore tool under any dialect
// spelling, or carries fak MCP tools the gateway auto-advertises the restore tool to.
// A harness without one (OpenCode Code Mode: read/shell/glob/execute) would see an
// unrecoverable hole and re-request the same result in a loop.
func resolveChatCtxPaging(r *http.Request, tools []agent.ToolDef) ctxPagingDecision {
	mode := ""
	if r != nil {
		mode = normalizeCtxPagingMode(r.Header.Get(HeaderCtxPaging))
	}
	if mode == "" {
		mode = normalizeCtxPagingMode(os.Getenv(envCtxPaging))
	}
	switch mode {
	case "off":
		return ctxPagingDecision{suppressReason: ctxmmu.PagingSuppressedOperatorOff, header: "off;reason=" + ctxmmu.PagingSuppressedOperatorOff}
	case "on":
		return ctxPagingDecision{header: "on;reason=operator_on"}
	}
	if _, present, autoAdvertise := determineChatRestoreTool(tools); present || autoAdvertise {
		return ctxPagingDecision{header: "auto-on;reason=harness_can_restore"}
	}
	return ctxPagingDecision{
		suppressReason: ctxmmu.PagingSuppressedHarnessCannotRestore,
		header:         "off;reason=" + ctxmmu.PagingSuppressedHarnessCannotRestore,
	}
}

func normalizeCtxPagingMode(v string) string {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "on", "1", "true":
		return "on"
	case "off", "0", "false":
		return "off"
	case "auto":
		return "auto"
	}
	return ""
}

// writeCtxPagingMetrics renders the oversize page-outs ctxmmu skipped, by reason, across
// both admission passes. Every known reason renders (0 when unseen) so the row never
// vanishes from a scrape.
func writeCtxPagingMetrics(b *strings.Builder) {
	writeHelpType(b, "fak_gateway_ctx_paging_suppressed_total", "WITNESSED (fak authored): oversize-but-benign tool results admitted whole instead of paged out to a restore stub, by reason (harness_cannot_restore: the client declared no restore tool; trailing_result: the result answers the current turn; operator_off: FAK_CTX_PAGING or X-Fak-Ctx-Paging disabled paging).", "counter")
	for _, row := range ctxmmu.SuppressedPagingCounts() {
		fmt.Fprintf(b, "fak_gateway_ctx_paging_suppressed_total{reason=\"%s\"} %d\n", promQuote(row.Reason), row.Count)
	}
}
