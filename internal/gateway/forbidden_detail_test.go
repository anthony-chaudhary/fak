package gateway

import (
	"strings"
	"testing"
	"time"
)

// scrubForbiddenDetail must strip token-shaped runs while keeping the KIND of denial legible, and
// bound the result — the operator-side /debug/vars drilldown must never persist a credential an
// upstream echoes into an error body (the trust-boundary reason the raw 403 body is withheld from
// the client in the first place).
func TestScrubForbiddenDetail_RedactsSecretsKeepsReason(t *testing.T) {
	body := `{"error":{"type":"permission_error","message":"key sk-ant-oat01-abcdefgh12345678 not entitled"}}`
	got := scrubForbiddenDetail(body)
	if got == "" {
		t.Fatal("a non-empty body must scrub to a non-empty detail")
	}
	if strings.Contains(got, "sk-ant-oat01-abcdefgh12345678") {
		t.Fatalf("scrubbed detail must not contain the raw token: %q", got)
	}
	if !strings.Contains(got, "permission_error") {
		t.Fatalf("scrubbed detail should keep the denial KIND: %q", got)
	}
}

// A Bearer token and a bare long high-entropy run must both be redacted, and a blank/whitespace
// body must scrub to empty (nothing to show).
func TestScrubForbiddenDetail_RedactsBearerAndEntropy(t *testing.T) {
	if got := scrubForbiddenDetail("Authorization: Bearer abcdEFGH1234ijklMNOP5678"); strings.Contains(got, "abcdEFGH1234ijklMNOP5678") {
		t.Fatalf("a Bearer token must be redacted: %q", got)
	}
	if got := scrubForbiddenDetail("   "); got != "" {
		t.Fatalf("a whitespace-only body must scrub to empty, got %q", got)
	}
}

// The stored detail is bounded so a hostile/huge body cannot bloat the operator surface.
func TestScrubForbiddenDetail_Bounded(t *testing.T) {
	long := strings.Repeat("denied ", 200) // ~1400 chars of harmless words
	got := scrubForbiddenDetail(long)
	if len(got) > forbiddenDetailMax+len("…") {
		t.Fatalf("scrubbed detail must be bounded to ~%d, got %d", forbiddenDetailMax, len(got))
	}
}

// recordForbiddenDetail must land on the operator-only /debug/vars drilldown, scrubbed, and a blank
// 403 body must NOT erase a useful earlier detail.
func TestRecordForbiddenDetail_SurfacesScrubbedAndKeepsPrevious(t *testing.T) {
	m := newGatewayMetrics(time.Unix(0, 0))
	m.recordForbiddenDetail(`{"error":{"type":"permission_error"}}`)
	if got := m.debugUpstreamVars().LastForbiddenDetail; !strings.Contains(got, "permission_error") {
		t.Fatalf("recorded detail should surface on /debug/vars, got %q", got)
	}
	// A blank body must not clobber the previous useful detail.
	m.recordForbiddenDetail("")
	if got := m.debugUpstreamVars().LastForbiddenDetail; !strings.Contains(got, "permission_error") {
		t.Fatalf("a blank 403 body must not erase the previous detail, got %q", got)
	}
}
