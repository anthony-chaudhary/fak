package gateway

import (
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGuardAuditFootprintDebugEndpoint(t *testing.T) {
	SetGuardAuditFootprintProvider(func() GuardAuditFootprint { return GuardAuditFootprint{Files: 7, Bytes: 99, OldestUnix: 123} })
	t.Cleanup(func() { SetGuardAuditFootprintProvider(nil) })
	rr := httptest.NewRecorder()
	handleGuardAuditDebug(rr, httptest.NewRequest("GET", "/debug/guard-audit", nil))
	if rr.Code != 200 {
		t.Fatalf("status = %d body=%s", rr.Code, rr.Body.String())
	}
	for _, want := range []string{`"files":7`, `"bytes":99`, `"oldest_unix":123`} {
		if !strings.Contains(rr.Body.String(), want) {
			t.Fatalf("body %s missing %s", rr.Body.String(), want)
		}
	}
}
