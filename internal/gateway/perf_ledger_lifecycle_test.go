package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// fak-test:runtime fast est=20ms lane=default
func TestPerfLedgerLifecycleIdentityOnSuccessAndTimeout(t *testing.T) {
	srv := newTestServer(t)
	srv.SetModelLoadProfile(&ModelLoadProfile{TotalSeconds: 1.5})
	request := func() perfSource {
		r := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"messages":[],"max_tokens":8}`))
		src := perfSource{observation: newRequestObservation(r)}
		r = r.WithContext(withPerfSource(context.Background(), src))
		if _, err := io.Copy(io.Discard, r.Body); err != nil {
			t.Fatal(err)
		}
		srv.admitRequestObservation(r, "shared-session")
		return src
	}
	first, later := request(), request()
	// A replacement after admission must not relabel already-running requests.
	srv.SetModelLoadProfile(&ModelLoadProfile{TotalSeconds: 2})
	srv.metrics.recordPerfFailure(first, localitySelfHosted, perfledger.ErrorClientWriteTimeout, 0, time.Second, 0)
	srv.metrics.recordPerf(localitySelfHosted, 128, 2, 0, "stop", time.Second, time.Millisecond, perfDetail{src: later})
	rows, _, _ := srv.metrics.perfRecordsSnapshot()
	if len(rows) != 2 {
		t.Fatal("success and failed request must both produce a row")
	}
	a, b := rows[0], rows[1]
	if a.RequestID == "" || a.RequestID == b.RequestID || a.TraceID != "shared-session" || b.TraceID != a.TraceID {
		t.Fatal("request identity must be unique independently of session trace")
	}
	if a.LoadLifecycle == nil || b.LoadLifecycle == nil || a.LoadLifecycle.LoadID != b.LoadLifecycle.LoadID || a.LoadLifecycle.LoadID == srv.modelLoadProfile().LoadID {
		t.Fatal("admission load identity lost or replaced")
	}
	if !a.LoadLifecycle.InitialAfterLoad || a.LoadLifecycle.RequestOrdinal != 1 || b.LoadLifecycle.InitialAfterLoad || b.LoadLifecycle.RequestOrdinal != 2 || a.LoadLifecycle.LoadMS != 1500 {
		t.Fatal("first failed admission and later success were conflated")
	}
	if a.WorkloadSHA256 == "" || a.WorkloadSHA256 != b.WorkloadSHA256 || a.WorkloadIdentityMode != "raw_request_body_sha256_v1" || a.ConfigSHA256 == "" || a.ConfigIdentityMode != "scalar_v1" {
		t.Fatal("consumed request identity was not retained")
	}
	if a.Error != perfledger.ErrorClientWriteTimeout || a.TTFTMS != 0 || a.NativeTiming != nil {
		t.Fatal("failure must retain identity without inventing missing phases")
	}
	raw, err := json.Marshal(a)
	if err != nil {
		t.Fatal(err)
	}
	var roundtrip perfledger.Record
	if err = json.Unmarshal(raw, &roundtrip); err != nil {
		t.Fatal(err)
	}
	if roundtrip.RequestID != a.RequestID || roundtrip.LoadLifecycle == nil || *roundtrip.LoadLifecycle != *a.LoadLifecycle {
		t.Fatal("durable row lost lifecycle identity")
	}
	var legacy perfledger.Record
	if err = json.Unmarshal([]byte(`{"schema":"fak.gateway.perf-record.v1","unix_ms":1}`), &legacy); err != nil {
		t.Fatal(err)
	}
	if legacy.RequestID != "" || legacy.LoadLifecycle != nil || legacy.WorkloadSHA256 != "" {
		t.Fatal("legacy evidence must remain unknown")
	}
}
