package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountobs"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/goalpark"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

func guardUpstreamObserver(quotaHarvester *accountobs.Harvester, parkStore goalpark.Store, parkGoal string, parkTemplate goalpark.Record, log io.Writer) func(int, http.Header) {
	longRetryParked := false
	return func(status int, header http.Header) {
		// Passive, zero-request harvest; persistence failures never fail the response path.
		_ = quotaHarvester.Observe(status, header)
		if parkGoal == "" || longRetryParked {
			return
		}
		parked, err := parkStore.RecordLongRetry(status, header, time.Now(), parkTemplate)
		if err != nil {
			fmt.Fprintf(log, "fak guard: long Retry-After park failed open: %v\n", err)
			return
		}
		if !parked {
			return
		}
		longRetryParked = true
		if rec, err := parkStore.Load(parkGoal); err == nil {
			fmt.Fprintf(log, "fak guard: PARKED goal=%q parked_until=%d reason=%s account=%q pool=%q next=%q\n", rec.Goal, rec.ParkedUntil, rec.Reason, rec.Account, rec.Pool, rec.NextAction)
		}
	}
}

func guardUpstreamFailureObserver(
	audit *journal.Journal,
	traceID string,
	wireGauge *guardWireErrorGauge,
	logf func(string, ...any),
	debugStats func(string, ...any),
) func(gateway.UpstreamFailureReceipt) {
	return func(r gateway.UpstreamFailureReceipt) {
		if logf != nil {
			logf("gateway: upstream failure: status=%d layer=%s target=%s cause=%q prov_req=%q proxy_req=%q attempt=%d/%d",
				r.HTTPStatus, r.EmittingLayer, r.TargetID, r.Cause, r.ProviderRequestID, r.ProxyRequestID, r.Attempt, r.RetryBudget)
		}
		if debugStats != nil && (r.HTTPStatus == http.StatusBadGateway || r.HTTPStatus >= 500 || r.EmittingLayer == "transport") {
			debugStats("fak-turn trace=%s UPSTREAM_FAILURE layer=%s status=%d target=%s cause=%q",
				traceID, r.EmittingLayer, r.HTTPStatus, r.TargetID, r.Cause)
		}
		if wireGauge != nil && (r.HTTPStatus == http.StatusBadGateway || r.HTTPStatus == http.StatusServiceUnavailable || r.HTTPStatus == http.StatusGatewayTimeout || r.EmittingLayer == "transport") {
			wireGauge.Observe(time.Now(), fmt.Errorf("upstream %s status %d: %s", r.EmittingLayer, r.HTTPStatus, r.Cause))
		}
		if audit != nil {
			eventType := "UPSTREAM_FAILURE"
			if r.HTTPStatus == http.StatusBadGateway {
				eventType = "UPSTREAM_502"
			}
			raw, err := json.Marshal(r)
			if err == nil {
				audit.AppendAgentEvent(eventType, traceID, string(raw))
			} else {
				audit.AppendAgentEvent(eventType, traceID, fmt.Sprintf("status=%d layer=%s target=%s cause=%s", r.HTTPStatus, r.EmittingLayer, r.TargetID, r.Cause))
			}
		}
	}
}
