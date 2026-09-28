package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// leaseCoordinatorClient is selected only by explicit environment configuration.
// Once selected, transport and authority failures never fall back to local refs.
type leaseCoordinatorClient struct {
	origin string
	key    string
	http   *http.Client
}

func leaseCoordinatorRequested() bool {
	return strings.TrimSpace(os.Getenv("FAK_LEASE_COORDINATOR_URL")) != "" ||
		strings.TrimSpace(os.Getenv("FAK_LEASE_COORDINATOR_KEY_FILE")) != ""
}

func configuredLeaseCoordinator() (*leaseCoordinatorClient, bool, error) {
	origin := strings.TrimSpace(os.Getenv("FAK_LEASE_COORDINATOR_URL"))
	keyFile := strings.TrimSpace(os.Getenv("FAK_LEASE_COORDINATOR_KEY_FILE"))
	if origin == "" && keyFile == "" {
		return nil, false, nil
	}
	if origin == "" || keyFile == "" {
		return nil, true, errors.New("FAK_LEASE_COORDINATOR_URL and FAK_LEASE_COORDINATOR_KEY_FILE must both be set")
	}
	u, err := url.Parse(origin)
	if err != nil || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || (u.Scheme != "https" && u.Scheme != "http") || (u.Path != "" && u.Path != "/" && u.Path != "/v1") {
		return nil, true, errors.New("invalid FAK_LEASE_COORDINATOR_URL: expected an HTTP(S) gateway origin")
	}
	if u.Scheme == "http" && !leaseCoordinatorLoopback(u.Hostname()) {
		return nil, true, errors.New("FAK_LEASE_COORDINATOR_URL requires HTTPS outside loopback")
	}
	keyBytes, err := os.ReadFile(pathutil.ExpandTilde(keyFile))
	if err != nil {
		return nil, true, fmt.Errorf("read FAK_LEASE_COORDINATOR_KEY_FILE: %w", err)
	}
	key := strings.TrimSpace(string(keyBytes))
	if key == "" || strings.ContainsAny(key, "\r\n") {
		return nil, true, errors.New("FAK_LEASE_COORDINATOR_KEY_FILE contains no single-line key")
	}
	u.Path = ""
	client := &leaseCoordinatorClient{origin: strings.TrimRight(u.String(), "/"), key: key, http: &http.Client{Timeout: 20 * time.Second}}
	if !probeRouterAuthProof(client.http, client.origin, client.key) {
		return nil, true, errors.New("coordinator authentication proof unavailable or invalid")
	}
	return client, true, nil
}

func leaseCoordinatorLoopback(host string) bool {
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func (c *leaseCoordinatorClient) write(ctx context.Context, op string, req gateway.LeaseWriteRequest) (gateway.LeaseWriteResult, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return gateway.LeaseWriteResult{}, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.origin+"/v1/leases/"+op, bytes.NewReader(body))
	if err != nil {
		return gateway.LeaseWriteResult{}, err
	}
	httpReq.Header.Set("Authorization", "Bearer "+c.key)
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(httpReq)
	if err != nil {
		return gateway.LeaseWriteResult{}, fmt.Errorf("coordinator lease %s transport: %w", op, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return gateway.LeaseWriteResult{}, fmt.Errorf("coordinator lease %s returned HTTP %d", op, resp.StatusCode)
	}
	const maxVerdictBytes = 64 << 10
	verdictBytes, err := io.ReadAll(io.LimitReader(resp.Body, maxVerdictBytes+1))
	if err != nil {
		return gateway.LeaseWriteResult{}, fmt.Errorf("read coordinator lease %s: %w", op, err)
	}
	if len(verdictBytes) > maxVerdictBytes {
		return gateway.LeaseWriteResult{}, fmt.Errorf("coordinator lease %s verdict exceeds %d bytes", op, maxVerdictBytes)
	}
	var result gateway.LeaseWriteResult
	if err := json.Unmarshal(verdictBytes, &result); err != nil {
		return gateway.LeaseWriteResult{}, fmt.Errorf("decode coordinator lease %s: %w", op, err)
	}
	if result.Op != op || result.ID != req.ID || result.Source == "" || result.ObservedUnix <= 0 {
		return gateway.LeaseWriteResult{}, fmt.Errorf("coordinator lease %s returned an invalid verdict", op)
	}
	if result.OK {
		if result.Reason != "" || result.Holder != req.Holder ||
			(op != "release" && (result.Generation <= 0 || result.CurrentGeneration != result.Generation)) ||
			(op == "acquire" && !slices.Equal(result.TreeGlobs, req.TreeGlobs)) ||
			(op == "renew" && result.Generation != req.Generation) ||
			(op == "release" && (result.Generation != 0 || len(result.TreeGlobs) != 0 ||
				(result.CurrentGeneration != 0 && result.CurrentGeneration != req.Generation))) {
			return gateway.LeaseWriteResult{}, fmt.Errorf("coordinator lease %s verdict does not match the requested authority", op)
		}
	} else {
		switch result.Reason {
		case leaseref.ReasonLeaseHeld, leaseref.ReasonStaleLease, leaseref.ReasonLeaseContended, leaseref.ReasonNoLease:
		default:
			return gateway.LeaseWriteResult{}, fmt.Errorf("coordinator lease %s returned an unknown refusal", op)
		}
		if result.Generation != req.Generation || len(result.TreeGlobs) != 0 {
			return gateway.LeaseWriteResult{}, fmt.Errorf("coordinator lease %s refusal does not match the requested authority", op)
		}
	}
	return result, nil
}

func coordinatorFenceVerdict(result gateway.LeaseWriteResult, presented int64) leaseref.FenceVerdict {
	v := leaseref.FenceVerdict{
		OK: result.OK, Reason: result.Reason, Presented: presented,
		Current: result.CurrentGeneration, Holder: result.Holder, Detail: result.Detail,
	}
	if result.OK && result.Op != "release" {
		v.Presented = result.Generation
		v.Current = result.Generation
	}
	return v
}
