package hil

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"
)

const (
	// DefaultDialTimeout is the per-port TCP probe deadline.
	DefaultDialTimeout = 50 * time.Millisecond
	// DefaultHealthzTimeout is the /healthz HTTP probe deadline.
	DefaultHealthzTimeout = 50 * time.Millisecond
	// DefaultGatewayPort is the default LAN agent/serving gateway port.
	DefaultGatewayPort = "8080"
	// DefaultSSHPort is the default appliance SSH management port.
	DefaultSSHPort = "22"
)

var ipv4Pattern = regexp.MustCompile(`\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b`)

// ScrubLANTelemetry replaces raw IP patterns with <LAN_IP> to prevent private network disclosure.
func ScrubLANTelemetry(text string) string {
	if text == "" {
		return ""
	}
	text = ipv4Pattern.ReplaceAllString(text, "<LAN_IP>")
	if strings.Contains(text, "::1") {
		text = strings.ReplaceAll(text, "::1", "<LAN_IP>")
	}
	return text
}

// ProbeLANNode performs fast, non-blocking reachability and capability probing of a LAN accelerator node.
func ProbeLANNode(ctx context.Context, hostOverride string) LANNodeInfo {
	start := time.Now()

	rawHost := strings.TrimSpace(hostOverride)
	if rawHost == "" {
		rawHost = strings.TrimSpace(os.Getenv("FAK_STRIX_HOST"))
	}
	if rawHost == "" {
		rawHost = strings.TrimSpace(os.Getenv("FAK_LAN_HOST"))
	}
	if rawHost == "" || strings.EqualFold(rawHost, "unconfigured") || strings.EqualFold(rawHost, "none") {
		return LANNodeInfo{
			Status:        LANNodeUnconfigured,
			Host:          "",
			Reachable:     false,
			LatencyMicros: 0,
			Transport:     "offline",
			Error:         "lan host unconfigured",
		}
	}

	cleanHost := rawHost
	cleanHost = strings.TrimPrefix(cleanHost, "http://")
	cleanHost = strings.TrimPrefix(cleanHost, "https://")
	cleanHost = strings.TrimRight(cleanHost, "/")

	gatewayPort := DefaultGatewayPort
	if envPort := os.Getenv("FAK_LAN_GATEWAY_PORT"); envPort != "" {
		gatewayPort = envPort
	}

	sshPort := DefaultSSHPort
	if envPort := os.Getenv("FAK_LAN_SSH_PORT"); envPort != "" {
		sshPort = envPort
	}

	dialHost := cleanHost
	h, p, err := net.SplitHostPort(cleanHost)
	if err == nil {
		dialHost = h
		if p == sshPort {
			// SSH port explicitly targeted in host string
		} else {
			gatewayPort = p
		}
	}

	gatewayAddr := net.JoinHostPort(dialHost, gatewayPort)
	sshAddr := net.JoinHostPort(dialHost, sshPort)
	healthzURL := fmt.Sprintf("http://%s/healthz", gatewayAddr)

	dialTimeout := DefaultDialTimeout
	dialer := &net.Dialer{Timeout: dialTimeout}

	type dialRes struct {
		conn net.Conn
		err  error
	}

	chGateway := make(chan dialRes, 1)
	chSSH := make(chan dialRes, 1)

	dialCtx, dialCancel := context.WithTimeout(ctx, dialTimeout)
	defer dialCancel()

	go func() {
		c, dErr := dialer.DialContext(dialCtx, "tcp", gatewayAddr)
		chGateway <- dialRes{conn: c, err: dErr}
	}()

	go func() {
		c, dErr := dialer.DialContext(dialCtx, "tcp", sshAddr)
		chSSH <- dialRes{conn: c, err: dErr}
	}()

	var resGateway, resSSH dialRes
	gwRecv, sshRecv := false, false
	gwDone, sshDone := false, false

	defer func() {
		if !sshRecv {
			go func() {
				res := <-chSSH
				if res.conn != nil {
					res.conn.Close()
				}
			}()
		}
		if !gwRecv {
			go func() {
				res := <-chGateway
				if res.conn != nil {
					res.conn.Close()
				}
			}()
		}
	}()

	scrubbedHost := ScrubLANTelemetry(dialHost)

	for !gwDone || !sshDone {
		select {
		case res := <-chGateway:
			resGateway = res
			gwRecv = true
			gwDone = true
			if res.err == nil {
				if res.conn != nil {
					res.conn.Close()
				}
				if checkHealthz(ctx, healthzURL, DefaultHealthzTimeout) {
					if resSSH.conn != nil {
						resSSH.conn.Close()
					}
					return LANNodeInfo{
						Status:        LANNodeOnline,
						Host:          scrubbedHost,
						Reachable:     true,
						LatencyMicros: time.Since(start).Microseconds(),
						Transport:     "http_healthz",
						Endpoint:      fmt.Sprintf("http://%s:%s", scrubbedHost, gatewayPort),
						Appliance:     "AMD Ryzen AI MAX+ 395",
						GPU:           "AMD Radeon 8060S (40 CUs, gfx1151, 64 GB UMA)",
					}
				}
			}
			if sshDone {
				goto evaluate
			}
		case res := <-chSSH:
			resSSH = res
			sshRecv = true
			sshDone = true
			if gwDone {
				goto evaluate
			}
		case <-dialCtx.Done():
			if !gwDone {
				resGateway = dialRes{err: dialCtx.Err()}
				gwDone = true
			}
			if !sshDone {
				resSSH = dialRes{err: dialCtx.Err()}
				sshDone = true
			}
			goto evaluate
		}
	}

evaluate:
	if resGateway.conn != nil {
		resGateway.conn.Close()
	}
	if resSSH.conn != nil {
		resSSH.conn.Close()
	}

	// 1. If 8080 connected but healthz was not 200 OK:
	if resGateway.err == nil {
		if resSSH.err == nil {
			return LANNodeInfo{
				Status:        LANNodeOnline,
				Host:          scrubbedHost,
				Reachable:     true,
				LatencyMicros: time.Since(start).Microseconds(),
				Transport:     "tcp_ssh",
				Appliance:     "strix-halo (SSH port 22 reachable; gateway offline)",
				Endpoint:      fmt.Sprintf("%s:%s", scrubbedHost, sshPort),
			}
		}
		return LANNodeInfo{
			Status:        LANNodeOnline,
			Host:          scrubbedHost,
			Reachable:     true,
			LatencyMicros: time.Since(start).Microseconds(),
			Transport:     "tcp_port",
			Appliance:     "strix-halo (gateway port reachable; healthz offline)",
			Endpoint:      fmt.Sprintf("http://%s:%s", scrubbedHost, gatewayPort),
		}
	}

	// 2. If port 22 responds: Status = LANNodeOnline, Reachable = true, Transport = "tcp_ssh"
	if resSSH.err == nil {
		return LANNodeInfo{
			Status:        LANNodeOnline,
			Host:          scrubbedHost,
			Reachable:     true,
			LatencyMicros: time.Since(start).Microseconds(),
			Transport:     "tcp_ssh",
			Appliance:     "strix-halo (SSH port 22 reachable; gateway offline)",
			Endpoint:      fmt.Sprintf("%s:%s", scrubbedHost, sshPort),
		}
	}

	// 3. Unreachable: fail closed within timeout (<60ms)
	errMsg := "connection refused or timed out"
	if resGateway.err != nil {
		errMsg = resGateway.err.Error()
	} else if resSSH.err != nil {
		errMsg = resSSH.err.Error()
	}

	return LANNodeInfo{
		Status:        LANNodeOffline,
		Host:          scrubbedHost,
		Reachable:     false,
		LatencyMicros: time.Since(start).Microseconds(),
		Transport:     "offline",
		Error:         ScrubLANTelemetry(errMsg),
	}
}

func checkHealthz(ctx context.Context, url string, timeout time.Duration) bool {
	hCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(hCtx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}

	client := &http.Client{
		Timeout: timeout,
		Transport: &http.Transport{
			DisableKeepAlives: true,
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)

	return resp.StatusCode == http.StatusOK
}

// ProbeInventory returns a combined local silicon + LAN node dynamic hardware inventory report.
func ProbeInventory(ctx context.Context, lanHost string) InventoryReport {
	start := time.Now().UTC()
	local := ProbeHardware()
	lan := ProbeLANNode(ctx, lanHost)
	local.LANNode = &lan

	nextAction := "lan node unconfigured; set FAK_STRIX_HOST or pass lanHost to enable remote acceleration"
	switch lan.Status {
	case LANNodeOnline:
		nextAction = "lan node online; remote hardware offload available"
	case LANNodeOffline:
		nextAction = "lan node offline; using local silicon only"
	case LANNodeUnconfigured:
		nextAction = "lan node unconfigured; set FAK_STRIX_HOST or pass lanHost to enable remote acceleration"
	}

	return InventoryReport{
		Schema:     InventorySchema,
		Timestamp:  start,
		Local:      local,
		LAN:        lan,
		Sanitized:  true,
		NextAction: nextAction,
	}
}
