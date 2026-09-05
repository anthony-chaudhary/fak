package researcharm

import (
	"context"
	"net"
	"net/http"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	socketCacheMu sync.RWMutex
	socketCache   = make(map[int]cachedProcess)
)

type cachedProcess struct {
	pid     int
	name    string
	expires time.Time
}

// ExtractOrigin derives caller attribution from an incoming HTTP request and trace ID.
func ExtractOrigin(r *http.Request, traceID string) Origin {
	var origin Origin
	if r != nil {
		origin.RemoteAddr = r.RemoteAddr
		origin.UserAgent = r.UserAgent()

		// 1. Check explicit headers.
		if h := strings.TrimSpace(r.Header.Get("X-Fak-Project-Arm")); h != "" {
			origin.ArmID = h
			origin.Explicit = true
		} else if h := strings.TrimSpace(r.Header.Get("X-Fak-Research-Arm")); h != "" {
			origin.ArmID = h
			origin.Explicit = true
		} else if h := strings.TrimSpace(r.Header.Get("X-Fak-Arm")); h != "" {
			origin.ArmID = h
			origin.Explicit = true
		} else if q := strings.TrimSpace(r.URL.Query().Get("arm")); q != "" {
			origin.ArmID = q
			origin.Explicit = true
		}

		// 2. Check explicit caller PID header.
		if pidStr := strings.TrimSpace(r.Header.Get("X-Fak-Caller-PID")); pidStr != "" {
			if p, err := strconv.Atoi(pidStr); err == nil && p > 0 {
				origin.CallerPID = p
			}
		}
	}

	// 3. Infer Arm from trace ID if not explicitly specified.
	traceID = strings.TrimSpace(traceID)
	if origin.ArmID == "" && traceID != "" {
		origin.ArmID, origin.ArmGroup = inferArmFromTrace(traceID)
	}

	// 4. Infer Arm from User-Agent or client characteristics if still empty.
	if origin.ArmID == "" && r != nil {
		origin.ArmID, origin.ArmGroup = inferArmFromUserAgent(origin.UserAgent, origin.RemoteAddr)
	}

	if origin.ArmID == "" {
		origin.ArmID = "default"
		origin.ArmGroup = "default"
	} else if origin.ArmGroup == "" {
		origin.ArmGroup = deriveGroup(origin.ArmID)
	}

	// 5. If CallerPID is not known and connection is local, resolve socket on host.
	if origin.CallerPID == 0 && r != nil && r.RemoteAddr != "" {
		pid, proc := ResolveSocketCaller(r.RemoteAddr)
		if pid > 0 {
			origin.CallerPID = pid
			origin.CallerProcess = proc
		}
	}

	return origin
}

func inferArmFromTrace(traceID string) (string, string) {
	lower := strings.ToLower(traceID)
	switch {
	case strings.HasPrefix(lower, "orch-"):
		// e.g. orch-62ef4cd4e1ce-worker-1 -> arm "orch", group "orch"
		return "orch", "orch"
	case strings.HasPrefix(lower, "codex-"):
		return "codex", "codex"
	case strings.HasPrefix(lower, "guard-"):
		return "guard", "guard"
	case strings.HasPrefix(lower, "gw-"):
		return "gw-bench", "bench"
	case strings.HasPrefix(lower, "fleet-"):
		return "fleet-worker", "fleet"
	case strings.HasPrefix(lower, "npc1_"):
		return "npc-sim", "sim"
	case strings.HasPrefix(lower, "bench-"):
		return "benchmark", "bench"
	default:
		return "session-" + summarizeTrace(traceID), "session"
	}
}

func summarizeTrace(traceID string) string {
	if len(traceID) <= 12 {
		return traceID
	}
	return traceID[:12]
}

func inferArmFromUserAgent(ua, remoteAddr string) (string, string) {
	lower := strings.ToLower(ua)
	switch {
	case strings.Contains(lower, "curl"):
		return "curl-client", "direct"
	case strings.Contains(lower, "prometheus"):
		return "prometheus", "monitoring"
	case strings.Contains(lower, "opencode"):
		return "opencode", "agent"
	case strings.Contains(lower, "codex"):
		return "codex", "agent"
	case strings.Contains(lower, "python"):
		return "python-script", "script"
	default:
		if isLoopback(remoteAddr) {
			return "local-direct", "local"
		}
		return "external-client", "external"
	}
}

func deriveGroup(armID string) string {
	lower := strings.ToLower(armID)
	switch {
	case strings.HasPrefix(lower, "orch"):
		return "orch"
	case strings.HasPrefix(lower, "codex"):
		return "codex"
	case strings.HasPrefix(lower, "guard"):
		return "guard"
	case strings.HasPrefix(lower, "bench") || strings.HasPrefix(lower, "eval"):
		return "bench"
	case strings.HasPrefix(lower, "chat") || strings.HasPrefix(lower, "interactive"):
		return "interactive"
	default:
		return "general"
	}
}

func isLoopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr
	}
	ip := net.ParseIP(host)
	if ip != nil && ip.IsLoopback() {
		return true
	}
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}

// ResolveSocketCaller looks up the local PID and executable name on loopback connections.
func ResolveSocketCaller(remoteAddr string) (int, string) {
	host, portStr, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		return 0, ""
	}
	if !isLoopback(host) {
		return 0, ""
	}
	port, err := strconv.Atoi(portStr)
	if err != nil || port <= 0 {
		return 0, ""
	}

	// Check cache
	now := time.Now()
	socketCacheMu.RLock()
	cached, ok := socketCache[port]
	socketCacheMu.RUnlock()
	if ok && now.Before(cached.expires) {
		return cached.pid, cached.name
	}

	pid, name := lookupPortProcess(port)

	// Update cache with 3s TTL
	socketCacheMu.Lock()
	socketCache[port] = cachedProcess{
		pid:     pid,
		name:    name,
		expires: now.Add(3 * time.Second),
	}
	// Bound cache size
	if len(socketCache) > 1024 {
		for k, v := range socketCache {
			if now.After(v.expires) {
				delete(socketCache, k)
			}
		}
	}
	socketCacheMu.Unlock()

	return pid, name
}

func lookupPortProcess(port int) (int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()

	if runtime.GOOS == "darwin" {
		// On Darwin, lsof -iTCP:<port> -sTCP:ESTABLISHED -P -n -F pc
		cmd := exec.CommandContext(ctx, "lsof", "-iTCP:"+strconv.Itoa(port), "-sTCP:ESTABLISHED", "-P", "-n", "-F", "pc")
		out, err := cmd.Output()
		if err != nil {
			return 0, ""
		}
		var pid int
		var comm string
		for _, line := range strings.Split(string(out), "\n") {
			if strings.HasPrefix(line, "p") && pid == 0 {
				pid, _ = strconv.Atoi(strings.TrimPrefix(line, "p"))
			} else if strings.HasPrefix(line, "c") && comm == "" {
				comm = strings.TrimPrefix(line, "c")
			}
		}
		return pid, comm
	} else if runtime.GOOS == "linux" {
		// On Linux, ss -t -p '( sport = :<port> )' or fuser
		cmd := exec.CommandContext(ctx, "ss", "-t", "-p", "( sport = :"+strconv.Itoa(port)+" )")
		out, err := cmd.Output()
		if err == nil {
			s := string(out)
			if idx := strings.Index(s, "pid="); idx != -1 {
				sub := s[idx+4:]
				if comma := strings.IndexAny(sub, ",)"); comma != -1 {
					if p, err := strconv.Atoi(sub[:comma]); err == nil {
						return p, ""
					}
				}
			}
		}
	}
	return 0, ""
}
