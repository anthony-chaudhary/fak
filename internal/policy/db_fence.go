package policy

import (
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

// ReasonProductionDBBlock is the policy refusal token emitted when an action targets a remote database.
const ReasonProductionDBBlock = "POLICY_BLOCK (PRODUCTION_DB_BLOCK)"

// ErrProductionDBBlock is the sentinel error returned when a non-loopback database target is blocked.
var ErrProductionDBBlock = errors.New(ReasonProductionDBBlock)

var allowedLocalHosts = map[string]bool{
	"localhost":            true,
	"127.0.0.1":            true,
	"::1":                  true,
	"0.0.0.0":              true,
	"host.docker.internal": true,
	"test-db":              true,
	"postgres":             true,
	"mysql":                true,
	"mariadb":              true,
	"redis":                true,
	"db":                   true,
	"database":             true,
}

// IsLocalDBHost reports whether a database host is a permitted local development endpoint.
func IsLocalDBHost(host string) bool {
	h := strings.ToLower(strings.TrimSpace(host))
	// Strip port if present
	if idx := strings.LastIndex(h, ":"); idx != -1 {
		// Ensure it's not an IPv6 address without brackets
		if !strings.Contains(h[:idx], ":") || strings.HasPrefix(h, "[") {
			h = strings.Trim(h[:idx], "[]")
		}
	}
	return allowedLocalHosts[h]
}

var (
	dbURLRE    = regexp.MustCompile(`(?i)\b(postgres|postgresql|mysql|mariadb|redis|rediss|mongodb|clickhouse)://([^\s"']+)`)
	cliHostRE  = regexp.MustCompile(`(?i)(?:^|[\s;&|])(?:psql|mysql|mariadb|redis-cli|mongosh|clickhouse-client)\b.*?(?:-h\s+|--host[=\s]+)([^\s;&|]+)`)
	dbTargetRE = regexp.MustCompile(`(?i)\bDATABASE_URL=([^\s]+)`)
)

// ClassifyDBDestination inspects tool arguments or command text for database connection endpoints.
// If a non-local/production destination is detected, it returns the target host, true, and the refusal reason.
func ClassifyDBDestination(tool string, args map[string]any) (string, bool, string) {
	// 1. Inspect structured connection arguments
	for _, key := range []string{"database_url", "db_url", "connection_string", "url", "endpoint"} {
		if val, ok := args[key]; ok {
			if s, ok := val.(string); ok && s != "" {
				if host, blocked := checkConnectionString(s); blocked {
					return host, true, ReasonProductionDBBlock
				}
			}
		}
	}

	// 2. Inspect host argument
	for _, key := range []string{"host", "hostname", "db_host"} {
		if val, ok := args[key]; ok {
			if s, ok := val.(string); ok && s != "" {
				if !IsLocalDBHost(s) {
					return s, true, ReasonProductionDBBlock
				}
			}
		}
	}

	// 3. Inspect command text for CLI flags or inline connection strings
	cmd := extractCommandText(args)
	if cmd == "" {
		return "", false, ""
	}

	// 3a. Inline DATABASE_URL
	if m := dbTargetRE.FindStringSubmatch(cmd); len(m) > 1 {
		if host, blocked := checkConnectionString(m[1]); blocked {
			return host, true, ReasonProductionDBBlock
		}
	}

	// 3b. URLs in command line
	for _, m := range dbURLRE.FindAllStringSubmatch(cmd, -1) {
		if len(m) > 0 {
			if host, blocked := checkConnectionString(m[0]); blocked {
				return host, true, ReasonProductionDBBlock
			}
		}
	}

	// 3c. CLI host flag (-h or --host)
	if m := cliHostRE.FindStringSubmatch(cmd); len(m) > 1 {
		host := strings.Trim(m[1], `"'`)
		if host != "" && !IsLocalDBHost(host) {
			return host, true, ReasonProductionDBBlock
		}
	}

	return "", false, ""
}

// CheckDBDestination validates that database targets in the tool call remain strictly local.
func CheckDBDestination(tool string, args map[string]any) error {
	host, blocked, reason := ClassifyDBDestination(tool, args)
	if blocked {
		return fmt.Errorf("%s: remote database host %q is refused by production connection fence", reason, host)
	}
	return nil
}

func checkConnectionString(raw string) (string, bool) {
	u, err := url.Parse(raw)
	if err != nil {
		return "", false
	}
	h := u.Hostname()
	if h == "" {
		return "", false
	}
	if !IsLocalDBHost(h) {
		return h, true
	}
	return h, false
}

func extractCommandText(args map[string]any) string {
	for _, k := range []string{"command", "cmd", "shell", "script"} {
		if v, ok := args[k]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
	}
	return ""
}
