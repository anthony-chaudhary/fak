package adjudicator

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"regexp"
	"strings"
	"sync"
)

// ReasonProductionDBBlock is the policy refusal token emitted when an action targets a remote database.
const ReasonProductionDBBlock = "POLICY_BLOCK (PRODUCTION_DB_BLOCK)"

// ErrProductionDBBlock is the sentinel error returned when a non-loopback database target is blocked.
var ErrProductionDBBlock = errors.New(ReasonProductionDBBlock)

// defaultLoopbackHosts contains only strict loopback addresses. Generic service aliases
// (e.g. postgres, mysql, database) must be explicitly configured to prevent bypasses
// in containerized/VPC environments (#11310).
var defaultLoopbackHosts = map[string]bool{
	"localhost": true,
	"127.0.0.1": true,
	"::1":       true,
}

var (
	allowedDBHostsMu     sync.RWMutex
	configuredDBHostsSet = map[string]bool{}
)

// AllowDBHost explicitly whitelists database host identifiers (e.g. container service aliases).
func AllowDBHost(hosts ...string) {
	allowedDBHostsMu.Lock()
	defer allowedDBHostsMu.Unlock()
	for _, h := range hosts {
		ch := cleanHost(h)
		if ch != "" {
			configuredDBHostsSet[ch] = true
		}
	}
}

// SetAllowedDBHosts replaces the set of explicitly configured allowed database hosts.
func SetAllowedDBHosts(hosts []string) {
	allowedDBHostsMu.Lock()
	defer allowedDBHostsMu.Unlock()
	configuredDBHostsSet = make(map[string]bool, len(hosts))
	for _, h := range hosts {
		ch := cleanHost(h)
		if ch != "" {
			configuredDBHostsSet[ch] = true
		}
	}
}

// ResetAllowedDBHosts clears all explicitly whitelisted database hosts.
func ResetAllowedDBHosts() {
	allowedDBHostsMu.Lock()
	defer allowedDBHostsMu.Unlock()
	configuredDBHostsSet = make(map[string]bool)
}

// AllowedDBHosts returns a copy of the explicitly configured allowed database hosts.
func AllowedDBHosts() []string {
	allowedDBHostsMu.RLock()
	defer allowedDBHostsMu.RUnlock()
	out := make([]string, 0, len(configuredDBHostsSet))
	for h := range configuredDBHostsSet {
		out = append(out, h)
	}
	return out
}

// cleanHost normalizes a host string by trimming whitespace, quotes, brackets, and ports.
func cleanHost(host string) string {
	h := strings.ToLower(strings.TrimSpace(host))
	h = strings.Trim(h, `"'`)
	if strings.HasPrefix(h, "[") {
		if end := strings.Index(h, "]"); end != -1 {
			return h[1:end]
		}
	}
	if idx := strings.LastIndex(h, ":"); idx != -1 {
		// Only strip port if it contains a single colon (host:port, not unbracketed IPv6)
		if strings.Count(h, ":") == 1 {
			h = h[:idx]
		}
	}
	return strings.Trim(h, "[]")
}

// IsLocalDBHost reports whether a database host is a permitted local development endpoint.
func IsLocalDBHost(host string) bool {
	return isAllowedDBHost(host, nil)
}

func isLocalDBHost(host string) bool {
	return isAllowedDBHost(host, nil)
}

func isAllowedDBHost(host string, extraAllowed []string) bool {
	h := cleanHost(host)
	if h == "" {
		return false
	}
	if defaultLoopbackHosts[h] {
		return true
	}
	for _, a := range extraAllowed {
		if cleanHost(a) == h {
			return true
		}
	}
	allowedDBHostsMu.RLock()
	allowed := configuredDBHostsSet[h]
	allowedDBHostsMu.RUnlock()
	if allowed {
		return true
	}
	if env := os.Getenv("FAK_ALLOWED_DB_HOSTS"); env != "" {
		for _, part := range strings.Split(env, ",") {
			if cleanHost(part) == h {
				return true
			}
		}
	}
	return false
}

var (
	dbURLRE    = regexp.MustCompile(`(?i)\b(postgres|postgresql|mysql|mariadb|redis|rediss|mongodb|clickhouse)://([^\s"']+)`)
	cliHostRE  = regexp.MustCompile(`(?i)(?:^|[\s;&|])(?:psql|pg_dump|pg_restore|pg_isready|vacuumdb|createdb|dropdb|mysql|mysqldump|mysqladmin|mariadb|mariadb-dump|redis-cli|mongosh|mongo|clickhouse-client)\b.*?(?:-h\s+|--host[=\s]+)([^\s;&|]+)`)
	dbEnvVarRE = regexp.MustCompile(`(?i)(?:^|[\s;&|]|(?:export|set|env)\s+|\$env:)(PGHOST|PGHOSTADDR|MYSQL_HOST|DATABASE_URL|POSTGRES_URL|PGDATABASE)\s*[=:]\s*(?:"([^"]+)"|'([^']+)'|([^\s;&|]+))`)
	dbExecRE   = regexp.MustCompile(`(?i)(?:^|[\s;&|])(?:psql|pg_dump|pg_restore|pg_isready|vacuumdb|createdb|dropdb|mysql|mysqldump|mysqladmin|mariadb|mariadb-dump|redis-cli|mongosh|mongo|clickhouse-client)\b`)
)

func isDBExecution(tool string, cmd string) bool {
	lowerTool := strings.ToLower(strings.TrimSpace(tool))
	if lowerTool == "psql" || lowerTool == "mysql" || lowerTool == "mariadb" ||
		lowerTool == "redis-cli" || lowerTool == "mongosh" || lowerTool == "clickhouse-client" ||
		lowerTool == "db_query" || lowerTool == "database" || lowerTool == "sql_query" ||
		strings.Contains(lowerTool, "db") || strings.Contains(lowerTool, "sql") {
		return true
	}
	return dbExecRE.MatchString(cmd)
}

func hasExplicitHostInCmd(cmd string) bool {
	return cliHostRE.MatchString(cmd) || dbEnvVarRE.MatchString(cmd) || dbURLRE.MatchString(cmd)
}

func extractExtraAllowed(args map[string]any) []string {
	var out []string
	for _, k := range []string{"allowed_db_hosts", "allowed_hosts", "db_allowed_hosts"} {
		v, ok := args[k]
		if !ok || v == nil {
			continue
		}
		switch val := v.(type) {
		case []string:
			out = append(out, val...)
		case []any:
			for _, item := range val {
				if s, ok := item.(string); ok && s != "" {
					out = append(out, s)
				}
			}
		case string:
			for _, item := range strings.Split(val, ",") {
				s := strings.TrimSpace(item)
				if s != "" {
					out = append(out, s)
				}
			}
		}
	}
	return out
}

// ClassifyDatabaseDestination inspects tool arguments or command text for database connection endpoints.
// If a non-local/production destination is detected, it returns the target host, true, and the refusal reason.
func ClassifyDatabaseDestination(tool string, args map[string]any) (string, bool, string) {
	return ClassifyDatabaseDestinationWithContext(tool, args, nil)
}

// ClassifyDBDestination is an alias for ClassifyDatabaseDestination.
func ClassifyDBDestination(tool string, args map[string]any) (string, bool, string) {
	return ClassifyDatabaseDestinationWithContext(tool, args, nil)
}

// ClassifyDatabaseDestinationWithContext inspects tool arguments, command text, and environment context
// for database connection endpoints.
func ClassifyDatabaseDestinationWithContext(tool string, args map[string]any, envCtx map[string]string) (string, bool, string) {
	extraAllowed := extractExtraAllowed(args)

	// 1. Inspect structured connection arguments in args
	for _, key := range []string{"database_url", "db_url", "connection_string", "url", "endpoint", "postgres_url", "DATABASE_URL", "POSTGRES_URL"} {
		if val, ok := args[key]; ok {
			if s, ok := val.(string); ok && s != "" {
				if host, blocked := checkDBConnectionString(s, extraAllowed); blocked {
					return host, true, ReasonProductionDBBlock
				}
			}
		}
	}

	// 2. Inspect direct host arguments in args
	for _, key := range []string{"host", "hostname", "db_host", "pghost", "pghostaddr", "mysql_host", "PGHOST", "PGHOSTADDR", "MYSQL_HOST"} {
		if val, ok := args[key]; ok {
			if s, ok := val.(string); ok && s != "" {
				h := cleanHost(s)
				if h != "" && !isAllowedDBHost(h, extraAllowed) {
					return s, true, ReasonProductionDBBlock
				}
			}
		}
	}

	// 2b. Inspect PGDATABASE if it carries a connection URL
	for _, key := range []string{"pgdatabase", "PGDATABASE"} {
		if val, ok := args[key]; ok {
			if s, ok := val.(string); ok && strings.Contains(s, "://") {
				if host, blocked := checkDBConnectionString(s, extraAllowed); blocked {
					return host, true, ReasonProductionDBBlock
				}
			}
		}
	}

	// 3. Inspect environment maps or slices embedded in args
	for _, envKey := range []string{"env", "environment", "env_vars", "envvars"} {
		if rawEnv, ok := args[envKey]; ok && rawEnv != nil {
			if host, blocked := checkEmbeddedEnv(rawEnv, extraAllowed); blocked {
				return host, true, ReasonProductionDBBlock
			}
		}
	}

	// 4. Inspect command text for CLI flags, inline env vars, or inline connection URLs
	cmd := extractCommandString(args)
	if cmd != "" {
		// 4a. Environment variables in command string (e.g. export PGHOST=..., PGHOST=... psql)
		for _, m := range dbEnvVarRE.FindAllStringSubmatch(cmd, -1) {
			if len(m) > 1 {
				varName := strings.ToUpper(m[1])
				val := m[2]
				if val == "" {
					val = m[3]
				}
				if val == "" && len(m) > 4 {
					val = m[4]
				}
				val = strings.TrimSpace(val)
				if val != "" {
					switch varName {
					case "PGHOST", "PGHOSTADDR", "MYSQL_HOST":
						h := cleanHost(val)
						if h != "" && !isAllowedDBHost(h, extraAllowed) {
							return h, true, ReasonProductionDBBlock
						}
					case "DATABASE_URL", "POSTGRES_URL":
						if host, blocked := checkDBConnectionString(val, extraAllowed); blocked {
							return host, true, ReasonProductionDBBlock
						}
					case "PGDATABASE":
						if strings.Contains(val, "://") {
							if host, blocked := checkDBConnectionString(val, extraAllowed); blocked {
								return host, true, ReasonProductionDBBlock
							}
						}
					}
				}
			}
		}

		// 4b. URLs in command line
		for _, m := range dbURLRE.FindAllStringSubmatch(cmd, -1) {
			if len(m) > 0 {
				if host, blocked := checkDBConnectionString(m[0], extraAllowed); blocked {
					return host, true, ReasonProductionDBBlock
				}
			}
		}

		// 4c. CLI host flag (-h or --host)
		for _, m := range cliHostRE.FindAllStringSubmatch(cmd, -1) {
			if len(m) > 1 {
				host := strings.Trim(m[1], `"'`)
				h := cleanHost(host)
				if h != "" && !isAllowedDBHost(h, extraAllowed) {
					return host, true, ReasonProductionDBBlock
				}
			}
		}
	}

	// 5. Inspect environment context before permitting database executions
	if isDBExecution(tool, cmd) && !hasExplicitHostInCmd(cmd) {
		if host, blocked := checkContextEnv(envCtx, extraAllowed); blocked {
			return host, true, ReasonProductionDBBlock
		}
	}

	return "", false, ""
}

func checkEmbeddedEnv(raw any, extraAllowed []string) (string, bool) {
	switch env := raw.(type) {
	case map[string]string:
		for k, v := range env {
			if h, blocked := checkEnvKeyValue(k, v, extraAllowed); blocked {
				return h, true
			}
		}
	case map[string]any:
		for k, v := range env {
			if s, ok := v.(string); ok {
				if h, blocked := checkEnvKeyValue(k, s, extraAllowed); blocked {
					return h, true
				}
			}
		}
	case []string:
		for _, entry := range env {
			if k, v, ok := parseEnvEntry(entry); ok {
				if h, blocked := checkEnvKeyValue(k, v, extraAllowed); blocked {
					return h, true
				}
			}
		}
	case []any:
		for _, item := range env {
			if entry, ok := item.(string); ok {
				if k, v, ok := parseEnvEntry(entry); ok {
					if h, blocked := checkEnvKeyValue(k, v, extraAllowed); blocked {
						return h, true
					}
				}
			}
		}
	}
	return "", false
}

func parseEnvEntry(entry string) (string, string, bool) {
	idx := strings.Index(entry, "=")
	if idx == -1 {
		return "", "", false
	}
	return strings.TrimSpace(entry[:idx]), strings.TrimSpace(entry[idx+1:]), true
}

func checkEnvKeyValue(key, val string, extraAllowed []string) (string, bool) {
	uk := strings.ToUpper(strings.TrimSpace(key))
	v := strings.TrimSpace(val)
	if v == "" {
		return "", false
	}
	switch uk {
	case "PGHOST", "PGHOSTADDR", "MYSQL_HOST":
		h := cleanHost(v)
		if h != "" && !isAllowedDBHost(h, extraAllowed) {
			return h, true
		}
	case "DATABASE_URL", "POSTGRES_URL":
		if host, blocked := checkDBConnectionString(v, extraAllowed); blocked {
			return host, true
		}
	case "PGDATABASE":
		if strings.Contains(v, "://") {
			if host, blocked := checkDBConnectionString(v, extraAllowed); blocked {
				return host, true
			}
		}
	}
	return "", false
}

func checkContextEnv(envCtx map[string]string, extraAllowed []string) (string, bool) {
	lookup := func(k string) string {
		if envCtx != nil {
			if v, ok := envCtx[k]; ok {
				return v
			}
			if v, ok := envCtx[strings.ToLower(k)]; ok {
				return v
			}
		}
		return os.Getenv(k)
	}

	for _, k := range []string{"PGHOST", "PGHOSTADDR", "MYSQL_HOST"} {
		if val := lookup(k); val != "" {
			h := cleanHost(val)
			if h != "" && !isAllowedDBHost(h, extraAllowed) {
				return h, true
			}
		}
	}
	for _, k := range []string{"DATABASE_URL", "POSTGRES_URL"} {
		if val := lookup(k); val != "" {
			if host, blocked := checkDBConnectionString(val, extraAllowed); blocked {
				return host, true
			}
		}
	}
	return "", false
}

// CheckDBDestination validates that database targets in the tool call remain strictly local.
func CheckDBDestination(tool string, args map[string]any) error {
	host, blocked, reason := ClassifyDBDestination(tool, args)
	if blocked {
		return fmt.Errorf("%w: remote database host %q is refused by production connection fence (reason=%s)", ErrProductionDBBlock, host, reason)
	}
	return nil
}

// CheckDBFence validates that database targets in the tool call remain strictly local.
func CheckDBFence(tool string, args map[string]any) error {
	return CheckDBDestination(tool, args)
}

func checkDBConnectionString(raw string, extraAllowed []string) (string, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", false
	}

	// Handle host=... in key-value connection strings (libpq format)
	if strings.Contains(raw, "host=") {
		for _, part := range strings.Fields(raw) {
			if strings.HasPrefix(strings.ToLower(part), "host=") {
				h := cleanHost(part[5:])
				if h != "" && !isAllowedDBHost(h, extraAllowed) {
					return h, true
				}
			}
		}
	}

	// Try standard url.Parse first
	if u, err := url.Parse(raw); err == nil && u.Hostname() != "" {
		h := cleanHost(u.Hostname())
		if h != "" {
			if !isAllowedDBHost(h, extraAllowed) {
				return h, true
			}
			return h, false
		}
	}

	// Fallback slicing for custom/unparseable URI formats
	idx := strings.Index(raw, "://")
	if idx == -1 {
		return "", false
	}
	rest := raw[idx+3:]
	if at := strings.LastIndex(rest, "@"); at != -1 {
		rest = rest[at+1:]
	}
	if slash := strings.Index(rest, "/"); slash != -1 {
		rest = rest[:slash]
	}
	if q := strings.Index(rest, "?"); q != -1 {
		rest = rest[:q]
	}
	h := cleanHost(rest)
	if h == "" {
		return "", false
	}
	if !isAllowedDBHost(h, extraAllowed) {
		return h, true
	}
	return h, false
}

func extractCommandString(args map[string]any) string {
	for _, k := range []string{"command", "cmd", "shell", "script"} {
		if v, ok := args[k]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
	}
	return ""
}
