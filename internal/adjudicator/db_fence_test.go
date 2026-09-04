package adjudicator

import (
	"errors"
	"testing"
)

func TestDBFence(t *testing.T) {
	// Ensure a clean state before running tests
	ResetAllowedDBHosts()
	defer ResetAllowedDBHosts()

	cases := []struct {
		name        string
		tool        string
		args        map[string]any
		wantBlocked bool
		wantHost    string
	}{
		// 1. Loopback addresses permitted by default
		{
			name:        "local localhost url",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@localhost:5432/app"},
			wantBlocked: false,
		},
		{
			name:        "local 127.0.0.1 url",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@127.0.0.1:5432/app"},
			wantBlocked: false,
		},
		{
			name:        "local ::1 url",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@[::1]:5432/app"},
			wantBlocked: false,
		},
		{
			name:        "local localhost cli flag",
			tool:        "bash",
			args:        map[string]any{"command": "psql -h localhost -U admin app"},
			wantBlocked: false,
		},
		{
			name:        "local 127.0.0.1 cli flag",
			tool:        "bash",
			args:        map[string]any{"command": "mysql -h 127.0.0.1 -u root -p secret app"},
			wantBlocked: false,
		},
		{
			name:        "structured argument local url",
			tool:        "db_query",
			args:        map[string]any{"database_url": "postgres://user:pass@localhost:5432/app"},
			wantBlocked: false,
		},
		{
			name:        "structured argument local host",
			tool:        "db_query",
			args:        map[string]any{"host": "127.0.0.1"},
			wantBlocked: false,
		},

		// 2. Generic service identifiers and container aliases BLOCKED by default (#11310 Requirement 1)
		{
			name:        "container alias postgres blocked by default",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@postgres:5432/app"},
			wantBlocked: true,
			wantHost:    "postgres",
		},
		{
			name:        "container alias mysql blocked by default",
			tool:        "bash",
			args:        map[string]any{"command": "mysql -h mysql -u root -p secret app"},
			wantBlocked: true,
			wantHost:    "mysql",
		},
		{
			name:        "container alias db blocked by default",
			tool:        "bash",
			args:        map[string]any{"command": "mysql -h db -u root -p secret app"},
			wantBlocked: true,
			wantHost:    "db",
		},
		{
			name:        "container alias database blocked by default",
			tool:        "bash",
			args:        map[string]any{"command": "psql -h database app"},
			wantBlocked: true,
			wantHost:    "database",
		},
		{
			name:        "container alias mariadb blocked by default",
			tool:        "bash",
			args:        map[string]any{"command": "mysql -h mariadb app"},
			wantBlocked: true,
			wantHost:    "mariadb",
		},
		{
			name:        "container alias redis blocked by default",
			tool:        "bash",
			args:        map[string]any{"command": "redis-cli -h redis ping"},
			wantBlocked: true,
			wantHost:    "redis",
		},
		{
			name:        "container alias test-db blocked by default",
			tool:        "bash",
			args:        map[string]any{"command": "psql -h test-db app"},
			wantBlocked: true,
			wantHost:    "test-db",
		},
		{
			name:        "host.docker.internal blocked by default",
			tool:        "bash",
			args:        map[string]any{"command": "psql -h host.docker.internal app"},
			wantBlocked: true,
			wantHost:    "host.docker.internal",
		},

		// 3. Remote hosts blocked
		{
			name:        "remote internal host blocked",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@prod-db.internal:5432/app"},
			wantBlocked: true,
			wantHost:    "prod-db.internal",
		},
		{
			name:        "remote amazon rds host blocked",
			tool:        "bash",
			args:        map[string]any{"command": "psql -h remote.rds.amazonaws.com -U admin app"},
			wantBlocked: true,
			wantHost:    "remote.rds.amazonaws.com",
		},
		{
			name:        "structured argument remote url",
			tool:        "db_query",
			args:        map[string]any{"database_url": "postgres://user:pass@staging-db.corp.net:5432/app"},
			wantBlocked: true,
			wantHost:    "staging-db.corp.net",
		},
		{
			name:        "structured argument remote host",
			tool:        "db_query",
			args:        map[string]any{"host": "prod-db.example.com"},
			wantBlocked: true,
			wantHost:    "prod-db.example.com",
		},

		// 4. Environment variables in command strings (#11310 Requirement 2)
		{
			name:        "export PGHOST remote host with semicolon",
			tool:        "bash",
			args:        map[string]any{"command": "export PGHOST=prod-db.example.com; psql"},
			wantBlocked: true,
			wantHost:    "prod-db.example.com",
		},
		{
			name:        "inline PGHOST prefix command",
			tool:        "bash",
			args:        map[string]any{"command": "PGHOST=remote-db psql"},
			wantBlocked: true,
			wantHost:    "remote-db",
		},
		{
			name:        "export PGHOST double quoted with and-chain",
			tool:        "bash",
			args:        map[string]any{"command": `export PGHOST="prod-db.example.com" && psql`},
			wantBlocked: true,
			wantHost:    "prod-db.example.com",
		},
		{
			name:        "export PGHOST single quoted",
			tool:        "bash",
			args:        map[string]any{"command": `export PGHOST='prod-db.example.com'; psql`},
			wantBlocked: true,
			wantHost:    "prod-db.example.com",
		},
		{
			name:        "powershell env PGHOST assignment",
			tool:        "bash",
			args:        map[string]any{"command": `$env:PGHOST="prod-db.example.com"; psql`},
			wantBlocked: true,
			wantHost:    "prod-db.example.com",
		},
		{
			name:        "windows cmd set PGHOST",
			tool:        "bash",
			args:        map[string]any{"command": `set PGHOST=prod-db.example.com & psql`},
			wantBlocked: true,
			wantHost:    "prod-db.example.com",
		},
		{
			name:        "inline MYSQL_HOST remote host",
			tool:        "bash",
			args:        map[string]any{"command": "MYSQL_HOST=remote.mysql.com mysql -u root"},
			wantBlocked: true,
			wantHost:    "remote.mysql.com",
		},
		{
			name:        "export MYSQL_HOST quoted",
			tool:        "bash",
			args:        map[string]any{"command": `export MYSQL_HOST="remote.mysql.com"; mysql`},
			wantBlocked: true,
			wantHost:    "remote.mysql.com",
		},
		{
			name:        "inline DATABASE_URL remote",
			tool:        "bash",
			args:        map[string]any{"command": "DATABASE_URL=postgres://u:p@db.production.company.com:5432/db go test ./..."},
			wantBlocked: true,
			wantHost:    "db.production.company.com",
		},
		{
			name:        "export DATABASE_URL quoted",
			tool:        "bash",
			args:        map[string]any{"command": `export DATABASE_URL="postgres://u:p@db.production.company.com:5432/db"; psql`},
			wantBlocked: true,
			wantHost:    "db.production.company.com",
		},
		{
			name:        "inline POSTGRES_URL remote",
			tool:        "bash",
			args:        map[string]any{"command": "POSTGRES_URL=postgres://u:p@staging-db.net:5432/app psql"},
			wantBlocked: true,
			wantHost:    "staging-db.net",
		},
		{
			name:        "export POSTGRES_URL remote",
			tool:        "bash",
			args:        map[string]any{"command": `export POSTGRES_URL="postgres://u:p@staging-db.net:5432/app"; psql`},
			wantBlocked: true,
			wantHost:    "staging-db.net",
		},
		{
			name:        "local PGHOST permitted",
			tool:        "bash",
			args:        map[string]any{"command": "PGHOST=localhost psql"},
			wantBlocked: false,
		},
		{
			name:        "local 127.0.0.1 PGHOST permitted",
			tool:        "bash",
			args:        map[string]any{"command": "PGHOST=127.0.0.1 psql"},
			wantBlocked: false,
		},

		// 5. Environment variables in tool arguments (#11310 Requirement 2)
		{
			name:        "direct PGHOST in tool args",
			tool:        "run_process",
			args:        map[string]any{"command": "psql", "PGHOST": "prod-db.example.com"},
			wantBlocked: true,
			wantHost:    "prod-db.example.com",
		},
		{
			name:        "direct MYSQL_HOST in tool args",
			tool:        "run_process",
			args:        map[string]any{"command": "mysql", "MYSQL_HOST": "remote-mysql.corp.net"},
			wantBlocked: true,
			wantHost:    "remote-mysql.corp.net",
		},
		{
			name:        "direct DATABASE_URL in tool args",
			tool:        "run_process",
			args:        map[string]any{"command": "npm test", "DATABASE_URL": "postgres://u:p@remote-db.corp.net:5432/db"},
			wantBlocked: true,
			wantHost:    "remote-db.corp.net",
		},
		{
			name:        "env map in tool args with remote PGHOST",
			tool:        "bash",
			args:        map[string]any{"command": "psql", "env": map[string]any{"PGHOST": "prod-db.example.com"}},
			wantBlocked: true,
			wantHost:    "prod-db.example.com",
		},
		{
			name:        "env map in tool args with remote MYSQL_HOST",
			tool:        "bash",
			args:        map[string]any{"command": "mysql", "env": map[string]string{"MYSQL_HOST": "remote.mysql.com"}},
			wantBlocked: true,
			wantHost:    "remote.mysql.com",
		},
		{
			name:        "env map in tool args with remote DATABASE_URL",
			tool:        "bash",
			args:        map[string]any{"command": "go test", "env": map[string]any{"DATABASE_URL": "postgres://u:p@staging.corp.com:5432/db"}},
			wantBlocked: true,
			wantHost:    "staging.corp.com",
		},
		{
			name:        "env slice in tool args with remote PGHOST",
			tool:        "bash",
			args:        map[string]any{"command": "psql", "env": []string{"PGHOST=prod-db.example.com"}},
			wantBlocked: true,
			wantHost:    "prod-db.example.com",
		},
		{
			name:        "env slice in tool args with remote DATABASE_URL",
			tool:        "bash",
			args:        map[string]any{"command": "npm start", "env": []any{"DATABASE_URL=postgres://u:p@remote-db.corp.net:5432/db"}},
			wantBlocked: true,
			wantHost:    "remote-db.corp.net",
		},
		{
			name:        "env map in tool args with local PGHOST permitted",
			tool:        "bash",
			args:        map[string]any{"command": "psql", "env": map[string]any{"PGHOST": "localhost"}},
			wantBlocked: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			host, blocked, reason := ClassifyDatabaseDestination(tc.tool, tc.args)
			if blocked != tc.wantBlocked {
				t.Fatalf("blocked = %v, want %v (host=%s, reason=%s)", blocked, tc.wantBlocked, host, reason)
			}
			if tc.wantBlocked && host != tc.wantHost {
				t.Errorf("host = %q, want %q", host, tc.wantHost)
			}
			if tc.wantBlocked && reason != ReasonProductionDBBlock {
				t.Errorf("reason = %q, want %q", reason, ReasonProductionDBBlock)
			}

			err := CheckDBDestination(tc.tool, tc.args)
			if tc.wantBlocked && err == nil {
				t.Errorf("expected CheckDBDestination error for blocked host, got nil")
			}
			if !tc.wantBlocked && err != nil {
				t.Errorf("unexpected CheckDBDestination error: %v", err)
			}
			if tc.wantBlocked && err != nil && !errors.Is(err, ErrProductionDBBlock) {
				t.Errorf("expected error to wrap ErrProductionDBBlock, got %v", err)
			}
		})
	}
}

func TestDBFenceWhitelisting(t *testing.T) {
	ResetAllowedDBHosts()
	defer ResetAllowedDBHosts()

	// Initially container aliases are blocked
	cmdPostgres := map[string]any{"command": "psql postgres://user:pass@postgres:5432/app"}
	cmdMySQL := map[string]any{"command": "mysql -h mysql -u root app"}
	cmdDb := map[string]any{"command": "mysql -h db app"}

	if _, blocked, _ := ClassifyDBDestination("bash", cmdPostgres); !blocked {
		t.Fatalf("expected unconfigured postgres to be blocked")
	}
	if _, blocked, _ := ClassifyDBDestination("bash", cmdMySQL); !blocked {
		t.Fatalf("expected unconfigured mysql to be blocked")
	}
	if _, blocked, _ := ClassifyDBDestination("bash", cmdDb); !blocked {
		t.Fatalf("expected unconfigured db to be blocked")
	}

	// Explicitly whitelist container service aliases (#11310 Requirement 1)
	AllowDBHost("postgres", "mysql", "db")

	if host, blocked, reason := ClassifyDBDestination("bash", cmdPostgres); blocked {
		t.Fatalf("expected whitelisted postgres to be permitted, got blocked: %s (%s)", host, reason)
	}
	if host, blocked, reason := ClassifyDBDestination("bash", cmdMySQL); blocked {
		t.Fatalf("expected whitelisted mysql to be permitted, got blocked: %s (%s)", host, reason)
	}
	if host, blocked, reason := ClassifyDBDestination("bash", cmdDb); blocked {
		t.Fatalf("expected whitelisted db to be permitted, got blocked: %s (%s)", host, reason)
	}

	// Remote production host must still be blocked even when aliases are whitelisted
	cmdRemote := map[string]any{"command": "psql postgres://user:pass@prod-db.example.com:5432/app"}
	if host, blocked, _ := ClassifyDBDestination("bash", cmdRemote); !blocked || host != "prod-db.example.com" {
		t.Fatalf("expected remote host to remain blocked, got host=%q, blocked=%v", host, blocked)
	}

	// Reset clears whitelisted hosts
	ResetAllowedDBHosts()
	if _, blocked, _ := ClassifyDBDestination("bash", cmdPostgres); !blocked {
		t.Fatalf("expected postgres to be blocked after ResetAllowedDBHosts")
	}

	// Per-request whitelisting via tool args
	argsWithAllow := map[string]any{
		"command":          "psql postgres://user:pass@postgres:5432/app",
		"allowed_db_hosts": []string{"postgres"},
	}
	if host, blocked, reason := ClassifyDBDestination("bash", argsWithAllow); blocked {
		t.Fatalf("expected per-request allowed_db_hosts to permit postgres, got blocked: %s (%s)", host, reason)
	}
}

func TestDBFenceEnvironmentContext(t *testing.T) {
	ResetAllowedDBHosts()
	defer ResetAllowedDBHosts()

	// 1. Process environment variable PGHOST set to remote host
	t.Setenv("PGHOST", "prod-db.example.com")

	// psql without explicit host in command line must be blocked by environment context
	cmdPsql := map[string]any{"command": "psql app"}
	host, blocked, reason := ClassifyDBDestination("bash", cmdPsql)
	if !blocked || host != "prod-db.example.com" {
		t.Fatalf("expected ambient PGHOST to block psql, got host=%q, blocked=%v, reason=%s", host, blocked, reason)
	}
	if reason != ReasonProductionDBBlock {
		t.Errorf("reason = %q, want %q", reason, ReasonProductionDBBlock)
	}

	// Explicit loopback host flag in command line overrides ambient PGHOST
	cmdPsqlLocal := map[string]any{"command": "psql -h localhost app"}
	if host, blocked, _ := ClassifyDBDestination("bash", cmdPsqlLocal); blocked {
		t.Fatalf("expected explicit -h localhost to override ambient PGHOST, got blocked: %s", host)
	}

	// Non-database command must NOT be blocked by ambient PGHOST
	cmdNonDB := map[string]any{"command": "ls -la"}
	if _, blocked, _ := ClassifyDBDestination("bash", cmdNonDB); blocked {
		t.Fatalf("non-database command must not be blocked by ambient PGHOST")
	}

	// 2. Explicit environment context passed via ClassifyDatabaseDestinationWithContext
	t.Setenv("PGHOST", "")
	envCtx := map[string]string{"MYSQL_HOST": "remote.mysql.corp.com"}
	cmdMySQL := map[string]any{"command": "mysql -u root app"}
	host, blocked, reason = ClassifyDatabaseDestinationWithContext("bash", cmdMySQL, envCtx)
	if !blocked || host != "remote.mysql.corp.com" {
		t.Fatalf("expected envCtx MYSQL_HOST to block mysql, got host=%q, blocked=%v", host, blocked)
	}
	if reason != ReasonProductionDBBlock {
		t.Errorf("reason = %q, want %q", reason, ReasonProductionDBBlock)
	}
}

func TestDatabaseDestinationFence(t *testing.T) {
	ResetAllowedDBHosts()
	defer ResetAllowedDBHosts()

	cases := []struct {
		name        string
		tool        string
		args        map[string]any
		wantBlocked bool
		wantHost    string
	}{
		{
			name:        "local localhost url",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@localhost:5432/app"},
			wantBlocked: false,
		},
		{
			name:        "local 127.0.0.1 url",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@127.0.0.1:5432/app"},
			wantBlocked: false,
		},
		{
			name:        "remote internal host blocked",
			tool:        "bash",
			args:        map[string]any{"command": "psql postgres://user:pass@prod-db.internal:5432/app"},
			wantBlocked: true,
			wantHost:    "prod-db.internal",
		},
		{
			name:        "remote amazon rds host blocked",
			tool:        "bash",
			args:        map[string]any{"command": "psql -h remote.rds.amazonaws.com -U admin app"},
			wantBlocked: true,
			wantHost:    "remote.rds.amazonaws.com",
		},
		{
			name:        "structured argument remote url",
			tool:        "db_query",
			args:        map[string]any{"database_url": "postgres://user:pass@staging-db.corp.net:5432/app"},
			wantBlocked: true,
			wantHost:    "staging-db.corp.net",
		},
		{
			name:        "structured argument local url",
			tool:        "db_query",
			args:        map[string]any{"database_url": "postgres://user:pass@localhost:5432/app"},
			wantBlocked: false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			host, blocked, reason := ClassifyDatabaseDestination(tc.tool, tc.args)
			if blocked != tc.wantBlocked {
				t.Fatalf("blocked = %v, want %v (host=%s, reason=%s)", blocked, tc.wantBlocked, host, reason)
			}
			if tc.wantBlocked && host != tc.wantHost {
				t.Errorf("host = %q, want %q", host, tc.wantHost)
			}
			if tc.wantBlocked && reason != ReasonProductionDBBlock {
				t.Errorf("reason = %q, want %q", reason, ReasonProductionDBBlock)
			}
		})
	}
}

func TestSentinelError(t *testing.T) {
	if !errors.Is(ErrProductionDBBlock, ErrProductionDBBlock) {
		t.Errorf("expected sentinel error match")
	}
}
