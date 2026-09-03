package policy

import (
	"errors"
	"testing"
)

func TestDatabaseDestinationFence(t *testing.T) {
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
			name:        "local container alias db",
			tool:        "bash",
			args:        map[string]any{"command": "mysql -h db -u root -p secret app"},
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
		{
			name:        "inline DATABASE_URL remote",
			tool:        "bash",
			args:        map[string]any{"command": "DATABASE_URL=postgres://u:p@db.production.company.com:5432/db go test ./..."},
			wantBlocked: true,
			wantHost:    "db.production.company.com",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			host, blocked, reason := ClassifyDBDestination(tc.tool, tc.args)
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
		})
	}
}

func TestSentinelError(t *testing.T) {
	if !errors.Is(ErrProductionDBBlock, ErrProductionDBBlock) {
		t.Errorf("expected sentinel error match")
	}
}
