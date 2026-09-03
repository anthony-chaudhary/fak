package policy

import (
	"errors"
	"reflect"
	"testing"
)

func TestCombineAllowlists(t *testing.T) {
	tests := []struct {
		name   string
		parent []string
		child  []string
		want   []string
	}{
		{
			name:   "both nil",
			parent: nil,
			child:  nil,
			want:   nil,
		},
		{
			name:   "parent nil child unconstrained",
			parent: nil,
			child:  []string{"tool_b", "tool_a", "tool_a"},
			want:   []string{"tool_a", "tool_b"},
		},
		{
			name:   "child nil inherits parent",
			parent: []string{"tool_y", "tool_x"},
			child:  nil,
			want:   []string{"tool_x", "tool_y"},
		},
		{
			name:   "parent nil child empty slice",
			parent: nil,
			child:  []string{},
			want:   []string{},
		},
		{
			name:   "parent empty slice child nil",
			parent: []string{},
			child:  nil,
			want:   []string{},
		},
		{
			name:   "parent empty slice child has items",
			parent: []string{},
			child:  []string{"tool_a", "tool_b"},
			want:   []string{},
		},
		{
			name:   "parent has items child empty slice",
			parent: []string{"tool_a", "tool_b"},
			child:  []string{},
			want:   []string{},
		},
		{
			name:   "disjoint sets result in empty slice",
			parent: []string{"read_file", "search_kb"},
			child:  []string{"refund_payment", "execute_command"},
			want:   []string{},
		},
		{
			name:   "overlapping sets intersect and sort",
			parent: []string{"search_kb", "read_file", "list_files"},
			child:  []string{"read_file", "grep_search", "search_kb"},
			want:   []string{"read_file", "search_kb"},
		},
		{
			name:   "duplicate items in parent and child deduplicated",
			parent: []string{"a", "b", "a", "c", "b"},
			child:  []string{"c", "b", "c", "b", "d"},
			want:   []string{"b", "c"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CombineAllowlists(tt.parent, tt.child)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CombineAllowlists(%v, %v) = %v; want %v", tt.parent, tt.child, got, tt.want)
			}
		})
	}
}

func TestCombineDenylists(t *testing.T) {
	tests := []struct {
		name   string
		parent []string
		child  []string
		want   []string
	}{
		{
			name:   "both nil",
			parent: nil,
			child:  nil,
			want:   nil,
		},
		{
			name:   "parent nil child has items",
			parent: nil,
			child:  []string{"tool_b", "tool_a", "tool_b"},
			want:   []string{"tool_a", "tool_b"},
		},
		{
			name:   "child nil parent has items",
			parent: []string{"tool_z", "tool_y"},
			child:  nil,
			want:   []string{"tool_y", "tool_z"},
		},
		{
			name:   "both empty slices",
			parent: []string{},
			child:  []string{},
			want:   []string{},
		},
		{
			name:   "parent empty slice child nil",
			parent: []string{},
			child:  nil,
			want:   []string{},
		},
		{
			name:   "disjoint union",
			parent: []string{"rm_rf", "drop_db"},
			child:  []string{"format_disk", "chmod_777"},
			want:   []string{"chmod_777", "drop_db", "format_disk", "rm_rf"},
		},
		{
			name:   "overlapping union with deduplication and sorting",
			parent: []string{"shell_exec", "eval", "reboot"},
			child:  []string{"eval", "su", "reboot"},
			want:   []string{"eval", "reboot", "shell_exec", "su"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CombineDenylists(tt.parent, tt.child)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CombineDenylists(%v, %v) = %v; want %v", tt.parent, tt.child, got, tt.want)
			}
		})
	}
}

func TestCombinePrivacy(t *testing.T) {
	tests := []struct {
		name   string
		parent PrivacyPolicy
		child  PrivacyPolicy
		want   PrivacyPolicy
	}{
		{
			name: "both permissive defaults",
			parent: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: true,
			},
			child: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: true,
			},
			want: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: true,
			},
		},
		{
			name: "parent enforces zero retention",
			parent: PrivacyPolicy{
				ZeroRetention:   true,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: true,
			},
			child: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: true,
			},
			want: PrivacyPolicy{
				ZeroRetention:   true,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: true,
			},
		},
		{
			name: "child enforces mask PII",
			parent: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: true,
			},
			child: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         true,
				LogContent:      true,
				StoreTranscript: true,
			},
			want: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         true,
				LogContent:      true,
				StoreTranscript: true,
			},
		},
		{
			name: "parent disallows logging content",
			parent: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      false,
				StoreTranscript: true,
			},
			child: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: true,
			},
			want: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      false,
				StoreTranscript: true,
			},
		},
		{
			name: "child disallows storing transcripts",
			parent: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: true,
			},
			child: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: false,
			},
			want: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: false,
			},
		},
		{
			name: "least privilege combination across all fields",
			parent: PrivacyPolicy{
				ZeroRetention:   true,
				MaskPII:         false,
				LogContent:      true,
				StoreTranscript: false,
			},
			child: PrivacyPolicy{
				ZeroRetention:   false,
				MaskPII:         true,
				LogContent:      false,
				StoreTranscript: true,
			},
			want: PrivacyPolicy{
				ZeroRetention:   true,
				MaskPII:         true,
				LogContent:      false,
				StoreTranscript: false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CombinePrivacy(tt.parent, tt.child)
			if got != tt.want {
				t.Errorf("CombinePrivacy(%+v, %+v) = %+v; want %+v", tt.parent, tt.child, got, tt.want)
			}
		})
	}
}

func TestEvaluateBudgets(t *testing.T) {
	tests := []struct {
		name    string
		budgets []Budget
		usage   Usage
		wantErr bool
	}{
		{
			name:    "empty budgets allows any usage",
			budgets: nil,
			usage: Usage{
				Tokens:  100000,
				CostUSD: 50.0,
				Calls:   500,
			},
			wantErr: false,
		},
		{
			name: "within budget caps",
			budgets: []Budget{
				{MaxTokens: 1000, MaxCostUSD: 2.50, MaxCalls: 50},
			},
			usage: Usage{
				Tokens:  500,
				CostUSD: 1.25,
				Calls:   25,
			},
			wantErr: false,
		},
		{
			name: "exactly at budget caps",
			budgets: []Budget{
				{MaxTokens: 1000, MaxCostUSD: 2.50, MaxCalls: 50},
			},
			usage: Usage{
				Tokens:  1000,
				CostUSD: 2.50,
				Calls:   50,
			},
			wantErr: false,
		},
		{
			name: "token cap exceeded",
			budgets: []Budget{
				{MaxTokens: 1000, MaxCostUSD: 2.50, MaxCalls: 50},
			},
			usage: Usage{
				Tokens:  1001,
				CostUSD: 1.0,
				Calls:   10,
			},
			wantErr: true,
		},
		{
			name: "cost cap exceeded",
			budgets: []Budget{
				{MaxTokens: 1000, MaxCostUSD: 2.50, MaxCalls: 50},
			},
			usage: Usage{
				Tokens:  500,
				CostUSD: 2.51,
				Calls:   10,
			},
			wantErr: true,
		},
		{
			name: "calls cap exceeded",
			budgets: []Budget{
				{MaxTokens: 1000, MaxCostUSD: 2.50, MaxCalls: 50},
			},
			usage: Usage{
				Tokens:  500,
				CostUSD: 1.0,
				Calls:   51,
			},
			wantErr: true,
		},
		{
			name: "unconstrained zero caps do not fail",
			budgets: []Budget{
				{MaxTokens: 0, MaxCostUSD: 0, MaxCalls: 0},
			},
			usage: Usage{
				Tokens:  999999,
				CostUSD: 999.99,
				Calls:   99999,
			},
			wantErr: false,
		},
		{
			name: "hierarchical lowest cap failure in parent",
			budgets: []Budget{
				{MaxTokens: 500},  // parent cap
				{MaxTokens: 1000}, // child cap
			},
			usage: Usage{
				Tokens: 600,
			},
			wantErr: true,
		},
		{
			name: "hierarchical lowest cap failure in child",
			budgets: []Budget{
				{MaxTokens: 1000}, // parent cap
				{MaxTokens: 300},  // child cap
			},
			usage: Usage{
				Tokens: 400,
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := EvaluateBudgets(tt.budgets, tt.usage)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("EvaluateBudgets(%v, %v) expected error, got nil", tt.budgets, tt.usage)
				}
				if !errors.Is(err, ErrBudgetExceeded) {
					t.Errorf("expected ErrBudgetExceeded, got %v", err)
				}
			} else {
				if err != nil {
					t.Fatalf("EvaluateBudgets(%v, %v) unexpected error: %v", tt.budgets, tt.usage, err)
				}
			}
		})
	}
}

func TestCombinePolicies(t *testing.T) {
	// Simulate 4-layer hierarchy: Organization -> Workspace -> Member -> Key
	orgPolicy := CombinedPolicy{
		Allow: []string{"search_kb", "read_file", "list_files", "send_email"},
		Deny:  []string{"rm_rf", "drop_db"},
		Privacy: PrivacyPolicy{
			ZeroRetention:   false,
			MaskPII:         true,
			LogContent:      true,
			StoreTranscript: true,
		},
		Budgets: []Budget{
			{MaxCostUSD: 100.0, MaxTokens: 1000000},
		},
	}

	workspacePolicy := CombinedPolicy{
		Allow: []string{"search_kb", "read_file", "write_file"},
		Deny:  []string{"drop_db", "reboot"},
		Privacy: PrivacyPolicy{
			ZeroRetention:   true,
			MaskPII:         false,
			LogContent:      true,
			StoreTranscript: true,
		},
		Budgets: []Budget{
			{MaxCostUSD: 25.0, MaxTokens: 250000},
		},
	}

	memberPolicy := CombinedPolicy{
		Allow: []string{"search_kb", "read_file"},
		Deny:  []string{"eval"},
		Privacy: PrivacyPolicy{
			ZeroRetention:   false,
			MaskPII:         false,
			LogContent:      false,
			StoreTranscript: true,
		},
		Budgets: []Budget{
			{MaxCostUSD: 10.0, MaxCalls: 100},
		},
	}

	keyPolicy := CombinedPolicy{
		Allow: []string{"read_file"},
		Deny:  nil,
		Privacy: PrivacyPolicy{
			ZeroRetention:   false,
			MaskPII:         false,
			LogContent:      true,
			StoreTranscript: false,
		},
		Budgets: []Budget{
			{MaxCalls: 50},
		},
	}

	// Combine Org -> Workspace
	orgWs := CombinePolicies(orgPolicy, workspacePolicy)
	// Combine (Org->Ws) -> Member
	orgWsMem := CombinePolicies(orgWs, memberPolicy)
	// Combine ((Org->Ws)->Mem) -> Key
	final := CombinePolicies(orgWsMem, keyPolicy)

	// Verify Allowlists:
	// Org: {search_kb, read_file, list_files, send_email}
	// Ws:  {search_kb, read_file, write_file} -> intersection: {read_file, search_kb}
	// Mem: {search_kb, read_file} -> intersection: {read_file, search_kb}
	// Key: {read_file} -> intersection: {read_file}
	wantAllow := []string{"read_file"}
	if !reflect.DeepEqual(final.Allow, wantAllow) {
		t.Errorf("final.Allow = %v, want %v", final.Allow, wantAllow)
	}

	// Verify Denylists:
	// Org: {rm_rf, drop_db}
	// Ws:  {drop_db, reboot}
	// Mem: {eval}
	// Key: nil
	// Union: {drop_db, eval, reboot, rm_rf}
	wantDeny := []string{"drop_db", "eval", "reboot", "rm_rf"}
	if !reflect.DeepEqual(final.Deny, wantDeny) {
		t.Errorf("final.Deny = %v, want %v", final.Deny, wantDeny)
	}

	// Verify Privacy:
	// ZeroRetention: Org(F) || Ws(T) || Mem(F) || Key(F) = true
	// MaskPII:       Org(T) || Ws(F) || Mem(F) || Key(F) = true
	// LogContent:    Org(T) && Ws(T) && Mem(F) && Key(T) = false
	// StoreTranscript: Org(T) && Ws(T) && Mem(T) && Key(F) = false
	wantPrivacy := PrivacyPolicy{
		ZeroRetention:   true,
		MaskPII:         true,
		LogContent:      false,
		StoreTranscript: false,
	}
	if final.Privacy != wantPrivacy {
		t.Errorf("final.Privacy = %+v, want %+v", final.Privacy, wantPrivacy)
	}

	// Verify Budgets concatenation:
	// Org (100 USD, 1M tokens), Ws (25 USD, 250k tokens), Mem (10 USD, 100 calls), Key (50 calls)
	if len(final.Budgets) != 4 {
		t.Fatalf("expected 4 budgets, got %d", len(final.Budgets))
	}

	// Test budget evaluation against multi-layer policy:
	// Usage passing all layers:
	validUsage := Usage{Tokens: 1000, CostUSD: 5.0, Calls: 20}
	if err := EvaluateBudgets(final.Budgets, validUsage); err != nil {
		t.Errorf("expected validUsage to pass, got: %v", err)
	}

	// Usage exceeding key budget (calls > 50):
	exceededKeyCalls := Usage{Tokens: 1000, CostUSD: 5.0, Calls: 51}
	if err := EvaluateBudgets(final.Budgets, exceededKeyCalls); !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("expected exceededKeyCalls to fail with ErrBudgetExceeded, got: %v", err)
	}

	// Usage exceeding member budget (cost > 10.0):
	exceededMemberCost := Usage{Tokens: 1000, CostUSD: 12.0, Calls: 30}
	if err := EvaluateBudgets(final.Budgets, exceededMemberCost); !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("expected exceededMemberCost to fail with ErrBudgetExceeded, got: %v", err)
	}

	// Usage exceeding workspace budget (tokens > 250000):
	exceededWsTokens := Usage{Tokens: 300000, CostUSD: 8.0, Calls: 30}
	if err := EvaluateBudgets(final.Budgets, exceededWsTokens); !errors.Is(err, ErrBudgetExceeded) {
		t.Errorf("expected exceededWsTokens to fail with ErrBudgetExceeded, got: %v", err)
	}
}
