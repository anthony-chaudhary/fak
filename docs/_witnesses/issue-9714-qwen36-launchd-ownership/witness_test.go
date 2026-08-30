package issue9714witness

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

const (
	incumbentCommandSHA = "a0c13c8d7e84fa92c76db532db0a6fed13b3b42313a75e10b8f69e252f76a91d"
	alternateOwnerSHA   = "b567298df044cecdac3cf921d7cb971e4665db7b5407fb634647a910e3abbdb7"
)

type receipt struct {
	Schema     string `json:"schema"`
	Issue      int    `json:"issue"`
	ObservedAt string `json:"observed_at"`
	Verdict    string `json:"verdict"`
	Reason     string `json:"reason"`
	Source     struct {
		Revision string `json:"revision"`
	} `json:"source"`
	Scope struct {
		OwnershipOnly         bool `json:"ownership_only"`
		Qwen38ArtifactTouched bool `json:"qwen38_artifact_touched"`
		Qwen38ModelArms       int  `json:"qwen38_model_arms"`
	} `json:"scope"`
	ExpectedOwner struct {
		ServiceLabel            string `json:"service_label"`
		LaunchdJobPresent       bool   `json:"launchd_job_present"`
		LaunchAgentPlistPresent bool   `json:"launchagent_plist_present"`
		ListenerPIDBound        bool   `json:"listener_pid_bound"`
	} `json:"expected_owner"`
	Incumbent struct {
		ListenerCount                   int    `json:"listener_count"`
		CommandSHA256                   string `json:"command_sha256"`
		CommandMatchesPreservedIdentity bool   `json:"command_matches_preserved_identity"`
		OwnerLabelSHA256                string `json:"owner_label_sha256"`
		OwnerMatchesExpected            bool   `json:"owner_matches_expected"`
		HealthStatus                    int    `json:"health_status"`
		ModelsStatus                    int    `json:"models_status"`
		ModelAlias                      string `json:"model_alias"`
	} `json:"incumbent"`
	Stability struct {
		SampleCount            int    `json:"sample_count"`
		FirstObservedAt        string `json:"first_observed_at"`
		LastObservedAt         string `json:"last_observed_at"`
		ElapsedSeconds         int    `json:"elapsed_seconds"`
		ListenerCountStable    bool   `json:"listener_count_stable"`
		CommandSHA256Stable    bool   `json:"command_sha256_stable"`
		OwnerLabelSHA256Stable bool   `json:"owner_label_sha256_stable"`
		Health200Continuous    bool   `json:"health_200_continuous"`
		Models200Continuous    bool   `json:"models_200_continuous"`
		ModelAliasStable       bool   `json:"model_alias_stable"`
		Classification         string `json:"classification"`
	} `json:"stability"`
	Operations struct {
		GPULeaseAcquired              bool `json:"gpu_lease_acquired"`
		SignalsSent                   int  `json:"signals_sent"`
		BootoutAttempts               int  `json:"bootout_attempts"`
		BootstrapAttempts             int  `json:"bootstrap_attempts"`
		ServiceConfigurationMutations int  `json:"service_configuration_mutations"`
		UnmatchedSignals              int  `json:"unmatched_signals"`
		PIDReuseSignals               int  `json:"pid_reuse_signals"`
	} `json:"operations"`
	ExternalState struct {
		Mutated                 bool `json:"mutated"`
		RestorationRequired     bool `json:"restoration_required"`
		IncumbentHealthyAtExit  bool `json:"incumbent_healthy_at_exit"`
		ExpectedJobAbsentAtExit bool `json:"expected_job_absent_at_exit"`
	} `json:"external_state"`
	DoneCondition struct {
		Complete bool   `json:"complete"`
		Missing  string `json:"missing"`
	} `json:"done_condition"`
}

func TestReceiptFailsClosedOnAlternateOwner(t *testing.T) {
	r := readReceipt(t)
	if err := validateReceipt(r); err != nil {
		t.Fatal(err)
	}
	mutants := []func(*receipt){
		func(v *receipt) { v.Verdict = "PASS" },
		func(v *receipt) { v.ExpectedOwner.LaunchdJobPresent = true },
		func(v *receipt) { v.Incumbent.OwnerMatchesExpected = true },
		func(v *receipt) { v.Operations.SignalsSent = 1 },
		func(v *receipt) { v.DoneCondition.Complete = true },
	}
	for i, mutate := range mutants {
		v := r
		mutate(&v)
		if err := validateReceipt(v); err == nil {
			t.Fatalf("tamper mutant %d passed", i)
		}
	}
}

func TestReadOnlyStabilitySpansAtLeastNinetySeconds(t *testing.T) {
	f, err := os.Open("stability-samples.tsv")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.Comma = '\t'
	rows, err := r.ReadAll()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 92 {
		t.Fatalf("rows = %d, want header plus 91 samples", len(rows))
	}
	if got := strings.Join(rows[0], ","); got != "sample,observed_at,listener_count,command_sha256,owner_label_sha256,health_status,models_status,model_alias" {
		t.Fatalf("header = %q", got)
	}
	for i, row := range rows[1:] {
		if len(row) != 8 || row[0] != stringInt(i) || row[2] != "1" ||
			row[3] != incumbentCommandSHA || row[4] != alternateOwnerSHA ||
			row[5] != "200" || row[6] != "200" || row[7] != "qwen3.6-27b" {
			t.Fatalf("sample row %d drifted: %v", i+2, row)
		}
	}
	first := parseTime(t, rows[1][1])
	last := parseTime(t, rows[len(rows)-1][1])
	if elapsed := last.Sub(first); elapsed < 90*time.Second {
		t.Fatalf("sample span = %s, want at least 90s", elapsed)
	}
}

func TestPacketContainsNoPrivateControlData(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		b, err := os.ReadFile(entry.Name())
		if err != nil {
			t.Fatal(err)
		}
		s := strings.ToLower(string(b))
		for _, forbidden := range []string{
			"/" + "users/", "/" + "private/", "host" + "name=", "ssh" + " ",
			"token" + "=", "password" + "=", "secret" + "=", "channel" + "_id",
		} {
			if strings.Contains(s, forbidden) {
				t.Fatalf("%s contains private marker %q", entry.Name(), forbidden)
			}
		}
	}
}

func validateReceipt(r receipt) error {
	if r.Schema != "fak.issue-9714-qwen36-launchd-ownership-hold/1" || r.Issue != 9714 ||
		r.Verdict != "HOLD" || r.Reason != "alternate_launchd_supervisor_owns_incumbent" ||
		r.Source.Revision != "d9ac41475c0c52e9e32ec3a0122ac0d2e5402aa1" {
		return errors.New("receipt identity drift")
	}
	if !r.Scope.OwnershipOnly || r.Scope.Qwen38ArtifactTouched || r.Scope.Qwen38ModelArms != 0 {
		return errors.New("ownership-only scope drift")
	}
	if r.ExpectedOwner.ServiceLabel != "com.fak.qwen36-model" || r.ExpectedOwner.LaunchdJobPresent ||
		r.ExpectedOwner.LaunchAgentPlistPresent || r.ExpectedOwner.ListenerPIDBound {
		return errors.New("expected owner gate drift")
	}
	if r.Incumbent.ListenerCount != 1 || r.Incumbent.CommandSHA256 != incumbentCommandSHA ||
		!r.Incumbent.CommandMatchesPreservedIdentity || r.Incumbent.OwnerLabelSHA256 != alternateOwnerSHA ||
		r.Incumbent.OwnerMatchesExpected || r.Incumbent.HealthStatus != 200 || r.Incumbent.ModelsStatus != 200 ||
		r.Incumbent.ModelAlias != "qwen3.6-27b" {
		return errors.New("incumbent identity drift")
	}
	if r.Stability.SampleCount != 91 || r.Stability.ElapsedSeconds < 90 || !r.Stability.ListenerCountStable ||
		!r.Stability.CommandSHA256Stable || !r.Stability.OwnerLabelSHA256Stable || !r.Stability.Health200Continuous ||
		!r.Stability.Models200Continuous || !r.Stability.ModelAliasStable ||
		r.Stability.Classification != "preexisting_incumbent_stability_not_post_restore_credit" {
		return errors.New("stability classification drift")
	}
	if r.Operations.GPULeaseAcquired || r.Operations.SignalsSent != 0 || r.Operations.BootoutAttempts != 0 ||
		r.Operations.BootstrapAttempts != 0 || r.Operations.ServiceConfigurationMutations != 0 ||
		r.Operations.UnmatchedSignals != 0 || r.Operations.PIDReuseSignals != 0 {
		return errors.New("fail-closed operation cardinality drift")
	}
	if r.ExternalState.Mutated || r.ExternalState.RestorationRequired || !r.ExternalState.IncumbentHealthyAtExit ||
		!r.ExternalState.ExpectedJobAbsentAtExit || r.DoneCondition.Complete ||
		!strings.Contains(r.DoneCondition.Missing, "explicit authority") {
		return errors.New("external-state or done-condition drift")
	}
	return nil
}

func readReceipt(t *testing.T) receipt {
	t.Helper()
	var r receipt
	b, err := os.ReadFile("receipt.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func parseTime(t *testing.T, value string) time.Time {
	t.Helper()
	v, err := time.Parse(time.RFC3339, value)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func stringInt(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for ; v > 0; v /= 10 {
		i--
		buf[i] = byte('0' + v%10)
	}
	return string(buf[i:])
}
