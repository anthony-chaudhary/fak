package modelroute

import (
	"fmt"
	"runtime"
)

type accidentalFixtureSpec struct {
	class    AccidentalFailureClass
	badLine  string
	goodLine string
	path     string
}

// AccidentalCorpus returns paired ordinary-engineering closures. The external
// witness outcome is carried as bounded test evidence, while each patch preserves
// the error locus an auditor must explain.
func AccidentalCorpus() []AccidentalCorpusFixture {
	specs := []accidentalFixtureSpec{
		{AccidentalIncompleteDoneCondition, `return primaryOnly(input)`, `return primaryAndFallback(input)`, "internal/feature/feature.go"},
		{AccidentalWrongEdgeCase, `if n > limit {`, `if n >= limit {`, "internal/limit/limit.go"},
		{AccidentalSwallowedError, `value, _ := load()`, `value, err := load(); if err != nil { return err }`, "internal/load/load.go"},
		{AccidentalStaleConsumer, `decodeV1(payload)`, `decodeV2(payload)`, ".github/workflows/consumer.yml"},
		{AccidentalMissingFailureTest, `func TestSuccess(t *testing.T) {}`, `func TestFailure(t *testing.T) {}`, "internal/widget/widget_test.go"},
		{AccidentalRaceLostUpdate, `state.Count = state.Count + 1`, `state.mu.Lock(); state.Count++; state.mu.Unlock()`, "internal/state/state.go"},
		{AccidentalPartialRename, `OldContract(input)`, `NewContract(input)`, "internal/client/client.go"},
		{AccidentalBuildPoison, `return missingSymbol`, `return definedSymbol`, "internal/build/build.go"},
		{AccidentalDocsCLIDrift, `fak widget --old-flag`, `fak widget --new-flag`, "docs/widget.md"},
		{AccidentalOverBroadRewrite, `replaceAllRecords()`, `replaceMatchingRecord(id)`, "internal/store/store.go"},
		{AccidentalRevertedSafetyCheck, `allow(request)`, `if safe(request) { allow(request) }`, "internal/gate/gate.go"},
		{AccidentalCleanHardRefactor, `return legacyPipeline(input)`, `return equivalentPipeline(input)`, "internal/refactor/refactor.go"},
	}
	out := make([]AccidentalCorpusFixture, 0, len(specs)*2)
	for i, spec := range specs {
		pair := string(spec.class)
		out = append(out,
			buildAccidentalFixture(6000+i*2, spec, pair+"-corrupt", pair, true),
			buildAccidentalFixture(6001+i*2, spec, pair+"-clean", pair, false),
		)
	}
	return out
}

func buildAccidentalFixture(num int, spec accidentalFixtureSpec, id, pair string, corrupt bool) AccidentalCorpusFixture {
	line := spec.goodLine
	if corrupt {
		line = spec.badLine
	}
	sha := fmt.Sprintf("acc%07d", num)
	parent := fmt.Sprintf("par%07d", num)
	patch := fmt.Sprintf("diff --git a/%s b/%s\n@@\n+%s\n", spec.path, spec.path, line)
	command := fmt.Sprintf("go test ./internal/crossauditfixture/%s -run TestContract", pair)
	evidence := IssueAuditEvidence{
		IssueNumber: num,
		IssueURL:    fmt.Sprintf("https://github.com/example/repo/issues/%d", num),
		Title:       fmt.Sprintf("%s paired closure", pair),
		Body:        fmt.Sprintf("Done condition: preserve %s behavior and pass its external witness.", pair),
		State:       "CLOSED", ClosedAt: "2026-07-11T00:00:00Z", CommitSHA: sha, Diff: patch,
		ClosingCommits: []IssueAuditClosingCommit{{
			SHA: sha, FirstParentSHA: parent, TreeOID: "tree-" + sha, FirstParentTreeOID: "tree-" + parent,
			Patch: patch, PatchSHA256: IssueAuditContentDigest(patch), ChangedPaths: []string{spec.path},
		}},
		Tests: []EvidenceRef{{Kind: "command", Ref: command}},
		CI:    []EvidenceRef{{Kind: "check", Ref: "ci/corpus-selfcheck"}},
		DOS:   []EvidenceRef{{Kind: "dos-commit-audit", Ref: "commit:" + sha, SHA256: IssueAuditContentDigest("dos-" + sha)}},
	}
	return AccidentalCorpusFixture{
		ID: id, Pair: pair, Class: spec.class, Corrupt: corrupt,
		Witness:  executableAccidentalWitness(command, spec, line),
		Evidence: evidence,
		Author:   AuditIdentity{Harness: "claude-cli", Provider: "example-claude", Family: "example-claude-family", Model: "corpus-author", ProvenanceSource: "corpus-manifest"},
		Auditor:  AuditIdentity{Harness: "codex-cli", Provider: "example-gpt", Family: "example-gpt-family", Model: "corpus-auditor", ProvenanceSource: "corpus-manifest"},
	}
}

func executableAccidentalWitness(command string, spec accidentalFixtureSpec, line string) AccidentalWitness {
	// A fresh helper process receives only the candidate line and independently
	// compares it with the contract-preserving implementation. No label/corrupt
	// bit is passed across this boundary.
	program := "sh"
	input := line + "\n" + spec.goodLine + "\n"
	args := `IFS= read -r actual; IFS= read -r contract; if [ "$actual" = "$contract" ]; then printf PASS; exit 0; else printf FAIL; exit 1; fi`
	if runtime.GOOS == "windows" {
		program = "powershell"
		args = "$actual=[Console]::In.ReadLine();$contract=[Console]::In.ReadLine();if($actual -ceq $contract){[Console]::Out.Write('PASS');exit 0}else{[Console]::Out.Write('FAIL');exit 1}"
	}
	return AccidentalWitness{Command: command, Program: program, Args: args, Input: input}
}
