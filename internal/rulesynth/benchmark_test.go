package rulesynth

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// BenchmarkDetect_NearMiss measures evaluating an unrecognized write targeting a guarded path.
func BenchmarkDetect_NearMiss(b *testing.B) {
	c := Call{
		Tool:    "Bash",
		Arg:     "command",
		Command: `php -r 'file_put_contents("internal/adjudicator/decide.go", $x);'`,
	}
	globs := DefaultHarnessGlobs

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nm, ok := Detect(c, globs)
		if !ok || nm.GuardedGlob == "" {
			b.Fatal("expected near-miss detection")
		}
	}
}

// BenchmarkDetect_RecognizedWrite measures evaluating a write with a recognized write verb.
func BenchmarkDetect_RecognizedWrite(b *testing.B) {
	c := Call{
		Tool:    "Bash",
		Arg:     "command",
		Command: `sed -i s/a/b/ internal/adjudicator/decide.go`,
	}
	globs := DefaultHarnessGlobs

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, ok := Detect(c, globs)
		if ok {
			b.Fatal("recognized write should not be detected as near-miss")
		}
	}
}

// BenchmarkDetect_Unguarded measures evaluating a command touching no guarded tree.
func BenchmarkDetect_Unguarded(b *testing.B) {
	c := Call{
		Tool:    "Bash",
		Arg:     "command",
		Command: `php -r 'file_put_contents("/tmp/output.txt", $x);'`,
	}
	globs := DefaultHarnessGlobs

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, ok := Detect(c, globs)
		if ok {
			b.Fatal("unguarded write should not be detected as near-miss")
		}
	}
}

// BenchmarkPropose measures clustering near-misses into synthesized candidate ArgRules.
func BenchmarkPropose(b *testing.B) {
	corpus := []NearMiss{
		{Call: Call{Tool: "Bash", Arg: "command", Command: `ruby -e 'File.write("internal/adjudicator/x.go","")'`}, GuardedGlob: "internal/adjudicator/"},
		{Call: Call{Tool: "Bash", Arg: "command", Command: `ruby -e 'File.write("internal/abi/y.go","")'`}, GuardedGlob: "internal/abi/"},
		{Call: Call{Tool: "Bash", Arg: "command", Command: `sponge internal/adjudicator/z.go`}, GuardedGlob: "internal/adjudicator/"},
		{Call: Call{Tool: "Bash", Arg: "command", Command: `php -r 'file_put_contents("internal/policy/p.go", "")'`}, GuardedGlob: "internal/policy/"},
		{Call: Call{Tool: "Bash", Arg: "command", Command: `perl -e 'print "x"' internal/kernel/k.go`}, GuardedGlob: "internal/kernel/"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cands := Propose(corpus)
		if len(cands) == 0 {
			b.Fatal("expected proposed candidates")
		}
	}
}

// BenchmarkPropose_LargeCorpus measures candidate synthesis over a scaled near-miss corpus.
func BenchmarkPropose_LargeCorpus(b *testing.B) {
	verbs := []string{"ruby -e", "sponge", "php -r", "perl -e", "python3 -c"}
	globs := []string{"internal/adjudicator/", "internal/abi/", "internal/policy/", "internal/kernel/", "dos.toml"}
	corpus := make([]NearMiss, 100)
	for i := 0; i < 100; i++ {
		v := verbs[i%len(verbs)]
		g := globs[i%len(globs)]
		corpus[i] = NearMiss{
			Call: Call{
				Tool:    "Bash",
				Arg:     "command",
				Command: fmt.Sprintf("%s 'write(%s/file_%d.go)'", v, g, i),
			},
			GuardedGlob: g,
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cands := Propose(corpus)
		if len(cands) != len(verbs) {
			b.Fatalf("expected %d candidates, got %d", len(verbs), len(cands))
		}
	}
}

// BenchmarkValidate_Clean measures model-free candidate validation yielding a KEEP decision.
func BenchmarkValidate_Clean(b *testing.B) {
	corpus := []NearMiss{
		{Call: Call{Tool: "Bash", Arg: "command", Command: `ruby -e 'File.write("internal/adjudicator/x.go","")'`}, GuardedGlob: "internal/adjudicator/"},
		{Call: Call{Tool: "Bash", Arg: "command", Command: `ruby -e 'File.write("internal/abi/y.go","")'`}, GuardedGlob: "internal/abi/"},
	}
	benign := []Call{
		{Tool: "Bash", Arg: "command", Command: `ruby app.rb`},
		{Tool: "Bash", Arg: "command", Command: `cat internal/adjudicator/decide.go`},
		{Tool: "Bash", Arg: "command", Command: `ls -la internal/abi/`},
	}
	cands := Propose(corpus)
	if len(cands) == 0 {
		b.Fatal("expected candidate")
	}
	cand := cands[0]

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Validate(cand, corpus, benign)
		if err != nil {
			b.Fatalf("validate: %v", err)
		}
		if !v.Kept {
			b.Fatal("expected clean candidate to be kept")
		}
	}
}

// BenchmarkValidate_Regressing measures candidate validation yielding a REVERT decision on regression.
func BenchmarkValidate_Regressing(b *testing.B) {
	corpus := []NearMiss{
		{Call: Call{Tool: "Bash", Arg: "command", Command: `ruby -e 'File.write("internal/adjudicator/x.go","")'`}, GuardedGlob: "internal/adjudicator/"},
	}
	cand := Propose(corpus)[0]
	benign := []Call{
		{Tool: "Bash", Arg: "command", Command: `ruby -e 'puts File.read("internal/adjudicator/decide.go")'`},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, err := Validate(cand, corpus, benign)
		if err != nil {
			b.Fatalf("validate: %v", err)
		}
		if v.Kept {
			b.Fatal("expected regressing candidate to be reverted")
		}
	}
}

// BenchmarkManifestDiff measures emitting the reviewable manifest diff fragment.
func BenchmarkManifestDiff(b *testing.B) {
	corpus := []NearMiss{
		{Call: Call{Tool: "Bash", Arg: "command", Command: `sponge internal/adjudicator/x.go`}, GuardedGlob: "internal/adjudicator/"},
	}
	cand := Propose(corpus)[0]

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := cand.ManifestDiff()
		if len(m.ArgRules) != 1 {
			b.Fatal("expected 1 arg rule in manifest diff")
		}
	}
}

// BenchmarkHarvester_Emit_NearMiss measures stream event ingestion for an admitted near-miss.
func BenchmarkHarvester_Emit_NearMiss(b *testing.B) {
	corpus := NewNearMissCorpus()
	h := NewHarvester(corpus, DefaultHarnessGlobs)
	payload, _ := json.Marshal(map[string]string{
		"command": `php -r 'file_put_contents("internal/adjudicator/x.go", $x);'`,
	})
	ev := abi.Event{
		Kind: abi.EvDecide,
		Call: &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: payload},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Emit(ev)
	}
}

// BenchmarkHarvester_Emit_FilterDeny measures short-circuit bypass for non-admitted events.
func BenchmarkHarvester_Emit_FilterDeny(b *testing.B) {
	corpus := NewNearMissCorpus()
	h := NewHarvester(corpus, DefaultHarnessGlobs)
	payload, _ := json.Marshal(map[string]string{
		"command": `php -r 'file_put_contents("internal/adjudicator/x.go", $x);'`,
	})
	ev := abi.Event{
		Kind: abi.EvDecide,
		Call: &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: payload},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictDeny},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Emit(ev)
	}
}

// BenchmarkHarvester_Emit_FilterNonDecide measures short-circuit bypass for non-decide event kinds.
func BenchmarkHarvester_Emit_FilterNonDecide(b *testing.B) {
	corpus := NewNearMissCorpus()
	h := NewHarvester(corpus, DefaultHarnessGlobs)
	payload, _ := json.Marshal(map[string]string{
		"command": `php -r 'file_put_contents("internal/adjudicator/x.go", $x);'`,
	})
	ev := abi.Event{
		Kind: abi.EvDispatch,
		Call: &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: payload},
		},
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		h.Emit(ev)
	}
}

// BenchmarkNearMissCorpus_Snapshot measures thread-safe extraction of captured corpus rows.
func BenchmarkNearMissCorpus_Snapshot(b *testing.B) {
	corpus := NewNearMissCorpus()
	for i := 0; i < 50; i++ {
		corpus.add(NearMiss{
			Call: Call{
				Tool:    "Bash",
				Arg:     "command",
				Command: fmt.Sprintf("ruby -e 'write(%d)'", i),
			},
			GuardedGlob: "internal/adjudicator/",
		})
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows := corpus.Rows()
		if len(rows) != 50 {
			b.Fatalf("expected 50 rows, got %d", len(rows))
		}
	}
}

// BenchmarkWriteVerb measures write verb extraction across representative shell commands.
func BenchmarkWriteVerb(b *testing.B) {
	cmds := []string{
		`ruby -e 'File.write("internal/adjudicator/x.go","")'`,
		`python3 -c 'import os; os.remove("...")'`,
		`sponge internal/adjudicator/z.go`,
		`sed -i s/a/b/ file.go`,
		`ls -la`,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, cmd := range cmds {
			_ = writeVerb(cmd)
		}
	}
}

// BenchmarkSynthRegex measures synthesis and regex escaping for rule predicates.
func BenchmarkSynthRegex(b *testing.B) {
	globs := []string{"internal/adjudicator/", "internal/abi/", "dos.toml"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = synthRegex("ruby -e", globs)
	}
}
