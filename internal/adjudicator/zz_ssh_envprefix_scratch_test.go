package adjudicator

import (
	"fmt"
	"testing"
)

func commandSelfModifyNoStrip(cmd string, globs []string) string {
	if !commandWrites(cmd) {
		return ""
	}
	for _, target := range commandWriteTargets(cmd) {
		if g := matchGlob(target, globs); g != "" {
			return g
		}
	}
	return ""
}

// scratch: prove removing the strip keeps both existing corpora correct and closes
// the laundering hole. Prints FAIL lines only for wrong verdicts.
func TestScratchRemovalSafe(t *testing.T) {
	cred := guardCredentialGlobs
	def := DefaultPolicy().SelfModifyGlobs

	// #1086 allow-cases (TestSSHIdentityReadIsNotSelfModify) — must ALLOW without strip.
	allow1086 := []string{
		"ssh -i ~/.ssh/id_rsa anthony@node 'cd ~/fak && cp build/out /tmp/x && go test ./internal/metalgemm'",
		"ssh -i ~/.ssh/id_ed25519 anthony@node 'go build ./... && mv a b'",
		"scp -i ~/.ssh/id_ed25519 ./fak anthony@node:~/",
		"scp -i ~/.ssh/id_rsa anthony@node:~/result.json ./",
		"ssh -i~/.ssh/id_rsa anthony@node 'cp a b'",
	}
	// real-key-write deny-cases (TestRealKeyWriteStillDenied) — must DENY without strip.
	deny := []string{
		"tee ~/.ssh/id_rsa < attacker_key",
		"cp attacker_key ~/.ssh/id_ed25519",
		"echo pwn > ~/.ssh/id_rsa",
		"ssh -i ~/.ssh/id_rsa anthony@node 'cp evil ~/.ssh/id_ed25519'",
		"cp payload .git/hooks/pre-commit",
	}
	// laundering cases — must DENY without strip (they wrongly ALLOW WITH strip).
	launder := []string{
		"sed -i s/a/b/ internal/abi/x.go && ssh host echo done",
		"perl -i -pe s/a/b/ internal/kernel/x.go ; scp -i ~/.ssh/id_rsa f host:~/",
		"ssh -i ~/.ssh/id_rsa host 'echo hi' && sed -i s/x/y/ internal/adjudicator/decide.go",
	}
	fail := 0
	for _, c := range allow1086 {
		if g := commandSelfModifyNoStrip(c, cred); g != "" {
			fail++
			fmt.Printf("FAIL want ALLOW got DENY:%s  %s\n", g, c)
		}
	}
	for _, c := range deny {
		if g := commandSelfModifyNoStrip(c, cred); g == "" {
			fail++
			fmt.Printf("FAIL want DENY got ALLOW  %s\n", c)
		}
	}
	for _, c := range launder {
		if g := commandSelfModifyNoStrip(c, def); g == "" {
			fail++
			fmt.Printf("FAIL want DENY got ALLOW (launder)  %s\n", c)
		}
	}
	fmt.Printf("RESULT: %d wrong verdicts under no-strip\n", fail)
}
