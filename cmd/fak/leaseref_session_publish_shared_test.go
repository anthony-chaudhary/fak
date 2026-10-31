package main

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/leaserefcli"
)

// fak-test:runtime unit est=1ms lane=default
func TestLeaserefSessionPublishSharedSpelling(t *testing.T) {
	for _, args := range [][]string{{"--help"}, {}, {"--session", "invalid/id"}, {"--ttl", "bad"}, {"unexpected"}} {
		var oldOut, oldErr, leafOut, leafErr bytes.Buffer
		old := runLeaseref(&oldOut, &oldErr, append([]string{"session-publish"}, args...))
		leaf := leaserefcli.RunSessionPublish(&leafOut, &leafErr, args)
		if old != leaf || oldOut.String() != leafOut.String() || oldErr.String() != leafErr.String() {
			t.Fatalf("args=%v existing=(%d,%q,%q) leaf=(%d,%q,%q)", args, old, oldOut.String(), oldErr.String(), leaf, leafOut.String(), leafErr.String())
		}
	}
}
