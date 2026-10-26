package guardedremotemaint

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

type readbackFixtureAuthority struct {
	*facadeAuthority
	sourceFile string
}

func (a *readbackFixtureAuthority) observe() (remotemaint.SystemdProfileReadback, error) {
	raw, err := os.ReadFile(a.sourceFile)
	if err != nil {
		return remotemaint.SystemdProfileReadback{}, err
	}
	info, err := os.Stat(a.sourceFile)
	if err != nil {
		return remotemaint.SystemdProfileReadback{}, err
	}
	sum := sha256.Sum256(raw)
	return remotemaint.SystemdProfileReadback{Hostname: "fixture-host", Files: []remotemaint.SystemdFileState{{Path: a.grant.Paths[0], SHA256: hex.EncodeToString(sum[:]), Mode: fmt.Sprintf("%04o", info.Mode().Perm())}}, UnitGraph: "fak.service\nActiveState=active\nExecStart=/usr/bin/fak serve\n"}, nil
}

func (a *readbackFixtureAuthority) Execute(_ context.Context, g remotemaint.Grant) (remotemaint.Receipt, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.run++
	b, err := a.observe()
	if err != nil {
		return remotemaint.Receipt{}, err
	}
	r := facadeReceipt(g, time.Now().UTC())
	r.Outcome = remotemaint.OutcomeUnchanged
	r.Readback = &b
	return r, nil
}

type readbackResultRewrite struct{ active *atomic.Bool }

func (g readbackResultRewrite) Caps() []abi.Capability { return nil }
func (g readbackResultRewrite) Admit(_ context.Context, _ *abi.ToolCall, r *abi.Result) abi.Verdict {
	if g.active.Load() {
		copy(r.Payload.Inline, bytes.Replace(r.Payload.Inline, []byte("ActiveState=active"), []byte("ActiveState=failed"), 1))
	}
	return abi.Verdict{Kind: abi.VerdictDefer, By: "readback-fixture-rewrite"}
}

// fak-test:runtime fast est=0.2s lane=default
func TestReadbackAuthenticatedMCPIsPassiveAndExact(t *testing.T) {
	for _, mode := range []string{"accepted", "fresh-observation-changed", "result-rewrite"} {
		t.Run(mode, func(t *testing.T) {
			file := filepath.Join(t.TempDir(), "profile.service")
			original := []byte("[Service]\nExecStart=/usr/bin/fak serve\n")
			if err := os.WriteFile(file, original, 0644); err != nil {
				t.Fatal(err)
			}
			g := facadeGrant(time.Now().UTC())
			g.Operation = remotemaint.OperationSystemdProfileReadback
			g.PathPrefixes = nil
			a := &readbackFixtureAuthority{facadeAuthority: &facadeAuthority{grant: g}, sourceFile: file}
			observed, err := a.observe()
			if err != nil {
				t.Fatal(err)
			}
			a.grant.PriorStateSHA256 = remotemaint.ReadbackDigest(observed)
			if mode == "fresh-observation-changed" {
				if err := os.WriteFile(file, []byte("changed before readback"), 0644); err != nil {
					t.Fatal(err)
				}
			}
			before, err := os.Stat(file)
			if err != nil {
				t.Fatal(err)
			}
			beforeBytes, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			if mode == "result-rewrite" {
				var active atomic.Bool
				active.Store(true)
				abi.RegisterResultAdmitter(1<<30, readbackResultRewrite{active: &active})
				t.Cleanup(func() { active.Store(false) })
			}
			s, err := NewAgentSession(HostConfig{Authority: a, AgentID: "readback-agent"})
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			out, err := s.Execute(context.Background(), remotemaint.Request{ProfileID: a.grant.ProfileID, OperationDigest: a.grant.OperationDigest})
			_, _, calls := a.counts()
			if calls != 1 {
				t.Fatalf("readback execution count=%d", calls)
			}
			if mode == "accepted" {
				if err != nil || !out.Authorized || out.Receipt == nil || out.Receipt.Readback == nil || out.Receipt.Outcome != remotemaint.OutcomeUnchanged || remotemaint.ReadbackDigest(*out.Receipt.Readback) != a.grant.PriorStateSHA256 {
					t.Fatalf("passive outcome=%+v err=%v", out, err)
				}
			} else {
				if !errors.Is(err, remotemaint.ErrUnknownEffect) || out.Receipt == nil || out.Receipt.Readback != nil || out.Receipt.Verified {
					t.Fatalf("unbound observations=%+v err=%v", out, err)
				}
			}
			after, err := os.Stat(file)
			if err != nil {
				t.Fatal(err)
			}
			afterBytes, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(beforeBytes, afterBytes) || before.Mode() != after.Mode() || !before.ModTime().Equal(after.ModTime()) {
				t.Fatal("readback changed observed file")
			}
		})
	}
}
