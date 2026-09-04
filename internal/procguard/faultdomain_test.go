package procguard

import (
	"os"
	"os/exec"
	"runtime"
	"testing"
	"time"
)

func TestFaultDomainReceiptNeverOverclaimsUnsupportedLimits(t *testing.T) {
	d, err := NewFaultDomain("instance/a", ResourceEnvelope{MemoryBytes: 64 << 20, ProcessCount: 4, OpenFiles: 32, ScratchBytes: 1 << 20, CoordinatorReserve: ResourceReserve{MemoryBytes: 32 << 20, CPUPercent: 10, ProcessCount: 2}})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	r := d.Receipt()
	if r.OwnerID != "instance/a" {
		t.Fatalf("owner=%q", r.OwnerID)
	}
	if r.Primitive == "" || r.Mode == "" {
		t.Fatalf("untyped receipt: %+v", r)
	}
	if r.Mode == EnforcementHard {
		t.Fatalf("hard mode overclaims unsupported FD/scratch controls: %+v", r)
	}
	unsupported := map[string]bool{}
	for _, l := range r.Limits {
		if !l.Enforced {
			unsupported[l.Resource] = true
		}
	}
	if !unsupported["open_files"] || !unsupported["scratch"] {
		t.Fatalf("unsupported limits hidden: %+v", r.Limits)
	}
	if r.CoordinatorReserve.MemoryBytes != 32<<20 || r.CoordinatorReserve.ProcessCount != 2 {
		t.Fatalf("reserve missing: %+v", r.CoordinatorReserve)
	}
	if r.InvalidatingAssumption == "" {
		t.Fatal("missing invalidating assumption")
	}
}

func TestFaultDomainsHaveIndependentNativeIdentity(t *testing.T) {
	a, err := NewFaultDomain("a", ResourceEnvelope{MemoryBytes: 128 << 20, ProcessCount: 8})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := NewFaultDomain("b", ResourceEnvelope{MemoryBytes: 128 << 20, ProcessCount: 8})
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	aReceipt, bReceipt := a.Receipt(), b.Receipt()
	if aReceipt.OwnerID == bReceipt.OwnerID {
		t.Fatal("owner identities collapsed")
	}
	if runtime.GOOS == "windows" && (aReceipt.Mode != EnforcementHard || bReceipt.Mode != EnforcementHard) {
		t.Fatalf("Windows must expose hard native isolation: a=%+v b=%+v", aReceipt, bReceipt)
	}
	if aReceipt.Mode == EnforcementHard && bReceipt.Mode == EnforcementHard && a.native == b.native {
		t.Fatal("hard-isolated owners unexpectedly share a native fault domain")
	}
}

func TestFaultDomainBindCurrentAndUsage(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("native bind witness is Windows-specific on this host")
	}
	if os.Getenv("FAK_FAULTDOMAIN_BIND_HELPER") == "1" {
		d, err := NewFaultDomain("bind-current", ResourceEnvelope{MemoryBytes: 512 << 20, ProcessCount: 32, CPUPercent: 100, CPUTime: time.Minute})
		if err != nil {
			t.Fatal(err)
		}
		defer d.Close()
		r, err := d.BindCurrent()
		if err != nil {
			t.Fatal(err)
		}
		if !r.DescendantsBound || r.Primitive != "windows-job-object" {
			t.Fatalf("receipt=%+v", r)
		}
		usage, err := d.Usage()
		if err != nil {
			t.Fatal(err)
		}
		if usage.Processes < 1 {
			t.Fatalf("current process absent: %+v", usage)
		}
		pressure, err := d.Pressure()
		if err != nil {
			t.Fatal(err)
		}
		if len(pressure) != 3 {
			t.Fatalf("pressure=%+v", pressure)
		}
		for _, p := range pressure {
			if p.OwnerID != "bind-current" {
				t.Fatalf("event lost owner: %+v", p)
			}
		}
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestFaultDomainBindCurrentAndUsage$")
	cmd.Env = append(os.Environ(), "FAK_FAULTDOMAIN_BIND_HELPER=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("helper: %v: %s", err, out)
	}
}
