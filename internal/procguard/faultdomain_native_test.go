package procguard

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestFaultDomainNativeIsolation(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows Job Object contract witness")
	}
	if os.Getenv("FAK_FAULTDOMAIN_HELPER") != "" {
		runFaultDomainHelper()
		return
	}
	survivor := exec.Command(os.Args[0], "-test.run=^TestFaultDomainNativeIsolation$")
	survivor.Env = append(os.Environ(), "FAK_FAULTDOMAIN_HELPER=survivor")
	stdout, err := survivor.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	if err = survivor.Start(); err != nil {
		t.Fatal(err)
	}
	defer survivor.Process.Kill()
	scanner := bufio.NewScanner(stdout)
	if !scanner.Scan() || scanner.Text() != "checkpoint-1" {
		t.Fatalf("survivor did not start: %q", scanner.Text())
	}

	hog := exec.Command(os.Args[0], "-test.run=^TestFaultDomainNativeIsolation$")
	hog.Env = append(os.Environ(), "FAK_FAULTDOMAIN_HELPER=hog")
	started := time.Now()
	err = hog.Run()
	if err == nil {
		t.Fatal("memory hog escaped its fault-domain limit")
	}
	if time.Since(started) > 5*time.Second {
		t.Fatal("memory fault containment exceeded 5s latency budget")
	}
	if !scanner.Scan() || scanner.Text() != "checkpoint-2" {
		t.Fatalf("unrelated owner stopped checkpointing: %q", scanner.Text())
	}
}

func TestFaultDomainProcessLimitIncludesDescendants(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows Job Object contract witness")
	}
	if os.Getenv("FAK_FAULTDOMAIN_HELPER") != "" {
		runFaultDomainHelper()
		return
	}
	child := exec.Command(os.Args[0], "-test.run=^TestFaultDomainProcessLimitIncludesDescendants$")
	child.Env = append(os.Environ(), "FAK_FAULTDOMAIN_HELPER=fork")
	out, err := child.CombinedOutput()
	if err != nil {
		t.Fatalf("helper: %v: %s", err, out)
	}
	if strings.TrimSpace(string(out)) != "grandchild-contained" {
		t.Fatalf("unexpected witness: %q", out)
	}
}

func runFaultDomainHelper() {
	mode := os.Getenv("FAK_FAULTDOMAIN_HELPER")
	envelope := ResourceEnvelope{MemoryBytes: 256 << 20, ProcessCount: 8}
	if mode == "hog" {
		envelope.MemoryBytes = 48 << 20
	}
	if mode == "fork" {
		envelope.ProcessCount = 1
	}
	d, err := NewFaultDomain(mode, envelope)
	if err != nil {
		panic(err)
	}
	if _, err = d.BindCurrent(); err != nil {
		panic(err)
	}
	switch mode {
	case "survivor":
		fmt.Println("checkpoint-1")
		time.Sleep(750 * time.Millisecond)
		fmt.Println("checkpoint-2")
		time.Sleep(time.Second)
	case "hog":
		blocks := make([][]byte, 0, 64)
		for {
			b := make([]byte, 4<<20)
			for i := range b {
				b[i] = 1
			}
			blocks = append(blocks, b)
		}
	case "fork":
		grandchild := exec.Command(os.Args[0], "-test.run=^$")
		if err := grandchild.Start(); err == nil {
			_ = grandchild.Process.Kill()
			panic("grandchild escaped process limit")
		}
		fmt.Println("grandchild-contained")
	}
	os.Exit(0)
}
