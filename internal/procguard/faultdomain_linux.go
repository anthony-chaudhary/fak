//go:build linux

package procguard

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type cgroupFaultDomain struct {
	path  string
	bound bool
}

func newNativeFaultDomain(owner string, e ResourceEnvelope) (nativeFaultDomain, FaultDomainReceipt, error) {
	root := "/sys/fs/cgroup"
	if _, err := os.Stat(filepath.Join(root, "cgroup.controllers")); err != nil {
		limits := requestedSupport(e, nil)
		return &observeFaultDomain{}, FaultDomainReceipt{Mode: EnforcementObserveOnly, Primitive: "process-observation", Limits: limits}, nil
	}
	path := filepath.Join(root, "fak-procguard-"+sanitizeFaultDomainID(owner))
	if err := os.Mkdir(path, 0750); err != nil {
		limits := requestedSupport(e, nil)
		return &observeFaultDomain{}, FaultDomainReceipt{Mode: EnforcementObserveOnly, Primitive: "cgroup-v2-unavailable", Limits: limits}, nil
	}
	c := &cgroupFaultDomain{path: path}
	enforced := map[string]string{}
	write := func(name, value, resource string) error {
		if err := os.WriteFile(filepath.Join(path, name), []byte(value), 0644); err != nil {
			return err
		}
		enforced[resource] = name
		return nil
	}
	if e.MemoryBytes > 0 {
		if err := write("memory.max", strconv.FormatUint(e.MemoryBytes, 10), "memory"); err != nil {
			_ = c.close()
			return nil, FaultDomainReceipt{}, err
		}
	}
	if e.ProcessCount > 0 {
		if err := write("pids.max", strconv.FormatUint(uint64(e.ProcessCount), 10), "processes"); err != nil {
			_ = c.close()
			return nil, FaultDomainReceipt{}, err
		}
	}
	if e.CPUPercent > 0 {
		quota := int64(e.CPUPercent) * 1000
		if err := write("cpu.max", fmt.Sprintf("%d 100000", quota), "cpu_share"); err != nil {
			_ = c.close()
			return nil, FaultDomainReceipt{}, err
		}
	}
	limits := requestedSupport(e, enforced)
	return c, FaultDomainReceipt{Mode: modeFor(limits), Primitive: "linux-cgroup-v2", Limits: limits}, nil
}
func (c *cgroupFaultDomain) bindCurrent() error {
	if err := os.WriteFile(filepath.Join(c.path, "cgroup.procs"), []byte(strconv.Itoa(os.Getpid())), 0644); err != nil {
		return err
	}
	c.bound = true
	return nil
}
func (c *cgroupFaultDomain) usage() (ResourceUsage, error) {
	mem, err := readUint(filepath.Join(c.path, "memory.current"))
	if err != nil {
		return ResourceUsage{}, err
	}
	procs, err := readLines(filepath.Join(c.path, "cgroup.procs"))
	if err != nil {
		return ResourceUsage{}, err
	}
	cpu, err := readCPUUsec(filepath.Join(c.path, "cpu.stat"))
	if err != nil {
		return ResourceUsage{}, err
	}
	return ResourceUsage{MemoryBytes: mem, Processes: uint64(procs), CPUTime: time.Duration(cpu) * time.Microsecond}, nil
}
func (c *cgroupFaultDomain) close() error { return os.Remove(c.path) }

type observeFaultDomain struct{}

func (*observeFaultDomain) bindCurrent() error {
	return errors.New("cgroup v2 is not delegated to this process")
}
func (*observeFaultDomain) usage() (ResourceUsage, error) {
	return ResourceUsage{}, errors.New("fault-domain usage unavailable without cgroup v2")
}
func (*observeFaultDomain) close() error { return nil }
func readUint(path string) (uint64, error) {
	b, e := os.ReadFile(path)
	if e != nil {
		return 0, e
	}
	return strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
}
func readLines(path string) (int, error) {
	f, e := os.Open(path)
	if e != nil {
		return 0, e
	}
	defer f.Close()
	n := 0
	s := bufio.NewScanner(f)
	for s.Scan() {
		n++
	}
	return n, s.Err()
}
func readCPUUsec(path string) (uint64, error) {
	f, e := os.Open(path)
	if e != nil {
		return 0, e
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		parts := strings.Fields(s.Text())
		if len(parts) == 2 && parts[0] == "usage_usec" {
			return strconv.ParseUint(parts[1], 10, 64)
		}
	}
	if e := s.Err(); e != nil {
		return 0, e
	}
	return 0, errors.New("cpu.stat lacks usage_usec")
}
