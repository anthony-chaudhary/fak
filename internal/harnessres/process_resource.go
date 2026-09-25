package harnessres

// ProcessResource is the portable per-PID view used by the guarded process-tree
// monitor. It deliberately exposes presence bits: a platform without a reader
// must not turn an unavailable counter into an observed zero.
type ProcessResource struct {
	CPUSeconds   float64
	RSSBytes     uint64
	PeakRSSBytes uint64
	IOReadBytes  uint64
	IOWriteBytes uint64
	HaveCPU      bool
	HaveRSS      bool
	HavePeakRSS  bool
	HaveIO       bool
}

// ReadProcessResource reads another process without granting termination or
// mutation rights. It is the small host seam shared by fleet sampling and the
// guard's owned-tree resource accounting.
func ReadProcessResource(pid int) (ProcessResource, bool) {
	s, ok := readProcPID(pid)
	if !ok {
		return ProcessResource{}, false
	}
	return ProcessResource{
		CPUSeconds:   s.cpuUser.Seconds() + s.cpuSys.Seconds(),
		RSSBytes:     s.rss,
		PeakRSSBytes: s.peakRSS,
		IOReadBytes:  s.ioRead,
		IOWriteBytes: s.ioWrite,
		HaveCPU:      s.haveCPU,
		HaveRSS:      s.haveRSS,
		HavePeakRSS:  s.havePeakRSS,
		HaveIO:       s.haveIO,
	}, true
}
