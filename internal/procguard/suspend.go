package procguard

// SuspendProcess pauses execution of the process identified by pid.
func SuspendProcess(pid int) error {
	return suspendProcess(pid)
}

// ResumeProcess resumes execution of the process identified by pid.
func ResumeProcess(pid int) error {
	return resumeProcess(pid)
}
