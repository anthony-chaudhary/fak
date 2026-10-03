package model

var beginSessionGPUKeepAlive = func(*Session) func() { return func() {} }

// BeginGPUKeepAlive opens a balanced optional Metal generation scope. Manual
// Step callers are covered per token; generation loops can hold it across gaps.
// The returned function must be deferred, including on error and cancellation.
func (s *Session) BeginGPUKeepAlive() func() { return beginSessionGPUKeepAlive(s) }
