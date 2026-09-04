package procguard

import (
	"runtime"
	"runtime/debug"
)

// YieldMemory releases unused memory back to the operating system.
// It forces runtime garbage collection, signals the runtime to return physical
// memory to the OS via debug.FreeOSMemory(), and on Windows empties the working
// set of the current process and any provided valid process IDs.
func YieldMemory(pids ...int) {
	runtime.GC()
	debug.FreeOSMemory()
	yieldWorkingSets(pids...)
}
