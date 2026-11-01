//go:build fak_native_output_isolation_qualification

package modelengine

// Explicit opt-in only. Record the selected Go build command and binary build
// info; source presence alone is not execution or performance evidence.
const nativeOutputIsolationEnabled = true
