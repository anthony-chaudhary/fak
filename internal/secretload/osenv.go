package secretload

import "os"

// OSEnv is a SecretSource backed by the process environment (os.LookupEnv). It is the
// default, highest-priority source in most loaders: an operator-set env var overrides a
// stored or .env value. An optional Prefix scopes lookups to a namespace (e.g. "FAK_"),
// so Lookup("API_KEY") reads FAK_API_KEY.
//
// An env var set to the EMPTY string is treated as absent (not a hit), so an exported-but-
// blank variable cannot shadow a real value in a lower-priority source.
type OSEnv struct {
	// Prefix, if non-empty, is prepended to every looked-up key.
	Prefix string
	// lookup is the env reader, injectable for tests; nil means os.LookupEnv.
	lookup func(string) (string, bool)
}

// OSEnvSource returns an OSEnv reading the real process environment with no prefix.
func OSEnvSource() *OSEnv { return &OSEnv{} }

// Name identifies the source for diagnostics.
func (e *OSEnv) Name() string { return "os-env" }

// Lookup reads Prefix+key from the environment, reporting a hit only for a non-empty
// value.
func (e *OSEnv) Lookup(key string) (string, bool) {
	get := e.lookup
	if get == nil {
		get = os.LookupEnv
	}
	v, ok := get(e.Prefix + key)
	if !ok || v == "" {
		return "", false
	}
	return v, true
}
