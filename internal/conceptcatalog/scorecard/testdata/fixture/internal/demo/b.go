package demo

// KVCache holds engine pages; PromptCache is the provider-side twin.
type KVCache struct{ CacheTTL5m, CacheTTL1h int }

type PromptCache struct{ cacheMeta string }

func MetaCache() string { return "FAK_DEMO_CACHE_TTL" }

func CacheEvictor() {}

func cacheIgnored() {}

func CacheGateway() {}

type Decision struct{ AdmissionGate bool }

func GatePolicy() {}

type SessionRef struct{ SessionRow, SessionKey, SessionKeys int }

func SessionSweep() {}

func SessionEtat() {}

var other = "unrelated.key"
