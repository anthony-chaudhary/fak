package metrics

import "sync/atomic"

type ShardMetrics struct {
	shardID          int
	gets             atomic.Int64
	sets             atomic.Int64
	deletes          atomic.Int64
	exists           atomic.Int64
	existsHits       atomic.Int64
	existsMisses     atomic.Int64
	hits             atomic.Int64
	misses           atomic.Int64
	evictions        atomic.Int64
	ttlExpirations   atomic.Int64
	bytesIn          atomic.Int64
	bytesOut         atomic.Int64
	rdmaReadBytesOut atomic.Int64
	keyBytesIn       atomic.Int64
	valueBytesIn     atomic.Int64

	evictionsKeyPressure   atomic.Int64
	evictionsValuePressure atomic.Int64
	evictionsFailed        atomic.Int64
	evictionsLeaseSkip     atomic.Int64
	evictionsRebalance     atomic.Int64

	promotions atomic.Int64

	migrationActive       atomic.Int32
	migrationDurationMs   atomic.Int64
	migrationEntries      atomic.Int64
	migrationsTotal       atomic.Int64
	migrationPreRegWaitMs atomic.Int64

	migrationP99Us     atomic.Int64
	migrationBatchSize atomic.Int32

	opsDropped atomic.Int64

	oomRejections         atomic.Int64
	lifetimeOOMRejections atomic.Int64
	memoryPressure        atomic.Int32

	panics       atomic.Int64
	circuitTrips atomic.Int64
	shardHalted  atomic.Int32

	lifetimeGets                   atomic.Int64
	lifetimeSets                   atomic.Int64
	lifetimeDeletes                atomic.Int64
	lifetimeExists                 atomic.Int64
	lifetimeExistsHits             atomic.Int64
	lifetimeExistsMisses           atomic.Int64
	lifetimeHits                   atomic.Int64
	lifetimeMisses                 atomic.Int64
	lifetimeEvictions              atomic.Int64
	lifetimeTTLExpirations         atomic.Int64
	lifetimeBytesIn                atomic.Int64
	lifetimeBytesOut               atomic.Int64
	lifetimeRDMAReadBytesOut       atomic.Int64
	lifetimeKeyBytesIn             atomic.Int64
	lifetimeValueBytesIn           atomic.Int64
	lifetimeEvictionsKeyPressure   atomic.Int64
	lifetimeEvictionsValuePressure atomic.Int64
	lifetimeEvictionsFailed        atomic.Int64
	lifetimeEvictionsLeaseSkip     atomic.Int64
	lifetimeEvictionsRebalance     atomic.Int64
	lifetimePromotions             atomic.Int64
}

func NewShardMetrics(shardID int) *ShardMetrics {
	return &ShardMetrics{shardID: shardID}
}

func (m *ShardMetrics) IncrGets()           { m.gets.Add(1); m.lifetimeGets.Add(1) }
func (m *ShardMetrics) IncrSets()           { m.sets.Add(1); m.lifetimeSets.Add(1) }
func (m *ShardMetrics) IncrDeletes()        { m.deletes.Add(1); m.lifetimeDeletes.Add(1) }
func (m *ShardMetrics) IncrExists()         { m.exists.Add(1); m.lifetimeExists.Add(1) }
func (m *ShardMetrics) IncrExistsHits()     { m.existsHits.Add(1); m.lifetimeExistsHits.Add(1) }
func (m *ShardMetrics) IncrExistsMisses()   { m.existsMisses.Add(1); m.lifetimeExistsMisses.Add(1) }
func (m *ShardMetrics) IncrHits()           { m.hits.Add(1); m.lifetimeHits.Add(1) }
func (m *ShardMetrics) IncrMisses()         { m.misses.Add(1); m.lifetimeMisses.Add(1) }
func (m *ShardMetrics) IncrEvictions()      { m.evictions.Add(1); m.lifetimeEvictions.Add(1) }
func (m *ShardMetrics) IncrTTLExpirations() { m.ttlExpirations.Add(1); m.lifetimeTTLExpirations.Add(1) }
func (m *ShardMetrics) IncrEvictionsKeyPressure() {
	m.evictionsKeyPressure.Add(1)
	m.lifetimeEvictionsKeyPressure.Add(1)
}
func (m *ShardMetrics) IncrEvictionsValuePressure() {
	m.evictionsValuePressure.Add(1)
	m.lifetimeEvictionsValuePressure.Add(1)
}
func (m *ShardMetrics) IncrEvictionsFailed() {
	m.evictionsFailed.Add(1)
	m.lifetimeEvictionsFailed.Add(1)
}
func (m *ShardMetrics) IncrEvictionsLeaseSkip() {
	m.evictionsLeaseSkip.Add(1)
	m.lifetimeEvictionsLeaseSkip.Add(1)
}
func (m *ShardMetrics) IncrEvictionsRebalance() {
	m.evictionsRebalance.Add(1)
	m.lifetimeEvictionsRebalance.Add(1)
}
func (m *ShardMetrics) IncrPromotions()           { m.promotions.Add(1); m.lifetimePromotions.Add(1) }
func (m *ShardMetrics) IncrOpsDropped()           { m.opsDropped.Add(1) }
func (m *ShardMetrics) IncrPanics()               { m.panics.Add(1) }
func (m *ShardMetrics) IncrCircuitTrips()         { m.circuitTrips.Add(1) }
func (m *ShardMetrics) SetShardHalted(v int32)    { m.shardHalted.Store(v) }
func (m *ShardMetrics) ShardHalted() int32        { return m.shardHalted.Load() }
func (m *ShardMetrics) IncrOOMRejections()        { m.oomRejections.Add(1); m.lifetimeOOMRejections.Add(1) }
func (m *ShardMetrics) SetMemoryPressure(v int32) { m.memoryPressure.Store(v) }

func (m *ShardMetrics) AddGets(n int64)         { m.gets.Add(n); m.lifetimeGets.Add(n) }
func (m *ShardMetrics) AddHits(n int64)         { m.hits.Add(n); m.lifetimeHits.Add(n) }
func (m *ShardMetrics) AddMisses(n int64)       { m.misses.Add(n); m.lifetimeMisses.Add(n) }
func (m *ShardMetrics) AddExists(n int64)       { m.exists.Add(n); m.lifetimeExists.Add(n) }
func (m *ShardMetrics) AddExistsHits(n int64)   { m.existsHits.Add(n); m.lifetimeExistsHits.Add(n) }
func (m *ShardMetrics) AddExistsMisses(n int64) { m.existsMisses.Add(n); m.lifetimeExistsMisses.Add(n) }
func (m *ShardMetrics) AddSets(n int64)         { m.sets.Add(n); m.lifetimeSets.Add(n) }
func (m *ShardMetrics) AddDeletes(n int64)      { m.deletes.Add(n); m.lifetimeDeletes.Add(n) }

func (m *ShardMetrics) SetMigrationActive(v int32)       { m.migrationActive.Store(v) }
func (m *ShardMetrics) SetMigrationDurationMs(v int64)   { m.migrationDurationMs.Store(v) }
func (m *ShardMetrics) SetMigrationEntries(v int64)      { m.migrationEntries.Store(v) }
func (m *ShardMetrics) IncrMigrationsTotal()             { m.migrationsTotal.Add(1) }
func (m *ShardMetrics) SetMigrationPreRegWaitMs(v int64) { m.migrationPreRegWaitMs.Store(v) }

func (m *ShardMetrics) SetMigrationP99Us(v int64)     { m.migrationP99Us.Store(v) }
func (m *ShardMetrics) SetMigrationBatchSize(v int32) { m.migrationBatchSize.Store(v) }
func (m *ShardMetrics) MigrationP99Us() int64         { return m.migrationP99Us.Load() }
func (m *ShardMetrics) MigrationBatchSize() int32     { return m.migrationBatchSize.Load() }
func (m *ShardMetrics) MigrationActive() int32        { return m.migrationActive.Load() }
func (m *ShardMetrics) MigrationDurationMs() int64    { return m.migrationDurationMs.Load() }
func (m *ShardMetrics) MigrationEntries() int64       { return m.migrationEntries.Load() }
func (m *ShardMetrics) MigrationsTotal() int64        { return m.migrationsTotal.Load() }
func (m *ShardMetrics) MigrationPreRegWaitMs() int64  { return m.migrationPreRegWaitMs.Load() }

func (m *ShardMetrics) AddBytesIn(n int64)  { m.bytesIn.Add(n); m.lifetimeBytesIn.Add(n) }
func (m *ShardMetrics) AddBytesOut(n int64) { m.bytesOut.Add(n); m.lifetimeBytesOut.Add(n) }
func (m *ShardMetrics) AddRDMAReadBytesOut(n int64) {
	m.rdmaReadBytesOut.Add(n)
	m.lifetimeRDMAReadBytesOut.Add(n)
}
func (m *ShardMetrics) AddKeyBytesIn(n int64)   { m.keyBytesIn.Add(n); m.lifetimeKeyBytesIn.Add(n) }
func (m *ShardMetrics) AddValueBytesIn(n int64) { m.valueBytesIn.Add(n); m.lifetimeValueBytesIn.Add(n) }

func (m *ShardMetrics) Gets() int64                   { return m.gets.Load() }
func (m *ShardMetrics) Sets() int64                   { return m.sets.Load() }
func (m *ShardMetrics) Deletes() int64                { return m.deletes.Load() }
func (m *ShardMetrics) Exists() int64                 { return m.exists.Load() }
func (m *ShardMetrics) ExistsHits() int64             { return m.existsHits.Load() }
func (m *ShardMetrics) ExistsMisses() int64           { return m.existsMisses.Load() }
func (m *ShardMetrics) Hits() int64                   { return m.hits.Load() }
func (m *ShardMetrics) Misses() int64                 { return m.misses.Load() }
func (m *ShardMetrics) Evictions() int64              { return m.evictions.Load() }
func (m *ShardMetrics) TTLExpirations() int64         { return m.ttlExpirations.Load() }
func (m *ShardMetrics) EvictionsKeyPressure() int64   { return m.evictionsKeyPressure.Load() }
func (m *ShardMetrics) EvictionsValuePressure() int64 { return m.evictionsValuePressure.Load() }
func (m *ShardMetrics) EvictionsFailed() int64        { return m.evictionsFailed.Load() }
func (m *ShardMetrics) EvictionsLeaseSkip() int64     { return m.evictionsLeaseSkip.Load() }
func (m *ShardMetrics) EvictionsRebalance() int64     { return m.evictionsRebalance.Load() }
func (m *ShardMetrics) Promotions() int64             { return m.promotions.Load() }
func (m *ShardMetrics) OpsDropped() int64             { return m.opsDropped.Load() }
func (m *ShardMetrics) Panics() int64                 { return m.panics.Load() }
func (m *ShardMetrics) CircuitTrips() int64           { return m.circuitTrips.Load() }
func (m *ShardMetrics) OOMRejections() int64          { return m.oomRejections.Load() }
func (m *ShardMetrics) MemoryPressure() int32         { return m.memoryPressure.Load() }
func (m *ShardMetrics) BytesIn() int64                { return m.bytesIn.Load() }
func (m *ShardMetrics) BytesOut() int64               { return m.bytesOut.Load() }
func (m *ShardMetrics) RDMAReadBytesOut() int64       { return m.rdmaReadBytesOut.Load() }
func (m *ShardMetrics) KeyBytesIn() int64             { return m.keyBytesIn.Load() }
func (m *ShardMetrics) ValueBytesIn() int64           { return m.valueBytesIn.Load() }

func (m *ShardMetrics) LifetimeGets() int64             { return m.lifetimeGets.Load() }
func (m *ShardMetrics) LifetimeSets() int64             { return m.lifetimeSets.Load() }
func (m *ShardMetrics) LifetimeDeletes() int64          { return m.lifetimeDeletes.Load() }
func (m *ShardMetrics) LifetimeExists() int64           { return m.lifetimeExists.Load() }
func (m *ShardMetrics) LifetimeExistsHits() int64       { return m.lifetimeExistsHits.Load() }
func (m *ShardMetrics) LifetimeExistsMisses() int64     { return m.lifetimeExistsMisses.Load() }
func (m *ShardMetrics) LifetimeHits() int64             { return m.lifetimeHits.Load() }
func (m *ShardMetrics) LifetimeMisses() int64           { return m.lifetimeMisses.Load() }
func (m *ShardMetrics) LifetimeEvictions() int64        { return m.lifetimeEvictions.Load() }
func (m *ShardMetrics) LifetimeTTLExpirations() int64   { return m.lifetimeTTLExpirations.Load() }
func (m *ShardMetrics) LifetimeBytesIn() int64          { return m.lifetimeBytesIn.Load() }
func (m *ShardMetrics) LifetimeBytesOut() int64         { return m.lifetimeBytesOut.Load() }
func (m *ShardMetrics) LifetimeRDMAReadBytesOut() int64 { return m.lifetimeRDMAReadBytesOut.Load() }
func (m *ShardMetrics) LifetimeKeyBytesIn() int64       { return m.lifetimeKeyBytesIn.Load() }
func (m *ShardMetrics) LifetimeValueBytesIn() int64     { return m.lifetimeValueBytesIn.Load() }
func (m *ShardMetrics) LifetimeEvictionsKeyPressure() int64 {
	return m.lifetimeEvictionsKeyPressure.Load()
}
func (m *ShardMetrics) LifetimeEvictionsValuePressure() int64 {
	return m.lifetimeEvictionsValuePressure.Load()
}
func (m *ShardMetrics) LifetimeEvictionsFailed() int64    { return m.lifetimeEvictionsFailed.Load() }
func (m *ShardMetrics) LifetimeEvictionsLeaseSkip() int64 { return m.lifetimeEvictionsLeaseSkip.Load() }
func (m *ShardMetrics) LifetimeEvictionsRebalance() int64 { return m.lifetimeEvictionsRebalance.Load() }
func (m *ShardMetrics) LifetimePromotions() int64         { return m.lifetimePromotions.Load() }
func (m *ShardMetrics) LifetimeOOMRejections() int64      { return m.lifetimeOOMRejections.Load() }

func (m *ShardMetrics) ResetEpoch() {
	m.gets.Store(0)
	m.sets.Store(0)
	m.deletes.Store(0)
	m.exists.Store(0)
	m.existsHits.Store(0)
	m.existsMisses.Store(0)
	m.hits.Store(0)
	m.misses.Store(0)
	m.evictions.Store(0)
	m.ttlExpirations.Store(0)
	m.bytesIn.Store(0)
	m.bytesOut.Store(0)
	m.rdmaReadBytesOut.Store(0)
	m.keyBytesIn.Store(0)
	m.valueBytesIn.Store(0)
	m.evictionsKeyPressure.Store(0)
	m.evictionsValuePressure.Store(0)
	m.evictionsFailed.Store(0)
	m.evictionsLeaseSkip.Store(0)
	m.evictionsRebalance.Store(0)
	m.promotions.Store(0)
	m.oomRejections.Store(0)
	m.memoryPressure.Store(0)
	m.migrationActive.Store(0)
	m.migrationDurationMs.Store(0)
	m.migrationEntries.Store(0)
	m.migrationPreRegWaitMs.Store(0)
	m.migrationP99Us.Store(0)
	m.migrationBatchSize.Store(0)
}
