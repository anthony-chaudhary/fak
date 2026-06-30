// fak-native ultracode Workflow SEED (epic #1494 / C4 #1502).
// This is the template an ultracode session running INSIDE the fak kernel should
// emit by construction — fak self-index + fak memory algebra + shared-path leasing
// are first-class, not bolted on. A generated workflow that lacks any of the three
// fak concept classes below is "fak-blind" and is REFUSED by `fak workflow lint`.
//
// Concepts embedded (the lint checks for all three):
//   - SELF-INDEX : `fak index leaf|claims` / fak_index_*  (query, don't survey)
//   - MEMORY     : `fak memory run --driver recall|compact|clean` / fak_memory_*  (recall before, compact after)
//   - SHARED PATH: dos_arbitrate lease per agent so a spawn wave never collides

export const meta = {
  name: 'fak-native-review-fix',
  description: 'Review changed files for bugs and fix them — fak-native: self-index understand, per-agent memory recall/compact, arbitrated shared-path leases',
  phases: [
    { title: 'Understand', detail: 'fak self-index: lane ownership + shipped/stub status of the touched code' },
    { title: 'Lease', detail: 'dos_arbitrate a disjoint lease per file group so the wave shares the path without colliding' },
    { title: 'ReviewFix', detail: 'per group: recall memory -> review+fix -> compact memory' },
  ],
}

// ---- Understand: ask fak its own facts instead of re-surveying prose ----
phase('Understand')
const changed = await agent(
  `List the repo's changed files (\`git diff --name-only HEAD\`). For each, run \`fak index lane <path>\` to get its owning lane + (fak <leaf>) stamp, and \`fak index claims <lane>\` to learn what is SHIPPED vs STUB there (don't "fix" a stub as if it were shipped). ` +
  `End with one JSON line: GROUPS=[{lane, files:[...], stamp}]`,
  { label: 'understand:self-index', phase: 'Understand' }
)
const groups = parseGroups(changed) // JSON.parse the GROUPS=... line

// ---- Lease + ReviewFix: each lane group is a shared-path lease, worked independently ----
const results = await pipeline(
  groups,
  // Stage 1 — take a disjoint lease so two agents never mutate the same tree (COLLISION_RISK floor).
  (g) => agent(
    `Acquire a fak lease for lane "${g.lane}" over files ${JSON.stringify(g.files)} via dos_arbitrate ` +
    `(mode=exclusive, tree=the lane's files). If it refuses COLLISION_RISK, report blocked and stop. ` +
    `Otherwise end with: LEASE=ok lane=${g.lane}`,
    { label: `lease:${g.lane}`, phase: 'Lease' }
  ).then(() => g),
  // Stage 2 — recall fak memory BEFORE the work (page in only what's relevant), review+fix, then COMPACT after.
  (g) => agent(
    `For lane "${g.lane}" files ${JSON.stringify(g.files)}:\n` +
    `1. RECALL: run \`fak memory run --driver recall --intent "bugs in ${g.lane}"\` (or fak_memory_run) to page in relevant prior memory — do NOT carry the whole store.\n` +
    `2. Review each file for correctness bugs and fix them in place. Commit by explicit path with the ${g.stamp} stamp.\n` +
    `3. COMPACT: run \`fak memory run --driver compact --apply\` to fold this turn's ephemeral notes into a derived disposition (and \`--driver clean\` to tombstone turn-class cells) so the next agent inherits a small, relevant store, not your scratch.\n` +
    `End with: DONE lane=${g.lane} fixed=<n>`,
    { label: `reviewfix:${g.lane}`, phase: 'ReviewFix' }
  ),
)

return { groups: groups.length, results: results.filter(Boolean) }

// Helper — pull the GROUPS=[...] JSON the understand agent emitted.
function parseGroups(text) {
  const m = (text || '').match(/GROUPS=(\[.*\])\s*$/m)
  if (!m) return []
  try { return JSON.parse(m[1]) } catch { return [] }
}
