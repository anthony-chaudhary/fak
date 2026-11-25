package projectassets

// DefaultOpenCodePlugin is the canonical dos-proof-guard.js script enforcing
// cross-harness FAK reference fences and DOS lane lease admission before OpenCode native mutations.
// The admission chain runs under a shared preflight wall-clock budget
// (FAK_WRITE_GUARD_BUDGET_MS); availability failures degrade through the bounded
// QUEUEUS ledger (<root>/.fak/queueus/) instead of failing closed, and
// FAK_WRITE_GUARD_STRICT=1 restores the strict fail-closed throw.
const DefaultOpenCodePlugin = `import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, realpath, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

/**
 * dos-proof-guard.js — opencode plugin for DOS on-device proof & cross-validation lifecycle.
 *
 * Shape: OpenCode 2.x default export { id, server, setup }. ` + "`" + `setup(api)` + "`" + ` registers
 * exactly two hooks (execute.before / execute.after) and returns a cleanup that
 * disposes both and clears in-memory session state; ` + "`" + `server({directory})` + "`" + ` keeps
 * the 1.x dotted-hook contract for direct callers.
 *
 * execute.before fences every write/edit/apply_patch/patch through lease
 * admission. Out-of-session-root targets are refused OUT_OF_LEASE unless they
 * lie in OS temp/scratch (admitted outright, fak-private#12166) or inside a
 * DECLARED sibling workspace root (FAK_PUBLIC_ROOT / FAK_ADMITTED_ROOTS), which
 * is re-based through the SAME admission chain against that root — never by
 * bypassing the fence. A declaration that contains the session root (an
 * ancestor) widens nothing and admits nothing.
 *
 * PREFLIGHT BUDGET: the admission chain (fak leaseref liveness -> fak leaseref
 * fence -> fak loop region -> python DOS snapshot -> dos arbitrate) runs under
 * one shared wall-clock deadline of FAK_WRITE_GUARD_BUDGET_MS ms (default 5000,
 * clamped to [250,120000]; invalid/absent env falls back to the default). Each
 * subprocess call gets the remaining budget as its timeout; when the budget is
 * spent BEFORE a call, the call is not spawned and reads as an availability
 * failure. Chain order is unchanged; each sibling group gets its own budget.
 *
 * AVAILABILITY vs VERDICT: a subprocess that cannot produce a parsed verdict
 * (spawn error, nonzero exit, timeout, stdout that fails JSON.parse) is an
 * AVAILABILITY failure. A PARSED verdict that refuses — region.admit !== true,
 * decision.outcome !== "acquire", fence.ok !== true, wrong-shape snapshots — is
 * a VERDICT and stays fail-closed exactly as before. Only availability degrades.
 *
 * DEGRADATION LADDER (availability path only; skipped entirely when
 * FAK_WRITE_GUARD_STRICT=1, the exact escape hatch that restores today's
 * fail-closed throw ` + "`" + `[dos-proof-guard] Lease admission unavailable or refused: ...` + "`" + `):
 *   1. live probe within budget succeeds -> admit, refresh the cache, reconcile;
 *   2. availability failure + fresh cache (age <= FAK_WRITE_GUARD_CACHE_TTL_MS,
 *      default 600000 via the same bounded-env clamp; cache owner+workspace
 *      match AND cache.trees cover ALL requested trees of the group) -> admit
 *      WITHOUT live consultation, DEGRADED_ADMISSION note;
 *   3. availability failure with no usable cache -> admit WITHOUT lease
 *      consultation, ADMISSION_UNVERIFIED note, pending record appended.
 * Policy refusals (OUT_OF_LEASE, COLLISION_RISK, verdict-plane refusals) always
 * run ahead of and dominate the ladder.
 *
 * QUEUEUS LEDGER — the guard's durable, bounded admission ledger under
 * <sessionRoot>/.fak/queueus/, best-effort: it NEVER blocks, throws into, or
 * meaningfully delays the admission decision (all IO is try/catch-swallowed
 * with at most one silent retry; corrupt files self-heal, ignored on read and
 * replaced on the next successful write; dir created recursively on first
 * write; every file written atomically via tmp + rename):
 *   admission-cache.json   schema fak.queueus.admission-cache.v1 —
 *                          {schema, at, owner, session, workspace, trees};
 *                          refreshed after EVERY fully-successful admission.
 *   admission-pending.json schema fak.queueus.pending.v1 —
 *                          {schema, records:[{at, owner, session, trees,
 *                          kind:"admission_unverified"}]}; appended on every
 *                          unverified admit; hard cap 256 records, drop oldest.
 *
 * NOTE BRIDGE: degraded/reconcile notes are held per session (bounded with the
 * same eviction discipline as session state) and PREPENDED to the next
 * after-event mutation note when that mutation SUCCEEDS; a failed mutation
 * drops the note (the failure dominates).
 *
 * RECONCILIATION (resume safely): a fully successful admission drains the
 * pending records for its workspace against the FRESH DOS peer snapshot (rows
 * whose holder/run_id differ from this owner/session); a record whose trees
 * intersect a peer row's tree array is collision_risk (overlapping trees +
 * holder named), else clean. Survivors are unadjudicable records only,
 * normally zero. When anything drained, a QUEUEUS_RECONCILE note names
 * drained=N collision_risk=M and, on collision, the overlapping trees plus
 * 'fak recover COLLISION_RISK'.
 *
 * execute.after annotates completed mutations: a claimed write whose target is
 * absent on disk carries a PHANTOM_WRITE note instead of the proof reminder.
 * It also observes successful capability-catalog reconnaissance (fak-dev
 * feature query / capabilities / index, fak-featureindex orient/refresh, or the
 * corresponding MCP invocations) in the same session+workspace; the first
 * direct mutation WITHOUT such a receipt gets one short DISCOVERY_ADVISORY,
 * labeled as a post-edit notice, at most once per session. The advisory never
 * blocks, throws, runs commands, or touches lease decisions; state is bounded
 * and cleared on disposal; child sessions (distinct sessionIDs) inherit nothing.
 */

const execute = promisify(execFile);
// patch is the 2.x mutation tool; apply_patch its 1.x name.
const mutations = new Set(["write", "edit", "apply_patch", "patch"]);
const SESSION_STATE_CAP = 64;

// Bounded env int: invalid/absent -> default, out of [min,max] -> clamped.
function boundedIntEnv(name, def, min, max) {
  const raw = process.env[name];
  if (raw === undefined || String(raw).trim() === "") return def;
  const n = Number(raw);
  if (!Number.isInteger(n)) return def;
  if (n < min) return min;
  if (n > max) return max;
  return n;
}
// The write-guard preflight budget and the cached-verdict TTL it degrades to.
const WRITE_GUARD_BUDGET_MS = boundedIntEnv("FAK_WRITE_GUARD_BUDGET_MS", 5000, 250, 120000);
const WRITE_GUARD_CACHE_TTL_MS = boundedIntEnv("FAK_WRITE_GUARD_CACHE_TTL_MS", 600000, 250, 3600000);
// QUEUEUS pending-ledger cap: drop oldest past this.
const QUEUEUS_PENDING_CAP = 256;

// AVAILABILITY failure marker: the lease plane could not produce a parsed
// verdict (spawn error, nonzero exit, timeout, unparseable stdout). A PARSED
// verdict that refuses is NOT this — it stays fail-closed as a plain Error.
class AdmissionUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = "AdmissionUnavailableError";
  }
}

function hashString(str) {
  if (typeof str !== "string" || !str) return "";
  return createHash("sha256").update(str).digest("hex").slice(0, 16);
}

function extractHolder(errorOrOutput, peers) {
  if (!errorOrOutput) return (peers && peers[0]?.holder) || "peer";
  if (typeof errorOrOutput === "object") {
    if (errorOrOutput.conflict?.holder) return errorOrOutput.conflict.holder;
    if (errorOrOutput.holder) return errorOrOutput.holder;
  }
  const text = typeof errorOrOutput === "string" ? errorOrOutput : (errorOrOutput.stdout || errorOrOutput.stderr || errorOrOutput.message || "");
  const match = text.match(/"holder"\s*:\s*"([^"]+)"/);
  if (match) return match[1];
  if (peers && peers.length > 0 && peers[0]?.holder) return peers[0].holder;
  return "peer";
}

function formatCollisionRecovery(holder) {
  return "[RECOVERY]: Lane conflict detected with holder '" + (holder || "peer") + "'. DO NOT retry editing this path. Run 'fak recover COLLISION_RISK' or select a tree-disjoint task.";
}

// Out-of-tree scratch (OS temp dirs, the harness's own scratch) is not leased
// by the kernel, so a repo mutation tool must neither be blocked nor steered to
// shell for it: that steerage is what taught agents to route around the guard
// (fak-private#12166). Admit it outright; containment still governs every
// in-repo path. Scratch is matched ONLY under a temp/scratch root so a repo
// path that merely contains the word cannot escape admission.
function isAdmittedScratchPath(target, filename) {
  const normTarget = (target || "").replace(/\\/g, "/");
  const normName = (filename || "").replace(/\\/g, "/");
  const scratchRoot = /(?:^|\/)(?:appdata\/local\/temp|temp|tmp|scratch)\//i;
  const scratchLeaf = /(?:^|\/)(?:opencode|scratch)(?:\/|$)/i;
  return (scratchRoot.test(normTarget) || scratchRoot.test(normName)) &&
    (scratchLeaf.test(normTarget) || scratchLeaf.test(normName));
}

function extractOutputText(output) {
  if (!output) return "";
  if (typeof output === "string") return output;
  if (typeof output.error === "string") return output.error;
  if (output.error instanceof Error) return output.error.message;
  if (typeof output.output === "string") return output.output;
  if (typeof output.result === "string") return output.result;
  if (typeof output.text === "string") return output.text;
  if (typeof output.content === "string") return output.content;
  if (Array.isArray(output.content)) {
    return output.content.map((c) => (typeof c === "string" ? c : c?.text || "")).join("\n");
  }
  if (Array.isArray(output.result)) {
    return output.result.map((c) => (typeof c === "string" ? c : c?.text || "")).join("\n");
  }
  return "";
}

function isMutationFailure(output) {
  if (!output) return false;
  if (output.error || output.isError === true || output.metadata?.error) return true;
  const text = extractOutputText(output);
  if (!text) return false;
  if (/could not find oldstring/i.test(text)) return true;
  if (/found multiple matches/i.test(text)) return true;
  if (/oldstring not found/i.test(text)) return true;
  if (/^error:\s+/im.test(text)) return true;
  return false;
}

// Completed-call success: explicit failure status, result error markers, or a
// nonzero shell exit all read as not-ok. Absent metadata stays unknown-ok.
function callOk(ev) {
  if (ev?.status === "failed") return false;
  const r = ev?.result;
  if (r && typeof r === "object") {
    if (r.error || r.isError === true || r.metadata?.error) return false;
    const exit = r.metadata?.exit;
    if (exit !== undefined && exit !== null && exit !== 0) return false;
  }
  return true;
}

// timeoutMs is the per-call budget slice; omitted/<=0 keeps the historical
// 120000 default for any other caller. Every failure here is AVAILABILITY —
// the subprocess could not produce a parsed verdict — so it throws the marker
// class, keeping the strict-path message text intact.
async function jsonCommand(command, args, cwd, timeoutMs) {
  let stdout;
  const binary = process.platform === "win32" && !command.endsWith(".exe") ? ` + "`" + `${command}.exe` + "`" + ` : command;
  try {
    ({ stdout } = await execute(binary, args, {
      cwd, timeout: timeoutMs > 0 ? timeoutMs : 120000, maxBuffer: 4 * 1024 * 1024, windowsHide: true,
    }));
  } catch (error) {
    const rawOut = error.stdout || error.stderr || error.message || "";
    let recovery = "";
    if (rawOut.includes("COLLISION_RISK")) {
      const holder = extractHolder(error);
      recovery = "\n" + formatCollisionRecovery(holder);
    }
    throw new AdmissionUnavailableError("[dos-proof-guard] Lease admission unavailable or refused: " + command + ": " + rawOut + recovery);
  }
  try {
    return JSON.parse(stdout);
  } catch {
    throw new AdmissionUnavailableError("[dos-proof-guard] Malformed lease admission response from " + command);
  }
}

// QUEUEUS ledger IO: best-effort and bounded. EVERY failure (missing dir,
// unwritable, corrupt JSON) is swallowed after at most one silent retry, so the
// ledger never blocks, throws into, or meaningfully delays the admission
// decision. Writes are atomic (tmp file + rename); a corrupt file is ignored on
// read and replaced by the next successful write (self-heal).
async function queueusWrite(file, obj) {
  const payload = JSON.stringify(obj);
  const attempt = async () => {
    await mkdir(path.dirname(file), { recursive: true });
    const tmp = ` + "`" + `${file}.tmp-${process.pid}-${Date.now()}` + "`" + `;
    await writeFile(tmp, payload, "utf8");
    await rename(tmp, file);
  };
  try {
    await attempt();
  } catch {
    try { await attempt(); } catch {}
  }
}

async function queueusReadJson(file) {
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch {
    return null; // absent or corrupt: the next successful write replaces it
  }
}

// Resolve existing ancestors as well as new files so a symlink cannot disguise
// a write to a leased path. External writes need their own workspace admission.
async function canonicalPath(filename) {
  try {
    return await realpath(filename);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    const parent = path.dirname(filename);
    if (parent === filename) throw error;
    return path.join(await canonicalPath(parent), path.basename(filename));
  }
}

function mutationPaths(tool, args) {
  if (tool !== "apply_patch" && tool !== "patch") {
    const candidates = [args?.filePath, args?.file_path, args?.path, args?.target];
    return [...new Set(candidates
      .filter((p) => typeof p === "string" && p.trim() !== "")
      .map((p) => p.trim()))];
  }
  const patch = typeof args?.patchText === "string" ? args.patchText : (typeof args?.patch === "string" ? args.patch : "");
  if (!patch) return [];

  const paths = new Set();
  const addPath = (raw) => {
    if (!raw || typeof raw !== "string") return;
    let s = raw.trim();
    const tabIdx = s.indexOf("\t");
    if (tabIdx !== -1) s = s.slice(0, tabIdx).trim();
    if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
      s = s.slice(1, -1).trim();
    }
    if (s.startsWith("a/") || s.startsWith("b/")) {
      s = s.slice(2).trim();
    }
    if (s && s !== "/dev/null" && s !== "dev/null") {
      paths.add(s);
    }
  };

  for (const m of patch.matchAll(/^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)\r?$/gm)) {
    addPath(m[1]);
  }
  for (const m of patch.matchAll(/^diff --git\s+(?:"([^"]+)"|'([^']+)'|(\S+))\s+(?:"([^"]+)"|'([^']+)'|(\S+))\r?$/gm)) {
    addPath(m[1] || m[2] || m[3]);
    addPath(m[4] || m[5] || m[6]);
  }
  for (const m of patch.matchAll(/^---\s+(?:"([^"]+)"|'([^']+)'|([^\t\r\n]+?))(?:\t.*)?\r?$/gm)) {
    addPath(m[1] || m[2] || m[3]);
  }
  for (const m of patch.matchAll(/^\+\+\+\s+(?:"([^"]+)"|'([^']+)'|([^\t\r\n]+?))(?:\t.*)?\r?$/gm)) {
    addPath(m[1] || m[2] || m[3]);
  }
  return [...paths];
}

// Paths a patch deletes: absence on disk after a completed delete is expected,
// not a phantom.
function deletedPaths(args) {
  const patch = typeof args?.patchText === "string" ? args.patchText : (typeof args?.patch === "string" ? args.patch : "");
  if (!patch) return [];
  const paths = new Set();
  for (const m of patch.matchAll(/^\*\*\* Delete File: (.+)\r?$/gm)) {
    let s = m[1].trim();
    if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
      s = s.slice(1, -1).trim();
    }
    if (s.startsWith("a/") || s.startsWith("b/")) {
      s = s.slice(2).trim();
    }
    if (s) paths.add(s);
  }
  return [...paths];
}

function normPath(p) {
  const s = String(p || "").replace(/\\/g, "/");
  return process.platform === "win32" ? s.toLowerCase() : s;
}

function isInside(target, container) {
  const t = normPath(target);
  const c = normPath(container);
  return t === c || t.startsWith(c + "/");
}

// Declared sibling workspace roots (FAK_PUBLIC_ROOT, FAK_ADMITTED_ROOTS list).
// A declaration that CONTAINS the session root (ancestor, or identical) must
// never widen the fence: it adds nothing beyond session containment itself.
function declaredAdmissionRoots(sessionRoot, env) {
  const raw = [];
  const pub = String(env.FAK_PUBLIC_ROOT || "").trim();
  if (pub) raw.push(pub);
  const list = String(env.FAK_ADMITTED_ROOTS || "");
  if (list) {
    for (const piece of list.split(path.delimiter)) {
      const p = piece.trim();
      if (p) raw.push(p);
    }
  }
  const out = [];
  for (const r of raw) {
    const abs = path.resolve(r);
    if (isInside(sessionRoot, abs)) continue;
    if (!out.includes(abs)) out.push(abs);
  }
  return out;
}

// The deepest declared root containing the target, or null.
function admissionRootFor(target, roots) {
  let best = null;
  for (const r of roots) {
    if (isInside(target, r) && (best === null || normPath(r).length > normPath(best).length)) {
      best = r;
    }
  }
  return best;
}

// Closed discovery contract: the capability-catalog CLI family (bare or wrapped
// in fak-run / go run) and the corresponding MCP tool names. Printed commands,
// schema-only search, and everything else do not qualify.
const DISCOVERY_SHELL_FAK_DEV = /\bfak-dev\s+(?:feature\s+query|capabilities|index)\b/i;
const DISCOVERY_SHELL_FEATUREINDEX = /\bfak-featureindex\s+(?:orient|refresh)\b/i;

function isDiscoveryShellCommand(args) {
  const cmd = typeof args?.command === "string" ? args.command : "";
  if (!cmd) return false;
  return DISCOVERY_SHELL_FAK_DEV.test(cmd) || DISCOVERY_SHELL_FEATUREINDEX.test(cmd);
}

function isDiscoveryToolName(tool) {
  const n = String(tool || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return /^(?:mcp_)?(?:fak|dos)_(?:feature_query|featureindex|capabilities|catalog_index|index)$/.test(n);
}

async function createGuard({ directory }) {
  let fileModifiedCount = 0;
  const failureHistory = new Map();
  // Capture host configuration once; tool arguments never supply ownership.
  const hostOwner = process.env.FAK_LEASE_OWNER;
  const hostSession = process.env.FAK_LEASE_SESSION;
  const hostLease = process.env.FAK_LEASE_ID;
  const root = await realpath(directory || process.cwd());
  // Bounded per-session discovery state: probed = a successful same-session
  // capability-catalog probe was observed; advised = the advisory fired (or was
  // suppressed by a probe). Keyed by sessionID, so child sessions inherit
  // nothing; cleared on disposal.
  const sessions = new Map();
  // Note bridge: before -> after. A degraded/reconcile note set by
  // admitWorkspace for a session is consumed (and cleared) by that session's
  // next afterEvent: prepended when the mutation succeeded, dropped when it
  // failed (the failure dominates). Same cap discipline as ` + "`" + `sessions` + "`" + `.
  const pendingNotes = new Map();
  // QUEUEUS ledger location: <sessionRoot>/.fak/queueus/, created on first write.
  const queueusDir = path.join(root, ".fak", "queueus");
  const queueusCacheFile = path.join(queueusDir, "admission-cache.json");
  const queueusPendingFile = path.join(queueusDir, "admission-pending.json");

  function setPendingNote(session, note) {
    if (!pendingNotes.has(session) && pendingNotes.size >= SESSION_STATE_CAP) {
      pendingNotes.delete(pendingNotes.keys().next().value);
    }
    pendingNotes.set(session, note);
  }

  function takePendingNote(session) {
    const note = pendingNotes.get(session);
    pendingNotes.delete(session);
    return note || "";
  }

  function normalizeKey(session, filename) {
    const full = path.resolve(root, filename);
    return session + ":" + full.toLowerCase().replace(/\\/g, "/");
  }

  function checkCircuitBreaker(session, filename) {
    const key = normalizeKey(session, filename);
    const entry = failureHistory.get(key);
    if (entry && entry.count >= 2) {
      entry.count++;
      throw new Error("[dos-proof-guard-circuit-breaker] Refusing 3rd consecutive failing edit on " + filename + ". Stop retrying. Inspect the file with 'read' or pivot to an alternate lane.");
    }
  }

  function recordFailure(session, filename, tool, args, reason) {
    const key = normalizeKey(session, filename);
    const existing = failureHistory.get(key);
    const count = (existing?.count || 0) + 1;
    const oldString = args?.oldString || args?.old_string || "";
    const oldStringHash = oldString ? hashString(oldString) : (existing?.oldStringHash || "");
    failureHistory.set(key, {
      tool: tool || existing?.tool || "unknown",
      filePath: filename,
      oldStringHash,
      failureReason: reason || existing?.failureReason || "",
      count,
      lastFailedAt: Date.now(),
    });
  }

  function resetFailure(session, filename) {
    const key = normalizeKey(session, filename);
    failureHistory.delete(key);
  }

  function sessionState(session) {
    let st = sessions.get(session);
    if (!st) {
      if (sessions.size >= SESSION_STATE_CAP) {
        sessions.delete(sessions.keys().next().value);
      }
      st = { probed: false, advised: false };
      sessions.set(session, st);
    }
    return st;
  }

  // A cached verdict is usable for degraded admission only while fresh, owned
  // by the same owner in the same workspace, and covering every requested tree.
  function cacheUsable(cache, workspace, owner, trees) {
    if (!cache || typeof cache !== "object" || cache.schema !== "fak.queueus.admission-cache.v1") return false;
    if (typeof cache.at !== "number" || !Number.isFinite(cache.at)) return false;
    if (Date.now() - cache.at > WRITE_GUARD_CACHE_TTL_MS) return false;
    if (cache.owner !== owner || normPath(cache.workspace) !== normPath(workspace)) return false;
    const cached = Array.isArray(cache.trees) ? cache.trees.map(normPath) : [];
    return trees.every((t) => cached.includes(normPath(t)));
  }

  // Reconciliation (resume safely): drain this workspace's pending records
  // against the FRESH peer snapshot. A record whose trees intersect a peer
  // row's tree array is collision_risk (overlapping trees + holder named),
  // else clean. Only unadjudicable records (no parseable tree list) survive;
  // normally zero. Never throws; sets the note bridge only when it is free.
  async function reconcileQueueus(session, owner, peers) {
    try {
      const doc = await queueusReadJson(queueusPendingFile);
      const records = (doc && Array.isArray(doc.records)) ? doc.records : [];
      if (records.length === 0) return;
      let drained = 0;
      let collisionRisk = 0;
      const overlapTrees = new Set();
      const peerHolders = new Set();
      const kept = [];
      for (const record of records) {
        const rt = record && Array.isArray(record.trees) ? record.trees.map(normPath) : null;
        if (rt === null) { kept.push(record); continue; } // unadjudicable: no tree list to intersect
        let hit = false;
        for (const row of peers) {
          const pt = Array.isArray(row?.tree) ? row.tree.map(normPath) : [];
          const shared = rt.filter((t) => pt.includes(t));
          if (shared.length > 0) {
            hit = true;
            for (const t of shared) overlapTrees.add(t);
            if (row?.holder) peerHolders.add(row.holder);
            break;
          }
        }
        drained++;
        if (hit) collisionRisk++;
      }
      const note = "[dos-proof-guard:QUEUEUS_RECONCILE] drained=" + drained + " collision_risk=" + collisionRisk +
        (collisionRisk > 0
          ? " — overlapping trees: " + [...overlapTrees].join(", ") + (peerHolders.size ? " (peer holder: " + [...peerHolders].join(", ") + ")" : "") + ". Run 'fak recover COLLISION_RISK' before editing those paths."
          : "");
      if (!pendingNotes.has(session)) setPendingNote(session, note);
      await queueusWrite(queueusPendingFile, { schema: "fak.queueus.pending.v1", records: kept });
    } catch {}
  }

  async function admitWorkspace(workspace, trees, session, owner) {
    // One shared wall-clock deadline for the whole chain: each call gets the
    // remaining budget as its timeout, and a spent budget skips the remaining
    // calls entirely instead of spawning doomed subprocesses.
    const deadline = Date.now() + WRITE_GUARD_BUDGET_MS;
    const consult = async (command, args) => {
      if (deadline - Date.now() <= 0) {
        throw new AdmissionUnavailableError("[dos-proof-guard] Lease admission unavailable or refused: " + command + ": budget exhausted before consultation");
      }
      return jsonCommand(command, args, workspace, Math.max(250, deadline - Date.now()));
    };
    let refs, own, regionArgs, lease, fence, region, snapshot, peers, decision;
    try {
      refs = await consult("fak", ["leaseref", "liveness", "--dir", workspace, "--session", session]);
      if (!Array.isArray(refs)) throw new Error("[dos-proof-guard] Invalid FAK lease snapshot");
      own = refs.filter((row) => row.holder === owner && row.session_id === session && (!hostLease || row.id === hostLease));
      if (hostLease && own.length !== 1) throw new Error("[dos-proof-guard] Host lease identity is not current");
      if (own.length > 1) throw new Error("[dos-proof-guard] Set host FAK_LEASE_ID to identify the active lease");
      regionArgs = ["loop", "region", "--dir", workspace, "--actor", owner, "--no-queue", "--json"];
      for (const tree of trees) regionArgs.push("--tree", tree);
      if (own.length === 1) {
        lease = own[0];
        if (!lease.id || !Number.isInteger(lease.generation) || lease.generation < 1) {
          throw new Error("[dos-proof-guard] Own FAK lease lacks a fencing generation");
        }
        fence = await consult("fak", ["leaseref", "fence", "--dir", workspace, "--id", lease.id, "--holder", owner, "--generation", String(lease.generation)]);
        if (fence.ok !== true) throw new Error("[dos-proof-guard] Own FAK lease fence refused");
        regionArgs.push("--self", lease.id);
      }
      region = await consult("fak", regionArgs);
      if (region.schema !== "fak.loop-region.v1" || region.admit !== true) {
        let recovery = "";
        if (region.reason === "COLLISION_RISK" || JSON.stringify(region).includes("COLLISION_RISK")) {
          const holder = extractHolder(region);
          recovery = "\n" + formatCollisionRecovery(holder);
        }
        throw new Error("[dos-proof-guard] FAK lease admission refused" + recovery);
      }

      // DOS owns WAL parsing, corruption handling, expiry and locking. DOS >= 0.22.0
      // removed the strict kwarg; emitting it raises TypeError and blocks mutations.
      // FAK_PYTHON overrides python resolution (the witness recording stubs use it).
      snapshot = await consult(process.env.FAK_PYTHON || "python", ["-c",
        "import json,sys; from dos import config,lane_lease; cfg=config.load_workspace_config(sys.argv[1],gather_env=False); print(json.dumps(lane_lease.live_leases(cfg, expire_dead=True)))", workspace]);
      if (!Array.isArray(snapshot) || snapshot.some((row) => !row || typeof row.lane !== "string" || !Array.isArray(row.tree))) {
        throw new Error("[dos-proof-guard] Invalid DOS lease snapshot");
      }
      peers = snapshot.filter((row) => !(row.holder === owner && row.run_id === session));
      decision = await consult("dos", ["arbitrate", "--workspace", workspace, "--lane", "opencode-native-write", "--kind", "keyword", "--output", "json", "--leases", JSON.stringify(peers), "--tree", ...trees]);
      if (decision.outcome !== "acquire") {
        let recovery = "";
        if (decision.reason === "COLLISION_RISK" || JSON.stringify(decision).includes("COLLISION_RISK")) {
          const holder = extractHolder(decision, peers);
          recovery = "\n" + formatCollisionRecovery(holder);
        }
        throw new Error("[dos-proof-guard] DOS lease admission refused" + recovery);
      }
    } catch (err) {
      // VERDICT refusals (and identity/snapshot-shape errors) propagate exactly
      // as before: fail-closed. Only AVAILABILITY degrades.
      if (!(err instanceof AdmissionUnavailableError)) throw err;
      // Strict escape hatch: restore today's fail-closed behavior verbatim.
      if (String(process.env.FAK_WRITE_GUARD_STRICT) === "1") throw err;
      // Ladder step 2: a fresh cached verdict admits without live consultation.
      const cache = await queueusReadJson(queueusCacheFile);
      if (cacheUsable(cache, workspace, owner, trees)) {
        setPendingNote(session, "[dos-proof-guard:DEGRADED_ADMISSION] Lease plane unreadable within " + WRITE_GUARD_BUDGET_MS + "ms; admitted on the cached verdict from " + Math.max(0, Math.round((Date.now() - cache.at) / 1000)) + "s ago (no live fence was taken). Re-probe when the plane recovers.");
        return;
      }
      // Ladder step 3: admit WITHOUT lease consultation, recorded pending.
      setPendingNote(session, "[dos-proof-guard:ADMISSION_UNVERIFIED] Lease plane unreadable within " + WRITE_GUARD_BUDGET_MS + "ms and no cached verdict; write admitted WITHOUT lease admission and recorded pending in the QUEUEUS ledger. Reconcile with 'fak leaseref liveness' when the plane recovers; if a peer collision surfaces afterward, run 'fak recover COLLISION_RISK'.");
      const doc = await queueusReadJson(queueusPendingFile);
      const records = (doc && Array.isArray(doc.records)) ? doc.records : [];
      records.push({ at: Date.now(), owner, session, trees: [...trees], kind: "admission_unverified" });
      while (records.length > QUEUEUS_PENDING_CAP) records.shift();
      await queueusWrite(queueusPendingFile, { schema: "fak.queueus.pending.v1", records });
      return;
    }
    // Fully successful admission: refresh the durable cache, then drain the
    // pending ledger against the FRESH peer snapshot (resume safely).
    await queueusWrite(queueusCacheFile, {
      schema: "fak.queueus.admission-cache.v1",
      at: Date.now(),
      owner, session, workspace,
      trees: [...trees],
    });
    await reconcileQueueus(session, owner, peers);
    // These are pre-execution observations, not an atomic filesystem fence.
    // Shell/MCP mutations and late lease changes need separate mediation.
  }

  /** execute.before: fence every mutation through lease admission. Throws OUT_OF_LEASE at the escape pre-check before any external command runs. */
  async function beforeEvent(ev) {
    const tool = ev?.tool;
    if (!mutations.has(tool)) return;
    const sessionID = ev?.sessionID;
    if (!sessionID || (!!hostOwner !== !!hostSession)) {
      throw new Error("[dos-proof-guard] A harness session and paired host lease owner/session are required");
    }
    const session = hostSession || sessionID;
    const owner = hostOwner || ` + "`" + `opencode:${session}` + "`" + `;
    const args = ev?.args;
    const filenames = mutationPaths(tool, args);
    if (!filenames.length || filenames.some((name) => typeof name !== "string" || !name.trim())) {
      throw new Error("[dos-proof-guard] Explicit mutation paths are required for lease admission");
    }

    for (const filename of filenames) {
      checkCircuitBreaker(session, filename);
    }

    try {
      const siblings = declaredAdmissionRoots(root, process.env);
      const groups = new Map(); // admission workspace root -> [tree, ...]
      let scratchOnly = true;
      for (const filename of filenames) {
        if (filename.includes("\0")) {
          throw new Error("[dos-proof-guard] Malformed mutation path: contains null byte");
        }
        const target = await canonicalPath(path.resolve(root, filename));
        const relative = path.relative(root, target);
        if (!relative || relative === ".." || relative.startsWith(` + "`" + `..${path.sep}` + "`" + `) || path.isAbsolute(relative)) {
          // Out-of-tree scratch is admitted outright: the kernel does not lease
          // OS temp, and the old "use shell" steerage taught agents to route
          // around the guard (fak-private#12166). In-repo containment is
          // unaffected; only temp/scratch subtrees short-circuit.
          if (isAdmittedScratchPath(target, filename)) {
            continue;
          }
          // A declared sibling root re-bases the target through the SAME
          // admission chain against that root — consultation, not bypass.
          const sibling = admissionRootFor(target, siblings);
          if (sibling) {
            scratchOnly = false;
            const srel = path.relative(sibling, target);
            const list = groups.get(sibling);
            if (list) list.push(srel.split(path.sep).join("/"));
            else groups.set(sibling, [srel.split(path.sep).join("/")]);
            continue;
          }
          throw new Error("[dos-proof-guard:OUT_OF_LEASE] Mutation path is outside this lease workspace: '" + filename + "'. The kernel leases in-repo trees only; this path is not covered by any lane. Take the lane that owns this path (fak leaseref acquire ... or dos arbitrate), or record a handoff - do NOT re-attempt the same write through a different tool (shell/edit/write all cross the same lease boundary). Out-of-tree SCRATCH belongs under a temp/scratch directory and is admitted automatically.");
        }
        scratchOnly = false;
        const list = groups.get(root);
        if (list) list.push(relative.split(path.sep).join("/"));
        else groups.set(root, [relative.split(path.sep).join("/")]);
      }

      if (scratchOnly) {
        // Nothing in-repo to admit; scratch writes need no lease consultation.
        return;
      }

      for (const [workspace, trees] of groups) {
        await admitWorkspace(workspace, trees, session, owner);
      }
    } catch (err) {
      for (const filename of filenames) {
        recordFailure(session, filename, tool, args, err.message);
      }
      throw err;
    }
  }

  /** execute.after: annotate completed mutations; observe discovery; advisory. Never throws. */
  async function afterEvent(ev) {
    try {
      const tool = ev?.tool || "";
      const result = ev?.result;
      const ok = callOk(ev);
      // Observer: a completed successful catalog probe in this session/workspace
      // earns the receipt that suppresses the advisory. Failed or empty-output
      // calls stay unknown and earn nothing.
      if (ok && (isDiscoveryShellCommand(ev?.args) || isDiscoveryToolName(tool))) {
        const text = extractOutputText(result);
        if (text.trim()) {
          sessionState(hostSession || ev?.sessionID || "default").probed = true;
        }
      }
      if (!mutations.has(tool)) return;

      const session = hostSession || ev?.sessionID || "default";
      const args = ev?.args;
      const files = mutationPaths(tool, args);

      // Note bridge: consume the degraded/reconcile note set by beforeEvent. A
      // failed mutation drops it (the failure dominates); a successful one
      // prepends it below.
      const pendingNote = takePendingNote(session);

      if (!ok || isMutationFailure(result)) {
        for (const f of files) {
          recordFailure(session, f, tool, args, extractOutputText(result) || "mutation_failed");
        }
        return;
      }

      for (const f of files) {
        resetFailure(session, f);
      }

      fileModifiedCount++;

      // Phantom write: the tool reported success but the target is absent on
      // disk. Deletes expect absence; everything else must exist.
      const deleted = new Set(deletedPaths(args));
      const phantoms = [];
      for (const f of files) {
        if (deleted.has(f)) continue;
        let target;
        try {
          target = await canonicalPath(path.resolve(root, f));
        } catch {
          continue;
        }
        try {
          if (!existsSync(target)) phantoms.push(f);
        } catch {}
      }
      let note = "";
      if (phantoms.length) {
        note = pendingNote + "\n\n[dos-proof-guard] PHANTOM_WRITE: " + phantoms.map((p) => "'" + p + "'").join(", ") +
          " reported written but absent on disk. Confirm the write actually landed before claiming completion.";
      } else {

        let reminder = "\n\n[dos-proof-guard] Code modified. On-device proof required before completion:\n" +
          "1. Run scoped tests for the touched packages: go test ./<touched-pkg>/... -count=1\n" +
          "2. Check the diff against its claim with the DOS truth syscall (dos verify)\n" +
          "3. Spawn cross-validator subagent to adversarial-audit the diff\n" +
          "4. Auto-ticket follow-ons/edge-cases by default (gh issue create / fak issue fanout)\n";

        const hasHardware = files.some((f) => typeof f === "string" && (f.includes("internal/compute") || f.includes("internal/amdgpu") || f.includes("vulkan") || f.includes("strix")));
        if (hasHardware) {
          reminder += "\n[dos-proof-guard] Halo-related compute/hardware code modified. Physical hardware evidence required:\n" +
            "- Run ` + "`" + `fak-dev amd-strix-probe` + "`" + ` early to discover the LAN AMD Strix Halo and choose a bounded device correctness witness.\n" +
            "- Hold an exclusive lease across the entire baseline/candidate performance comparison.\n";
        }

        // The discovery advisory precedes the standing proof reminder: it is
        // the one-time post-edit notice that this session skipped the catalog.
        // The pending note bridge (degraded/reconcile admission) precedes both.
        const advisory = maybeAdvisory(sessionState(session));
        note = pendingNote + (advisory || "") + reminder;
      }
      // Emit against the hook result under an injection-root receiver name so
      // the self-observation audit sees this text reach the model context.
      if (note && result && typeof result === "object") {
        if (typeof result.output === "string") {
          result.output += note;
        } else if (typeof result.content === "string") {
          result.content += note;
        } else if (Array.isArray(result.content)) {
          result.content.push({ type: "text", text: note });
        } else if (typeof result.result === "string") {
          result.result += note;
        } else if (Array.isArray(result.result)) {
          result.result.push({ type: "text", text: note });
        } else if (typeof result.text === "string") {
          result.text += note;
        }
      }
    } catch {}
  }

  /** Bounded state, cleared on disposal: child sessions inherit nothing. */
  function dispose() {
    sessions.clear();
    pendingNotes.clear();
    failureHistory.clear();
    fileModifiedCount = 0;
  }

  return {
    root,
    dispose,
    resetCounter: () => { fileModifiedCount = 0; },
    beforeEvent,
    afterEvent,
  };
}

// One short DISCOVERY_ADVISORY per session: the first landed mutation in a
// session with no successful same-session catalog probe yet. Labeled a
// post-edit notice because it arrives after the fact; it never blocks, throws,
// runs commands, or touches lease decisions.
function maybeAdvisory(st) {
  if (st.probed || st.advised) return "";
  st.advised = true;
  return "\n\n[dos-proof-guard:DISCOVERY_ADVISORY] Post-edit notice: this session made a direct code change without a prior capability-catalog probe. Query the catalog before the next edit - fak-dev feature query \"<axis>\", fak-featureindex orient \"<intent>\", or the corresponding MCP catalog tool - so work reuses existing capability instead of rediscovering it.";
}

/** OpenCode 1.x dotted-hook surface: the same guard behind a stable contract. */
async function dosProofGuardServer({ client, directory }) {
  const guard = await createGuard({ directory });
  return {
    "tool.execute.before": async (input, output) => {
      // Fail closed: OUT_OF_LEASE must reject this hook call, not leak as an
      // unhandled rejection after the caller has moved on.
      await guard.beforeEvent({
        tool: input?.tool,
        sessionID: input?.sessionID,
        args: output?.args ?? input?.args,
      });
    },
    "tool.execute.after": async (input, output) => {
      await guard.afterEvent({
        tool: input?.tool,
        sessionID: input?.sessionID,
        args: input?.args ?? output?.args,
        result: output,
        status: input?.status ?? output?.status,
      });
    },
    "chat.message": async (input) => {
      if (input?.role === "user") guard.resetCounter();
    },
  };
}

/** OpenCode 2.x entrypoint: exactly two hooks; cleanup disposes both and clears state. */
async function dosProofGuardSetup(api) {
  const guard = await createGuard({ directory: api?.location?.directory || process.cwd() });
  const registrations = [];
  registrations.push(await api.tool.hook("execute.before", async (event) => {
    // Fail closed on the fence: OUT_OF_LEASE must propagate to the caller.
    await guard.beforeEvent({
      tool: event?.tool,
      sessionID: event?.sessionID,
      args: event?.input ?? event?.args,
    });
  }));
  registrations.push(await api.tool.hook("execute.after", async (event) => {
    try {
      await guard.afterEvent({
        tool: event?.tool,
        sessionID: event?.sessionID,
        args: event?.input ?? event?.args,
        result: event?.result,
        status: event?.status,
      });
    } catch {}
  }));
  return async () => {
    for (const registration of registrations) await registration.dispose();
    guard.dispose();
  };
}

export default { id: "dos-proof-guard", server: dosProofGuardServer, setup: dosProofGuardSetup };`
