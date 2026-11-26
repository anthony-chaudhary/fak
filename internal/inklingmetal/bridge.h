#ifndef FAK_INKLING_METAL_H
#define FAK_INKLING_METAL_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct FakInkModel FakInkModel;
typedef struct FakInkSession FakInkSession;
typedef struct {
  uint64_t target_images, draft_images, prefill_scratch, decode_scratch, state,
      kv;
} FakInkMemory;
// What a model's kernels fix beyond its files. A session's state repeats only
// under the same values.
typedef struct {
  uint32_t layers, kv_format, state_format, chunk_rows, draft_window,
      context_tokens;
} FakInkKernels;
// A model's pipeline archive: pipelines Metal compiled, pipelines created
// from the archive, pipelines its list named when attached, the time creating
// those took, and the time of the prefill run at load when it listed none.
typedef struct {
  uint64_t compiled, archive_hits, archived;
  double archive_warm_ms, load_warmup_ms;
  int attached;
} FakInkPipelineStats;
int fak_ink_available(void);
char *fak_ink_plan(const char *, const char *, uint32_t, FakInkMemory *);
// pipelines is the directory the model's Metal pipeline archive persists
// under; empty keeps none.
char *fak_ink_model(const char *gguf, const char *draft, const char *pipelines,
                    FakInkModel **model);
void fak_ink_free_model(FakInkModel *model);
char *fak_ink_session(FakInkModel *model, uint32_t capacity,
                      FakInkSession **session);
char *fak_ink_reset(FakInkSession *, uint32_t);
void fak_ink_kernels(FakInkModel *, FakInkKernels *);
// When the model's attached archive lists no pipelines, runs a short prefill
// so a cold start compiles at load rather than inside the first prompt.
char *fak_ink_warm_pipelines(FakInkModel *);
// Writes the archive when pipelines were added; nonzero when it wrote it.
int fak_ink_pipeline_flush(FakInkModel *);
void fak_ink_pipeline_stats(FakInkModel *, FakInkPipelineStats *);
// Keeps the state after each of a session's newest leading prefill chunks.
char *fak_ink_keep_boundaries(FakInkSession *, uint32_t most);
// Reports the committed tokens a session shares with ids, its count of leading
// full prefill chunks, and the fewest of them it can still restart after.
char *fak_ink_shared(FakInkSession *, const int32_t *ids, uint32_t n,
                     uint32_t *shared, uint32_t *oldest, uint32_t *chunks);
// Resets a session for a new request that keeps its first chunks.
char *fak_ink_rewind(FakInkSession *, uint32_t capacity, uint32_t chunks);
void fak_ink_free_session(FakInkSession *session);
char *fak_ink_step(FakInkSession *s, const int32_t *tokens, uint32_t n,
                   int prefill, float *scores, int32_t *argmax);
char *fak_ink_step_batch(FakInkSession *const *sessions, const int32_t *tokens,
                         uint32_t n, int wide, float *scores, int32_t *argmax);
uint64_t fak_ink_batch_bytes(FakInkModel *model);
char *fak_ink_tree(FakInkSession *, const int32_t *rows, uint32_t count,
                   int32_t *argmax);
char *fak_ink_propose_verify(FakInkSession *, int32_t, uint32_t, uint32_t,
                             int32_t *, int32_t *, uint32_t *);
char *fak_ink_cycle(FakInkSession *, int32_t anchor, uint32_t depth,
                    uint32_t width, uint32_t maximum, int32_t eos,
                    int32_t *rows, int32_t *ids, uint32_t *count,
                    uint32_t *retained);
char *fak_ink_commit_tree(FakInkSession *, int32_t leaf);
char *fak_ink_draft_tree(FakInkSession *, int32_t anchor, uint32_t depth,
                         uint32_t width, int32_t *rows, uint32_t *count);
char *fak_ink_commit(FakInkSession *s, uint32_t rows);
char *fak_ink_propose(FakInkSession *s, int32_t anchor, uint32_t depth,
                      int32_t *proposal, uint32_t *count);
#ifdef __cplusplus
}
#endif
#endif
