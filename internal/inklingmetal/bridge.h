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
int fak_ink_available(void);
char *fak_ink_plan(const char *, const char *, uint32_t, FakInkMemory *);
char *fak_ink_model(const char *gguf, const char *draft, FakInkModel **model);
void fak_ink_free_model(FakInkModel *model);
char *fak_ink_session(FakInkModel *model, uint32_t capacity,
                      FakInkSession **session);
char *fak_ink_reset(FakInkSession *, uint32_t);
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
