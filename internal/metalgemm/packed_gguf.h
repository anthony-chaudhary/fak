#include <stddef.h>
#include <stdint.h>

typedef struct {
    uintptr_t weight;
    size_t byte_offset;
    uint32_t type, rows, cols;
    size_t input_offset, output_offset;
} mg_packed_job;

void *mg_packed_create(const char *shader);
size_t mg_packed_max_buffer(void);
int mg_packed_prepare(void *ctx, size_t nx, size_t ny, float **x, float **y);
typedef struct {
    uint32_t count, job[4];
} mg_packed_panel;
int mg_packed_run(void *ctx, const mg_packed_job *jobs, size_t count, const mg_packed_panel *panels);
typedef struct {
    uint32_t slots, hidden, width, activation_stride;
    float beta, linear_beta;
} mg_packed_moe_shape;
int mg_packed_moe(void *ctx, const mg_packed_job *jobs, mg_packed_moe_shape shape);
typedef struct {
    uint32_t experts, slots;
    float weight_scale;
} mg_packed_route_shape;
int mg_packed_routed_ffn(void *ctx, const mg_packed_job *jobs, mg_packed_moe_shape shape,
                         mg_packed_route_shape route, size_t ids_at, size_t table_at);
void mg_packed_close(void *ctx);

void *mg_zerocopy_create_shared_buffer(void *ptr, size_t length);
void mg_zerocopy_buffer_free(void *handle);
