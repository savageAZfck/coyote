#ifndef COYOTE_H
#define COYOTE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CoyoteVm CoyoteVm;

const char *coyote_last_error(void);

CoyoteVm *coyote_open(const char *cy_path, const char *out_dir, bool use_cpu);
void coyote_close(CoyoteVm *vm);

bool coyote_bind_extern_i64(
    CoyoteVm *vm,
    const char *name,
    int64_t (*cb)(int64_t, int64_t)
);

bool coyote_bind_extern_str(
    CoyoteVm *vm,
    const char *name,
    ssize_t (*cb)(const char *input, char *out_buf, size_t cap)
);

bool coyote_run_main_i64(CoyoteVm *vm, int64_t *out_val);

bool coyote_call_i64(
    CoyoteVm *vm,
    const char *fn_name,
    const int64_t *args,
    size_t nargs,
    int64_t *out_val
);

ssize_t coyote_call_str(
    CoyoteVm *vm,
    const char *fn_name,
    const char *arg,
    char *out_buf,
    size_t cap
);

bool coyote_resume(CoyoteVm *vm, const char *snap_path, int64_t *out_val);

size_t coyote_witness_count(const CoyoteVm *vm);
ssize_t coyote_witness_json(const CoyoteVm *vm, char *out_buf, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* COYOTE_H */
