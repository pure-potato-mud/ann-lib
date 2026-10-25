#ifndef INTERNAL_MLP_CREATION_H_
#define INTERNAL_MLP_CREATION_H_

#include <stdbool.h>
#include <stddef.h>

struct layer_size_data {
  const size_t *const layer_sizes_ref;
  const size_t layer_size_count;
};

struct addr_idx_offsets {
  size_t *data; // first element is max `offset+1`
  size_t size;
};

struct double_layer {
  float **element_addrs;
  float *elements;
  struct addr_idx_offsets *idx_offsets;
};

struct creation_functions {
  struct addr_idx_offsets *(*const get_idx_offsets)(size_t);
  int (*const get_current_size)(size_t, const size_t *, const size_t *,
                                size_t *);
};

void destory_addr_idx_offsets(struct addr_idx_offsets *offsets);

struct double_layer *
create_double_layer(bool has_extra_double_buffers,
                    const struct creation_functions *funcs,
                    const struct layer_size_data *layer_size);

void destory_double_layer(struct double_layer *obj);

struct addr_idx_offsets *model_get_idx_offsets(size_t layer_size_count);
int model_get_current_size(size_t i, const size_t *layer_sizes,
                           const size_t *idx_offsets, size_t *current_size);

struct addr_idx_offsets *
learning_buffer_get_idx_offsets(size_t layer_size_count);
int learning_buffer_get_current_size(size_t i, const size_t *layer_sizes,
                                     const size_t *idx_offsets,
                                     size_t *current_size);

struct addr_idx_offsets *
gradiant_buffer_get_idx_offsets(size_t layer_size_count);
int gradiant_buffer_get_current_size(size_t i, const size_t *layer_sizes,
                                     const size_t *idx_offsets,
                                     size_t *current_size);

#endif /* INTERNAL_MLP_CREATION_H_ */