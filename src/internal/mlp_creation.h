#ifndef INTERNAL_MLP_CREATION_H_
#define INTERNAL_MLP_CREATION_H_

#include <stdbool.h>
#include <stddef.h>

struct mlp_layer_size_data {
  const size_t *layer_sizes_ref;
  size_t layer_size_count;
};

struct mlp_field_offsets {
  size_t *field_offsets; // last element is `max_offset + 1`
  size_t field_offset_count;
};

struct mlp_double_layer {
  float **element_ptrs;
  float *elements;
  struct mlp_field_offsets *field_offsets;
};

struct mlp_creation_functions {
  struct mlp_field_offsets *(*const get_field_offsets)(size_t);
  int (*const get_current_size)(size_t, const size_t *, const size_t *,
                                size_t *);
};

void mlp_destory_field_offsets(struct mlp_field_offsets *offsets);

struct mlp_double_layer *
mlp_create_double_layer(bool has_extra_double_buffers,
                        const struct mlp_creation_functions *funcs,
                        const struct mlp_layer_size_data *layer_size);

void mlp_destory_double_layer(struct mlp_double_layer *obj);

struct mlp_field_offsets *mlp_model_get_field_offsets(size_t layer_size_count);
int mlp_model_get_current_size(size_t i, const size_t *layer_sizes,
                               const size_t *field_offsets,
                               size_t *current_size);

struct mlp_field_offsets *
mlp_backward_buffer_get_field_offsets(size_t layer_size_count);
int mlp_backward_buffer_get_current_size(size_t i, const size_t *layer_sizes,
                                         const size_t *field_offsets,
                                         size_t *current_size);

struct mlp_field_offsets *
mlp_gradient_buffer_get_field_offsets(size_t layer_size_count);
int mlp_gradient_buffer_get_current_size(size_t i, const size_t *layer_sizes,
                                         const size_t *field_offsets,
                                         size_t *current_size);

#endif /* INTERNAL_MLP_CREATION_H_ */