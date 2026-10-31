#include <stdint.h>
#include <stdlib.h>

#include <internal/mlp_creation.h>

void mlp_destory_field_offsets(struct mlp_field_offsets *offsets) {
  if (offsets == NULL) { return; }

  size_t *data = offsets->field_offsets;
  free(offsets);
  if (data == NULL) { return; }

  free(data);
}

struct mlp_double_layer *
mlp_create_double_layer(bool has_extra_double_buffers,
                        const struct mlp_creation_functions *funcs,
                        const struct mlp_layer_size_data *layer_size) {
  if (funcs == NULL || layer_size == NULL) { return NULL; }
  struct mlp_field_offsets *(*const get_field_offsets)(size_t) =
      funcs->get_field_offsets;
  int (*const get_current_size)(size_t, const size_t *, const size_t *,
                                size_t *) = funcs->get_current_size;
  if (get_field_offsets == NULL || get_current_size == NULL) { return NULL; }
  const size_t *const layer_sizes_ref = layer_size->layer_sizes_ref;
  const size_t layer_size_count = layer_size->layer_size_count;
  if (layer_sizes_ref == NULL || layer_size_count < 2U) { return NULL; }

  // allocate elements_ptrs and addr offsets
  // need layer_full_size and layer_offsets
  struct mlp_field_offsets *field_offsets_obj =
      get_field_offsets(layer_size_count);
  if (field_offsets_obj == NULL) { return NULL; }
  const size_t field_offset_count = field_offsets_obj->field_offset_count;
  const size_t *field_offsets = field_offsets_obj->field_offsets;
  if (field_offset_count < 1U) {
    mlp_destory_field_offsets(field_offsets_obj);
    return NULL;
  }
  if (field_offsets == NULL) {
    mlp_destory_field_offsets(field_offsets_obj);
    return NULL;
  }
  const size_t layer_full_size = field_offsets[0U];
  if (SIZE_MAX / sizeof(size_t) < layer_full_size ||
      SIZE_MAX / sizeof(float *) < layer_full_size) {
    mlp_destory_field_offsets(field_offsets_obj);
    return NULL;
  }
  size_t *addr_offsets = (size_t *)malloc(layer_full_size * sizeof(size_t));
  if (addr_offsets == NULL) {
    mlp_destory_field_offsets(field_offsets_obj);
    return NULL;
  }
  float **element_addrs = (float **)malloc(layer_full_size * sizeof(float *));
  if (element_addrs == NULL) {
    free(addr_offsets);
    mlp_destory_field_offsets(field_offsets_obj);
    return NULL;
  }

  // compute addr offsets
  // using layer_full_size and layer_offsets
  // optional: calculate max_layer_size, and adjust offset_sum
  size_t i, offset_sum = 0, current_size;
  int result;
  for (i = 0; i < layer_size_count; ++i) {
    if (layer_sizes_ref[i] == 0U) {
      mlp_destory_field_offsets(field_offsets_obj);
      free(addr_offsets);
      free(element_addrs);
      return NULL;
    }
  }
  for (i = 0; i < layer_full_size; ++i) {
    result = get_current_size(i, layer_sizes_ref, field_offsets, &current_size);
    if (result != 0) {
      mlp_destory_field_offsets(field_offsets_obj);
      free(addr_offsets);
      free(element_addrs);
      return NULL;
    }
    if (SIZE_MAX - current_size < offset_sum) {
      mlp_destory_field_offsets(field_offsets_obj);
      free(addr_offsets);
      free(element_addrs);
      return NULL;
    }
    // get address offset
    addr_offsets[i] = offset_sum;
    offset_sum += current_size;
  }

  size_t max_layer_size = layer_sizes_ref[0U];
  if (max_layer_size == 0) {
    mlp_destory_field_offsets(field_offsets_obj);
    free(addr_offsets);
    free(element_addrs);
    return NULL;
  }
  for (i = 0; i < layer_size_count; ++i) {
    if (max_layer_size < layer_sizes_ref[i]) {
      max_layer_size = layer_sizes_ref[i];
    }
  }

  // deal with max_layer_size
  if (has_extra_double_buffers) {
    if ((SIZE_MAX - offset_sum) / 2U < max_layer_size) {
      mlp_destory_field_offsets(field_offsets_obj);
      free(addr_offsets);
      free(element_addrs);
      return NULL;
    }
    offset_sum += 2U * max_layer_size;
  }

  // allocate elements

  if (SIZE_MAX / sizeof(float) < offset_sum) {
    mlp_destory_field_offsets(field_offsets_obj);
    free(addr_offsets);
    free(element_addrs);
    return NULL;
  }
  float *elements = (float *)malloc(offset_sum * sizeof(float));
  if (elements == NULL) {
    mlp_destory_field_offsets(field_offsets_obj);
    free(addr_offsets);
    free(element_addrs);
    return NULL;
  }

  // apply elements addr to elements_ptrs
  if (has_extra_double_buffers) {
    element_addrs[0U] = elements;
    element_addrs[1U] = elements + max_layer_size;
    for (i = field_offsets[1]; i < layer_full_size; ++i) {
      element_addrs[i] = elements + addr_offsets[i] + 2U * max_layer_size;
    }
  } else {
    for (i = 0U; i < layer_full_size; ++i) {
      element_addrs[i] = elements + addr_offsets[i];
    }
  }
  free(addr_offsets);

  // return field_offsets
  struct mlp_double_layer *ret =
      (struct mlp_double_layer *)malloc(sizeof(struct mlp_double_layer));
  if (ret == NULL) {
    mlp_destory_field_offsets(field_offsets_obj);
    free(elements);
    free(element_addrs);
    return NULL;
  }

  ret->element_ptrs = element_addrs;
  ret->elements = elements;
  ret->field_offsets = field_offsets_obj;

  return ret;
}

void mlp_destory_double_layer(struct mlp_double_layer *obj) {
  if (obj == NULL) { return; }

  if (obj->element_ptrs != NULL) { free(obj->element_ptrs); }
  if (obj->elements != NULL) { free(obj->elements); }
  if (obj->field_offsets != NULL) {
    mlp_destory_field_offsets(obj->field_offsets);
  }
  free(obj);
}

struct mlp_field_offsets *mlp_model_get_field_offsets(size_t layer_size_count) {
  if (SIZE_MAX / 2U < layer_size_count) { return NULL; }
  const size_t layer_full_size = layer_size_count * 2U;

  const size_t data[] = {layer_full_size, 2U, layer_size_count + 1U};
  const size_t data_size = sizeof(data) / sizeof(size_t);
  size_t *copied_data = (size_t *)malloc(data_size * sizeof(size_t));
  if (copied_data == NULL) { return NULL; }
  size_t i;
  for (i = 0; i < data_size; ++i) {
    copied_data[i] = data[i];
  }

  struct mlp_field_offsets *ret =
      (struct mlp_field_offsets *)malloc(sizeof(struct mlp_field_offsets));
  if (ret == NULL) {
    free(copied_data);
    return NULL;
  }

  ret->field_offsets = copied_data;
  ret->field_offset_count = data_size;

  return ret;
}

int mlp_model_get_current_size(size_t i, const size_t *layer_sizes,
                               const size_t *field_offsets,
                               size_t *current_size) {
  const size_t weight_offset = field_offsets[1U],
               bias_offset = field_offsets[2U];
  size_t current_layer_idx, current_neruon_count;
  if (i < weight_offset) {
    *current_size = 0;
  } else if (i < bias_offset) { // offsets for weights
    current_layer_idx = i - weight_offset + 1U;
    current_neruon_count = layer_sizes[current_layer_idx];
    // multification overflow test
    if (SIZE_MAX / current_neruon_count < layer_sizes[current_layer_idx - 1U]) {
      return 1;
      /* free(parameter_addr_offsets);
      free(element_ptrs);
      return NULL; */
    }
    *current_size = current_neruon_count * layer_sizes[current_layer_idx - 1U];
  } else { // offsets for bias
    *current_size = layer_sizes[i - bias_offset + 1U];
  }
  return 0;
}

struct mlp_field_offsets *
mlp_backward_buffer_get_field_offsets(size_t layer_size_count) {
  if (SIZE_MAX / 2U < layer_size_count) { return NULL; }
  const size_t layer_full_size = layer_size_count * 2U - 1U;

  const size_t data[] = {layer_full_size, layer_size_count};
  const size_t data_size = sizeof(data) / sizeof(size_t);
  size_t *copied_data = (size_t *)malloc(data_size * sizeof(size_t));
  if (copied_data == NULL) { return NULL; }
  size_t i;
  for (i = 0; i < data_size; ++i) {
    copied_data[i] = data[i];
  }

  struct mlp_field_offsets *ret =
      (struct mlp_field_offsets *)malloc(sizeof(struct mlp_field_offsets));
  if (ret == NULL) {
    free(copied_data);
    return NULL;
  }

  ret->field_offsets = copied_data;
  ret->field_offset_count = data_size;

  return ret;
}

int mlp_backward_buffer_get_current_size(size_t i, const size_t *layer_sizes,
                                         const size_t *field_offsets,
                                         size_t *current_size) {
  size_t raw_output_offset = field_offsets[1U];
  if (i < raw_output_offset) {
    *current_size = layer_sizes[i];
  } else {
    *current_size = layer_sizes[i - raw_output_offset + 1U];
  }
  return 0;
}

struct mlp_field_offsets *
mlp_gradient_buffer_get_field_offsets(size_t layer_size_count) {
  if (SIZE_MAX / 2U < layer_size_count) { return NULL; }
  const size_t layer_full_size = layer_size_count - 1U;

  const size_t data[] = {layer_full_size};
  const size_t data_size = sizeof(data) / sizeof(size_t);
  size_t *copied_data = (size_t *)malloc(data_size * sizeof(size_t));
  if (copied_data == NULL) { return NULL; }
  size_t i;
  for (i = 0; i < data_size; ++i) {
    copied_data[i] = data[i];
  }

  struct mlp_field_offsets *ret =
      (struct mlp_field_offsets *)malloc(sizeof(struct mlp_field_offsets));
  if (ret == NULL) {
    free(copied_data);
    return NULL;
  }

  ret->field_offsets = copied_data;
  ret->field_offset_count = data_size;

  return ret;
}

int mlp_gradient_buffer_get_current_size(size_t i, const size_t *layer_sizes,
                                         const size_t *field_offsets,
                                         size_t *current_size) {
  *current_size = layer_sizes[i + 1U];
  return 0;
}
