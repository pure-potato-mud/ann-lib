#include <stdint.h>
#include <stdlib.h>

#include <internal/mlp_creation.h>

void destory_addr_idx_offsets(struct mlp_addr_idx_offsets *offsets) {
  if (offsets == NULL) { return; }

  size_t *data = offsets->data;
  free(offsets);
  if (data == NULL) { return; }

  free(data);
}

struct mlp_double_layer *
create_double_layer(bool has_extra_double_buffers,
                    const struct mlp_creation_functions *funcs,
                    const struct mlp_layer_size_data *layer_size) {
  if (funcs == NULL || layer_size == NULL) { return NULL; }
  struct mlp_addr_idx_offsets *(*const get_idx_offsets)(size_t) =
      funcs->get_idx_offsets;
  int (*const get_current_size)(size_t, const size_t *, const size_t *,
                                size_t *) = funcs->get_current_size;
  if (get_idx_offsets == NULL || get_current_size == NULL) { return NULL; }
  const size_t *const layer_sizes_ref = layer_size->layer_sizes_ref;
  const size_t layer_size_count = layer_size->layer_size_count;
  if (layer_sizes_ref == NULL || layer_size_count < 2U) { return NULL; }

  // allocate elements_ptrs and addr offsets
  // need layer_full_size and layer_offsets
  struct mlp_addr_idx_offsets *idx_offsets_obj =
      get_idx_offsets(layer_size_count);
  if (idx_offsets_obj == NULL) { return NULL; }
  const size_t idx_offsets_size = idx_offsets_obj->size;
  const size_t *idx_offsets = idx_offsets_obj->data;
  if (idx_offsets_size < 1U) {
    destory_addr_idx_offsets(idx_offsets_obj);
    return NULL;
  }
  if (idx_offsets == NULL) {
    destory_addr_idx_offsets(idx_offsets_obj);
    return NULL;
  }
  const size_t layer_full_size = idx_offsets[0U];
  if (SIZE_MAX / sizeof(size_t) < layer_full_size ||
      SIZE_MAX / sizeof(float *) < layer_full_size) {
    destory_addr_idx_offsets(idx_offsets_obj);
    return NULL;
  }
  size_t *addr_offsets = (size_t *)malloc(layer_full_size * sizeof(size_t));
  if (addr_offsets == NULL) {
    destory_addr_idx_offsets(idx_offsets_obj);
    return NULL;
  }
  float **element_addrs = (float **)malloc(layer_full_size * sizeof(float *));
  if (element_addrs == NULL) {
    free(addr_offsets);
    destory_addr_idx_offsets(idx_offsets_obj);
    return NULL;
  }

  // compute addr offsets
  // using layer_full_size and layer_offsets
  // optional: calculate max_layer_size, and adjust offset_sum
  size_t i, offset_sum = 0, current_size;
  int result;
  for (i = 0; i < layer_size_count; ++i) {
    if (layer_sizes_ref[i] == 0U) {
      destory_addr_idx_offsets(idx_offsets_obj);
      free(addr_offsets);
      free(element_addrs);
      return NULL;
    }
  }
  for (i = 0; i < layer_full_size; ++i) {
    result = get_current_size(i, layer_sizes_ref, idx_offsets, &current_size);
    if (result != 0) {
      destory_addr_idx_offsets(idx_offsets_obj);
      free(addr_offsets);
      free(element_addrs);
      return NULL;
    }
    if (SIZE_MAX - current_size < offset_sum) {
      destory_addr_idx_offsets(idx_offsets_obj);
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
    destory_addr_idx_offsets(idx_offsets_obj);
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
      destory_addr_idx_offsets(idx_offsets_obj);
      free(addr_offsets);
      free(element_addrs);
      return NULL;
    }
    offset_sum += 2U * max_layer_size;
  }

  // allocate elements

  if (SIZE_MAX / sizeof(float) < offset_sum) {
    destory_addr_idx_offsets(idx_offsets_obj);
    free(addr_offsets);
    free(element_addrs);
    return NULL;
  }
  float *elements = (float *)malloc(offset_sum * sizeof(float));
  if (elements == NULL) {
    destory_addr_idx_offsets(idx_offsets_obj);
    free(addr_offsets);
    free(element_addrs);
    return NULL;
  }

  // apply elements addr to elements_ptrs
  if (has_extra_double_buffers) {
    element_addrs[0U] = elements;
    element_addrs[1U] = elements + max_layer_size;
    for (i = idx_offsets[1]; i < layer_full_size; ++i) {
      element_addrs[i] = elements + addr_offsets[i] + 2U * max_layer_size;
    }
  } else {
    for (i = 0U; i < layer_full_size; ++i) {
      element_addrs[i] = elements + addr_offsets[i];
    }
  }
  free(addr_offsets);

  // return idx_offsets
  struct mlp_double_layer *ret =
      (struct mlp_double_layer *)malloc(sizeof(struct mlp_double_layer));
  if (ret == NULL) {
    destory_addr_idx_offsets(idx_offsets_obj);
    free(elements);
    free(element_addrs);
    return NULL;
  }

  ret->element_ptrs = element_addrs;
  ret->elements = elements;
  ret->idx_offsets = idx_offsets_obj;

  return ret;
}

void destory_double_layer(struct mlp_double_layer *obj) {
  if (obj == NULL) { return; }

  if (obj->element_ptrs != NULL) { free(obj->element_ptrs); }
  if (obj->elements != NULL) { free(obj->elements); }
  if (obj->idx_offsets != NULL) { destory_addr_idx_offsets(obj->idx_offsets); }
  free(obj);
}

struct mlp_addr_idx_offsets *model_get_idx_offsets(size_t layer_size_count) {
  if (SIZE_MAX / 2U < layer_size_count) { return NULL; }
  const size_t layer_full_size = layer_size_count * 2U;

  const size_t data_size = 3U;
  size_t *data = (size_t *)malloc(data_size * sizeof(size_t));
  if (data == NULL) { return NULL; }
  data[0U] = layer_full_size;
  data[1U] = 2U;
  data[2U] = layer_size_count + 1U;

  struct mlp_addr_idx_offsets *ret = (struct mlp_addr_idx_offsets *)malloc(
      sizeof(struct mlp_addr_idx_offsets));
  if (ret == NULL) {
    free(data);
    return NULL;
  }

  ret->data = data;
  ret->size = data_size;

  return ret;
}

int model_get_current_size(size_t i, const size_t *layer_sizes,
                           const size_t *idx_offsets, size_t *current_size) {
  const size_t weight_offset = idx_offsets[1U], bias_offset = idx_offsets[2U];
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

struct mlp_addr_idx_offsets *
backward_buffer_get_idx_offsets(size_t layer_size_count) {
  if (SIZE_MAX / 2U < layer_size_count) { return NULL; }
  const size_t layer_full_size = layer_size_count * 2U - 1U;

  const size_t data_size = 2U;
  size_t *data = (size_t *)malloc(data_size * sizeof(size_t));
  if (data == NULL) { return NULL; }
  data[0U] = layer_full_size;
  data[1U] = layer_size_count;

  struct mlp_addr_idx_offsets *ret = (struct mlp_addr_idx_offsets *)malloc(
      sizeof(struct mlp_addr_idx_offsets));
  if (ret == NULL) {
    free(data);
    return NULL;
  }

  ret->data = data;
  ret->size = data_size;

  return ret;
}

int backward_buffer_get_current_size(size_t i, const size_t *layer_sizes,
                                     const size_t *idx_offsets,
                                     size_t *current_size) {
  size_t raw_output_offset = idx_offsets[1U];
  if (i < raw_output_offset) {
    *current_size = layer_sizes[i];
  } else {
    *current_size = layer_sizes[i - raw_output_offset + 1U];
  }
  return 0;
}

struct mlp_addr_idx_offsets *
gradient_buffer_get_idx_offsets(size_t layer_size_count) {
  if (SIZE_MAX / 2U < layer_size_count) { return NULL; }
  const size_t layer_full_size = layer_size_count - 1U;

  const size_t data_size = 1U;
  size_t *data = (size_t *)malloc(data_size * sizeof(size_t));
  if (data == NULL) { return NULL; }
  data[0U] = layer_full_size;

  struct mlp_addr_idx_offsets *ret = (struct mlp_addr_idx_offsets *)malloc(
      sizeof(struct mlp_addr_idx_offsets));
  if (ret == NULL) {
    free(data);
    return NULL;
  }

  ret->data = data;
  ret->size = data_size;

  return ret;
}

int gradient_buffer_get_current_size(size_t i, const size_t *layer_sizes,
                                     const size_t *idx_offsets,
                                     size_t *current_size) {
  *current_size = layer_sizes[i + 1U];
  return 0;
}
