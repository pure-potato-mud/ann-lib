#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <internal/mlp.h>
#include <internal/mlp_creation.h>

void mlp_destroy_content_of_layer_data(struct mlp_layer_data *layer_data) {
  if (layer_data == NULL) { return; }
  if (layer_data->elements_begin != NULL) { free(layer_data->elements_begin); }
  if (layer_data->layers != NULL) { free(layer_data->layers); }
}

// compare two metadata.
// return true any one or its layer_sizes_ref is NULL.
// and check content, false if layer_size_count and layer_sizes_ref are both
// same.
bool is_metadata_not_same(const struct mlp_layer_metadata *a,
                          const struct mlp_layer_metadata *b) {
  if (a == NULL || b == NULL) { return true; }
  if (a->layer_size_count != b->layer_size_count) { return true; }
  if (a->layer_sizes_ref == NULL || b->layer_sizes_ref == NULL) { return true; }
  if (a->layer_sizes_ref == b->layer_sizes_ref) { return false; }
  size_t i;
  for (i = 0U; i < a->layer_size_count; ++i) {
    if (a->layer_sizes_ref[i] != b->layer_sizes_ref[i]) { return true; }
  }
  return false;
}

// WARN: mlp will store poenum mlp_status er layer_sizes, so DO NOT free it
// before mlp been free.
struct mlp_model *mlp_create_model(const struct mlp_layer_size_data *size_data,
                                   enum mlp_status *status) {
  if (size_data == NULL) {
    if (status != NULL) { *status = MLP_STATUS_INVALID_ARGUMENT; }
    return NULL;
  }
  const struct mlp_creation_functions funcs = {mlp_model_get_field_offsets,
                                               mlp_model_get_current_size};
  struct mlp_double_layers *dl =
      mlp_create_double_layers(true, &funcs, size_data, status);
  if (dl == NULL) { return NULL; } // pass status of mlp_create_double_layers
  if (dl->layers == NULL || dl->elements_begin == NULL) {
    if (status != NULL) { *status = MLP_STATUS_ALLOCATION_FAILED; }
    mlp_destroy_double_layers(dl);
    return NULL;
  }
  float **layers = dl->layers;

  struct mlp_model *mlp = (struct mlp_model *)malloc(sizeof(struct mlp_model));
  if (mlp == NULL) {
    if (status != NULL) { *status = MLP_STATUS_ALLOCATION_FAILED; }
    mlp_destroy_double_layers(dl);
    return NULL;
  }

  mlp->layer_metadata.layer_size_count = size_data->layer_size_count;
  mlp->layer_metadata.layer_sizes_ref = size_data->layer_sizes_ref;
  mlp->layer_data.layers = dl->layers;
  mlp->layer_data.elements_begin = dl->elements_begin;
  mlp->layer_data.elements_end = dl->elements_end;

  mlp->buffers = layers;
  mlp->weight_layers = layers + dl->field_offsets->field_offsets[0U];
  mlp->bias_layers = layers + dl->field_offsets->field_offsets[1U];
  mlp->activation_funcs_ref = NULL;

  mlp_set_learning_rate(mlp, 0.0f);
  mlp_fill_biases(mlp, 0.0f);
  mlp_fill_weights(mlp, 0.0f);

  dl->layers = NULL;
  dl->elements_begin = NULL;
  mlp_destroy_double_layers(dl);
  if (status != NULL) { *status = MLP_STATUS_SUCCESS; }
  return mlp;
}

void mlp_destroy_model(struct mlp_model *model) {
  if (model == NULL) { return; }

  mlp_destroy_content_of_layer_data(&model->layer_data);
  free(model);
}

struct mlp_backward_buffer *
mlp_create_backward_buffer(const struct mlp_layer_size_data *size_data,
                           enum mlp_status *status) {
  if (size_data == NULL) {
    if (status != NULL) { *status = MLP_STATUS_INVALID_ARGUMENT; }
    return NULL;
  }
  const struct mlp_creation_functions funcs = {
      mlp_backward_buffer_get_field_offsets,
      mlp_backward_buffer_get_current_size};
  struct mlp_double_layers *dl =
      mlp_create_double_layers(true, &funcs, size_data, status);
  if (dl == NULL) { return NULL; } // pass status of mlp_create_double_layers
  if (dl->layers == NULL || dl->elements_begin == NULL) {
    if (status != NULL) { *status = MLP_STATUS_ALLOCATION_FAILED; }
    mlp_destroy_double_layers(dl);
    return NULL;
  }
  float **layers = dl->layers;

  struct mlp_backward_buffer *buffer =
      (struct mlp_backward_buffer *)malloc(sizeof(struct mlp_backward_buffer));
  if (buffer == NULL) {
    if (status != NULL) { *status = MLP_STATUS_ALLOCATION_FAILED; }
    mlp_destroy_double_layers(dl);
    return NULL;
  }

  buffer->layer_metadata.layer_size_count = size_data->layer_size_count;
  buffer->layer_metadata.layer_sizes_ref = size_data->layer_sizes_ref;
  buffer->layer_data.layers = dl->layers;
  buffer->layer_data.elements_begin = dl->elements_begin;
  buffer->layer_data.elements_end = dl->elements_end;
  buffer->compute = layers;
  buffer->layers = layers + dl->field_offsets->field_offsets[0U];
  buffer->raw_output_layers = layers + dl->field_offsets->field_offsets[1U];

  dl->layers = NULL;
  dl->elements_begin = NULL;
  mlp_destroy_double_layers(dl);
  if (status != NULL) { *status = MLP_STATUS_SUCCESS; }
  return buffer;
}

void mlp_destroy_backward_buffer(struct mlp_backward_buffer *buffer) {
  if (buffer == NULL) { return; }

  mlp_destroy_content_of_layer_data(&buffer->layer_data);
  free(buffer);
}

struct mlp_gradient_buffer *
mlp_create_gradient_buffer(const struct mlp_layer_size_data *size_data,
                           enum mlp_status *status) {
  if (size_data == NULL) {
    if (status != NULL) { *status = MLP_STATUS_INVALID_ARGUMENT; }
    return NULL;
  }
  const struct mlp_creation_functions funcs = {
      mlp_gradient_buffer_get_field_offsets,
      mlp_gradient_buffer_get_current_size};
  struct mlp_double_layers *dl =
      mlp_create_double_layers(false, &funcs, size_data, status);
  if (dl == NULL) { return NULL; } // pass status of mlp_create_double_layers
  if (dl->layers == NULL || dl->elements_begin == NULL) {
    if (status != NULL) { *status = MLP_STATUS_ALLOCATION_FAILED; }
    mlp_destroy_double_layers(dl);
    return NULL;
  }
  float **layers = dl->layers;

  struct mlp_gradient_buffer *buffer =
      (struct mlp_gradient_buffer *)malloc(sizeof(struct mlp_gradient_buffer));
  if (buffer == NULL) {
    if (status != NULL) { *status = MLP_STATUS_ALLOCATION_FAILED; }
    mlp_destroy_double_layers(dl);
    return NULL;
  }

  buffer->layer_metadata.layer_size_count = size_data->layer_size_count;
  buffer->layer_metadata.layer_sizes_ref = size_data->layer_sizes_ref;
  buffer->layer_data.layers = dl->layers;
  buffer->layer_data.elements_begin = dl->elements_begin;
  buffer->layer_data.elements_end = dl->elements_end;
  buffer->weight_gradient_layers = layers;
  buffer->bias_gradient_layers = layers + dl->field_offsets->field_offsets[0U];

  dl->layers = NULL;
  dl->elements_begin = NULL;
  mlp_destroy_double_layers(dl);
  if (status != NULL) { *status = MLP_STATUS_SUCCESS; }
  return buffer;
}

void mlp_destroy_gradient_buffer(struct mlp_gradient_buffer *buffer) {
  if (buffer == NULL) { return; }

  mlp_destroy_content_of_layer_data(&buffer->layer_data);
  free(buffer);
}

enum mlp_status mlp_set_learning_rate(struct mlp_model *model,
                                      float learning_rate) {
  if (model == NULL) { return MLP_STATUS_INVALID_ARGUMENT; }
  model->learning_rate = learning_rate;
  return MLP_STATUS_SUCCESS;
}

enum mlp_status
mlp_set_activation_fns_ref(struct mlp_model *model,
                           float (**activation_fns_ref)(float)) {
  if (model == NULL || activation_fns_ref == NULL) {
    return MLP_STATUS_INVALID_ARGUMENT;
  }
  model->activation_funcs_ref = activation_fns_ref;
  return MLP_STATUS_SUCCESS;
}

enum mlp_status mlp_fill_biases(struct mlp_model *model, const float value) {
  if (model == NULL) { return MLP_STATUS_INVALID_ARGUMENT; }
  if (model->bias_layers == NULL) { return MLP_STATUS_INVALID_FIELD; }
  float *begin = model->bias_layers[0U], *end = model->layer_data.elements_end;
  if (begin == NULL) { return MLP_STATUS_INVALID_LAYER; }
  if (end == NULL) { return MLP_STATUS_INVALID_STATE; }
  float *i;
  for (i = begin; i < end; ++i) {
    *i = value;
  }
  return MLP_STATUS_SUCCESS;
}

enum mlp_status mlp_fill_weights(struct mlp_model *model, const float value) {
  if (model == NULL) { return MLP_STATUS_INVALID_ARGUMENT; }
  if (model->weight_layers == NULL || model->bias_layers == NULL) {
    return MLP_STATUS_INVALID_FIELD;
  }
  float *begin = model->weight_layers[0U], *end = model->bias_layers[0U];
  if (begin == NULL || end == NULL) { return MLP_STATUS_INVALID_LAYER; }
  float *i;
  for (i = begin; i < end; ++i) {
    *i = value;
  }
  return MLP_STATUS_SUCCESS;
}

enum mlp_status mlp_clear_gradient_buffer(struct mlp_gradient_buffer *buffer) {
  if (buffer == NULL) { return MLP_STATUS_INVALID_ARGUMENT; }
  if (buffer->weight_gradient_layers == NULL) {
    return MLP_STATUS_INVALID_FIELD;
  }
  float *begin = buffer->weight_gradient_layers[0U],
        *end = buffer->layer_data.elements_end;
  if (begin == NULL) { return MLP_STATUS_INVALID_LAYER; }
  if (end == NULL) { return MLP_STATUS_INVALID_STATE; }
  float *i;
  for (i = begin; i < end; ++i) {
    *i = 0.0f;
  }
  return MLP_STATUS_SUCCESS;
}

enum mlp_status mlp_scale_gradients(struct mlp_gradient_buffer *buffer,
                                    float scalar) {
  if (buffer == NULL) { return MLP_STATUS_INVALID_ARGUMENT; }
  if (buffer->weight_gradient_layers == NULL) {
    return MLP_STATUS_INVALID_FIELD;
  }
  float *begin = buffer->weight_gradient_layers[0U],
        *end = buffer->layer_data.elements_end;
  if (begin == NULL) { return MLP_STATUS_INVALID_LAYER; }
  if (end == NULL) { return MLP_STATUS_INVALID_STATE; }
  float *i;
  for (i = begin; i < end; ++i) {
    *i *= scalar;
  }
  return MLP_STATUS_SUCCESS;
}

enum mlp_status mlp_forward(struct mlp_model *model, const float *input,
                            float *output,
                            struct mlp_backward_buffer *backward_buffer) {
  struct mlp_backward_buffer *local_backward_buffer = backward_buffer;
  if (model == NULL || input == NULL) { return MLP_STATUS_INVALID_ARGUMENT; }
  if (local_backward_buffer != NULL &&
      is_metadata_not_same(&model->layer_metadata,
                           &local_backward_buffer->layer_metadata)) {
    return MLP_STATUS_TOPOLOGY_MISMATCH;
  }

  const size_t max_layer_idx = model->layer_metadata.layer_size_count - 1U;
  const size_t *layer_sizes = model->layer_metadata.layer_sizes_ref;
  float **const buffers = model->buffers;
  float **const biases = model->bias_layers;
  float **const weights = model->weight_layers;
  float (**activation_funcs_ref)(float) = model->activation_funcs_ref;
  if (layer_sizes == NULL || buffers == NULL || biases == NULL ||
      weights == NULL || activation_funcs_ref == NULL) {
    return MLP_STATUS_INVALID_FIELD;
  }
  float *const input_layer = buffers[0U];
  if (input_layer == NULL) { return MLP_STATUS_INVALID_LAYER; }

  size_t i, j, k;
  float **layers = NULL, **raw_output_layers = NULL;
  if (local_backward_buffer != NULL) {
    layers = local_backward_buffer->layers;
    raw_output_layers = local_backward_buffer->raw_output_layers;
  }
  if (layers != NULL && raw_output_layers != NULL) {
    for (i = 0U; i < max_layer_idx; ++i) {
      if (layers[i] == NULL || raw_output_layers[i] == NULL) {
        local_backward_buffer = NULL;
      }
    }
    if (layers[max_layer_idx] == NULL) { local_backward_buffer = NULL; }
  } else {
    local_backward_buffer = NULL;
  }

  for (i = 0U; i < layer_sizes[0U]; ++i) {
    input_layer[i] = input[i];
  }
  if (local_backward_buffer != NULL) {
    for (i = 0U; i < layer_sizes[0U]; ++i) {
      layers[0U][i] = input[i];
    }
  }

  size_t back_layer_size, front_layer_size;
  size_t j_idx;
  float *current_layer = NULL, *next_layer = NULL, *weight = NULL, *bias = NULL;
  float (*activation_function)(float) = NULL;

  for (i = 0U; i < max_layer_idx; ++i) {
    back_layer_size = layer_sizes[i];
    front_layer_size = layer_sizes[i + 1];

    current_layer = buffers[i % 2U];
    next_layer = buffers[(i + 1U) % 2U];
    weight = weights[i];
    bias = biases[i];
    activation_function = activation_funcs_ref[i];

    if (current_layer == NULL || next_layer == NULL || weight == NULL ||
        bias == NULL || activation_function == NULL) {
      return MLP_STATUS_INVALID_LAYER;
    }

    for (j = 0U; j < front_layer_size; ++j) {
      next_layer[j] = bias[j];
      j_idx = j * back_layer_size;

      for (k = 0U; k < back_layer_size; ++k) {
        next_layer[j] += weight[j_idx + k] * current_layer[k];
      }
      if (local_backward_buffer != NULL) {
        raw_output_layers[i][j] = next_layer[j];
      }

      next_layer[j] = activation_function(next_layer[j]);
    }
    if (local_backward_buffer != NULL) {
      for (j = 0U; j < front_layer_size; ++j) {
        layers[i + 1U][j] = next_layer[j];
      }
    }
  }
  if (output != NULL) {
    float *final_layer = buffers[max_layer_idx % 2U];
    for (i = 0U; i < layer_sizes[max_layer_idx]; ++i) {
      output[i] = final_layer[i];
    }
  }
  return MLP_STATUS_SUCCESS;
}

enum mlp_status mlp_backward(const struct mlp_model *model,
                             const struct mlp_backward_buffer *backward_buffer,
                             const float *const gradients,
                             struct mlp_gradient_buffer *gradient_total) {
  if (model == NULL || backward_buffer == NULL || gradient_total == NULL ||
      gradients == NULL) {
    return MLP_STATUS_INVALID_ARGUMENT;
  }
  if (is_metadata_not_same(&model->layer_metadata,
                           &backward_buffer->layer_metadata) ||
      is_metadata_not_same(&model->layer_metadata,
                           &gradient_total->layer_metadata)) {
    return MLP_STATUS_TOPOLOGY_MISMATCH;
  }

  const size_t layer_size_count = model->layer_metadata.layer_size_count;
  const size_t last_layer_idx = layer_size_count - 1U;

  const size_t *layer_sizes = model->layer_metadata.layer_sizes_ref;
  float **const compute_buffer = backward_buffer->compute;
  float **const layers = backward_buffer->layers;
  float **const raw_output_layers = backward_buffer->raw_output_layers;
  float **const weight_gradient_layers = gradient_total->weight_gradient_layers;
  float **const bias_gradient_layers = gradient_total->bias_gradient_layers;
  float **const weight_layers = model->weight_layers;
  float (**activation_funcs_ref)(float) = model->activation_funcs_ref;
  if (layer_sizes == NULL || compute_buffer == NULL || layers == NULL ||
      raw_output_layers == NULL || weight_gradient_layers == NULL ||
      bias_gradient_layers == NULL || weight_layers == NULL ||
      activation_funcs_ref == NULL) {
    return MLP_STATUS_INVALID_FIELD;
  }

  float *final_backward_layer = compute_buffer[last_layer_idx % 2U];
  if (final_backward_layer == NULL) { return MLP_STATUS_INVALID_LAYER; }
  size_t i, j, k;
  for (i = 0; i < layer_sizes[last_layer_idx]; ++i) {
    final_backward_layer[i] = gradients[i];
  }

  size_t back_layer_size, front_layer_size;
  size_t j_idx;
  float *layer = NULL, *output_layer = NULL, *current_backward_buffer = NULL,
        *last_backward_buffer = NULL, *weight_gradient_layer = NULL,
        *bias_gradient_layer = NULL, *weights = NULL;
  float (*activation_function)(float) = NULL;
  for (i = last_layer_idx; i > 0U;) {
    --i;

    front_layer_size = layer_sizes[i + 1U];
    back_layer_size = layer_sizes[i];

    layer = layers[i];
    current_backward_buffer = compute_buffer[(i + 1U) % 2U];
    last_backward_buffer = compute_buffer[i % 2U];
    weight_gradient_layer = weight_gradient_layers[i];
    bias_gradient_layer = bias_gradient_layers[i];
    output_layer = raw_output_layers[i];
    weights = weight_layers[i];
    activation_function = activation_funcs_ref[i];
    if (layer == NULL || current_backward_buffer == NULL ||
        last_backward_buffer == NULL || weight_gradient_layer == NULL ||
        bias_gradient_layer == NULL || output_layer == NULL ||
        weights == NULL || activation_function == NULL) {
      return MLP_STATUS_INVALID_LAYER;
    }

    for (j = 0U; j < front_layer_size; ++j) {
      // dL/da * da/dz * dz/dn
      // L is for loss function
      // a is for activate function
      // z is for output
      // n is for any part, there are weight(w), input(x) and bias(b).

      // = dL/da * da/dz
      current_backward_buffer[j] *=
          (activation_function(output_layer[j] + 4.76837e-7f) -
           activation_function(output_layer[j] - 4.76837e-7f)) /
          9.53674e-7f;

      j_idx = j * back_layer_size;
      for (k = 0U; k < back_layer_size; ++k) {

        // pass gradient to last layer first
        // dz/dx = w
        last_backward_buffer[k] =
            (j == 0U ? 0.0f : last_backward_buffer[k]) +
            weights[j_idx + k] * current_backward_buffer[j];

        // dz/dw = x
        weight_gradient_layer[j_idx + k] +=
            layer[k] * current_backward_buffer[j];
      }

      // dz/db = 1
      bias_gradient_layer[j] += current_backward_buffer[j];
    }
  }

  return MLP_STATUS_SUCCESS;
}

enum mlp_status
mlp_update_parameters_sgd(mlp_model_t *model,
                          const mlp_gradient_buffer_t *gradient_total) {
  if (model == NULL || gradient_total == NULL) {
    return MLP_STATUS_INVALID_ARGUMENT;
  }
  if (is_metadata_not_same(&model->layer_metadata,
                           &gradient_total->layer_metadata)) {
    return MLP_STATUS_TOPOLOGY_MISMATCH;
  }

  const float learn_rate = model->learning_rate;
  const size_t layer_size_count = model->layer_metadata.layer_size_count;
  const size_t last_layer_idx = layer_size_count - 1U;

  const size_t *layer_sizes = model->layer_metadata.layer_sizes_ref;
  float **const weight_gradient_layers = gradient_total->weight_gradient_layers;
  float **const bias_gradient_layers = gradient_total->bias_gradient_layers;
  float **const weight_layers = model->weight_layers;
  float **const bias_layers = model->bias_layers;
  float (**activation_funcs_ref)(float) = model->activation_funcs_ref;
  if (layer_sizes == NULL || weight_gradient_layers == NULL ||
      bias_gradient_layers == NULL || weight_layers == NULL ||
      bias_layers == NULL || activation_funcs_ref == NULL) {
    return MLP_STATUS_INVALID_FIELD;
  }

  size_t i, j, k;
  size_t back_layer_size, front_layer_size;
  size_t j_idx;
  float *weight_gradient_layer = NULL, *bias_gradient_layer = NULL,
        *weights = NULL, *biases = NULL;
  float (*activation_function)(float) = NULL;
  for (i = last_layer_idx; i > 0U;) {
    --i;

    front_layer_size = layer_sizes[i + 1U];
    back_layer_size = layer_sizes[i];

    weight_gradient_layer = weight_gradient_layers[i];
    bias_gradient_layer = bias_gradient_layers[i];
    weights = weight_layers[i];
    biases = bias_layers[i];
    activation_function = activation_funcs_ref[i];
    if (weight_gradient_layer == NULL || bias_gradient_layer == NULL ||
        weights == NULL || biases == NULL || activation_function == NULL) {
      return MLP_STATUS_INVALID_LAYER;
    }

    for (j = 0U; j < front_layer_size; ++j) {
      j_idx = j * back_layer_size;
      for (k = 0U; k < back_layer_size; ++k) {
        weights[j_idx + k] -= learn_rate * weight_gradient_layer[j_idx + k];
      }
      biases[j] -= learn_rate * bias_gradient_layer[j];
    }
  }

  return MLP_STATUS_SUCCESS;
}
