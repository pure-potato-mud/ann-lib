#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <internal/mlp.h>
#include <internal/mlp_creation.h>

// WARN: mlp will store pointer layer_sizes, so DO NOT free it before mlp been
// free.
struct mlp_model *mlp_create_model(const size_t *layer_sizes_ref,
                                   size_t layer_size_count) {
  const struct creation_functions funcs = {model_get_idx_offsets,
                                           model_get_current_size};
  const struct layer_size_data layer_size_data = {layer_sizes_ref,
                                                  layer_size_count};
  struct double_layer *dl = create_double_layer(true, &funcs, &layer_size_data);
  if (dl == NULL) { return NULL; }
  if (dl->element_addrs == NULL || dl->elements == NULL) {
    destory_double_layer(dl);
    return NULL;
  }
  float **element_addrs = dl->element_addrs;

  struct mlp_model *mlp = (struct mlp_model *)malloc(sizeof(struct mlp_model));
  if (mlp == NULL) {
    destory_double_layer(dl);
    return NULL;
  }

  mlp->layer_size_count = layer_size_count;
  mlp->layer_sizes_ref = layer_sizes_ref;
  mlp->buffers = element_addrs;
  mlp->weight_layers = element_addrs + dl->idx_offsets->data[1];
  mlp->bias_layers = element_addrs + dl->idx_offsets->data[2];
  mlp->activation_funcs_ref = NULL;

  mlp_set_learning_rate(mlp, 0.0f);
  mlp_fill_biases(mlp, 0.0f);
  mlp_fill_weights(mlp, 0.0f);

  dl->element_addrs = NULL;
  dl->elements = NULL;
  destory_double_layer(dl);
  return mlp;
}

void mlp_destroy_model(struct mlp_model *model) {
  if (model == NULL) { return; }

  float **element_ptrs = model->buffers;
  free(model);
  if (element_ptrs == NULL) { return; }

  float *elements = element_ptrs[0U];
  free(element_ptrs);
  if (elements == NULL) { return; }

  free(elements);
}

struct mlp_backward_buffer *
mlp_create_backward_buffer(struct mlp_model *model) {
  if (model == NULL) { return NULL; }

  const size_t layer_size_count = model->layer_size_count;
  const size_t *const layer_sizes_ref = model->layer_sizes_ref;

  const struct creation_functions funcs = {backward_buffer_get_idx_offsets,
                                           backward_buffer_get_current_size};
  const struct layer_size_data layer_size_data = {layer_sizes_ref,
                                                  layer_size_count};
  struct double_layer *dl =
      create_double_layer(false, &funcs, &layer_size_data);
  if (dl == NULL) { return NULL; }
  if (dl->element_addrs == NULL || dl->elements == NULL) {
    destory_double_layer(dl);
    return NULL;
  }
  float **element_addrs = dl->element_addrs;

  struct mlp_backward_buffer *buffer =
      (struct mlp_backward_buffer *)malloc(sizeof(struct mlp_backward_buffer));
  if (buffer == NULL) {
    destory_double_layer(dl);
    return NULL;
  }

  buffer->layers = element_addrs;
  buffer->raw_output_layers = element_addrs + dl->idx_offsets->data[1];

  dl->element_addrs = NULL;
  dl->elements = NULL;
  destory_double_layer(dl);
  return buffer;
}

void mlp_destroy_backward_buffer(struct mlp_backward_buffer *buffer) {
  if (buffer == NULL) { return; }

  float **layers = buffer->layers;
  free(buffer);
  if (layers == NULL) { return; }

  float *elements = layers[0U];
  free(layers);
  if (elements == NULL) { return; }

  free(elements);
}

struct mlp_gradient_buffer *
mlp_create_gradient_buffer(struct mlp_model *model) {
  if (model == NULL) { return NULL; }

  const size_t layer_size_count = model->layer_size_count;
  const size_t *const layer_sizes_ref = model->layer_sizes_ref;

  const struct creation_functions funcs = {gradient_buffer_get_idx_offsets,
                                           gradient_buffer_get_current_size};
  const struct layer_size_data layer_size_data = {layer_sizes_ref,
                                                  layer_size_count};
  struct double_layer *dl =
      create_double_layer(false, &funcs, &layer_size_data);
  if (dl == NULL) { return NULL; }
  if (dl->element_addrs == NULL || dl->elements == NULL) {
    destory_double_layer(dl);
    return NULL;
  }
  float **element_addrs = dl->element_addrs;

  struct mlp_gradient_buffer *buffer =
      (struct mlp_gradient_buffer *)malloc(sizeof(struct mlp_gradient_buffer));
  if (buffer == NULL) {
    destory_double_layer(dl);
    return NULL;
  }

  buffer->gradient_layers = element_addrs;

  dl->element_addrs = NULL;
  dl->elements = NULL;
  destory_double_layer(dl);
  return buffer;
}

void mlp_destroy_gradient_buffer(struct mlp_gradient_buffer *buffer) {
  if (buffer == NULL) { return; }

  float **gradient_layers = buffer->gradient_layers;
  free(buffer);
  if (gradient_layers == NULL) { return; }

  float *gradients = gradient_layers[0U];
  free(gradient_layers);
  if (gradients == NULL) { return; }

  free(gradients);
}

int mlp_set_learning_rate(struct mlp_model *model, float learning_rate) {
  if (model == NULL) { return 1; }
  model->learning_rate = learning_rate;
  return 0;
}

int mlp_set_activation_fns_ref(struct mlp_model *model,
                               float (**activation_fns_ref)(float)) {
  if (model == NULL) { return 1; }
  if (activation_fns_ref == NULL) { return 2; }
  model->activation_funcs_ref = activation_fns_ref;
  return 0;
}

int mlp_fill_biases(struct mlp_model *model, const float value) {
  if (model == NULL) { return 1; }
  size_t i, j, size;
  for (i = 0U; i < model->layer_size_count - 1U; ++i) {
    size = model->layer_sizes_ref[i + 1U];
    if (model->bias_layers[i] == NULL) { return 2; }
    for (j = 0U; j < size; ++j) {
      model->bias_layers[i][j] = value;
    }
  }
  return 0;
}

int mlp_fill_weights(struct mlp_model *model, const float value) {
  if (model == NULL) { return 1; }
  size_t i, j, size;
  for (i = 0U; i < model->layer_size_count - 1U; ++i) {
    size = model->layer_sizes_ref[i] * model->layer_sizes_ref[i + 1U];
    if (model->weight_layers[i] == NULL) { return 2; }
    for (j = 0U; j < size; ++j) {
      model->weight_layers[i][j] = value;
    }
  }
  return 0;
}

int mlp_forward(struct mlp_model *model,
                struct mlp_backward_buffer *backward_buffer, float const *input,
                float *output) {
  if (model == NULL || input == NULL) { return 1; }

  const size_t max_layer_idx = model->layer_size_count - 1U;
  const size_t *layer_sizes = model->layer_sizes_ref;
  float **const buffers = model->buffers;
  float **const biases = model->bias_layers;
  float **const weights = model->weight_layers;
  float (**activation_funcs_ref)(float) = model->activation_funcs_ref;
  if (layer_sizes == NULL || buffers == NULL || biases == NULL ||
      weights == NULL || activation_funcs_ref == NULL) {
    return 2;
  }
  float *const input_layer = buffers[0U];
  if (input_layer == NULL) { return 3; }

  size_t i, j, k;
  float **layers = NULL, **raw_output_layers = NULL;
  if (backward_buffer != NULL) {
    layers = backward_buffer->layers;
    raw_output_layers = backward_buffer->raw_output_layers;
  }
  if (layers != NULL && raw_output_layers != NULL) {
    for (i = 0U; i < max_layer_idx; ++i) {
      if (layers[i] == NULL || raw_output_layers[i] == NULL) {
        backward_buffer = NULL;
      }
    }
    if (layers[max_layer_idx] == NULL) { backward_buffer = NULL; }
  }

  for (i = 0U; i < layer_sizes[0U]; ++i) {
    input_layer[i] = input[i];
  }
  if (backward_buffer != NULL) {
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
      return 3;
    }

    for (j = 0U; j < front_layer_size; ++j) {
      next_layer[j] = bias[j];
      j_idx = j * back_layer_size;

      for (k = 0U; k < back_layer_size; ++k) {
        next_layer[j] += weight[j_idx + k] * current_layer[k];
      }
      if (backward_buffer != NULL) { raw_output_layers[i][j] = next_layer[j]; }

      next_layer[j] = activation_function(next_layer[j]);
    }
    if (backward_buffer != NULL) {
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
  return 0;
}

int mlp_backward(struct mlp_model *model,
                 struct mlp_backward_buffer *backward_buffer,
                 struct mlp_gradient_buffer *gradient_output,
                 const float *const answer) {
  if (model == NULL || backward_buffer == NULL || gradient_output == NULL ||
      answer == NULL) {
    return 1;
  }

  const float learn_rate = model->learning_rate;
  const size_t layer_size_count = model->layer_size_count;
  const size_t last_layer_idx = layer_size_count - 1U;

  const size_t *layer_sizes = model->layer_sizes_ref;
  float **const layers = backward_buffer->layers;
  float **const raw_output_layers = backward_buffer->raw_output_layers;
  float **const gradient_layers = gradient_output->gradient_layers;
  float **const bias_layers = model->bias_layers;
  float **const weight_layers = model->weight_layers;
  float (**activation_funcs_ref)(float) = model->activation_funcs_ref;
  if (layer_sizes == NULL || layers == NULL || raw_output_layers == NULL ||
      gradient_layers == NULL || bias_layers == NULL || weight_layers == NULL ||
      activation_funcs_ref == NULL) {
    return 2;
  }

  float *const final_output_layer = layers[last_layer_idx];
  float *const final_backward_layer = gradient_layers[last_layer_idx - 1U];
  if (final_output_layer == NULL || final_backward_layer == NULL) { return 3; }

  size_t i, j, k;
  for (i = 0; i < layer_sizes[last_layer_idx]; ++i) {
    // L(a) = 0.5 * (a - y)^2
    // L'(a) = a - y
    // y is for expect output
    final_backward_layer[i] = (final_output_layer[i] - answer[i]);
  }

  size_t back_layer_size, front_layer_size;
  size_t j_idx;
  float *layer = NULL, *output_layer = NULL, *current_backward_layer = NULL,
        *last_backward_layer = NULL, *weights = NULL, *biases = NULL;
  float (*activation_function)(float) = NULL;
  for (i = last_layer_idx; i > 0U;) {
    --i;

    front_layer_size = layer_sizes[i + 1U];
    back_layer_size = layer_sizes[i];

    layer = layers[i];
    current_backward_layer = gradient_layers[i];
    output_layer = raw_output_layers[i];
    weights = weight_layers[i];
    biases = bias_layers[i];
    activation_function = activation_funcs_ref[i];
    if (layer == NULL || output_layer == NULL ||
        current_backward_layer == NULL || weights == NULL || biases == NULL ||
        activation_function == NULL) {
      return 3;
    }

    if (i > 0U) {
      last_backward_layer = gradient_layers[i - 1U];
      if (last_backward_layer == NULL) { return 3; }
    }

    for (j = 0U; j < front_layer_size; ++j) {
      // dL/da * da/dz * dz/dn
      // L is for loss function
      // a is for activate function
      // z is for output
      // n is for any part, there are weight(w), input(x) and bias(b).

      // = dL/da * da/dz
      current_backward_layer[j] *=
          (activation_function(output_layer[j] + 4.76837e-7f) -
           activation_function(output_layer[j] - 4.76837e-7f)) /
          9.53674e-7f;

      j_idx = j * back_layer_size;
      for (k = 0U; k < back_layer_size; ++k) {

        // pass gradiant to last layer first
        // dz/dx = w
        if (i > 0U) {
          last_backward_layer[k] =
              (j == 0U ? 0.0f : last_backward_layer[k]) +
              weights[j_idx + k] * current_backward_layer[j];
        }

        // dz/dw = x
        weights[j_idx + k] -= learn_rate * layer[k] * current_backward_layer[j];
      }

      // dz/db = 1
      biases[j] -= learn_rate * current_backward_layer[j];
    }
  }

  return 0;
}
