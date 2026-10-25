#include <stdint.h>
#include <stdlib.h>

#include <mlp.h>

// WARN: mlp will store pointer layer_sizes, so DO NOT free it before mlp been
// free.
struct mlp_model *mlp_create_model(const size_t *layer_sizes_ref,
                                   size_t layer_size_count) {
  if (layer_sizes_ref == NULL || layer_size_count < 2U) {
    return NULL;
  }

  if (SIZE_MAX / 2U < layer_size_count) {
    return NULL;
  }
  size_t layer_full_size = layer_size_count * 2U;

  if (SIZE_MAX / sizeof(size_t) < layer_full_size) {
    return NULL;
  }
  size_t *parameter_addr_offsets =
      (size_t *)malloc(layer_full_size * sizeof(size_t));
  if (parameter_addr_offsets == NULL) {
    return NULL;
  }

  if (SIZE_MAX / sizeof(float **) < layer_full_size) {
    free(parameter_addr_offsets);
    return NULL;
  }
  float **element_ptrs = (float **)malloc(layer_full_size * sizeof(float *));
  if (element_ptrs == NULL) {
    free(parameter_addr_offsets);
    return NULL;
  }

  const size_t weight_offset = 2U;
  const size_t bias_offset = layer_size_count + 1U;

  // get offsets before malloc at once.
  size_t i, offset_sum = 0U, current_size, current_neruon_count,
            current_layer_idx, max_layer_size = layer_sizes_ref[0];
  if (max_layer_size == 0) {
    return NULL;
  }
  for (i = weight_offset; i < layer_full_size; ++i) {
    if (i < bias_offset) { // offsets for weights
      current_layer_idx = i - weight_offset + 1U;
      current_neruon_count = layer_sizes_ref[current_layer_idx];
      // zero neuron test and multification overflow test
      if (current_neruon_count == 0U ||
          SIZE_MAX / current_neruon_count <
              layer_sizes_ref[current_layer_idx - 1U]) {
        free(parameter_addr_offsets);
        free(element_ptrs);
        return NULL;
      }
      if (max_layer_size < current_neruon_count) {
        max_layer_size = current_neruon_count;
      }
      current_size =
          current_neruon_count * layer_sizes_ref[current_layer_idx - 1U];
    } else { // offsets for bias
      current_size = layer_sizes_ref[i - bias_offset + 1U];
    }

    if (SIZE_MAX - current_size < offset_sum) {
      free(parameter_addr_offsets);
      free(element_ptrs);
      return NULL;
    }
    // get address offset
    parameter_addr_offsets[i] = offset_sum;
    offset_sum += current_size;
  }

  // size in this malloc
  // = size sum of each layer(except first layer) of bias
  // + size sum of each layer(except first layer) of weights
  if ((SIZE_MAX - 2U * max_layer_size) / sizeof(float) < offset_sum) {
    free(parameter_addr_offsets);
    free(element_ptrs);
    return NULL;
  }
  float *elements =
      (float *)malloc((offset_sum + 2U * max_layer_size) * sizeof(float));
  if (elements == NULL) {
    free(parameter_addr_offsets);
    free(element_ptrs);
    return NULL;
  }
  // offset to real pointer
  element_ptrs[0] = elements;
  element_ptrs[1] = elements + max_layer_size;
  for (i = 2U; i < layer_full_size; ++i) {
    element_ptrs[i] =
        elements + parameter_addr_offsets[i] + 2U * max_layer_size;
  }
  free(parameter_addr_offsets);

  struct mlp_model *mlp = (struct mlp_model *)malloc(sizeof(struct mlp_model));
  if (mlp == NULL) {
    free(element_ptrs);
    free(elements);
    return NULL;
  }

  mlp->layer_size_count = layer_size_count;
  mlp->layer_sizes_ref = layer_sizes_ref;
  mlp->buffers = element_ptrs;
  mlp->weight_layers = element_ptrs + weight_offset;
  mlp->bias_layers = element_ptrs + bias_offset;
  mlp->activation_funcs_ref = NULL;

  mlp_set_learning_rate(mlp, 0.0f);
  mlp_fill_biases(mlp, 0.0f);
  mlp_fill_weights(mlp, 0.0f);

  return mlp;
}

void mlp_destroy_model(struct mlp_model *model) {
  if (model == NULL) {
    return;
  }

  float **element_ptrs = model->buffers;
  free(model);
  if (element_ptrs == NULL) {
    return;
  }

  float *elements = element_ptrs[0U];
  free(element_ptrs);
  if (elements == NULL) {
    return;
  }

  free(elements);
}

struct mlp_learning_cache *mlp_create_learning_cache(struct mlp_model *model) {
  if (model == NULL) {
    return NULL;
  }

  const size_t layer_size_count = model->layer_size_count;
  const size_t *const layer_sizes = model->layer_sizes_ref;
  if (layer_sizes == NULL) {
    return NULL;
  }

  if ((SIZE_MAX - 1U) / 3U + 1U < layer_size_count) {
    return NULL;
  }
  const size_t layer_full_size = layer_size_count * 3U - 2U;

  if (SIZE_MAX / sizeof(size_t) < layer_full_size) {
    return NULL;
  }

  size_t *layer_addr_offsets =
      (size_t *)malloc(layer_full_size * sizeof(size_t));
  if (layer_addr_offsets == NULL) {
    return NULL;
  }

  const size_t raw_output_offset = layer_size_count;
  const size_t gradiants_offset = layer_size_count * 2U - 1U;
  size_t i, current_size, offset_sum = 0U;
  for (i = 0U; i < layer_full_size; ++i) {

    if (i < raw_output_offset) {
      current_size = layer_sizes[i];
      if (current_size == 0) {
        free(layer_addr_offsets);
        return NULL;
      }
    } else if (i < gradiants_offset) {
      current_size = layer_sizes[i - raw_output_offset + 1U];
    } else {
      current_size = layer_sizes[i - gradiants_offset + 1U];
    }

    if (SIZE_MAX - current_size < offset_sum) {
      free(layer_addr_offsets);
      return NULL;
    }
    layer_addr_offsets[i] = offset_sum;
    offset_sum += current_size;
  }

  if (SIZE_MAX / sizeof(float *) < layer_full_size) {
    free(layer_addr_offsets);
    return NULL;
  }
  float **layers = (float **)malloc(layer_full_size * sizeof(float *));
  if (layers == NULL) {
    free(layer_addr_offsets);
    return NULL;
  }

  if (SIZE_MAX / sizeof(float) < offset_sum) {
    free(layer_addr_offsets);
    free(layers);
    return NULL;
  }
  float *elements = (float *)malloc(offset_sum * sizeof(float));
  if (elements == NULL) {
    free(layer_addr_offsets);
    free(layers);
    return NULL;
  }

  for (i = 0; i < layer_full_size; ++i) {
    layers[i] = elements + layer_addr_offsets[i];
  }
  free(layer_addr_offsets);

  struct mlp_learning_cache *cache =
      (struct mlp_learning_cache *)malloc(sizeof(struct mlp_learning_cache));
  if (cache == NULL) {
    free(elements);
    free(layers);
    return NULL;
  }

  cache->layers = layers;
  cache->raw_output_layers = layers + raw_output_offset;
  cache->gradient_layers = layers + gradiants_offset;

  return cache;
}

void mlp_destroy_learning_cache(struct mlp_learning_cache *cache) {
  if (cache == NULL) {
    return;
  }

  float **layers = cache->layers;
  free(cache);
  if (layers == NULL) {
    return;
  }

  float *elements = layers[0U];
  free(layers);
  if (elements == NULL) {
    return;
  }

  free(elements);
}

int mlp_set_learning_rate(struct mlp_model *model, float learning_rate) {
  if (model == NULL) {
    return 1;
  }
  model->learning_rate = learning_rate;
  return 0;
}

int mlp_set_activation_fns_ref(struct mlp_model *model,
                               float (**activation_fns_ref)(float)) {
  if (model == NULL) {
    return 1;
  }
  if (activation_fns_ref == NULL) {
    return 2;
  }
  model->activation_funcs_ref = activation_fns_ref;
  return 0;
}

int mlp_fill_biases(struct mlp_model *model, const float value) {
  if (model == NULL) {
    return 1;
  }
  size_t i, j, size;
  for (i = 0U; i < model->layer_size_count - 1U; ++i) {
    size = model->layer_sizes_ref[i + 1U];
    if (model->bias_layers[i] == NULL) {
      return 2;
    }
    for (j = 0U; j < size; ++j) {
      model->bias_layers[i][j] = value;
    }
  }
  return 0;
}

int mlp_fill_weights(struct mlp_model *model, const float value) {
  if (model == NULL) {
    return 1;
  }
  size_t i, j, size;
  for (i = 0U; i < model->layer_size_count - 1U; ++i) {
    size = model->layer_sizes_ref[i] * model->layer_sizes_ref[i + 1U];
    if (model->weight_layers[i] == NULL) {
      return 2;
    }
    for (j = 0U; j < size; ++j) {
      model->weight_layers[i][j] = value;
    }
  }
  return 0;
}

int mlp_forward(struct mlp_model *model,
                struct mlp_learning_cache *learning_cache, float const *input,
                float *output) {
  if (model == NULL || input == NULL) {
    return 1;
  }

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
  if (input_layer == NULL) {
    return 3;
  }

  size_t i, j, k;
  float **layers = NULL, **raw_output_layers = NULL;
  if (learning_cache != NULL) {
    layers = learning_cache->layers;
    raw_output_layers = learning_cache->raw_output_layers;
  }
  if (layers != NULL && raw_output_layers != NULL) {
    for (i = 0U; i < max_layer_idx; ++i) {
      if (layers[i] == NULL || raw_output_layers[i] == NULL) {
        learning_cache = NULL;
      }
    }
    if (layers[max_layer_idx] == NULL) {
      learning_cache = NULL;
    }
  }

  for (i = 0U; i < layer_sizes[0U]; ++i) {
    input_layer[i] = input[i];
  }
  if (learning_cache != NULL) {
    for (i = 0U; i < layer_sizes[0U]; ++i) {
      layers[0][i] = input[i];
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
      if (learning_cache != NULL) {
        raw_output_layers[i][j] = next_layer[j];
      }

      next_layer[j] = activation_function(next_layer[j]);
    }
    if (learning_cache != NULL) {
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
                 struct mlp_learning_cache *learning_cache,
                 const float *const answer) {
  if (model == NULL || learning_cache == NULL || answer == NULL) {
    return 1;
  }

  const float learn_rate = model->learning_rate;
  const size_t layer_size_count = model->layer_size_count;
  const size_t last_layer_idx = layer_size_count - 1U;

  const size_t *layer_sizes = model->layer_sizes_ref;
  float **const layers = learning_cache->layers;
  float **const raw_output_layers = learning_cache->raw_output_layers;
  float **const gradient_layers = learning_cache->gradient_layers;
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
  if (final_output_layer == NULL || final_backward_layer == NULL) {
    return 3;
  }

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
      if (last_backward_layer == NULL) {
        return 3;
      }
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