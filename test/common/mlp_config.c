#include <stddef.h>

#include <ann-lib/mlp.h>

float input[2] = {0.5f, 2.0f}, answer[2] = {1.0f, 0.0f};

size_t layer_sizes[3] = {2U, 3U, 2U};
const size_t layer_size_count = 3U;
struct mlp_layer_size_data size_data = {layer_size_count, layer_sizes};

float learning_rate = 0.125f;
float momentum = 0.875f;

float leaky_relu(float value) { return (value > 0) ? value : value * 0.03125f; }
float (*activation_funcs[3U])(float) = {leaky_relu, leaky_relu, leaky_relu};

int test_mlp_get_gradients(size_t size, const float *expect,
                           const float *actual, float *gradients_output) {
  if (expect == NULL || actual == NULL || gradients_output == NULL) {
    return 1;
  }
  size_t i;
  for (i = 0; i < size; ++i) {
    // L(a) = 0.5 * (a - y)^2
    // L'(a) = a - y
    // y is for expect output
    gradients_output[i] = actual[i] - expect[i];
  }
  return 0;
}