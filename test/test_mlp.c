#include <stdbool.h>

#include <unity.h>

#include <internal/mlp.h>

static float leaky_relu(float value) {
  return (value > 0) ? value : value * 0.03125f;
}

static struct mlp_model *model;
static struct mlp_backward_buffer *backward_buffer;
static struct mlp_gradient_buffer *gradient_buffer;
static enum mlp_status status;

static float output[2], gradients[2];
static float input[2] = {0.5f, 2.0f}, answer[2] = {1.0f, 0.0f};
static size_t layer_sizes[3] = {2U, 3U, 2U};
static const size_t layer_size_count = 3U;
static float learning_rate = 0.125f;
static float (*activation_funcs[3U])(float) = {leaky_relu, leaky_relu,
                                               leaky_relu};
static size_t layer_sizes_another[4] = {3U, 4U, 1U, 3U};
static const size_t layer_size_count_another = 4U;
static struct mlp_layer_size_data size_data = {layer_sizes, layer_size_count};
static struct mlp_layer_size_data size_data_another = {
    layer_sizes_another, layer_size_count_another};

static int get_gradients(size_t size, const float *expect, const float *actual,
                         float *gradients_output) {
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

void setUp() {
  model = NULL;
  backward_buffer = NULL;
  gradient_buffer = NULL;
  status = -1;
}

void tearDown() {
  mlp_destroy_model(model);
  mlp_destroy_backward_buffer(backward_buffer);
  mlp_destroy_gradient_buffer(gradient_buffer);
}

void test_mlp_create_model(void) {
  model = mlp_create_model(&size_data, &status);

  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
}

void test_mlp_create_model_null(void) {
  struct mlp_layer_size_data test_size_data = {NULL, layer_size_count};
  model = mlp_create_model(&test_size_data, &status);

  TEST_ASSERT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_model_one_layer(void) {
  size_t tiny_layer_sizes[1] = {2U};
  struct mlp_layer_size_data test_size_data = {tiny_layer_sizes, 1U};
  model = mlp_create_model(&test_size_data, &status);

  TEST_ASSERT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_model_zero_neuron(void) {
  size_t small_layer_sizes[3] = {1U, 0U, 1U};
  struct mlp_layer_size_data test_size_data = {small_layer_sizes, 3U};
  model = mlp_create_model(&test_size_data, &status);

  TEST_ASSERT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_model_large_layer_full_size(void) {
  size_t large[2] = {1U, 1U};
  struct mlp_layer_size_data test_size_data = {large,
                                               (SIZE_MAX - 1U) / 5U + 1U};
  model = mlp_create_model(&test_size_data, &status);

  TEST_ASSERT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_INT_OVERFLOW, status);
}

void test_mlp_create_model_large_layer_offsets(void) {
  size_t large[2] = {1U, 1U};
  struct mlp_layer_size_data test_size_data = {
      large, ((SIZE_MAX - 1U) / 5U + 1U) >> 3};
  model = mlp_create_model(&test_size_data, &status);

  TEST_ASSERT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_ALLOCATION_FAILED, status);
}

void test_mlp_create_model_large_offset_sum(void) {
  size_t large_layer_sizes[3] = {SIZE_MAX >> 1, (SIZE_MAX >> 2) / 3,
                                 (SIZE_MAX >> 2) / 3};
  struct mlp_layer_size_data test_size_data = {large_layer_sizes, 3U};
  model = mlp_create_model(&test_size_data, &status);

  TEST_ASSERT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_INT_OVERFLOW, status);
}

void test_mlp_create_backward_buffer(void) {
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);

  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
}

void test_mlp_create_backward_buffer_null(void) {
  struct mlp_layer_size_data test_size_data = {NULL, layer_size_count};
  backward_buffer = mlp_create_backward_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_backward_buffer_one_layer(void) {
  size_t tiny_layer_sizes[1] = {2U};
  struct mlp_layer_size_data test_size_data = {tiny_layer_sizes, 1U};
  backward_buffer = mlp_create_backward_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_backward_buffer_zero_neuron(void) {
  size_t small_layer_sizes[3] = {1U, 0U, 1U};
  struct mlp_layer_size_data test_size_data = {small_layer_sizes, 3U};
  backward_buffer = mlp_create_backward_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_backward_buffer_large_layer_full_size(void) {
  size_t large[2] = {1U, 1U};
  struct mlp_layer_size_data test_size_data = {large,
                                               (SIZE_MAX - 1U) / 5U + 1U};
  backward_buffer = mlp_create_backward_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INT_OVERFLOW, status);
}

void test_mlp_create_backward_buffer_large_layer_offsets(void) {
  size_t large[2] = {1U, 1U};
  struct mlp_layer_size_data test_size_data = {
      large, ((SIZE_MAX - 1U) / 5U + 1U) >> 3};
  backward_buffer = mlp_create_backward_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_ALLOCATION_FAILED, status);
}

void test_mlp_create_backward_buffer_large_offset_sum(void) {
  size_t large_layer_sizes[3] = {SIZE_MAX >> 1, (SIZE_MAX >> 2) / 3,
                                 (SIZE_MAX >> 2) / 3};
  struct mlp_layer_size_data test_size_data = {large_layer_sizes, 3U};
  backward_buffer = mlp_create_backward_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INT_OVERFLOW, status);
}

void test_mlp_create_gradient_buffer(void) {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);

  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
}

void test_mlp_create_gradient_buffer_null(void) {
  struct mlp_layer_size_data test_size_data = {NULL, layer_size_count};
  gradient_buffer = mlp_create_gradient_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_gradient_buffer_one_layer(void) {
  size_t tiny_layer_sizes[1] = {2U};
  struct mlp_layer_size_data test_size_data = {tiny_layer_sizes, 1U};
  gradient_buffer = mlp_create_gradient_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_gradient_buffer_zero_neuron(void) {
  size_t small_layer_sizes[3] = {1U, 0U, 1U};
  struct mlp_layer_size_data test_size_data = {small_layer_sizes, 3U};
  gradient_buffer = mlp_create_gradient_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_gradient_buffer_large_layer_full_size(void) {
  size_t large[2] = {1U, 1U};
  struct mlp_layer_size_data test_size_data = {large,
                                               (SIZE_MAX - 1U) / 5U + 1U};
  gradient_buffer = mlp_create_gradient_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INT_OVERFLOW, status);
}

void test_mlp_create_gradient_buffer_large_layer_offsets(void) {
  size_t large[2] = {1U, 1U};
  struct mlp_layer_size_data test_size_data = {
      large, ((SIZE_MAX - 1U) / 5U + 1U) >> 3};
  gradient_buffer = mlp_create_gradient_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_ALLOCATION_FAILED, status);
}

void test_mlp_create_gradient_buffer_large_offset_sum(void) {
  size_t large_layer_sizes[3] = {SIZE_MAX >> 1, SIZE_MAX >> 2,
                                 (SIZE_MAX >> 2) + 1};
  struct mlp_layer_size_data test_size_data = {large_layer_sizes, 3U};
  gradient_buffer = mlp_create_gradient_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INT_OVERFLOW, status);
}

void test_mlp_set_learning_rate(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);

  TEST_ASSERT_UNLESS(mlp_set_learning_rate(model, learning_rate));

  TEST_ASSERT_EQUAL_FLOAT(learning_rate, model->learning_rate);
}

void test_mlp_set_learning_rate_null(void) {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_set_learning_rate(NULL, learning_rate));
}

void test_mlp_set_activation_fns_ref(void) {
  float (*activation_funcs[3])(float) = {leaky_relu, leaky_relu, leaky_relu};
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);

  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  TEST_ASSERT_EQUAL_PTR(activation_funcs, model->activation_funcs_ref);
}

void test_mlp_set_activation_fns_ref_null_model(void) {
  float (*activation_funcs[3])(float) = {leaky_relu, leaky_relu, leaky_relu};

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_set_activation_fns_ref(NULL, activation_funcs));
}

void test_mlp_set_activation_fns_ref_null_fns_ref(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_set_activation_fns_ref(model, NULL));
}

void test_mlp_fill_biases(void) {
  float except_layer1_biases[3U] = {1.0f, 1.0f, 1.0f};
  float except_layer2_biases[2U] = {1.0f, 1.0f};
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);

  TEST_ASSERT_UNLESS(mlp_fill_biases(model, 1.0f));

  TEST_ASSERT_EQUAL_FLOAT_ARRAY(except_layer1_biases, model->bias_layers[0U],
                                3U);
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(except_layer2_biases, model->bias_layers[1U],
                                2U);
}

void test_mlp_fill_biases_null(void) {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, mlp_fill_biases(NULL, 1.0f));
}

void test_mlp_fill_biases_null_bias_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  model->bias_layers = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD, mlp_fill_biases(model, 1.0f));
}

void test_mlp_fill_biases_null_bias_layer(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  model->bias_layers[0U] = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER, mlp_fill_biases(model, 1.0f));
}

void test_mlp_fill_biases_null_element_end(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  model->layer_data.elements_end = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_STATE, mlp_fill_biases(model, 1.0f));
}

void test_mlp_fill_weights(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);

  TEST_ASSERT_UNLESS(mlp_fill_weights(model, 1.0f));

  size_t i;
  for (i = 0U; i < 6U; ++i) {
    float weight_first_layer = model->weight_layers[0U][i],
          weight_second_layer = model->weight_layers[1U][i];
    TEST_ASSERT_EQUAL_FLOAT(1.0f, weight_first_layer);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, weight_second_layer);
  }
}

void test_mlp_fill_weights_null(void) {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, mlp_fill_weights(NULL, 1.0f));
}

void test_mlp_fill_weights_null_weight_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  model->weight_layers = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD, mlp_fill_weights(model, 1.0f));
}

void test_mlp_fill_weights_null_bias_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  model->bias_layers = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD, mlp_fill_weights(model, 1.0f));
}

void test_mlp_fill_weights_null_weight_layer(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  model->weight_layers[0U] = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER, mlp_fill_weights(model, 1.0f));
}

void test_mlp_fill_weights_null_bias_layer(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  model->bias_layers[0U] = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER, mlp_fill_weights(model, 1.0f));
}

void test_mlp_clear_gradient_buffer(void) {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_NOT_NULL(gradient_buffer->weight_gradient_layers);
  TEST_ASSERT_UNLESS(status);
  float *i;
  for (i = gradient_buffer->weight_gradient_layers[0U];
       i < gradient_buffer->layer_data.elements_end; ++i) {
    *i = 0.5f;
  }

  TEST_ASSERT_UNLESS(mlp_clear_gradient_buffer(gradient_buffer));

  for (i = gradient_buffer->weight_gradient_layers[0U];
       i < gradient_buffer->layer_data.elements_end; ++i) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, *i);
  }
}

void test_mlp_clear_gradient_buffer_null(void) {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_clear_gradient_buffer(NULL));
}

void test_mlp_clear_gradient_buffer_null_bias_layers(void) {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer->weight_gradient_layers = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_clear_gradient_buffer(gradient_buffer));
}

void test_mlp_clear_gradient_buffer_null_bias_layer(void) {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_NOT_NULL(gradient_buffer->weight_gradient_layers);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer->weight_gradient_layers[0U] = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_clear_gradient_buffer(gradient_buffer));
}

void test_mlp_clear_gradient_buffer_null_element_end(void) {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer->layer_data.elements_end = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_STATE,
                    mlp_clear_gradient_buffer(gradient_buffer));
}

void test_mlp_scale_gradients(void) {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_NOT_NULL(gradient_buffer->weight_gradient_layers);
  TEST_ASSERT_UNLESS(status);
  float *i;
  for (i = gradient_buffer->weight_gradient_layers[0U];
       i < gradient_buffer->layer_data.elements_end; ++i) {
    *i = 0.5f;
  }

  TEST_ASSERT_UNLESS(mlp_scale_gradients(gradient_buffer, 0.5f));

  for (i = gradient_buffer->weight_gradient_layers[0U];
       i < gradient_buffer->layer_data.elements_end; ++i) {
    TEST_ASSERT_EQUAL_FLOAT(0.25f, *i);
  }
}

void test_mlp_scale_gradients_null(void) {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_scale_gradients(NULL, 0.5f));
}

void test_mlp_scale_gradients_null_bias_layers(void) {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer->weight_gradient_layers = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_scale_gradients(gradient_buffer, 0.5f));
}

void test_mlp_scale_gradients_null_bias_layer(void) {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_NOT_NULL(gradient_buffer->weight_gradient_layers);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer->weight_gradient_layers[0U] = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_scale_gradients(gradient_buffer, 0.5f));
}

void test_mlp_scale_gradients_null_element_end(void) {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer->layer_data.elements_end = NULL;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_STATE,
                    mlp_scale_gradients(gradient_buffer, 0.5f));
}

void test_mlp_forward(void) {
  float output[2U], except_output[2U] = {5.5f, 5.5f};
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(mlp_set_learning_rate(model, 0.125f));
  TEST_ASSERT_UNLESS(mlp_fill_biases(model, -0.5f));
  TEST_ASSERT_UNLESS(mlp_fill_weights(model, 1.0f));

  TEST_ASSERT_UNLESS(mlp_forward(model, NULL, input, output));

  TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-7f, except_output, output, 2U);
}

void test_mlp_forward_different_model_metadata(void) {
  model = mlp_create_model(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_TOPOLOGY_MISMATCH,
                        mlp_forward(model, backward_buffer, input, NULL));
}

void test_mlp_forward_different_backward_buffer_metadata(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  backward_buffer = mlp_create_backward_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_TOPOLOGY_MISMATCH,
                        mlp_forward(model, backward_buffer, input, NULL));
}

void test_mlp_forward_null_layer_sizes(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->layer_metadata.layer_sizes_ref = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD,
                        mlp_forward(model, NULL, input, NULL));
}

void test_mlp_forward_null_buffers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  float **buffers_temp = model->buffers;
  model->buffers = NULL;
  int result = mlp_forward(model, NULL, input, NULL);
  model->buffers = buffers_temp;

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD, result);
}

void test_mlp_forward_null_buffer(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  float *buffer_temp = model->buffers[0];
  model->buffers[0] = NULL;
  int result = mlp_forward(model, NULL, input, NULL);
  model->buffers[0] = buffer_temp;

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_LAYER, result);
}

void test_mlp_forward_null_weight_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD,
                        mlp_forward(model, NULL, input, NULL));
}

void test_mlp_forward_null_single_weights(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers[0] = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_LAYER,
                        mlp_forward(model, NULL, input, NULL));
}

void test_mlp_forward_null_bias_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD,
                        mlp_forward(model, NULL, input, NULL));
}

void test_mlp_forward_null_biases(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers[0] = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_LAYER,
                        mlp_forward(model, NULL, input, NULL));
}

void test_mlp_forward_null_activation_funcs(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD,
                        mlp_forward(model, NULL, input, NULL));
}

void test_mlp_backward(void) {
  float dump[17U] = {5.0f,  20.0f, 5.0f,  20.0f, 5.0f,  20.0f, 9.0f, 9.0f, 9.0f,
                     11.0f, 11.0f, 11.0f, 10.0f, 10.0f, 10.0f, 4.5f, 5.5f};
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(mlp_set_learning_rate(model, 0.125f));
  TEST_ASSERT_UNLESS(mlp_fill_biases(model, -0.5f));
  TEST_ASSERT_UNLESS(mlp_fill_weights(model, 1.0f));
  TEST_ASSERT_UNLESS(mlp_forward(model, backward_buffer, input, output));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  TEST_ASSERT_UNLESS(mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_UNLESS(
      mlp_backward(model, backward_buffer, gradient_buffer, gradients));

  float *begin = gradient_buffer->layer_data.elements_begin,
        *end = gradient_buffer->layer_data.elements_end;
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(dump, begin, end - begin);
}

void test_mlp_backward_different_model_metadata(void) {
  model = mlp_create_model(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  int result = mlp_backward(model, backward_buffer, gradient_buffer, gradients);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_backward_different_backward_buffer_metadata(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  int result = mlp_backward(model, backward_buffer, gradient_buffer, gradients);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_backward_different_gradient_buffer_metadata(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  int result = mlp_backward(model, backward_buffer, gradient_buffer, gradients);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_backward_null_layer_sizes(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  model->layer_metadata.layer_sizes_ref = NULL;
  backward_buffer->layer_metadata.layer_sizes_ref = NULL;
  gradient_buffer->layer_metadata.layer_sizes_ref = NULL;
  int result = mlp_backward(model, backward_buffer, gradient_buffer, gradients);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_backward_null_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  float **layers_temp = backward_buffer->layers;
  backward_buffer->layers = NULL;
  int result = mlp_backward(model, backward_buffer, gradient_buffer, gradients);
  backward_buffer->layers = layers_temp;

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD, result);
}

void test_mlp_backward_null_layer(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  float *layer_temp = backward_buffer->layers[0];
  backward_buffer->layers[0] = NULL;
  int result = mlp_backward(model, backward_buffer, gradient_buffer, gradients);
  backward_buffer->layers[0] = layer_temp;

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_LAYER, result);
}

void test_mlp_backward_null_output_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  backward_buffer->raw_output_layers = NULL;
  TEST_ASSERT_EQUAL_INT(
      MLP_STATUS_INVALID_FIELD,
      mlp_backward(model, backward_buffer, gradient_buffer, gradients));
}

void test_mlp_backward_null_output_layer(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  backward_buffer->raw_output_layers[0] = NULL;
  TEST_ASSERT_EQUAL_INT(
      MLP_STATUS_INVALID_LAYER,
      mlp_backward(model, backward_buffer, gradient_buffer, gradients));
}

void test_mlp_backward_null_weights(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  model->weight_layers = NULL;
  TEST_ASSERT_EQUAL_INT(
      MLP_STATUS_INVALID_FIELD,
      mlp_backward(model, backward_buffer, gradient_buffer, gradients));
}

void test_mlp_backward_null_single_layer_weights(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  model->weight_layers[0] = NULL;
  TEST_ASSERT_EQUAL_INT(
      MLP_STATUS_INVALID_LAYER,
      mlp_backward(model, backward_buffer, gradient_buffer, gradients));
}

void test_mlp_backward_null_activation_funcs(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  TEST_ASSERT_EQUAL_INT(
      MLP_STATUS_INVALID_FIELD,
      mlp_backward(model, backward_buffer, gradient_buffer, gradients));
}

void test_mlp_update_parameters_sgd(void) {
  float dump[17U] = {5.0f,  20.0f, 5.0f,  20.0f, 5.0f,  20.0f, 9.0f, 9.0f, 9.0f,
                     11.0f, 11.0f, 11.0f, 10.0f, 10.0f, 10.0f, 4.5f, 5.5f};
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(mlp_set_learning_rate(model, 0.125f));
  TEST_ASSERT_UNLESS(mlp_fill_biases(model, -0.5f));
  TEST_ASSERT_UNLESS(mlp_fill_weights(model, 1.0f));
  TEST_ASSERT_UNLESS(mlp_forward(model, backward_buffer, input, output));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  TEST_ASSERT_UNLESS(mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_UNLESS(
      mlp_backward(model, backward_buffer, gradient_buffer, gradients));
  TEST_ASSERT_UNLESS(mlp_update_parameters_sgd(model, gradient_buffer));

  float *begin = gradient_buffer->layer_data.elements_begin,
        *end = gradient_buffer->layer_data.elements_end;
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(dump, begin, end - begin);
}

void test_mlp_update_parameters_sgd_different_model_metadata(void) {
  model = mlp_create_model(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));

  int result = mlp_update_parameters_sgd(model, gradient_buffer);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_sgd_different_gradient_buffer_metadata(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  int result = mlp_update_parameters_sgd(model, gradient_buffer);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_sgd_null_layer_sizes(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->layer_metadata.layer_sizes_ref = NULL;
  gradient_buffer->layer_metadata.layer_sizes_ref = NULL;
  int result = mlp_update_parameters_sgd(model, gradient_buffer);

  TEST_ASSERT_EQUAL_INT(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_sgd_null_weights(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD,
                        mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_single_layer_weights(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers[0] = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_LAYER,
                        mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_bias_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD,
                        mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_biases(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers[0] = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_LAYER,
                        mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_weight_gradient_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->weight_gradient_layers = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD,
                        mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_weight_gradient_layer(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->weight_gradient_layers[0] = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_LAYER,
                        mlp_update_parameters_sgd(model, gradient_buffer));
}
void test_mlp_update_parameters_sgd_null_bias_gradient_layers(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->bias_gradient_layers = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_FIELD,
                        mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_bias_gradient_layer(void) {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_UNLESS(status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_UNLESS(status);
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->bias_gradient_layers[0] = NULL;
  TEST_ASSERT_EQUAL_INT(MLP_STATUS_INVALID_LAYER,
                        mlp_update_parameters_sgd(model, gradient_buffer));
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_mlp_create_model);
  RUN_TEST(test_mlp_create_model_null);
  RUN_TEST(test_mlp_create_model_one_layer);
  RUN_TEST(test_mlp_create_model_zero_neuron);
  RUN_TEST(test_mlp_create_model_large_layer_full_size);
  RUN_TEST(test_mlp_create_model_large_layer_offsets);
  RUN_TEST(test_mlp_create_model_large_offset_sum);

  RUN_TEST(test_mlp_create_backward_buffer);
  RUN_TEST(test_mlp_create_backward_buffer_null);
  RUN_TEST(test_mlp_create_backward_buffer_one_layer);
  RUN_TEST(test_mlp_create_backward_buffer_zero_neuron);
  RUN_TEST(test_mlp_create_backward_buffer_large_layer_full_size);
  RUN_TEST(test_mlp_create_backward_buffer_large_layer_offsets);
  RUN_TEST(test_mlp_create_backward_buffer_large_offset_sum);

  RUN_TEST(test_mlp_create_gradient_buffer);
  RUN_TEST(test_mlp_create_gradient_buffer_null);
  RUN_TEST(test_mlp_create_gradient_buffer_one_layer);
  RUN_TEST(test_mlp_create_gradient_buffer_zero_neuron);
  RUN_TEST(test_mlp_create_gradient_buffer_large_layer_full_size);
  RUN_TEST(test_mlp_create_gradient_buffer_large_layer_offsets);
  RUN_TEST(test_mlp_create_gradient_buffer_large_offset_sum);

  RUN_TEST(test_mlp_set_learning_rate);
  RUN_TEST(test_mlp_set_learning_rate_null);

  RUN_TEST(test_mlp_set_activation_fns_ref);
  RUN_TEST(test_mlp_set_activation_fns_ref_null_model);
  RUN_TEST(test_mlp_set_activation_fns_ref_null_fns_ref);

  RUN_TEST(test_mlp_fill_biases);
  RUN_TEST(test_mlp_fill_biases_null);
  RUN_TEST(test_mlp_fill_biases_null_bias_layers);
  RUN_TEST(test_mlp_fill_biases_null_bias_layer);
  RUN_TEST(test_mlp_fill_biases_null_element_end);

  RUN_TEST(test_mlp_fill_weights);
  RUN_TEST(test_mlp_fill_weights_null_weight_layers);
  RUN_TEST(test_mlp_fill_weights_null_bias_layers);
  RUN_TEST(test_mlp_fill_weights_null_weight_layer);
  RUN_TEST(test_mlp_fill_weights_null_bias_layer);

  RUN_TEST(test_mlp_clear_gradient_buffer);
  RUN_TEST(test_mlp_clear_gradient_buffer_null);
  RUN_TEST(test_mlp_clear_gradient_buffer_null_bias_layers);
  RUN_TEST(test_mlp_clear_gradient_buffer_null_bias_layer);
  RUN_TEST(test_mlp_clear_gradient_buffer_null_element_end);

  RUN_TEST(test_mlp_scale_gradients);
  RUN_TEST(test_mlp_scale_gradients_null);
  RUN_TEST(test_mlp_scale_gradients_null_bias_layers);
  RUN_TEST(test_mlp_scale_gradients_null_bias_layer);
  RUN_TEST(test_mlp_scale_gradients_null_element_end);

  RUN_TEST(test_mlp_forward);
  RUN_TEST(test_mlp_forward_different_model_metadata);
  RUN_TEST(test_mlp_forward_different_backward_buffer_metadata);
  RUN_TEST(test_mlp_forward_null_layer_sizes);
  RUN_TEST(test_mlp_forward_null_buffers);
  RUN_TEST(test_mlp_forward_null_buffer);
  RUN_TEST(test_mlp_forward_null_weight_layers);
  RUN_TEST(test_mlp_forward_null_single_weights);
  RUN_TEST(test_mlp_forward_null_bias_layers);
  RUN_TEST(test_mlp_forward_null_biases);
  RUN_TEST(test_mlp_forward_null_activation_funcs);

  RUN_TEST(test_mlp_backward);
  RUN_TEST(test_mlp_backward_different_model_metadata);
  RUN_TEST(test_mlp_backward_different_backward_buffer_metadata);
  RUN_TEST(test_mlp_backward_different_gradient_buffer_metadata);
  RUN_TEST(test_mlp_backward_null_layer_sizes);
  RUN_TEST(test_mlp_backward_null_layers);
  RUN_TEST(test_mlp_backward_null_layer);
  RUN_TEST(test_mlp_backward_null_output_layers);
  RUN_TEST(test_mlp_backward_null_output_layer);
  RUN_TEST(test_mlp_backward_null_weights);
  RUN_TEST(test_mlp_backward_null_single_layer_weights);
  RUN_TEST(test_mlp_backward_null_activation_funcs);

  RUN_TEST(test_mlp_update_parameters_sgd);
  RUN_TEST(test_mlp_update_parameters_sgd_different_model_metadata);
  RUN_TEST(test_mlp_update_parameters_sgd_different_gradient_buffer_metadata);
  RUN_TEST(test_mlp_update_parameters_sgd_null_layer_sizes);
  RUN_TEST(test_mlp_update_parameters_sgd_null_weights);
  RUN_TEST(test_mlp_update_parameters_sgd_null_single_layer_weights);
  RUN_TEST(test_mlp_update_parameters_sgd_null_bias_layers);
  RUN_TEST(test_mlp_update_parameters_sgd_null_biases);
  RUN_TEST(test_mlp_update_parameters_sgd_null_weight_gradient_layers);
  RUN_TEST(test_mlp_update_parameters_sgd_null_weight_gradient_layer);
  RUN_TEST(test_mlp_update_parameters_sgd_null_bias_gradient_layers);
  RUN_TEST(test_mlp_update_parameters_sgd_null_bias_gradient_layer);

  return UNITY_END();
}
