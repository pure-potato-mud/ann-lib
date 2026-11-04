#include <unity.h>

#include <ann-lib/mlp.h>

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
static struct mlp_layer_size_data size_data = {layer_size_count, layer_sizes};

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

void setUp(void) {
  model = NULL;
  backward_buffer = NULL;
  gradient_buffer = NULL;
  status = -1;
}

void tearDown(void) {
  mlp_destroy_model(model);
  mlp_destroy_backward_buffer(backward_buffer);
  mlp_destroy_gradient_buffer(gradient_buffer);
}

void test_mlp_propagation_and_sgd_update(void) {
  float expect_output_1[2] = {5.5f, 5.5f},
        expect_output_2[2] = {-3.153228e-2f, -3.209686e-2f};

  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_UNLESS(mlp_set_learning_rate(model, learning_rate));
  TEST_ASSERT_UNLESS(mlp_fill_weights(model, 1.0f));
  TEST_ASSERT_UNLESS(mlp_fill_biases(model, -0.5f));
  TEST_ASSERT_UNLESS(mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_UNLESS(mlp_forward(model, input, output, backward_buffer));
  TEST_ASSERT_UNLESS(mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(expect_output_1, output, 2U);
  TEST_ASSERT_UNLESS(get_gradients(2U, answer, output, gradients));
  TEST_ASSERT_UNLESS(
      mlp_backward(model, backward_buffer, gradients, gradient_buffer));
  TEST_ASSERT_UNLESS(mlp_update_parameters_sgd(model, gradient_buffer));
  TEST_ASSERT_UNLESS(mlp_forward(model, input, output, NULL));
  TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-8, expect_output_2, output, 2U);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_mlp_propagation_and_sgd_update);

  return UNITY_END();
}