#include <unity.h>

#include <ann-lib/mlp.h>

extern float input[2], answer[2];
extern size_t layer_sizes[3];
extern const size_t layer_size_count;
extern struct mlp_layer_size_data size_data;
extern float learning_rate;
extern float momentum;
extern float one(void);
extern float neg_half(void);
extern float leaky_relu(float value);
extern float (*activation_funcs[3U])(float);
extern int test_mlp_get_gradients(size_t size, const float *expect,
                                  const float *actual, float *gradients_output);

static struct mlp_model *model;
static struct mlp_backward_buffer *backward_buffer;
static struct mlp_gradient_buffer *gradient_buffer;
static mlp_msgd_buffer_t *msgd_buffer;
static enum mlp_status status;

static float output[2], gradients[2];

void setUp(void) {
  model = NULL;
  backward_buffer = NULL;
  gradient_buffer = NULL;
  msgd_buffer = NULL;
  status = -1;
}

void tearDown(void) {
  mlp_destroy_model(model);
  mlp_destroy_backward_buffer(backward_buffer);
  mlp_destroy_gradient_buffer(gradient_buffer);
  mlp_destroy_msgd_buffer(msgd_buffer);
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
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, learning_rate));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, one));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, neg_half));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, backward_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(expect_output_1, output, 2U);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_update_parameters_sgd(model, gradient_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, NULL));
  TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-8, expect_output_2, output, 2U);
}

void test_mlp_propagation_and_msgd_update(void) {

  float expect_output_1[2] = {5.5f, 5.5f},
        expect_output_2[2] = {-3.153228e-2f, -3.209686e-2f},
        expect_output_3[2] = {-1.494893e-2f, -8.2605e-3f};

  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, learning_rate));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, one));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, neg_half));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_momentum(msgd_buffer, momentum));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, backward_buffer));
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(expect_output_1, output, 2U);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_clear_msgd_buffer(msgd_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, backward_buffer));
  TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-8, expect_output_2, output, 2U);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, NULL));
  TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-8, expect_output_3, output, 2U);
}

int main(void) {
  UNITY_BEGIN();

  RUN_TEST(test_mlp_propagation_and_sgd_update);
  RUN_TEST(test_mlp_propagation_and_msgd_update);

  return UNITY_END();
}