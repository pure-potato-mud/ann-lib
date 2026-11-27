#include <cstdlib>

extern "C" {
#include <unity.h>
}

#include <internal/mlp.hpp>

extern float input[2], answer[2];
extern size_t layer_sizes[3];
extern const size_t layer_size_count;
extern struct mlp_layer_size_data size_data;
extern float learning_rate;
extern float momentum;
extern float one();
extern float neg_half();
extern float leaky_relu(float value);
extern mlp_activation_t *activation_funcs[3U];
extern int test_mlp_get_gradients(size_t size, const float *expect,
                                  const float *actual, float *gradients_output);

static struct mlp_double_layers *double_layers;
static struct mlp_model *model;
static struct mlp_backward_buffer *backward_buffer;
static struct mlp_gradient_buffer *gradient_buffer;
static struct mlp_model *model_2;
static struct mlp_backward_buffer *backward_buffer_2;
static struct mlp_gradient_buffer *gradient_buffer_2;
static struct mlp_msgd_buffer *msgd_buffer;
static enum mlp_status status;

static float output[2], gradients[2], middle[3], middle_gradient[3];
static size_t layer_sizes_another[4] = {3U, 4U, 1U, 3U};
static const size_t layer_size_count_another = 4U;
static struct mlp_layer_size_data size_data_another = {layer_size_count_another,
                                                       layer_sizes_another};
static struct mlp_creation_functions creation_functions = {
    mlp_model_get_field_offsets, mlp_model_get_current_size};

static struct mlp_field_offsets *
test_mlp_get_null_obj(size_t layer_size_count) {
  (void)layer_size_count;
  return nullptr;
}
static struct mlp_field_offsets *
test_mlp_get_zero_field(size_t layer_size_count) {
  (void)layer_size_count;

  auto *ret = ::new struct mlp_field_offsets();
  if (ret == nullptr) { return nullptr; }

  ret->field_offsets = nullptr;
  ret->field_offset_count = 0U;

  return ret;
}

static struct mlp_field_offsets *
test_mlp_get_null_offsets(size_t layer_size_count) {
  (void)layer_size_count;

  auto *ret = ::new struct mlp_field_offsets();
  if (ret == nullptr) { return nullptr; }

  ret->field_offsets = nullptr;
  ret->field_offset_count = 1U;

  return ret;
}

void setUp() {
  double_layers = nullptr;
  model = nullptr;
  backward_buffer = nullptr;
  gradient_buffer = nullptr;
  model_2 = nullptr;
  backward_buffer_2 = nullptr;
  gradient_buffer_2 = nullptr;
  msgd_buffer = nullptr;
  status = MLP_STATUS_UNKNOWN_ERROR;
}

void tearDown() {
  mlp_destroy_double_layers(double_layers);
  mlp_destroy_model(model);
  mlp_destroy_backward_buffer(backward_buffer);
  mlp_destroy_gradient_buffer(gradient_buffer);
  mlp_destroy_model(model_2);
  mlp_destroy_backward_buffer(backward_buffer_2);
  mlp_destroy_gradient_buffer(gradient_buffer_2);
  mlp_destroy_msgd_buffer(msgd_buffer);
}

void test_mlp_create_double_layers_with_buffers() {
  double_layers =
      mlp_create_double_layers(true, &creation_functions, &size_data, &status);

  TEST_ASSERT_NOT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
}

void test_mlp_create_double_layers_without_buffers() {
  struct mlp_creation_functions gradient_creation_functions = {
      mlp_gradient_buffer_get_field_offsets,
      mlp_gradient_buffer_get_current_size};
  double_layers = mlp_create_double_layers(false, &gradient_creation_functions,
                                           &size_data, &status);

  TEST_ASSERT_NOT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
}

void test_mlp_create_double_layers_null_funcs() {
  double_layers = mlp_create_double_layers(true, NULL, &size_data, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_double_layers_null_layer_size() {
  double_layers =
      mlp_create_double_layers(true, &creation_functions, NULL, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_double_layers_null_get_field_offset() {
  struct mlp_creation_functions funcs_without_get_offset = {
      NULL, mlp_model_get_current_size};
  double_layers = mlp_create_double_layers(false, &funcs_without_get_offset,
                                           &size_data, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_HARDCODE, status);
}

void test_mlp_create_double_layers_null_get_current_size() {
  struct mlp_creation_functions funcs_without_get_size = {
      mlp_model_get_field_offsets, NULL};
  double_layers = mlp_create_double_layers(false, &funcs_without_get_size,
                                           &size_data, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_HARDCODE, status);
}

void test_mlp_create_double_layers_null_layer_sizes() {
  struct mlp_layer_size_data layer_data_without_sizes = {layer_size_count,
                                                         NULL};
  double_layers = mlp_create_double_layers(true, &creation_functions,
                                           &layer_data_without_sizes, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_double_layers_one_layer() {
  size_t one_layer[1] = {2U};
  struct mlp_layer_size_data two_layer_data = {1U, one_layer};
  double_layers = mlp_create_double_layers(true, &creation_functions,
                                           &two_layer_data, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_double_layers_get_null_offsets_obj() {
  struct mlp_creation_functions funcs_with_get_null_obj = {
      test_mlp_get_null_obj, mlp_model_get_current_size};
  double_layers = mlp_create_double_layers(false, &funcs_with_get_null_obj,
                                           &size_data, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_ALLOCATION_FAILED, status);
}

void test_mlp_create_double_layers_get_zero_field() {
  struct mlp_creation_functions funcs_with_get_zero_field = {
      test_mlp_get_zero_field, mlp_model_get_current_size};
  double_layers = mlp_create_double_layers(false, &funcs_with_get_zero_field,
                                           &size_data, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_HARDCODE, status);
}

void test_mlp_create_double_layers_get_null_field_offsets() {
  struct mlp_creation_functions funcs_with_get_null_offset = {
      test_mlp_get_null_offsets, mlp_model_get_current_size};
  double_layers = mlp_create_double_layers(false, &funcs_with_get_null_offset,
                                           &size_data, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_ALLOCATION_FAILED, status);
}

void test_mlp_create_double_layers_zero_neuron() {
  size_t small_layer_sizes[3] = {1U, 0U, 1U};
  struct mlp_layer_size_data test_size_data = {3U, small_layer_sizes};
  double_layers = mlp_create_double_layers(true, &creation_functions,
                                           &test_size_data, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_double_layers_large_offset_sum() {
  size_t large_layer_sizes[3] = {SIZE_MAX >> 1, (SIZE_MAX >> 2) / 3,
                                 (SIZE_MAX >> 2) / 3};
  struct mlp_layer_size_data test_size_data = {3U, large_layer_sizes};
  double_layers = mlp_create_double_layers(true, &creation_functions,
                                           &test_size_data, &status);

  TEST_ASSERT_NULL(double_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_INT_OVERFLOW, status);
}

void test_mlp_create_model() {
  model = mlp_create_model(&size_data, &status);

  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
}

void test_mlp_create_model_null_status() {
  model = mlp_create_model(&size_data, NULL);

  TEST_ASSERT_NOT_NULL(model);
}

void test_mlp_create_model_null_layer_sizes_ref() {
  struct mlp_layer_size_data test_size_data = {layer_size_count, NULL};
  model = mlp_create_model(&test_size_data, &status);

  TEST_ASSERT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_backward_buffer() {
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);

  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
}

void test_mlp_create_backward_buffer_null_status() {
  backward_buffer = mlp_create_backward_buffer(&size_data, NULL);

  TEST_ASSERT_NOT_NULL(backward_buffer);
}

void test_mlp_create_backward_buffer_null_layer_sizes_ref() {
  struct mlp_layer_size_data test_size_data = {layer_size_count, NULL};
  backward_buffer = mlp_create_backward_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_gradient_buffer() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);

  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
}

void test_mlp_create_gradient_buffer_null_status() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, NULL);

  TEST_ASSERT_NOT_NULL(gradient_buffer);
}

void test_mlp_create_gradient_buffer_null_layer_sizes_ref() {
  struct mlp_layer_size_data test_size_data = {layer_size_count, NULL};
  gradient_buffer = mlp_create_gradient_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_create_msgd_buffer() {
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);

  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
}

void test_mlp_create_msgd_buffer_null_status() {
  msgd_buffer = mlp_create_msgd_buffer(&size_data, NULL);

  TEST_ASSERT_NOT_NULL(msgd_buffer);
}

void test_mlp_create_msgd_buffer_null_layer_sizes_ref() {
  struct mlp_layer_size_data test_size_data = {layer_size_count, NULL};
  msgd_buffer = mlp_create_msgd_buffer(&test_size_data, &status);

  TEST_ASSERT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, status);
}

void test_mlp_set_learning_rate() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, learning_rate));

  TEST_ASSERT_EQUAL_FLOAT(learning_rate, model->learning_rate);
}

void test_mlp_set_learning_rate_null_model() {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_set_learning_rate(NULL, learning_rate));
}

void test_mlp_set_activation_fns_ref() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  TEST_ASSERT_EQUAL_PTR(activation_funcs, model->activation_funcs_ref);
}

void test_mlp_set_activation_fns_ref_null_model() {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_set_activation_fns_ref(NULL, activation_funcs));
}

void test_mlp_set_activation_fns_ref_null_fns_ref() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_set_activation_fns_ref(model, NULL));
}

void test_mlp_fill_biases() {
  float except_layer1_biases[3U] = {1.0f, 1.0f, 1.0f};
  float except_layer2_biases[2U] = {1.0f, 1.0f};
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, one));

  TEST_ASSERT_EQUAL_FLOAT_ARRAY(except_layer1_biases, model->bias_layers[0U],
                                3U);
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(except_layer2_biases, model->bias_layers[1U],
                                2U);
}

void test_mlp_fill_biases_null_model() {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, mlp_fill_biases(NULL, one));
}

void test_mlp_fill_biases_null_bias_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  model->bias_layers = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD, mlp_fill_biases(model, one));
}

void test_mlp_fill_biases_null_bias_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  model->bias_layers[0U] = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER, mlp_fill_biases(model, one));
}

void test_mlp_fill_biases_null_element_end() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  model->layer_data.elements_end = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_STATE, mlp_fill_biases(model, one));
}

void test_mlp_fill_weights() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, one));

  size_t i;
  for (i = 0U; i < 6U; ++i) {
    float weight_first_layer = model->weight_layers[0U][i],
          weight_second_layer = model->weight_layers[1U][i];
    TEST_ASSERT_EQUAL_FLOAT(1.0f, weight_first_layer);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, weight_second_layer);
  }
}

void test_mlp_fill_weights_null_model() {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, mlp_fill_weights(NULL, one));
}

void test_mlp_fill_weights_null_weight_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  model->weight_layers = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD, mlp_fill_weights(model, one));
}

void test_mlp_fill_weights_null_bias_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  model->bias_layers = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD, mlp_fill_weights(model, one));
}

void test_mlp_fill_weights_null_weight_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  model->weight_layers[0U] = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER, mlp_fill_weights(model, one));
}

void test_mlp_fill_weights_null_bias_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  model->bias_layers[0U] = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER, mlp_fill_weights(model, one));
}

void test_mlp_clear_gradient_buffer() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_NOT_NULL(gradient_buffer->weight_gradient_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  float *i;
  for (i = gradient_buffer->weight_gradient_layers[0U];
       i < gradient_buffer->layer_data.elements_end; ++i) {
    *i = 0.5f;
  }

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));

  for (i = gradient_buffer->weight_gradient_layers[0U];
       i < gradient_buffer->layer_data.elements_end; ++i) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, *i);
  }
}

void test_mlp_clear_gradient_buffer_null_model() {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_clear_gradient_buffer(NULL));
}

void test_mlp_clear_gradient_buffer_null_bias_layers() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer->weight_gradient_layers = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_clear_gradient_buffer(gradient_buffer));
}

void test_mlp_clear_gradient_buffer_null_bias_layer() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_NOT_NULL(gradient_buffer->weight_gradient_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer->weight_gradient_layers[0U] = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_clear_gradient_buffer(gradient_buffer));
}

void test_mlp_clear_gradient_buffer_null_element_end() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer->layer_data.elements_end = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_STATE,
                    mlp_clear_gradient_buffer(gradient_buffer));
}

void test_mlp_scale_gradients() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_NOT_NULL(gradient_buffer->weight_gradient_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  float *i;
  for (i = gradient_buffer->weight_gradient_layers[0U];
       i < gradient_buffer->layer_data.elements_end; ++i) {
    *i = 0.5f;
  }

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_scale_gradients(gradient_buffer, 0.5f));

  for (i = gradient_buffer->weight_gradient_layers[0U];
       i < gradient_buffer->layer_data.elements_end; ++i) {
    TEST_ASSERT_EQUAL_FLOAT(0.25f, *i);
  }
}

void test_mlp_scale_gradients_null_model() {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_scale_gradients(NULL, 0.5f));
}

void test_mlp_scale_gradients_null_bias_layers() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer->weight_gradient_layers = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_scale_gradients(gradient_buffer, 0.5f));
}

void test_mlp_scale_gradients_null_bias_layer() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_NOT_NULL(gradient_buffer->weight_gradient_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer->weight_gradient_layers[0U] = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_scale_gradients(gradient_buffer, 0.5f));
}

void test_mlp_scale_gradients_null_element_end() {
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer->layer_data.elements_end = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_STATE,
                    mlp_scale_gradients(gradient_buffer, 0.5f));
}

void test_mlp_set_momentum() {
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_momentum(msgd_buffer, momentum));

  TEST_ASSERT_EQUAL_FLOAT(momentum, msgd_buffer->momentum);
}

void test_mlp_set_momentum_null_buffer() {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT,
                    mlp_set_momentum(NULL, momentum));
}

void test_mlp_clear_msgd_buffer() {
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_NOT_NULL(msgd_buffer->layer_data.layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  float *i;
  for (i = msgd_buffer->layer_data.layers[0U];
       i < msgd_buffer->layer_data.elements_end; ++i) {
    *i = 0.5f;
  }

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_clear_msgd_buffer(msgd_buffer));

  for (i = msgd_buffer->layer_data.layers[0U];
       i < msgd_buffer->layer_data.elements_end; ++i) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, *i);
  }
}

void test_mlp_clear_msgd_buffer_null_model() {
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_ARGUMENT, mlp_clear_msgd_buffer(NULL));
}

void test_mlp_clear_msgd_buffer_null_weight_velocity_layers() {
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer->weight_velocity_layers = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_clear_msgd_buffer(msgd_buffer));
}

void test_mlp_clear_msgd_buffer_null_weight_velocity_layer() {
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_NOT_NULL(msgd_buffer->bias_velocity_layers);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer->weight_velocity_layers[0U] = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_clear_msgd_buffer(msgd_buffer));
}

void test_mlp_clear_msgd_buffer_null_element_end() {
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer->layer_data.elements_end = nullptr;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_STATE,
                    mlp_clear_msgd_buffer(msgd_buffer));
}

void test_mlp_forward() {
  float output[2U], except_output[2U] = {5.5f, 5.5f};
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, learning_rate));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, neg_half));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, one));

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, NULL));

  TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-7f, except_output, output, 2U);
}

void test_mlp_forward_different_model_metadata() {
  model = mlp_create_model(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH,
                    mlp_forward(model, input, NULL, backward_buffer));
}

void test_mlp_forward_different_backward_buffer_metadata() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  backward_buffer = mlp_create_backward_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH,
                    mlp_forward(model, input, NULL, backward_buffer));
}

void test_mlp_forward_null_layer_sizes() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->layer_metadata.layer_sizes_ref = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_forward(model, input, NULL, NULL));
}

void test_mlp_forward_null_buffers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  float **buffers_temp = model->buffers;
  model->buffers = nullptr;
  int result = mlp_forward(model, input, NULL, NULL);
  model->buffers = buffers_temp;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD, result);
}

void test_mlp_forward_null_buffer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  float *buffer_temp = model->buffers[0];
  model->buffers[0] = nullptr;
  int result = mlp_forward(model, input, NULL, NULL);
  model->buffers[0] = buffer_temp;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER, result);
}

void test_mlp_forward_null_weight_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_forward(model, input, NULL, NULL));
}

void test_mlp_forward_null_weight_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_forward(model, input, NULL, NULL));
}

void test_mlp_forward_null_bias_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_forward(model, input, NULL, NULL));
}

void test_mlp_forward_null_bias_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_forward(model, input, NULL, NULL));
}

void test_mlp_forward_null_activation_funcs() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_forward(model, input, NULL, NULL));
}

void test_mlp_backward() {
  float dump[17U] = {5.0f,  20.0f, 5.0f,  20.0f, 5.0f,  20.0f, 9.0f, 9.0f, 9.0f,
                     11.0f, 11.0f, 11.0f, 10.0f, 10.0f, 10.0f, 4.5f, 5.5f};
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, learning_rate));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, neg_half));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, one));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, backward_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL));

  float *begin = gradient_buffer->layer_data.elements_begin,
        *end = gradient_buffer->layer_data.elements_end;
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(dump, begin, end - begin);
}

void test_mlp_backward_pass_gradient() {
  // float dump[17U] =
  // {5.0f,  20.0f, 5.0f,  20.0f, 5.0f,  20.0f, 9.0f, 9.0f, 9.0f, 11.0f, 11.0f, 11.0f,
  // 10.0f, 10.0f, 10.0f, 4.5f, 5.5f};
  float dump_first[9] = {5.0f,  20.0f, 5.0f,  20.0f, 5.0f,
                         20.0f, 10.0f, 10.0f, 10.0f};
  float dump_second[8] = {9.0f, 9.0f, 9.0f, 11.0f, 11.0f, 11.0f, 4.5f, 5.5f};

  size_t layer_sizes_first[2] = {2U, 3U};
  struct mlp_layer_size_data size_data_first = {2U, layer_sizes_first};
  model = mlp_create_model(&size_data_first, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  backward_buffer = mlp_create_backward_buffer(&size_data_first, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data_first, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, learning_rate));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, neg_half));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, one));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));

  size_t layer_sizes_second[2] = {3U, 2U};
  struct mlp_layer_size_data size_data_second = {2U, layer_sizes_second};
  model_2 = mlp_create_model(&size_data_second, &status);
  TEST_ASSERT_NOT_NULL(model_2);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  backward_buffer_2 = mlp_create_backward_buffer(&size_data_second, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer_2);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer_2 = mlp_create_gradient_buffer(&size_data_second, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer_2);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model_2, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model_2, learning_rate));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model_2, neg_half));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model_2, one));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer_2));

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, middle, backward_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model_2, middle, output, backward_buffer_2));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_backward(model_2, backward_buffer_2, gradients,
                                 gradient_buffer_2, middle_gradient));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_backward(model, backward_buffer, middle_gradient,
                                 gradient_buffer, NULL));

  float *begin = gradient_buffer->layer_data.elements_begin,
        *end = gradient_buffer->layer_data.elements_end;
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(dump_first, begin, end - begin);
  begin = gradient_buffer_2->layer_data.elements_begin;
  end = gradient_buffer_2->layer_data.elements_end;
  TEST_ASSERT_EQUAL_FLOAT_ARRAY(dump_second, begin, end - begin);
}

void test_mlp_backward_different_model_metadata() {
  model = mlp_create_model(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  int result =
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_backward_different_backward_buffer_metadata() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  backward_buffer = mlp_create_backward_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  int result =
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_backward_different_gradient_buffer_metadata() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  int result =
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_backward_null_layer_sizes() {
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  model->layer_metadata.layer_sizes_ref = nullptr;
  backward_buffer->layer_metadata.layer_sizes_ref = nullptr;
  gradient_buffer->layer_metadata.layer_sizes_ref = nullptr;
  int result =
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_backward_null_layers() {
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  float **layers_temp = backward_buffer->layers;
  backward_buffer->layers = nullptr;
  int result =
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL);
  backward_buffer->layers = layers_temp;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD, result);
}

void test_mlp_backward_null_layer() {
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  float *layer_temp = backward_buffer->layers[0];
  backward_buffer->layers[0] = nullptr;
  int result =
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL);
  backward_buffer->layers[0] = layer_temp;

  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER, result);
}

void test_mlp_backward_null_output_layers() {
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  backward_buffer->raw_output_layers = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_FIELD,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL));
}

void test_mlp_backward_null_output_layer() {
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  backward_buffer->raw_output_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_LAYER,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL));
}

void test_mlp_backward_null_weight_layers() {
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  model->weight_layers = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_FIELD,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL));
}

void test_mlp_backward_null_weight_layer() {
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  model->weight_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_LAYER,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL));
}

void test_mlp_backward_null_activation_funcs() {
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
                    test_mlp_get_gradients(2U, answer, output, gradients));

  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_FIELD,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL));
}

void test_mlp_update_parameters_sgd() {
  float dump[17U] = {0.375f,  -1.5f,   0.375f,  -1.5f,    0.375f,  -1.5f,
                     -0.125f, -0.125f, -0.125f, -0.375f,  -0.375f, -0.375f,
                     -1.75f,  -1.75f,  -1.75f,  -1.0625f, -1.1875f};
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, learning_rate));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, neg_half));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, one));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, backward_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_update_parameters_sgd(model, gradient_buffer));

  float *begin = model->weight_layers[0], *end = model->layer_data.elements_end;
  TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-6f, dump, begin, end - begin);
}

void test_mlp_update_parameters_sgd_different_model_metadata() {
  model = mlp_create_model(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  int result = mlp_update_parameters_sgd(model, gradient_buffer);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_sgd_different_gradient_buffer_metadata() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  int result = mlp_update_parameters_sgd(model, gradient_buffer);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_sgd_null_layer_sizes() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->layer_metadata.layer_sizes_ref = nullptr;
  gradient_buffer->layer_metadata.layer_sizes_ref = nullptr;
  int result = mlp_update_parameters_sgd(model, gradient_buffer);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_sgd_null_weight_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_weight_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_bias_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_bias_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_weight_gradient_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->weight_gradient_layers = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_weight_gradient_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->weight_gradient_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_update_parameters_sgd(model, gradient_buffer));
}
void test_mlp_update_parameters_sgd_null_bias_gradient_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->bias_gradient_layers = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_FIELD,
                    mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_sgd_null_bias_gradient_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->bias_gradient_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(MLP_STATUS_INVALID_LAYER,
                    mlp_update_parameters_sgd(model, gradient_buffer));
}

void test_mlp_update_parameters_msgd() {
  float dump[17U] = {-0.171884f, -3.687534f, -0.171884f, -3.687534f, -0.171884f,
                     -3.687534f, -1.109950f, -1.109950f, -1.109950f, -1.578143f,
                     -1.578143f, -1.578143f, -2.843767f, -2.843767f, -2.843767f,
                     -1.550658f, -1.788937f};
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
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, learning_rate));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_momentum(msgd_buffer, momentum));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, neg_half));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, one));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_clear_msgd_buffer(msgd_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, backward_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));

  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_clear_gradient_buffer(gradient_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_forward(model, input, output, backward_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_backward(model, backward_buffer, gradients, gradient_buffer, NULL));
  TEST_ASSERT_EQUAL(
      MLP_STATUS_SUCCESS,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));

  float *begin = model->weight_layers[0], *end = model->layer_data.elements_end;
  TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-6f, dump, begin, end - begin);
}

void test_mlp_update_parameters_msgd_different_model_metadata() {
  model = mlp_create_model(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  int result = mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_msgd_different_gradient_buffer_metadata() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  int result = mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_msgd_different_msgd_buffer_metadata() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data_another, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    test_mlp_get_gradients(2U, answer, output, gradients));

  int result = mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_msgd_null_layer_sizes() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->layer_metadata.layer_sizes_ref = nullptr;
  gradient_buffer->layer_metadata.layer_sizes_ref = nullptr;
  int result = mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer);

  TEST_ASSERT_EQUAL(MLP_STATUS_TOPOLOGY_MISMATCH, result);
}

void test_mlp_update_parameters_msgd_null_weight_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_FIELD,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_weight_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->weight_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_LAYER,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_bias_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_FIELD,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_bias_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  model->bias_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_LAYER,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_weight_gradient_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->weight_gradient_layers = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_FIELD,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_weight_gradient_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->weight_gradient_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_LAYER,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_bias_gradient_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->bias_gradient_layers = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_FIELD,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_bias_gradient_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  gradient_buffer->bias_gradient_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_LAYER,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_weight_velocity_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  msgd_buffer->weight_velocity_layers = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_FIELD,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_weight_velocity_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  msgd_buffer->weight_velocity_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_LAYER,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_bias_velocity_layers() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  msgd_buffer->bias_velocity_layers = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_FIELD,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

void test_mlp_update_parameters_msgd_null_bias_velocity_layer() {
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activation_funcs));

  msgd_buffer->bias_velocity_layers[0] = nullptr;
  TEST_ASSERT_EQUAL(
      MLP_STATUS_INVALID_LAYER,
      mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer));
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(test_mlp_create_double_layers_with_buffers);
  RUN_TEST(test_mlp_create_double_layers_without_buffers);
  RUN_TEST(test_mlp_create_double_layers_null_funcs);
  RUN_TEST(test_mlp_create_double_layers_null_layer_size);
  RUN_TEST(test_mlp_create_double_layers_null_get_field_offset);
  RUN_TEST(test_mlp_create_double_layers_null_get_current_size);
  RUN_TEST(test_mlp_create_double_layers_null_layer_sizes);
  RUN_TEST(test_mlp_create_double_layers_one_layer);
  RUN_TEST(test_mlp_create_double_layers_get_null_offsets_obj);
  RUN_TEST(test_mlp_create_double_layers_get_zero_field);
  RUN_TEST(test_mlp_create_double_layers_get_null_field_offsets);
  RUN_TEST(test_mlp_create_double_layers_zero_neuron);
  RUN_TEST(test_mlp_create_double_layers_large_offset_sum);

  RUN_TEST(test_mlp_create_model);
  RUN_TEST(test_mlp_create_model_null_status);
  RUN_TEST(test_mlp_create_model_null_layer_sizes_ref);

  RUN_TEST(test_mlp_create_backward_buffer);
  RUN_TEST(test_mlp_create_backward_buffer_null_status);
  RUN_TEST(test_mlp_create_backward_buffer_null_layer_sizes_ref);

  RUN_TEST(test_mlp_create_gradient_buffer);
  RUN_TEST(test_mlp_create_gradient_buffer_null_status);
  RUN_TEST(test_mlp_create_gradient_buffer_null_layer_sizes_ref);

  RUN_TEST(test_mlp_create_msgd_buffer);
  RUN_TEST(test_mlp_create_msgd_buffer_null_status);
  RUN_TEST(test_mlp_create_msgd_buffer_null_layer_sizes_ref);

  RUN_TEST(test_mlp_set_learning_rate);
  RUN_TEST(test_mlp_set_learning_rate_null_model);

  RUN_TEST(test_mlp_set_activation_fns_ref);
  RUN_TEST(test_mlp_set_activation_fns_ref_null_model);
  RUN_TEST(test_mlp_set_activation_fns_ref_null_fns_ref);

  RUN_TEST(test_mlp_fill_biases);
  RUN_TEST(test_mlp_fill_biases_null_model);
  RUN_TEST(test_mlp_fill_biases_null_bias_layers);
  RUN_TEST(test_mlp_fill_biases_null_bias_layer);
  RUN_TEST(test_mlp_fill_biases_null_element_end);

  RUN_TEST(test_mlp_fill_weights);
  RUN_TEST(test_mlp_fill_weights_null_model);
  RUN_TEST(test_mlp_fill_weights_null_weight_layers);
  RUN_TEST(test_mlp_fill_weights_null_bias_layers);
  RUN_TEST(test_mlp_fill_weights_null_weight_layer);
  RUN_TEST(test_mlp_fill_weights_null_bias_layer);

  RUN_TEST(test_mlp_clear_gradient_buffer);
  RUN_TEST(test_mlp_clear_gradient_buffer_null_model);
  RUN_TEST(test_mlp_clear_gradient_buffer_null_bias_layers);
  RUN_TEST(test_mlp_clear_gradient_buffer_null_bias_layer);
  RUN_TEST(test_mlp_clear_gradient_buffer_null_element_end);

  RUN_TEST(test_mlp_scale_gradients);
  RUN_TEST(test_mlp_scale_gradients_null_model);
  RUN_TEST(test_mlp_scale_gradients_null_bias_layers);
  RUN_TEST(test_mlp_scale_gradients_null_bias_layer);
  RUN_TEST(test_mlp_scale_gradients_null_element_end);

  RUN_TEST(test_mlp_set_momentum);
  RUN_TEST(test_mlp_set_momentum_null_buffer);

  RUN_TEST(test_mlp_clear_msgd_buffer);
  RUN_TEST(test_mlp_clear_msgd_buffer_null_model);
  RUN_TEST(test_mlp_clear_msgd_buffer_null_weight_velocity_layers);
  RUN_TEST(test_mlp_clear_msgd_buffer_null_weight_velocity_layer);
  RUN_TEST(test_mlp_clear_msgd_buffer_null_element_end);

  RUN_TEST(test_mlp_forward);
  RUN_TEST(test_mlp_forward_different_model_metadata);
  RUN_TEST(test_mlp_forward_different_backward_buffer_metadata);
  RUN_TEST(test_mlp_forward_null_layer_sizes);
  RUN_TEST(test_mlp_forward_null_buffers);
  RUN_TEST(test_mlp_forward_null_buffer);
  RUN_TEST(test_mlp_forward_null_weight_layers);
  RUN_TEST(test_mlp_forward_null_weight_layer);
  RUN_TEST(test_mlp_forward_null_bias_layers);
  RUN_TEST(test_mlp_forward_null_bias_layer);
  RUN_TEST(test_mlp_forward_null_activation_funcs);

  RUN_TEST(test_mlp_backward);
  RUN_TEST(test_mlp_backward_pass_gradient);
  RUN_TEST(test_mlp_backward_different_model_metadata);
  RUN_TEST(test_mlp_backward_different_backward_buffer_metadata);
  RUN_TEST(test_mlp_backward_different_gradient_buffer_metadata);
  RUN_TEST(test_mlp_backward_null_layer_sizes);
  RUN_TEST(test_mlp_backward_null_layers);
  RUN_TEST(test_mlp_backward_null_layer);
  RUN_TEST(test_mlp_backward_null_output_layers);
  RUN_TEST(test_mlp_backward_null_output_layer);
  RUN_TEST(test_mlp_backward_null_weight_layers);
  RUN_TEST(test_mlp_backward_null_weight_layer);
  RUN_TEST(test_mlp_backward_null_activation_funcs);

  RUN_TEST(test_mlp_update_parameters_sgd);
  RUN_TEST(test_mlp_update_parameters_sgd_different_model_metadata);
  RUN_TEST(test_mlp_update_parameters_sgd_different_gradient_buffer_metadata);
  RUN_TEST(test_mlp_update_parameters_sgd_null_layer_sizes);
  RUN_TEST(test_mlp_update_parameters_sgd_null_weight_layers);
  RUN_TEST(test_mlp_update_parameters_sgd_null_weight_layer);
  RUN_TEST(test_mlp_update_parameters_sgd_null_bias_layers);
  RUN_TEST(test_mlp_update_parameters_sgd_null_bias_layer);
  RUN_TEST(test_mlp_update_parameters_sgd_null_weight_gradient_layers);
  RUN_TEST(test_mlp_update_parameters_sgd_null_weight_gradient_layer);
  RUN_TEST(test_mlp_update_parameters_sgd_null_bias_gradient_layers);
  RUN_TEST(test_mlp_update_parameters_sgd_null_bias_gradient_layer);

  RUN_TEST(test_mlp_update_parameters_msgd);
  RUN_TEST(test_mlp_update_parameters_msgd_different_model_metadata);
  RUN_TEST(test_mlp_update_parameters_msgd_different_gradient_buffer_metadata);
  RUN_TEST(test_mlp_update_parameters_msgd_different_msgd_buffer_metadata);
  RUN_TEST(test_mlp_update_parameters_msgd_null_layer_sizes);
  RUN_TEST(test_mlp_update_parameters_msgd_null_weight_layers);
  RUN_TEST(test_mlp_update_parameters_msgd_null_weight_layer);
  RUN_TEST(test_mlp_update_parameters_msgd_null_bias_layers);
  RUN_TEST(test_mlp_update_parameters_msgd_null_bias_layer);
  RUN_TEST(test_mlp_update_parameters_msgd_null_weight_gradient_layers);
  RUN_TEST(test_mlp_update_parameters_msgd_null_weight_gradient_layer);
  RUN_TEST(test_mlp_update_parameters_msgd_null_bias_gradient_layers);
  RUN_TEST(test_mlp_update_parameters_msgd_null_bias_gradient_layer);
  RUN_TEST(test_mlp_update_parameters_msgd_null_weight_velocity_layers);
  RUN_TEST(test_mlp_update_parameters_msgd_null_weight_velocity_layer);
  RUN_TEST(test_mlp_update_parameters_msgd_null_bias_velocity_layers);
  RUN_TEST(test_mlp_update_parameters_msgd_null_bias_velocity_layer);

  return UNITY_END();
}
