#include <cmath>
#include <cstdint>
#include <random>

extern "C" {
#include <unity.h>

#include <ann-lib/mlp.h>
}

#define PI 3.1415926535897

extern "C" {
extern mlp_activation_t activation;
extern int test_mlp_get_gradients(size_t size, const float *expect,
                                  const float *actual, float *gradients_output);
}

static std::mt19937 rng;
static struct mlp_model *model;
static struct mlp_backward_buffer *backward_buffer;
static struct mlp_gradient_buffer *gradient_buffer;
static mlp_msgd_buffer_t *msgd_buffer;
static enum mlp_status status;

float mlp_rand() { return (float)rng() / UINT32_MAX - 0.5f; }

struct TrainingConfig {
  size_t set_size;
  size_t batch_size;
  size_t epoch_size;
};

struct TrainingPack {
  const size_t input_size;
  const size_t output_size;
  const size_t total_size;
  const float scalar;
  const float *data_set;
  float *input;
  float *output;
  float *answer;
  float *gradient;
};

struct TrainingEvents {
  mlp_status_t (*on_batch_begin)();
  mlp_status_t (*on_integration)(const size_t, TrainingPack &);
  mlp_status_t (*on_update)(TrainingPack &);
};

mlp_status_t train(const TrainingConfig &config, const TrainingEvents &events,
                   TrainingPack &context) {
  mlp_status_t status = MLP_STATUS_UNKNOWN_ERROR;
  size_t set_size = config.set_size, batch_size = config.batch_size,
         epoch_size = config.epoch_size;
  size_t last_batch_size = set_size % batch_size;
  size_t batch_count = set_size / batch_size;
  size_t epoch_idx, batch_idx, i;
  for (epoch_idx = 0; epoch_idx < epoch_size; ++epoch_idx) {
    for (batch_idx = 0; batch_idx <= batch_count; ++batch_idx) {
      status = events.on_batch_begin();
      if (status != MLP_STATUS_SUCCESS) { return status; }
      size_t current_batch_size =
          (batch_idx < batch_count ? batch_size : last_batch_size);
      if (current_batch_size == 0) { continue; }
      for (i = 0; i < current_batch_size; ++i) {
        status = events.on_integration((batch_idx * batch_size) + i, context);
        if (status != MLP_STATUS_SUCCESS) { return status; }
      }
      status = events.on_update(context);
      if (status != MLP_STATUS_SUCCESS) { return status; }
    }
  }
  return MLP_STATUS_SUCCESS;
}

static mlp_status_t on_batch_begin() {
  status = mlp_clear_gradient_buffer(gradient_buffer);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  return MLP_STATUS_SUCCESS;
}

static mlp_status_t on_integration(const size_t sample_idx,
                                   TrainingPack &pack) {
  const size_t input_size = pack.input_size, output_size = pack.output_size,
               total_size = pack.total_size;
  for (size_t i = 0U; i < input_size; ++i) {
    pack.input[i] = pack.data_set[sample_idx * total_size + i];
  }
  for (size_t i = 0U; i < output_size; ++i) {
    pack.answer[i] = pack.data_set[sample_idx * total_size + input_size + i];
  }
  status = mlp_forward(model, pack.input, pack.output, backward_buffer);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  if (1 == test_mlp_get_gradients(output_size, pack.answer, pack.output,
                                  pack.gradient)) {
    return MLP_STATUS_INVALID_ARGUMENT;
  }
  status = mlp_backward(model, backward_buffer, pack.gradient, gradient_buffer);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  return MLP_STATUS_SUCCESS;
}

static mlp_status_t on_update_sgd(TrainingPack &pack) {
  status = mlp_scale_gradients(gradient_buffer, pack.scalar);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  status = mlp_update_parameters_sgd(model, gradient_buffer);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  return MLP_STATUS_SUCCESS;
}

void setUp() {
  rng.seed(4067023588);
  model = nullptr;
  backward_buffer = nullptr;
  gradient_buffer = nullptr;
  msgd_buffer = nullptr;
  status = MLP_STATUS_UNKNOWN_ERROR;
}

void tearDown() {
  mlp_destroy_model(model);
  mlp_destroy_backward_buffer(backward_buffer);
  mlp_destroy_gradient_buffer(gradient_buffer);
  mlp_destroy_msgd_buffer(msgd_buffer);
}

void test_mlp_xor() {
  constexpr size_t layer_size_count = 4U;
  constexpr size_t input_size = 2U, output_size = 1U;
  constexpr size_t training_set_size = 4U;
  constexpr size_t batch_size = 2U;
  constexpr size_t total_size = input_size + output_size;
  float input[input_size], output[output_size], answer[output_size],
      gradient[output_size];
  size_t sizes[layer_size_count] = {input_size, 4U, 4U, output_size};
  mlp_layer_size_data_t size_data = {layer_size_count, sizes};

  float training_set[training_set_size][total_size] = {
      {0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 1.0f},
      {1.0f, 0.0f, 1.0f},
      {1.0f, 1.0f, 0.0f},
  };
  float scalar = 1.0f / batch_size;
  TrainingConfig config = {training_set_size, batch_size, 1500};
  mlp_activation_t *activate_fns[4] = {&activation, &activation, &activation,
                                       &activation};

  // allocate
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_NOT_NULL(model);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);

  // setup
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_set_learning_rate(model, 0.03125f));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activate_fns));
  TrainingPack pack = {
      input_size, output_size, total_size, scalar,   (float *)training_set,
      input,      output,      answer,     gradient,
  };
  TrainingEvents events = {on_batch_begin, on_integration, on_update_sgd};

  double init_loss = 0.0, tmp_loss;
  for (auto const &sample : training_set) {
    for (size_t i = 0; i < input_size; ++i) {
      input[i] = sample[i];
    }
    for (size_t i = 0; i < output_size; ++i) {
      answer[i] = sample[input_size + i];
    }
    TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                      mlp_forward(model, input, output, nullptr));
    tmp_loss = output[0] - answer[0];
    init_loss += tmp_loss * tmp_loss;
  }
  init_loss = sqrt(init_loss / training_set_size);

  // act
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, train(config, events, pack));

  // assert
  double trained_loss = 0.0;
  for (auto const &sample : training_set) {
    for (size_t i = 0; i < input_size; ++i) {
      input[i] = sample[i];
    }
    for (size_t i = 0; i < output_size; ++i) {
      answer[i] = sample[input_size + i];
    }
    TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                      mlp_forward(model, input, output, nullptr));
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(5e-2, answer, output, output_size);
    tmp_loss = output[0] - answer[0];
    trained_loss += tmp_loss * tmp_loss;
  }
  trained_loss = sqrt(trained_loss / training_set_size);

  TEST_ASSERT_LESS_THAN_FLOAT(init_loss * 0.1, trained_loss);
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(test_mlp_xor);

  return UNITY_END();
}
