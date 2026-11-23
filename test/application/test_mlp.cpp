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
static struct mlp_model *model_2;
static struct mlp_backward_buffer *backward_buffer_2;
static struct mlp_gradient_buffer *gradient_buffer_2;
static mlp_msgd_buffer_t *msgd_buffer_2;
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

struct TrainingPackEncoderDecoder {
  const size_t io_size;
  const float scalar;
  const float *data_set;
  float *input;
  float *middle;
  float *output;
  float *answer;
  float *gradient;
  float *middle_gradient;
};

template <typename T> struct TrainingEvents {
  mlp_status_t (*on_batch_begin)();
  mlp_status_t (*on_integration)(const size_t, T &);
  mlp_status_t (*on_update)(T &);
};

template <typename T>
mlp_status_t train(const TrainingConfig &config,
                   const TrainingEvents<T> &events, T &context) {
  mlp_status_t status = MLP_STATUS_UNKNOWN_ERROR;
  size_t set_size = config.set_size, batch_size = config.batch_size,
         epoch_size = config.epoch_size;
  if (batch_size == 0) { return MLP_STATUS_INVALID_FIELD; }
  size_t last_batch_size = set_size % batch_size;
  size_t batch_count = set_size / batch_size;
  size_t epoch_idx, batch_idx, i;
  for (epoch_idx = 0; epoch_idx < epoch_size; ++epoch_idx) {
    for (batch_idx = 0; batch_idx <= batch_count; ++batch_idx) {
      size_t current_batch_size =
          (batch_idx < batch_count ? batch_size : last_batch_size);
      if (current_batch_size == 0) { continue; }

      status = events.on_batch_begin();
      if (status != MLP_STATUS_SUCCESS) { return status; }
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
  status = mlp_backward(model, backward_buffer, pack.gradient, gradient_buffer,
                        nullptr);
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

static mlp_status_t on_update_msgd(TrainingPack &pack) {
  status = mlp_scale_gradients(gradient_buffer, pack.scalar);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  status = mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  return MLP_STATUS_SUCCESS;
}

static mlp_status_t on_batch_begin_auto_encoder() {
  status = mlp_clear_gradient_buffer(gradient_buffer);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  status = mlp_clear_gradient_buffer(gradient_buffer_2);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  return MLP_STATUS_SUCCESS;
}

static mlp_status_t
on_integration_auto_encoder(const size_t sample_idx,
                            TrainingPackEncoderDecoder &pack) {
  const size_t io_size = pack.io_size;
  for (size_t i = 0U; i < io_size; ++i) {
    pack.input[i] = pack.data_set[sample_idx * io_size + i];
  }
  for (size_t i = 0U; i < io_size; ++i) {
    pack.answer[i] = pack.data_set[sample_idx * io_size + i];
  }
  status = mlp_forward(model, pack.input, pack.middle, backward_buffer);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  status = mlp_forward(model_2, pack.middle, pack.output, backward_buffer_2);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  if (1 == test_mlp_get_gradients(io_size, pack.answer, pack.output,
                                  pack.gradient)) {
    return MLP_STATUS_INVALID_ARGUMENT;
  }
  status = mlp_backward(model_2, backward_buffer_2, pack.gradient,
                        gradient_buffer_2, pack.middle_gradient);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  status = mlp_backward(model, backward_buffer, pack.middle_gradient,
                        gradient_buffer, nullptr);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  return MLP_STATUS_SUCCESS;
}

static mlp_status_t
on_update_msgd_auto_encoder(TrainingPackEncoderDecoder &pack) {
  status = mlp_scale_gradients(gradient_buffer, pack.scalar);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  status = mlp_scale_gradients(gradient_buffer_2, pack.scalar);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  status = mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  status =
      mlp_update_parameters_msgd(model_2, msgd_buffer_2, gradient_buffer_2);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  return MLP_STATUS_SUCCESS;
}

void setUp() {
  rng.seed(4067023588);
  model = nullptr;
  backward_buffer = nullptr;
  gradient_buffer = nullptr;
  msgd_buffer = nullptr;
  model_2 = nullptr;
  backward_buffer_2 = nullptr;
  gradient_buffer_2 = nullptr;
  msgd_buffer_2 = nullptr;
  status = MLP_STATUS_UNKNOWN_ERROR;
}

void tearDown() {
  mlp_destroy_model(model);
  mlp_destroy_backward_buffer(backward_buffer);
  mlp_destroy_gradient_buffer(gradient_buffer);
  mlp_destroy_msgd_buffer(msgd_buffer);
  mlp_destroy_model(model_2);
  mlp_destroy_backward_buffer(backward_buffer_2);
  mlp_destroy_gradient_buffer(gradient_buffer_2);
  mlp_destroy_msgd_buffer(msgd_buffer_2);
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
  TrainingEvents<TrainingPack> events = {on_batch_begin, on_integration,
                                         on_update_sgd};

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

void test_mlp_noisy_regression() {
  constexpr size_t layer_size_count = 4U;
  constexpr size_t input_size = 1U, output_size = 1U;
  constexpr size_t training_set_size = 300U;
  constexpr size_t testing_set_size = 300U;
  constexpr size_t batch_size = 16U;
  constexpr size_t total_size = input_size + output_size;
  float input[input_size], output[output_size], answer[output_size],
      gradient[output_size];
  size_t sizes[layer_size_count] = {input_size, 16U, 16U, output_size};
  mlp_layer_size_data_t size_data = {layer_size_count, sizes};

  float training_set[training_set_size][total_size] = {
      {0.000000f, 0.496305f}, {0.003344f, 0.434620f}, {0.006689f, 0.497903f},
      {0.010033f, 0.467260f}, {0.013378f, 0.601679f}, {0.016722f, 0.634048f},
      {0.020067f, 0.542304f}, {0.023411f, 0.665624f}, {0.026756f, 0.643111f},
      {0.030100f, 0.686998f}, {0.033445f, 0.566984f}, {0.036789f, 0.658204f},
      {0.040134f, 0.616343f}, {0.043478f, 0.651687f}, {0.046823f, 0.675215f},
      {0.050167f, 0.641073f}, {0.053512f, 0.684548f}, {0.056856f, 0.755471f},
      {0.060201f, 0.766443f}, {0.063545f, 0.778489f}, {0.066890f, 0.731790f},
      {0.070234f, 0.825078f}, {0.073579f, 0.784347f}, {0.076923f, 0.713083f},
      {0.080268f, 0.796280f}, {0.083612f, 0.734526f}, {0.086957f, 0.824356f},
      {0.090301f, 0.757131f}, {0.093645f, 0.820599f}, {0.096990f, 0.803947f},
      {0.100334f, 0.758471f}, {0.103679f, 0.860047f}, {0.107023f, 0.897431f},
      {0.110368f, 0.883804f}, {0.113712f, 0.761007f}, {0.117057f, 0.760270f},
      {0.120401f, 0.888107f}, {0.123746f, 0.874823f}, {0.127090f, 0.898613f},
      {0.130435f, 0.971797f}, {0.133779f, 0.899129f}, {0.137124f, 0.886855f},
      {0.140468f, 0.838757f}, {0.143813f, 0.876820f}, {0.147157f, 0.984363f},
      {0.150502f, 0.963845f}, {0.153846f, 0.924860f}, {0.157191f, 0.919471f},
      {0.160535f, 0.896734f}, {0.163880f, 0.922883f}, {0.167224f, 0.812008f},
      {0.170569f, 0.976303f}, {0.173913f, 0.959037f}, {0.177258f, 0.901215f},
      {0.180602f, 0.903621f}, {0.183946f, 0.860620f}, {0.187291f, 0.889858f},
      {0.190635f, 0.815760f}, {0.193980f, 0.984420f}, {0.197324f, 0.872920f},
      {0.200669f, 0.935578f}, {0.204013f, 0.835941f}, {0.207358f, 0.805188f},
      {0.210702f, 0.899007f}, {0.214047f, 0.803050f}, {0.217391f, 0.778138f},
      {0.220736f, 0.937659f}, {0.224080f, 0.879536f}, {0.227425f, 0.837759f},
      {0.230769f, 0.875053f}, {0.234114f, 0.886670f}, {0.237458f, 0.878496f},
      {0.240803f, 0.726725f}, {0.244147f, 0.817069f}, {0.247492f, 0.751093f},
      {0.250836f, 0.753654f}, {0.254181f, 0.854721f}, {0.257525f, 0.837010f},
      {0.260870f, 0.694940f}, {0.264214f, 0.804955f}, {0.267559f, 0.770868f},
      {0.270903f, 0.661836f}, {0.274247f, 0.676930f}, {0.277592f, 0.672822f},
      {0.280936f, 0.755818f}, {0.284281f, 0.729110f}, {0.287625f, 0.581494f},
      {0.290970f, 0.635167f}, {0.294314f, 0.601031f}, {0.297659f, 0.697939f},
      {0.301003f, 0.626274f}, {0.304348f, 0.652166f}, {0.307692f, 0.548234f},
      {0.311037f, 0.520831f}, {0.314381f, 0.534680f}, {0.317726f, 0.547408f},
      {0.321070f, 0.498806f}, {0.324415f, 0.484362f}, {0.327759f, 0.577739f},
      {0.331104f, 0.445626f}, {0.334448f, 0.519423f}, {0.337793f, 0.409678f},
      {0.341137f, 0.494096f}, {0.344482f, 0.461407f}, {0.347826f, 0.522341f},
      {0.351171f, 0.481788f}, {0.354515f, 0.400657f}, {0.357860f, 0.376125f},
      {0.361204f, 0.494867f}, {0.364549f, 0.425005f}, {0.367893f, 0.305921f},
      {0.371237f, 0.445804f}, {0.374582f, 0.256314f}, {0.377926f, 0.394650f},
      {0.381271f, 0.296793f}, {0.384615f, 0.298077f}, {0.387960f, 0.248025f},
      {0.391304f, 0.195721f}, {0.394649f, 0.281344f}, {0.397993f, 0.365160f},
      {0.401338f, 0.340950f}, {0.404682f, 0.239156f}, {0.408027f, 0.294310f},
      {0.411371f, 0.319684f}, {0.414716f, 0.135532f}, {0.418060f, 0.183437f},
      {0.421405f, 0.220398f}, {0.424749f, 0.252611f}, {0.428094f, 0.146102f},
      {0.431438f, 0.141155f}, {0.434783f, 0.231805f}, {0.438127f, 0.254009f},
      {0.441472f, 0.247750f}, {0.444816f, 0.167677f}, {0.448161f, 0.117479f},
      {0.451505f, 0.065547f}, {0.454850f, 0.147912f}, {0.458194f, 0.131502f},
      {0.461538f, 0.122535f}, {0.464883f, 0.148809f}, {0.468227f, 0.099429f},
      {0.471572f, 0.121764f}, {0.474916f, 0.091999f}, {0.478261f, 0.147794f},
      {0.481605f, 0.020690f}, {0.484950f, 0.015772f}, {0.488294f, 0.087996f},
      {0.491639f, 0.020168f}, {0.494983f, 0.016082f}, {0.498328f, 0.108505f},
      {0.501672f, 0.163213f}, {0.505017f, 0.081322f}, {0.508361f, 0.195403f},
      {0.511706f, 0.115663f}, {0.515050f, 0.073141f}, {0.518395f, 0.153647f},
      {0.521739f, 0.141071f}, {0.525084f, 0.080423f}, {0.528428f, 0.144699f},
      {0.531773f, 0.206703f}, {0.535117f, 0.024019f}, {0.538462f, 0.094846f},
      {0.541806f, 0.033749f}, {0.545151f, 0.054358f}, {0.548495f, 0.138388f},
      {0.551839f, 0.161507f}, {0.555184f, 0.225174f}, {0.558528f, 0.211697f},
      {0.561873f, 0.161673f}, {0.565217f, 0.239112f}, {0.568562f, 0.113356f},
      {0.571906f, 0.177033f}, {0.575251f, 0.165321f}, {0.578595f, 0.290500f},
      {0.581940f, 0.294612f}, {0.585284f, 0.245349f}, {0.588629f, 0.267393f},
      {0.591973f, 0.289006f}, {0.595318f, 0.268960f}, {0.598662f, 0.211359f},
      {0.602007f, 0.243607f}, {0.605351f, 0.298934f}, {0.608696f, 0.256708f},
      {0.612040f, 0.267113f}, {0.615385f, 0.343133f}, {0.618729f, 0.364352f},
      {0.622074f, 0.294886f}, {0.625418f, 0.368006f}, {0.628763f, 0.385086f},
      {0.632107f, 0.357009f}, {0.635451f, 0.331552f}, {0.638796f, 0.392551f},
      {0.642140f, 0.432688f}, {0.645485f, 0.360494f}, {0.648829f, 0.438897f},
      {0.652174f, 0.480043f}, {0.655518f, 0.423475f}, {0.658863f, 0.496612f},
      {0.662207f, 0.399808f}, {0.665552f, 0.405894f}, {0.668896f, 0.582420f},
      {0.672241f, 0.548798f}, {0.675585f, 0.505160f}, {0.678930f, 0.492096f},
      {0.682274f, 0.562948f}, {0.685619f, 0.652311f}, {0.688963f, 0.607402f},
      {0.692308f, 0.603482f}, {0.695652f, 0.554938f}, {0.698997f, 0.589308f},
      {0.702341f, 0.602328f}, {0.705686f, 0.546364f}, {0.709030f, 0.733326f},
      {0.712375f, 0.567667f}, {0.715719f, 0.708755f}, {0.719064f, 0.723838f},
      {0.722408f, 0.716062f}, {0.725753f, 0.701713f}, {0.729097f, 0.710149f},
      {0.732441f, 0.770769f}, {0.735786f, 0.779654f}, {0.739130f, 0.664935f},
      {0.742475f, 0.833037f}, {0.745819f, 0.794780f}, {0.749164f, 0.756719f},
      {0.752508f, 0.818435f}, {0.755853f, 0.760894f}, {0.759197f, 0.825715f},
      {0.762542f, 0.730633f}, {0.765886f, 0.883665f}, {0.769231f, 0.766019f},
      {0.772575f, 0.845006f}, {0.775920f, 0.796778f}, {0.779264f, 0.812594f},
      {0.782609f, 0.761958f}, {0.785953f, 0.796428f}, {0.789298f, 0.828672f},
      {0.792642f, 0.907888f}, {0.795987f, 0.954970f}, {0.799331f, 0.803165f},
      {0.802676f, 0.844324f}, {0.806020f, 0.956964f}, {0.809365f, 0.865136f},
      {0.812709f, 0.793003f}, {0.816054f, 0.911215f}, {0.819398f, 0.942304f},
      {0.822742f, 0.855249f}, {0.826087f, 0.799987f}, {0.829431f, 0.945763f},
      {0.832776f, 0.886276f}, {0.836120f, 0.891786f}, {0.839465f, 0.925075f},
      {0.842809f, 0.961550f}, {0.846154f, 0.929763f}, {0.849498f, 0.895539f},
      {0.852843f, 0.843798f}, {0.856187f, 0.969463f}, {0.859532f, 0.948759f},
      {0.862876f, 0.852580f}, {0.866221f, 0.854143f}, {0.869565f, 0.857024f},
      {0.872910f, 0.950194f}, {0.876254f, 0.875106f}, {0.879599f, 0.957805f},
      {0.882943f, 0.835710f}, {0.886288f, 0.757770f}, {0.889632f, 0.785635f},
      {0.892977f, 0.768809f}, {0.896321f, 0.733810f}, {0.899666f, 0.809716f},
      {0.903010f, 0.848462f}, {0.906354f, 0.835110f}, {0.909699f, 0.900466f},
      {0.913043f, 0.719813f}, {0.916388f, 0.841593f}, {0.919732f, 0.755458f},
      {0.923077f, 0.799053f}, {0.926421f, 0.704789f}, {0.929766f, 0.757776f},
      {0.933110f, 0.723116f}, {0.936455f, 0.715878f}, {0.939799f, 0.679550f},
      {0.943144f, 0.607245f}, {0.946488f, 0.623214f}, {0.949833f, 0.611848f},
      {0.953177f, 0.607007f}, {0.956522f, 0.684696f}, {0.959866f, 0.726103f},
      {0.963211f, 0.677118f}, {0.966555f, 0.650628f}, {0.969900f, 0.640134f},
      {0.973244f, 0.511668f}, {0.976589f, 0.492396f}, {0.979933f, 0.631343f},
      {0.983278f, 0.510640f}, {0.986622f, 0.498396f}, {0.989967f, 0.530343f},
      {0.993311f, 0.561419f}, {0.996656f, 0.512955f}, {1.000000f, 0.556280f}};
  float testing_set[testing_set_size][total_size] = {
      {0.000000f, 0.500000f}, {0.003344f, 0.512604f}, {0.006689f, 0.525200f},
      {0.010033f, 0.537767f}, {0.013378f, 0.550300f}, {0.016722f, 0.562780f},
      {0.020067f, 0.575201f}, {0.023411f, 0.587543f}, {0.026756f, 0.599802f},
      {0.030100f, 0.611958f}, {0.033445f, 0.624007f}, {0.036789f, 0.635929f},
      {0.040134f, 0.647719f}, {0.043478f, 0.659360f}, {0.046823f, 0.670845f},
      {0.050167f, 0.682157f}, {0.053512f, 0.693291f}, {0.056856f, 0.704231f},
      {0.060201f, 0.714970f}, {0.063545f, 0.725493f}, {0.066890f, 0.735795f},
      {0.070234f, 0.745859f}, {0.073579f, 0.755683f}, {0.076923f, 0.765249f},
      {0.080268f, 0.774554f}, {0.083612f, 0.783585f}, {0.086957f, 0.792336f},
      {0.090301f, 0.800794f}, {0.093645f, 0.808953f}, {0.096990f, 0.816808f},
      {0.100334f, 0.824345f}, {0.103679f, 0.831563f}, {0.107023f, 0.838449f},
      {0.110368f, 0.845001f}, {0.113712f, 0.851208f}, {0.117057f, 0.857068f},
      {0.120401f, 0.862572f}, {0.123746f, 0.867717f}, {0.127090f, 0.872495f},
      {0.130435f, 0.876905f}, {0.133779f, 0.880938f}, {0.137124f, 0.884595f},
      {0.140468f, 0.887868f}, {0.143813f, 0.890757f}, {0.147157f, 0.893257f},
      {0.150502f, 0.895367f}, {0.153846f, 0.897083f}, {0.157191f, 0.898406f},
      {0.160535f, 0.899332f}, {0.163880f, 0.899862f}, {0.167224f, 0.899994f},
      {0.170569f, 0.899729f}, {0.173913f, 0.899068f}, {0.177258f, 0.898009f},
      {0.180602f, 0.896555f}, {0.183946f, 0.894707f}, {0.187291f, 0.892467f},
      {0.190635f, 0.889838f}, {0.193980f, 0.886820f}, {0.197324f, 0.883419f},
      {0.200669f, 0.879636f}, {0.204013f, 0.875477f}, {0.207358f, 0.870943f},
      {0.210702f, 0.866043f}, {0.214047f, 0.860777f}, {0.217391f, 0.855155f},
      {0.220736f, 0.849178f}, {0.224080f, 0.842855f}, {0.227425f, 0.836191f},
      {0.230769f, 0.829194f}, {0.234114f, 0.821868f}, {0.237458f, 0.814225f},
      {0.240803f, 0.806267f}, {0.244147f, 0.798007f}, {0.247492f, 0.789449f},
      {0.250836f, 0.780605f}, {0.254181f, 0.771481f}, {0.257525f, 0.762089f},
      {0.260870f, 0.752434f}, {0.264214f, 0.742531f}, {0.267559f, 0.732384f},
      {0.270903f, 0.722010f}, {0.274247f, 0.711415f}, {0.277592f, 0.700606f},
      {0.280936f, 0.689602f}, {0.284281f, 0.678406f}, {0.287625f, 0.667036f},
      {0.290970f, 0.655497f}, {0.294314f, 0.643806f}, {0.297659f, 0.631969f},
      {0.301003f, 0.620005f}, {0.304348f, 0.607918f}, {0.307692f, 0.595727f},
      {0.311037f, 0.583438f}, {0.314381f, 0.571069f}, {0.317726f, 0.558626f},
      {0.321070f, 0.546129f}, {0.324415f, 0.533582f}, {0.327759f, 0.521005f},
      {0.331104f, 0.508404f}, {0.334448f, 0.495798f}, {0.337793f, 0.483192f},
      {0.341137f, 0.470607f}, {0.344482f, 0.458048f}, {0.347826f, 0.445534f},
      {0.351171f, 0.433070f}, {0.354515f, 0.420676f}, {0.357860f, 0.408358f},
      {0.361204f, 0.396134f}, {0.364549f, 0.384010f}, {0.367893f, 0.372005f},
      {0.371237f, 0.360126f}, {0.374582f, 0.348384f}, {0.377926f, 0.336795f},
      {0.381271f, 0.325365f}, {0.384615f, 0.314112f}, {0.387960f, 0.303040f},
      {0.391304f, 0.292168f}, {0.394649f, 0.281498f}, {0.397993f, 0.271049f},
      {0.401338f, 0.260824f}, {0.404682f, 0.250840f}, {0.408027f, 0.241100f},
      {0.411371f, 0.231620f}, {0.414716f, 0.222405f}, {0.418060f, 0.213468f},
      {0.421405f, 0.204812f}, {0.424749f, 0.196453f}, {0.428094f, 0.188393f},
      {0.431438f, 0.180644f}, {0.434783f, 0.173211f}, {0.438127f, 0.166105f},
      {0.441472f, 0.159328f}, {0.444816f, 0.152892f}, {0.448161f, 0.146798f},
      {0.451505f, 0.141058f}, {0.454850f, 0.135672f}, {0.458194f, 0.130650f},
      {0.461538f, 0.125994f}, {0.464883f, 0.121709f}, {0.468227f, 0.117801f},
      {0.471572f, 0.114271f}, {0.474916f, 0.111126f}, {0.478261f, 0.108366f},
      {0.481605f, 0.105996f}, {0.484950f, 0.104017f}, {0.488294f, 0.102432f},
      {0.491639f, 0.101241f}, {0.494983f, 0.100447f}, {0.498328f, 0.100050f},
      {0.501672f, 0.100050f}, {0.505017f, 0.100447f}, {0.508361f, 0.101241f},
      {0.511706f, 0.102432f}, {0.515050f, 0.104017f}, {0.518395f, 0.105996f},
      {0.521739f, 0.108366f}, {0.525084f, 0.111126f}, {0.528428f, 0.114271f},
      {0.531773f, 0.117801f}, {0.535117f, 0.121709f}, {0.538462f, 0.125994f},
      {0.541806f, 0.130649f}, {0.545151f, 0.135673f}, {0.548495f, 0.141058f},
      {0.551839f, 0.146798f}, {0.555184f, 0.152892f}, {0.558528f, 0.159328f},
      {0.561873f, 0.166105f}, {0.565217f, 0.173211f}, {0.568562f, 0.180644f},
      {0.571906f, 0.188393f}, {0.575251f, 0.196453f}, {0.578595f, 0.204812f},
      {0.581940f, 0.213468f}, {0.585284f, 0.222405f}, {0.588629f, 0.231620f},
      {0.591973f, 0.241100f}, {0.595318f, 0.250840f}, {0.598662f, 0.260824f},
      {0.602007f, 0.271049f}, {0.605351f, 0.281498f}, {0.608696f, 0.292167f},
      {0.612040f, 0.303040f}, {0.615385f, 0.314112f}, {0.618729f, 0.325365f},
      {0.622074f, 0.336795f}, {0.625418f, 0.348384f}, {0.628763f, 0.360127f},
      {0.632107f, 0.372005f}, {0.635451f, 0.384010f}, {0.638796f, 0.396134f},
      {0.642140f, 0.408358f}, {0.645485f, 0.420676f}, {0.648829f, 0.433070f},
      {0.652174f, 0.445534f}, {0.655518f, 0.458048f}, {0.658863f, 0.470607f},
      {0.662207f, 0.483192f}, {0.665552f, 0.495798f}, {0.668896f, 0.508404f},
      {0.672241f, 0.521005f}, {0.675585f, 0.533582f}, {0.678930f, 0.546129f},
      {0.682274f, 0.558626f}, {0.685619f, 0.571069f}, {0.688963f, 0.583438f},
      {0.692308f, 0.595727f}, {0.695652f, 0.607918f}, {0.698997f, 0.620005f},
      {0.702341f, 0.631970f}, {0.705686f, 0.643806f}, {0.709030f, 0.655496f},
      {0.712375f, 0.667036f}, {0.715719f, 0.678406f}, {0.719064f, 0.689602f},
      {0.722408f, 0.700606f}, {0.725753f, 0.711415f}, {0.729097f, 0.722010f},
      {0.732441f, 0.732384f}, {0.735786f, 0.742531f}, {0.739130f, 0.752434f},
      {0.742475f, 0.762089f}, {0.745819f, 0.771481f}, {0.749164f, 0.780605f},
      {0.752508f, 0.789449f}, {0.755853f, 0.798007f}, {0.759197f, 0.806267f},
      {0.762542f, 0.814225f}, {0.765886f, 0.821868f}, {0.769231f, 0.829194f},
      {0.772575f, 0.836191f}, {0.775920f, 0.842855f}, {0.779264f, 0.849177f},
      {0.782609f, 0.855155f}, {0.785953f, 0.860777f}, {0.789298f, 0.866043f},
      {0.792642f, 0.870943f}, {0.795987f, 0.875477f}, {0.799331f, 0.879636f},
      {0.802676f, 0.883419f}, {0.806020f, 0.886820f}, {0.809365f, 0.889838f},
      {0.812709f, 0.892467f}, {0.816054f, 0.894707f}, {0.819398f, 0.896555f},
      {0.822742f, 0.898009f}, {0.826087f, 0.899068f}, {0.829431f, 0.899729f},
      {0.832776f, 0.899994f}, {0.836120f, 0.899862f}, {0.839465f, 0.899332f},
      {0.842809f, 0.898406f}, {0.846154f, 0.897084f}, {0.849498f, 0.895367f},
      {0.852843f, 0.893257f}, {0.856187f, 0.890757f}, {0.859532f, 0.887868f},
      {0.862876f, 0.884595f}, {0.866221f, 0.880938f}, {0.869565f, 0.876905f},
      {0.872910f, 0.872495f}, {0.876254f, 0.867717f}, {0.879599f, 0.862572f},
      {0.882943f, 0.857068f}, {0.886288f, 0.851208f}, {0.889632f, 0.845001f},
      {0.892977f, 0.838449f}, {0.896321f, 0.831563f}, {0.899666f, 0.824345f},
      {0.903010f, 0.816808f}, {0.906354f, 0.808955f}, {0.909699f, 0.800794f},
      {0.913043f, 0.792336f}, {0.916388f, 0.783585f}, {0.919732f, 0.774555f},
      {0.923077f, 0.765249f}, {0.926421f, 0.755683f}, {0.929766f, 0.745859f},
      {0.933110f, 0.735795f}, {0.936455f, 0.725493f}, {0.939799f, 0.714970f},
      {0.943144f, 0.704231f}, {0.946488f, 0.693291f}, {0.949833f, 0.682157f},
      {0.953177f, 0.670845f}, {0.956522f, 0.659360f}, {0.959866f, 0.647719f},
      {0.963211f, 0.635929f}, {0.966555f, 0.624007f}, {0.969900f, 0.611958f},
      {0.973244f, 0.599802f}, {0.976589f, 0.587543f}, {0.979933f, 0.575201f},
      {0.983278f, 0.562780f}, {0.986622f, 0.550300f}, {0.989967f, 0.537767f},
      {0.993311f, 0.525200f}, {0.996656f, 0.512604f}, {1.000000f, 0.500000f}};
  float scalar = 1.0f / batch_size;
  TrainingConfig config = {training_set_size, batch_size, 500};
  mlp_activation_t *activate_fns[layer_size_count] = {&activation, &activation,
                                                      &activation, &activation};

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
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);

  // setup
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, 0.00390625f));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activate_fns));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_clear_msgd_buffer(msgd_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_set_momentum(msgd_buffer, 0.99f));
  TrainingPack pack = {
      input_size, output_size, total_size, scalar,   (float *)training_set,
      input,      output,      answer,     gradient,
  };
  TrainingEvents<TrainingPack> events = {on_batch_begin, on_integration,
                                         on_update_msgd};

  double init_loss = 0.0, tmp_loss = 0.0;
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
  for (auto const &sample : testing_set) {
    for (size_t i = 0; i < input_size; ++i) {
      input[i] = sample[i];
    }
    for (size_t i = 0; i < output_size; ++i) {
      answer[i] = sample[input_size + i];
    }
    TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                      mlp_forward(model, input, output, nullptr));
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-1, answer, output, output_size);
    tmp_loss = output[0] - answer[0];
    trained_loss += tmp_loss * tmp_loss;
  }
  trained_loss = sqrt(trained_loss / training_set_size);

  TEST_ASSERT_LESS_THAN_FLOAT(init_loss * 0.1, trained_loss);
}

void test_mlp_spiral_ont_hot() {
  constexpr size_t layer_size_count = 5U;
  constexpr size_t input_size = 2U, output_size = 2U;
  constexpr size_t training_set_size = 300U;
  constexpr size_t testing_set_size = 300U;
  constexpr size_t batch_size = 16U;
  constexpr size_t total_size = input_size + output_size;
  float input[input_size], output[output_size], answer[output_size],
      gradient[output_size];
  size_t sizes[layer_size_count] = {input_size, 16U, 16U, 16U, output_size};
  mlp_layer_size_data_t size_data = {layer_size_count, sizes};

  float training_set[training_set_size][total_size] = {
      {0.662859f, 0.733260f, 1.0f, 0.0f}, {0.252165f, 0.351757f, 0.0f, 1.0f},
      {0.740587f, 0.721063f, 1.0f, 0.0f}, {0.520639f, 0.727700f, 0.0f, 1.0f},
      {0.222278f, 0.190374f, 1.0f, 0.0f}, {0.398479f, 0.650950f, 0.0f, 1.0f},
      {0.183659f, 0.656900f, 1.0f, 0.0f}, {0.702522f, 0.215524f, 0.0f, 1.0f},
      {0.359457f, 0.850686f, 1.0f, 0.0f}, {0.624908f, 0.607867f, 0.0f, 1.0f},
      {0.270545f, 0.722016f, 1.0f, 0.0f}, {0.552062f, 0.648569f, 0.0f, 1.0f},
      {0.467357f, 0.079535f, 1.0f, 0.0f}, {0.785242f, 0.312420f, 0.0f, 1.0f},
      {0.391040f, 0.852779f, 1.0f, 0.0f}, {0.732668f, 0.261178f, 0.0f, 1.0f},
      {0.154546f, 0.334770f, 1.0f, 0.0f}, {0.545040f, 0.221013f, 0.0f, 1.0f},
      {0.220400f, 0.182799f, 1.0f, 0.0f}, {0.659305f, 0.917222f, 0.0f, 1.0f},
      {0.089823f, 0.307600f, 1.0f, 0.0f}, {0.481584f, 0.174756f, 0.0f, 1.0f},
      {0.643172f, 0.331840f, 1.0f, 0.0f}, {0.700835f, 0.868815f, 0.0f, 1.0f},
      {0.371639f, 0.478946f, 1.0f, 0.0f}, {0.547928f, 0.685491f, 0.0f, 1.0f},
      {0.092332f, 0.433327f, 1.0f, 0.0f}, {0.619433f, 0.213953f, 0.0f, 1.0f},
      {0.132703f, 0.430877f, 1.0f, 0.0f}, {0.806216f, 0.713920f, 0.0f, 1.0f},
      {0.646852f, 0.767350f, 1.0f, 0.0f}, {0.236541f, 0.523594f, 0.0f, 1.0f},
      {0.610472f, 0.273617f, 1.0f, 0.0f}, {0.231640f, 0.576948f, 0.0f, 1.0f},
      {0.657213f, 0.749725f, 1.0f, 0.0f}, {0.222626f, 0.552155f, 0.0f, 1.0f},
      {0.504132f, 0.316081f, 1.0f, 0.0f}, {0.667602f, 0.886834f, 0.0f, 1.0f},
      {0.141687f, 0.439658f, 1.0f, 0.0f}, {0.814277f, 0.387056f, 0.0f, 1.0f},
      {0.417424f, 0.032084f, 1.0f, 0.0f}, {0.857756f, 0.299183f, 0.0f, 1.0f},
      {0.326986f, 0.101738f, 1.0f, 0.0f}, {0.768842f, 0.800271f, 0.0f, 1.0f},
      {0.742990f, 0.524770f, 1.0f, 0.0f}, {0.585582f, 0.573569f, 0.0f, 1.0f},
      {0.450141f, 0.356918f, 1.0f, 0.0f}, {0.917859f, 0.430927f, 0.0f, 1.0f},
      {0.381980f, 0.557755f, 1.0f, 0.0f}, {0.518933f, 0.956240f, 0.0f, 1.0f},
      {0.078954f, 0.429273f, 1.0f, 0.0f}, {0.896135f, 0.632788f, 0.0f, 1.0f},
      {0.473227f, 0.329964f, 1.0f, 0.0f}, {0.581337f, 0.171116f, 0.0f, 1.0f},
      {0.716925f, 0.630274f, 1.0f, 0.0f}, {0.869972f, 0.654327f, 0.0f, 1.0f},
      {0.675825f, 0.351943f, 1.0f, 0.0f}, {0.748932f, 0.873203f, 0.0f, 1.0f},
      {0.711344f, 0.423119f, 1.0f, 0.0f}, {0.304400f, 0.274223f, 0.0f, 1.0f},
      {0.404230f, 0.482291f, 1.0f, 0.0f}, {0.700475f, 0.865864f, 0.0f, 1.0f},
      {0.454756f, 0.828865f, 1.0f, 0.0f}, {0.539113f, 0.729516f, 0.0f, 1.0f},
      {0.390594f, 0.838951f, 1.0f, 0.0f}, {0.443979f, 0.670084f, 0.0f, 1.0f},
      {0.143185f, 0.466089f, 1.0f, 0.0f}, {0.561847f, 0.922229f, 0.0f, 1.0f},
      {0.158568f, 0.691752f, 1.0f, 0.0f}, {0.321281f, 0.299212f, 0.0f, 1.0f},
      {0.226328f, 0.197386f, 1.0f, 0.0f}, {0.904730f, 0.375919f, 0.0f, 1.0f},
      {0.412070f, 0.074787f, 1.0f, 0.0f}, {0.713797f, 0.179527f, 0.0f, 1.0f},
      {0.426732f, 0.840263f, 1.0f, 0.0f}, {0.295109f, 0.257243f, 0.0f, 1.0f},
      {0.449453f, 0.302789f, 1.0f, 0.0f}, {0.260386f, 0.351207f, 0.0f, 1.0f},
      {0.397892f, 0.093692f, 1.0f, 0.0f}, {0.871904f, 0.660944f, 0.0f, 1.0f},
      {0.422915f, 0.579788f, 1.0f, 0.0f}, {0.487992f, 0.476926f, 0.0f, 1.0f},
      {0.128176f, 0.247209f, 1.0f, 0.0f}, {0.924865f, 0.498901f, 0.0f, 1.0f},
      {0.235749f, 0.751439f, 1.0f, 0.0f}, {0.239589f, 0.549559f, 0.0f, 1.0f},
      {0.093452f, 0.467975f, 1.0f, 0.0f}, {0.632323f, 0.462410f, 0.0f, 1.0f},
      {0.710811f, 0.463474f, 1.0f, 0.0f}, {0.585771f, 0.549404f, 0.0f, 1.0f},
      {0.238827f, 0.696200f, 1.0f, 0.0f}, {0.233438f, 0.427568f, 0.0f, 1.0f},
      {0.488188f, 0.065255f, 1.0f, 0.0f}, {0.840539f, 0.714733f, 0.0f, 1.0f},
      {0.118812f, 0.390002f, 1.0f, 0.0f}, {0.611412f, 0.142475f, 0.0f, 1.0f},
      {0.716248f, 0.508035f, 1.0f, 0.0f}, {0.315280f, 0.605385f, 0.0f, 1.0f},
      {0.370302f, 0.034163f, 1.0f, 0.0f}, {0.858548f, 0.474925f, 0.0f, 1.0f},
      {0.184995f, 0.237773f, 1.0f, 0.0f}, {0.621436f, 0.921760f, 0.0f, 1.0f},
      {0.733282f, 0.735623f, 1.0f, 0.0f}, {0.692533f, 0.240064f, 0.0f, 1.0f},
      {0.286095f, 0.108445f, 1.0f, 0.0f}, {0.250981f, 0.564341f, 0.0f, 1.0f},
      {0.395222f, 0.587951f, 1.0f, 0.0f}, {0.839293f, 0.669040f, 0.0f, 1.0f},
      {0.271498f, 0.157410f, 1.0f, 0.0f}, {0.475313f, 0.480815f, 0.0f, 1.0f},
      {0.162697f, 0.585160f, 1.0f, 0.0f}, {0.660786f, 0.912964f, 0.0f, 1.0f},
      {0.718475f, 0.569667f, 1.0f, 0.0f}, {0.236898f, 0.434801f, 0.0f, 1.0f},
      {0.721655f, 0.379933f, 1.0f, 0.0f}, {0.475614f, 0.202657f, 0.0f, 1.0f},
      {0.639657f, 0.296547f, 1.0f, 0.0f}, {0.285156f, 0.536313f, 0.0f, 1.0f},
      {0.312176f, 0.098925f, 1.0f, 0.0f}, {0.647011f, 0.862865f, 0.0f, 1.0f},
      {0.443559f, 0.042429f, 1.0f, 0.0f}, {0.574464f, 0.176175f, 0.0f, 1.0f},
      {0.394716f, 0.539477f, 1.0f, 0.0f}, {0.864423f, 0.493956f, 0.0f, 1.0f},
      {0.585608f, 0.766479f, 1.0f, 0.0f}, {0.739264f, 0.890317f, 0.0f, 1.0f},
      {0.538094f, 0.823368f, 1.0f, 0.0f}, {0.283715f, 0.544936f, 0.0f, 1.0f},
      {0.153366f, 0.569728f, 1.0f, 0.0f}, {0.815558f, 0.795662f, 0.0f, 1.0f},
      {0.458521f, 0.859318f, 1.0f, 0.0f}, {0.558102f, 0.442208f, 0.0f, 1.0f},
      {0.156146f, 0.419463f, 1.0f, 0.0f}, {0.295875f, 0.529643f, 0.0f, 1.0f},
      {0.557629f, 0.800668f, 1.0f, 0.0f}, {0.611052f, 0.877321f, 0.0f, 1.0f},
      {0.371136f, 0.779349f, 1.0f, 0.0f}, {0.249869f, 0.319681f, 0.0f, 1.0f},
      {0.113947f, 0.569013f, 1.0f, 0.0f}, {0.767957f, 0.308652f, 0.0f, 1.0f},
      {0.399689f, 0.021717f, 1.0f, 0.0f}, {0.554418f, 0.929353f, 0.0f, 1.0f},
      {0.457264f, 0.041698f, 1.0f, 0.0f}, {0.763228f, 0.246125f, 0.0f, 1.0f},
      {0.534072f, 0.803046f, 1.0f, 0.0f}, {0.628321f, 0.546258f, 0.0f, 1.0f},
      {0.630267f, 0.335798f, 1.0f, 0.0f}, {0.640052f, 0.177252f, 0.0f, 1.0f},
      {0.210896f, 0.648484f, 1.0f, 0.0f}, {0.576564f, 0.897252f, 0.0f, 1.0f},
      {0.513247f, 0.299123f, 1.0f, 0.0f}, {0.923807f, 0.496983f, 0.0f, 1.0f},
      {0.542751f, 0.833088f, 1.0f, 0.0f}, {0.273235f, 0.385332f, 0.0f, 1.0f},
      {0.455094f, 0.352921f, 1.0f, 0.0f}, {0.360899f, 0.710490f, 0.0f, 1.0f},
      {0.497367f, 0.817460f, 1.0f, 0.0f}, {0.674654f, 0.876456f, 0.0f, 1.0f},
      {0.664583f, 0.725449f, 1.0f, 0.0f}, {0.894565f, 0.535199f, 0.0f, 1.0f},
      {0.330628f, 0.780203f, 1.0f, 0.0f}, {0.515794f, 0.138684f, 0.0f, 1.0f},
      {0.126955f, 0.371536f, 1.0f, 0.0f}, {0.377962f, 0.689728f, 0.0f, 1.0f},
      {0.403328f, 0.332213f, 1.0f, 0.0f}, {0.905094f, 0.663569f, 0.0f, 1.0f},
      {0.788923f, 0.617781f, 1.0f, 0.0f}, {0.317064f, 0.600175f, 0.0f, 1.0f},
      {0.618965f, 0.319265f, 1.0f, 0.0f}, {0.625754f, 0.597484f, 0.0f, 1.0f},
      {0.484472f, 0.352697f, 1.0f, 0.0f}, {0.578130f, 0.951090f, 0.0f, 1.0f},
      {0.070519f, 0.453328f, 1.0f, 0.0f}, {0.785665f, 0.277289f, 0.0f, 1.0f},
      {0.369776f, 0.401895f, 1.0f, 0.0f}, {0.446745f, 0.144544f, 0.0f, 1.0f},
      {0.267233f, 0.722357f, 1.0f, 0.0f}, {0.869476f, 0.626457f, 0.0f, 1.0f},
      {0.151122f, 0.242383f, 1.0f, 0.0f}, {0.802782f, 0.762268f, 0.0f, 1.0f},
      {0.094929f, 0.583966f, 1.0f, 0.0f}, {0.307449f, 0.676244f, 0.0f, 1.0f},
      {0.435196f, 0.859134f, 1.0f, 0.0f}, {0.337686f, 0.251129f, 0.0f, 1.0f},
      {0.405520f, 0.829259f, 1.0f, 0.0f}, {0.552057f, 0.990617f, 0.0f, 1.0f},
      {0.223359f, 0.679965f, 1.0f, 0.0f}, {0.340303f, 0.261426f, 0.0f, 1.0f},
      {0.364806f, 0.467252f, 1.0f, 0.0f}, {0.607420f, 0.882229f, 0.0f, 1.0f},
      {0.784118f, 0.642423f, 1.0f, 0.0f}, {0.582178f, 0.660850f, 0.0f, 1.0f},
      {0.434299f, 0.028054f, 1.0f, 0.0f}, {0.665742f, 0.924279f, 0.0f, 1.0f},
      {0.569662f, 0.826482f, 1.0f, 0.0f}, {0.617583f, 0.467830f, 0.0f, 1.0f},
      {0.157388f, 0.426958f, 1.0f, 0.0f}, {0.744855f, 0.262202f, 0.0f, 1.0f},
      {0.182748f, 0.660805f, 1.0f, 0.0f}, {0.601159f, 0.405327f, 0.0f, 1.0f},
      {0.154509f, 0.363415f, 1.0f, 0.0f}, {0.628173f, 0.184727f, 0.0f, 1.0f},
      {0.233273f, 0.799322f, 1.0f, 0.0f}, {0.421045f, 0.704228f, 0.0f, 1.0f},
      {0.246185f, 0.794011f, 1.0f, 0.0f}, {0.205793f, 0.422754f, 0.0f, 1.0f},
      {0.315320f, 0.134976f, 1.0f, 0.0f}, {0.766952f, 0.322657f, 0.0f, 1.0f},
      {0.636332f, 0.747662f, 1.0f, 0.0f}, {0.646622f, 0.572164f, 0.0f, 1.0f},
      {0.729253f, 0.721947f, 1.0f, 0.0f}, {0.778951f, 0.261796f, 0.0f, 1.0f},
      {0.733983f, 0.663001f, 1.0f, 0.0f}, {0.234279f, 0.533844f, 0.0f, 1.0f},
      {0.410003f, 0.376204f, 1.0f, 0.0f}, {0.871286f, 0.519923f, 0.0f, 1.0f},
      {0.530607f, 0.840981f, 1.0f, 0.0f}, {0.765140f, 0.761835f, 0.0f, 1.0f},
      {0.796003f, 0.510647f, 1.0f, 0.0f}, {0.223162f, 0.433333f, 0.0f, 1.0f},
      {0.141832f, 0.364643f, 1.0f, 0.0f}, {0.620539f, 0.211410f, 0.0f, 1.0f},
      {0.674254f, 0.729143f, 1.0f, 0.0f}, {0.474725f, 0.963317f, 0.0f, 1.0f},
      {0.640950f, 0.783624f, 1.0f, 0.0f}, {0.518087f, 0.167024f, 0.0f, 1.0f},
      {0.785663f, 0.552306f, 1.0f, 0.0f}, {0.907305f, 0.595216f, 0.0f, 1.0f},
      {0.410941f, 0.521850f, 1.0f, 0.0f}, {0.840809f, 0.337568f, 0.0f, 1.0f},
      {0.459533f, 0.052011f, 1.0f, 0.0f}, {0.296706f, 0.582341f, 0.0f, 1.0f},
      {0.766268f, 0.509161f, 1.0f, 0.0f}, {0.823272f, 0.738388f, 0.0f, 1.0f},
      {0.765210f, 0.697427f, 1.0f, 0.0f}, {0.558638f, 0.662704f, 0.0f, 1.0f},
      {0.554657f, 0.341640f, 1.0f, 0.0f}, {0.584591f, 0.946775f, 0.0f, 1.0f},
      {0.662574f, 0.726439f, 1.0f, 0.0f}, {0.898719f, 0.707123f, 0.0f, 1.0f},
      {0.795285f, 0.476492f, 1.0f, 0.0f}, {0.226235f, 0.548774f, 0.0f, 1.0f},
      {0.215360f, 0.711666f, 1.0f, 0.0f}, {0.864592f, 0.685483f, 0.0f, 1.0f},
      {0.451894f, 0.841437f, 1.0f, 0.0f}, {0.268128f, 0.614051f, 0.0f, 1.0f},
      {0.147433f, 0.473598f, 1.0f, 0.0f}, {0.915559f, 0.486274f, 0.0f, 1.0f},
      {0.370590f, 0.534626f, 1.0f, 0.0f}, {0.517034f, 0.974532f, 0.0f, 1.0f},
      {0.195559f, 0.761844f, 1.0f, 0.0f}, {0.435512f, 0.687791f, 0.0f, 1.0f},
      {0.497513f, 0.075178f, 1.0f, 0.0f}, {0.696303f, 0.831235f, 0.0f, 1.0f},
      {0.456762f, 0.561965f, 1.0f, 0.0f}, {0.817042f, 0.378238f, 0.0f, 1.0f},
      {0.472340f, 0.552025f, 1.0f, 0.0f}, {0.362556f, 0.622447f, 0.0f, 1.0f},
      {0.380272f, 0.534954f, 1.0f, 0.0f}, {0.595864f, 0.413460f, 0.0f, 1.0f},
      {0.651092f, 0.737687f, 1.0f, 0.0f}, {0.656755f, 0.864338f, 0.0f, 1.0f},
      {0.700483f, 0.747687f, 1.0f, 0.0f}, {0.853304f, 0.593301f, 0.0f, 1.0f},
      {0.438035f, 0.530070f, 1.0f, 0.0f}, {0.528569f, 0.684845f, 0.0f, 1.0f},
      {0.691818f, 0.347394f, 1.0f, 0.0f}, {0.810501f, 0.300153f, 0.0f, 1.0f},
      {0.139974f, 0.274704f, 1.0f, 0.0f}, {0.471629f, 0.968680f, 0.0f, 1.0f},
      {0.266798f, 0.124768f, 1.0f, 0.0f}, {0.755457f, 0.258430f, 0.0f, 1.0f},
      {0.434552f, 0.085346f, 1.0f, 0.0f}, {0.428316f, 0.666298f, 0.0f, 1.0f},
      {0.106882f, 0.436457f, 1.0f, 0.0f}, {0.549003f, 0.719146f, 0.0f, 1.0f},
      {0.484059f, 0.062742f, 1.0f, 0.0f}, {0.292492f, 0.232361f, 0.0f, 1.0f},
      {0.612670f, 0.775602f, 1.0f, 0.0f}, {0.541796f, 0.406145f, 0.0f, 1.0f},
      {0.745231f, 0.425909f, 1.0f, 0.0f}, {0.877267f, 0.539282f, 0.0f, 1.0f},
      {0.281142f, 0.734088f, 1.0f, 0.0f}, {0.656065f, 0.544950f, 0.0f, 1.0f},
      {0.144149f, 0.347120f, 1.0f, 0.0f}, {0.333754f, 0.662058f, 0.0f, 1.0f},
      {0.504592f, 0.308679f, 1.0f, 0.0f}, {0.911074f, 0.623803f, 0.0f, 1.0f},
      {0.205132f, 0.772343f, 1.0f, 0.0f}, {0.291300f, 0.340211f, 0.0f, 1.0f},
      {0.098123f, 0.326419f, 1.0f, 0.0f}, {0.607430f, 0.969576f, 0.0f, 1.0f},
      {0.288609f, 0.121845f, 1.0f, 0.0f}, {0.233412f, 0.411710f, 0.0f, 1.0f},
      {0.433418f, 0.826125f, 1.0f, 0.0f}, {0.206642f, 0.410737f, 0.0f, 1.0f},
      {0.670049f, 0.738726f, 1.0f, 0.0f}, {0.749374f, 0.261694f, 0.0f, 1.0f},
      {0.627829f, 0.814680f, 1.0f, 0.0f}, {0.886490f, 0.587525f, 0.0f, 1.0f}};
  float testing_set[testing_set_size][total_size] = {
      {0.311844f, 0.808034f, 1.0f, 0.0f}, {0.214602f, 0.341859f, 0.0f, 1.0f},
      {0.275927f, 0.128688f, 1.0f, 0.0f}, {0.870131f, 0.604220f, 0.0f, 1.0f},
      {0.387260f, 0.797808f, 1.0f, 0.0f}, {0.583484f, 0.673533f, 0.0f, 1.0f},
      {0.409530f, 0.595795f, 1.0f, 0.0f}, {0.640231f, 0.181438f, 0.0f, 1.0f},
      {0.145630f, 0.274106f, 1.0f, 0.0f}, {0.577800f, 0.681346f, 0.0f, 1.0f},
      {0.472199f, 0.527684f, 1.0f, 0.0f}, {0.882923f, 0.625450f, 0.0f, 1.0f},
      {0.116747f, 0.510166f, 1.0f, 0.0f}, {0.574301f, 0.950940f, 0.0f, 1.0f},
      {0.076472f, 0.385359f, 1.0f, 0.0f}, {0.365524f, 0.631270f, 0.0f, 1.0f},
      {0.397056f, 0.565997f, 1.0f, 0.0f}, {0.610239f, 0.186149f, 0.0f, 1.0f},
      {0.406046f, 0.819885f, 1.0f, 0.0f}, {0.588860f, 0.574728f, 0.0f, 1.0f},
      {0.522289f, 0.302774f, 1.0f, 0.0f}, {0.823835f, 0.411534f, 0.0f, 1.0f},
      {0.153713f, 0.619109f, 1.0f, 0.0f}, {0.541813f, 0.640611f, 0.0f, 1.0f},
      {0.112355f, 0.430844f, 1.0f, 0.0f}, {0.364273f, 0.638023f, 0.0f, 1.0f},
      {0.354369f, 0.089528f, 1.0f, 0.0f}, {0.269181f, 0.515829f, 0.0f, 1.0f},
      {0.764746f, 0.583448f, 1.0f, 0.0f}, {0.692255f, 0.881491f, 0.0f, 1.0f},
      {0.429530f, 0.317579f, 1.0f, 0.0f}, {0.545503f, 0.986090f, 0.0f, 1.0f},
      {0.188632f, 0.222639f, 1.0f, 0.0f}, {0.848963f, 0.634464f, 0.0f, 1.0f},
      {0.442457f, 0.609049f, 1.0f, 0.0f}, {0.544751f, 0.654480f, 0.0f, 1.0f},
      {0.399346f, 0.449028f, 1.0f, 0.0f}, {0.663213f, 0.198756f, 0.0f, 1.0f},
      {0.620573f, 0.830076f, 1.0f, 0.0f}, {0.369051f, 0.676125f, 0.0f, 1.0f},
      {0.671684f, 0.694390f, 1.0f, 0.0f}, {0.251599f, 0.434836f, 0.0f, 1.0f},
      {0.486991f, 0.330591f, 1.0f, 0.0f}, {0.583682f, 0.938027f, 0.0f, 1.0f},
      {0.746720f, 0.505905f, 1.0f, 0.0f}, {0.277621f, 0.515304f, 0.0f, 1.0f},
      {0.582298f, 0.797666f, 1.0f, 0.0f}, {0.874313f, 0.686775f, 0.0f, 1.0f},
      {0.557185f, 0.772101f, 1.0f, 0.0f}, {0.585846f, 0.944253f, 0.0f, 1.0f},
      {0.737398f, 0.717716f, 1.0f, 0.0f}, {0.538812f, 0.157962f, 0.0f, 1.0f},
      {0.538050f, 0.784461f, 1.0f, 0.0f}, {0.639880f, 0.943558f, 0.0f, 1.0f},
      {0.351319f, 0.107888f, 1.0f, 0.0f}, {0.828618f, 0.686625f, 0.0f, 1.0f},
      {0.400573f, 0.376534f, 1.0f, 0.0f}, {0.416852f, 0.720508f, 0.0f, 1.0f},
      {0.575175f, 0.803946f, 1.0f, 0.0f}, {0.247069f, 0.600095f, 0.0f, 1.0f},
      {0.623193f, 0.342231f, 1.0f, 0.0f}, {0.362064f, 0.652647f, 0.0f, 1.0f},
      {0.356541f, 0.436689f, 1.0f, 0.0f}, {0.249814f, 0.412673f, 0.0f, 1.0f},
      {0.333253f, 0.804463f, 1.0f, 0.0f}, {0.607784f, 0.411169f, 0.0f, 1.0f},
      {0.672031f, 0.357649f, 1.0f, 0.0f}, {0.407119f, 0.227623f, 0.0f, 1.0f},
      {0.143895f, 0.303901f, 1.0f, 0.0f}, {0.855574f, 0.682238f, 0.0f, 1.0f},
      {0.135636f, 0.320951f, 1.0f, 0.0f}, {0.488175f, 0.714369f, 0.0f, 1.0f},
      {0.166574f, 0.720021f, 1.0f, 0.0f}, {0.496757f, 0.708339f, 0.0f, 1.0f},
      {0.714750f, 0.415603f, 1.0f, 0.0f}, {0.388683f, 0.733276f, 0.0f, 1.0f},
      {0.708676f, 0.357811f, 1.0f, 0.0f}, {0.745503f, 0.869384f, 0.0f, 1.0f},
      {0.475046f, 0.288155f, 1.0f, 0.0f}, {0.885609f, 0.575898f, 0.0f, 1.0f},
      {0.418526f, 0.056235f, 1.0f, 0.0f}, {0.756370f, 0.820535f, 0.0f, 1.0f},
      {0.400373f, 0.588827f, 1.0f, 0.0f}, {0.642860f, 0.532707f, 0.0f, 1.0f},
      {0.191624f, 0.760553f, 1.0f, 0.0f}, {0.600744f, 0.644364f, 0.0f, 1.0f},
      {0.465186f, 0.581183f, 1.0f, 0.0f}, {0.765063f, 0.876575f, 0.0f, 1.0f},
      {0.595197f, 0.272177f, 1.0f, 0.0f}, {0.548177f, 0.427865f, 0.0f, 1.0f},
      {0.236706f, 0.160202f, 1.0f, 0.0f}, {0.798655f, 0.288688f, 0.0f, 1.0f},
      {0.103410f, 0.346543f, 1.0f, 0.0f}, {0.326188f, 0.670309f, 0.0f, 1.0f},
      {0.312178f, 0.124704f, 1.0f, 0.0f}, {0.837991f, 0.443101f, 0.0f, 1.0f},
      {0.691899f, 0.678828f, 1.0f, 0.0f}, {0.400588f, 0.662164f, 0.0f, 1.0f},
      {0.166083f, 0.652322f, 1.0f, 0.0f}, {0.266262f, 0.441672f, 0.0f, 1.0f},
      {0.363360f, 0.365502f, 1.0f, 0.0f}, {0.866451f, 0.496291f, 0.0f, 1.0f},
      {0.700796f, 0.390942f, 1.0f, 0.0f}, {0.373858f, 0.246430f, 0.0f, 1.0f},
      {0.121345f, 0.254130f, 1.0f, 0.0f}, {0.806618f, 0.311691f, 0.0f, 1.0f},
      {0.357451f, 0.514671f, 1.0f, 0.0f}, {0.303992f, 0.323638f, 0.0f, 1.0f},
      {0.253585f, 0.770640f, 1.0f, 0.0f}, {0.816606f, 0.682624f, 0.0f, 1.0f},
      {0.414083f, 0.567021f, 1.0f, 0.0f}, {0.404410f, 0.212134f, 0.0f, 1.0f},
      {0.171220f, 0.220974f, 1.0f, 0.0f}, {0.675975f, 0.234132f, 0.0f, 1.0f},
      {0.459288f, 0.018606f, 1.0f, 0.0f}, {0.684100f, 0.212794f, 0.0f, 1.0f},
      {0.743650f, 0.563790f, 1.0f, 0.0f}, {0.458230f, 0.708728f, 0.0f, 1.0f},
      {0.423699f, 0.567142f, 1.0f, 0.0f}, {0.727700f, 0.874100f, 0.0f, 1.0f},
      {0.400151f, 0.521495f, 1.0f, 0.0f}, {0.723135f, 0.883975f, 0.0f, 1.0f},
      {0.735084f, 0.562817f, 1.0f, 0.0f}, {0.778506f, 0.797820f, 0.0f, 1.0f},
      {0.616244f, 0.316347f, 1.0f, 0.0f}, {0.641219f, 0.173020f, 0.0f, 1.0f},
      {0.286573f, 0.767523f, 1.0f, 0.0f}, {0.831867f, 0.746235f, 0.0f, 1.0f},
      {0.619233f, 0.316365f, 1.0f, 0.0f}, {0.378076f, 0.704566f, 0.0f, 1.0f},
      {0.747555f, 0.372804f, 1.0f, 0.0f}, {0.544570f, 0.188605f, 0.0f, 1.0f},
      {0.445495f, 0.869397f, 1.0f, 0.0f}, {0.841393f, 0.512996f, 0.0f, 1.0f},
      {0.213013f, 0.224822f, 1.0f, 0.0f}, {0.715231f, 0.239334f, 0.0f, 1.0f},
      {0.795974f, 0.589788f, 1.0f, 0.0f}, {0.612351f, 0.634598f, 0.0f, 1.0f},
      {0.500178f, 0.262453f, 1.0f, 0.0f}, {0.929517f, 0.507064f, 0.0f, 1.0f},
      {0.129530f, 0.235827f, 1.0f, 0.0f}, {0.719065f, 0.229261f, 0.0f, 1.0f},
      {0.114542f, 0.305993f, 1.0f, 0.0f}, {0.530683f, 0.449770f, 0.0f, 1.0f},
      {0.741720f, 0.506756f, 1.0f, 0.0f}, {0.434689f, 0.662780f, 0.0f, 1.0f},
      {0.290159f, 0.736847f, 1.0f, 0.0f}, {0.749555f, 0.868518f, 0.0f, 1.0f},
      {0.709859f, 0.687183f, 1.0f, 0.0f}, {0.241798f, 0.442840f, 0.0f, 1.0f},
      {0.469404f, 0.311276f, 1.0f, 0.0f}, {0.864426f, 0.540341f, 0.0f, 1.0f},
      {0.358900f, 0.857725f, 1.0f, 0.0f}, {0.464851f, 0.718273f, 0.0f, 1.0f},
      {0.430997f, 0.347897f, 1.0f, 0.0f}, {0.507208f, 0.737265f, 0.0f, 1.0f},
      {0.502203f, 0.340416f, 1.0f, 0.0f}, {0.525021f, 0.151507f, 0.0f, 1.0f},
      {0.109008f, 0.458028f, 1.0f, 0.0f}, {0.616428f, 0.460061f, 0.0f, 1.0f},
      {0.754191f, 0.378894f, 1.0f, 0.0f}, {0.875266f, 0.602557f, 0.0f, 1.0f},
      {0.364294f, 0.579142f, 1.0f, 0.0f}, {0.296360f, 0.291929f, 0.0f, 1.0f},
      {0.369662f, 0.767825f, 1.0f, 0.0f}, {0.267536f, 0.300657f, 0.0f, 1.0f},
      {0.165583f, 0.585332f, 1.0f, 0.0f}, {0.787718f, 0.859028f, 0.0f, 1.0f},
      {0.379933f, 0.479502f, 1.0f, 0.0f}, {0.276402f, 0.525728f, 0.0f, 1.0f},
      {0.122875f, 0.401386f, 1.0f, 0.0f}, {0.418739f, 0.228763f, 0.0f, 1.0f},
      {0.711119f, 0.638475f, 1.0f, 0.0f}, {0.820019f, 0.782888f, 0.0f, 1.0f},
      {0.140345f, 0.526377f, 1.0f, 0.0f}, {0.591665f, 0.624738f, 0.0f, 1.0f},
      {0.422320f, 0.350922f, 1.0f, 0.0f}, {0.351037f, 0.695736f, 0.0f, 1.0f},
      {0.449101f, 0.348607f, 1.0f, 0.0f}, {0.530555f, 0.642334f, 0.0f, 1.0f},
      {0.110262f, 0.390252f, 1.0f, 0.0f}, {0.696302f, 0.859560f, 0.0f, 1.0f},
      {0.307927f, 0.833507f, 1.0f, 0.0f}, {0.428307f, 0.702221f, 0.0f, 1.0f},
      {0.631019f, 0.317039f, 1.0f, 0.0f}, {0.818810f, 0.313901f, 0.0f, 1.0f},
      {0.744382f, 0.622802f, 1.0f, 0.0f}, {0.351986f, 0.659023f, 0.0f, 1.0f},
      {0.386139f, 0.103830f, 1.0f, 0.0f}, {0.540376f, 0.698031f, 0.0f, 1.0f},
      {0.776005f, 0.470505f, 1.0f, 0.0f}, {0.813877f, 0.750825f, 0.0f, 1.0f},
      {0.300413f, 0.778016f, 1.0f, 0.0f}, {0.307648f, 0.349746f, 0.0f, 1.0f},
      {0.480952f, 0.820220f, 1.0f, 0.0f}, {0.287157f, 0.647943f, 0.0f, 1.0f},
      {0.120243f, 0.592249f, 1.0f, 0.0f}, {0.494203f, 0.960308f, 0.0f, 1.0f},
      {0.356862f, 0.113406f, 1.0f, 0.0f}, {0.627124f, 0.497303f, 0.0f, 1.0f},
      {0.527477f, 0.556185f, 1.0f, 0.0f}, {0.604396f, 0.535431f, 0.0f, 1.0f},
      {0.444778f, 0.804063f, 1.0f, 0.0f}, {0.590908f, 0.157939f, 0.0f, 1.0f},
      {0.282636f, 0.153948f, 1.0f, 0.0f}, {0.297028f, 0.363283f, 0.0f, 1.0f},
      {0.113811f, 0.454572f, 1.0f, 0.0f}, {0.302590f, 0.644044f, 0.0f, 1.0f},
      {0.526025f, 0.271581f, 1.0f, 0.0f}, {0.558130f, 0.613774f, 0.0f, 1.0f},
      {0.466439f, 0.331627f, 1.0f, 0.0f}, {0.525936f, 0.167207f, 0.0f, 1.0f},
      {0.294789f, 0.135452f, 1.0f, 0.0f}, {0.309693f, 0.554265f, 0.0f, 1.0f},
      {0.399956f, 0.119561f, 1.0f, 0.0f}, {0.420906f, 0.228509f, 0.0f, 1.0f},
      {0.441914f, 0.812416f, 1.0f, 0.0f}, {0.277582f, 0.633584f, 0.0f, 1.0f},
      {0.419977f, 0.076401f, 1.0f, 0.0f}, {0.334748f, 0.262501f, 0.0f, 1.0f},
      {0.408336f, 0.084335f, 1.0f, 0.0f}, {0.489785f, 0.483984f, 0.0f, 1.0f},
      {0.154393f, 0.512641f, 1.0f, 0.0f}, {0.688497f, 0.209590f, 0.0f, 1.0f},
      {0.735221f, 0.499368f, 1.0f, 0.0f}, {0.785583f, 0.801132f, 0.0f, 1.0f},
      {0.787764f, 0.605190f, 1.0f, 0.0f}, {0.752003f, 0.223204f, 0.0f, 1.0f},
      {0.451013f, 0.786754f, 1.0f, 0.0f}, {0.445283f, 0.713737f, 0.0f, 1.0f},
      {0.572383f, 0.771882f, 1.0f, 0.0f}, {0.475160f, 0.408578f, 0.0f, 1.0f},
      {0.345923f, 0.504110f, 1.0f, 0.0f}, {0.348918f, 0.681188f, 0.0f, 1.0f},
      {0.319445f, 0.799742f, 1.0f, 0.0f}, {0.596352f, 0.163634f, 0.0f, 1.0f},
      {0.390334f, 0.497083f, 1.0f, 0.0f}, {0.250146f, 0.543235f, 0.0f, 1.0f},
      {0.463658f, 0.006332f, 1.0f, 0.0f}, {0.605494f, 0.444653f, 0.0f, 1.0f},
      {0.376565f, 0.806364f, 1.0f, 0.0f}, {0.620264f, 0.229487f, 0.0f, 1.0f},
      {0.473155f, 0.794103f, 1.0f, 0.0f}, {0.238140f, 0.354003f, 0.0f, 1.0f},
      {0.546061f, 0.315308f, 1.0f, 0.0f}, {0.294917f, 0.606062f, 0.0f, 1.0f},
      {0.344613f, 0.458231f, 1.0f, 0.0f}, {0.230259f, 0.363986f, 0.0f, 1.0f},
      {0.698360f, 0.708601f, 1.0f, 0.0f}, {0.817598f, 0.715989f, 0.0f, 1.0f},
      {0.686800f, 0.408406f, 1.0f, 0.0f}, {0.837854f, 0.429600f, 0.0f, 1.0f},
      {0.685558f, 0.376947f, 1.0f, 0.0f}, {0.549180f, 0.914163f, 0.0f, 1.0f},
      {0.763092f, 0.529450f, 1.0f, 0.0f}, {0.907356f, 0.592368f, 0.0f, 1.0f},
      {0.350429f, 0.840457f, 1.0f, 0.0f}, {0.444762f, 0.650467f, 0.0f, 1.0f},
      {0.485657f, 0.518768f, 1.0f, 0.0f}, {0.787221f, 0.825039f, 0.0f, 1.0f},
      {0.143054f, 0.312127f, 1.0f, 0.0f}, {0.785880f, 0.837629f, 0.0f, 1.0f},
      {0.095726f, 0.422414f, 1.0f, 0.0f}, {0.552656f, 0.666285f, 0.0f, 1.0f},
      {0.382568f, 0.415420f, 1.0f, 0.0f}, {0.660891f, 0.185066f, 0.0f, 1.0f},
      {0.333678f, 0.770142f, 1.0f, 0.0f}, {0.488590f, 0.513618f, 0.0f, 1.0f},
      {0.145825f, 0.541255f, 1.0f, 0.0f}, {0.377212f, 0.683580f, 0.0f, 1.0f},
      {0.760052f, 0.629161f, 1.0f, 0.0f}, {0.391379f, 0.685325f, 0.0f, 1.0f},
      {0.151528f, 0.248662f, 1.0f, 0.0f}, {0.496131f, 0.668285f, 0.0f, 1.0f},
      {0.095265f, 0.600431f, 1.0f, 0.0f}, {0.898171f, 0.667062f, 0.0f, 1.0f},
      {0.094238f, 0.438744f, 1.0f, 0.0f}, {0.555140f, 0.642814f, 0.0f, 1.0f},
      {0.391843f, 0.039970f, 1.0f, 0.0f}, {0.649376f, 0.196563f, 0.0f, 1.0f},
      {0.246933f, 0.134524f, 1.0f, 0.0f}, {0.868553f, 0.391892f, 0.0f, 1.0f},
      {0.720669f, 0.684389f, 1.0f, 0.0f}, {0.843369f, 0.754099f, 0.0f, 1.0f},
      {0.381302f, 0.065352f, 1.0f, 0.0f}, {0.841132f, 0.725429f, 0.0f, 1.0f},
      {0.472380f, 0.543139f, 1.0f, 0.0f}, {0.539777f, 0.215302f, 0.0f, 1.0f},
      {0.389394f, 0.522768f, 1.0f, 0.0f}, {0.219495f, 0.409347f, 0.0f, 1.0f},
      {0.581458f, 0.320136f, 1.0f, 0.0f}, {0.885158f, 0.522778f, 0.0f, 1.0f},
      {0.155345f, 0.691705f, 1.0f, 0.0f}, {0.517251f, 0.967893f, 0.0f, 1.0f},
      {0.631879f, 0.377590f, 1.0f, 0.0f}, {0.850485f, 0.778641f, 0.0f, 1.0f},
      {0.401646f, 0.421305f, 1.0f, 0.0f}, {0.468181f, 0.715652f, 0.0f, 1.0f},
      {0.399208f, 0.052545f, 1.0f, 0.0f}, {0.261972f, 0.487710f, 0.0f, 1.0f}};
  float scalar = 1.0f / batch_size;
  TrainingConfig config = {training_set_size, batch_size, 500};
  mlp_activation_t *activate_fns[layer_size_count] = {
      &activation, &activation, &activation, &activation, &activation};

  // allocate
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);

  // setup
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, 0.00390625f));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activate_fns));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_clear_msgd_buffer(msgd_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_set_momentum(msgd_buffer, 0.99f));
  TrainingPack pack = {
      input_size, output_size, total_size, scalar,   (float *)training_set,
      input,      output,      answer,     gradient,
  };
  TrainingEvents<TrainingPack> events = {on_batch_begin, on_integration,
                                         on_update_msgd};

  double init_loss = 0.0, tmp_loss = 0.0;
  for (auto const &sample : testing_set) {
    for (size_t j = 0; j < input_size; ++j) {
      input[j] = sample[j];
    }
    for (size_t j = 0; j < output_size; ++j) {
      answer[j] = sample[input_size + j];
    }
    TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                      mlp_forward(model, input, output, nullptr));
    for (size_t i = 0; i < output_size; ++i) {
      tmp_loss = output[i] - answer[i];
      init_loss += tmp_loss * tmp_loss;
    }
  }
  init_loss = sqrt(init_loss / (output_size * testing_set_size));

  // act
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, train(config, events, pack));

  // assert
  double trained_loss = 0.0;
  size_t correct_count = 0;
  for (auto const &sample : testing_set) {
    for (size_t i = 0; i < input_size; ++i) {
      input[i] = sample[i];
    }
    for (size_t i = 0; i < output_size; ++i) {
      answer[i] = sample[input_size + i];
    }
    TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                      mlp_forward(model, input, output, nullptr));
    for (size_t i = 0; i < output_size; ++i) {
      tmp_loss = output[i] - answer[i];
      trained_loss += tmp_loss * tmp_loss;
    }
    bool is_equal = output[0] == output[1];
    if (is_equal) { continue; }
    bool is_first_larger = output[0] > output[1];
    correct_count += (sample[2] == 1.0f) ? (is_first_larger ? 1 : 0)
                                         : (is_first_larger ? 0 : 1);
  }
  trained_loss = sqrt(trained_loss / (output_size * testing_set_size));
  TEST_ASSERT_GREATER_THAN_FLOAT(95.0f,
                                 correct_count * 100.0f / testing_set_size);

  TEST_ASSERT_LESS_THAN_FLOAT(init_loss * 0.4, trained_loss);
}

void test_mlp_auto_encoder() {
  constexpr size_t training_set_size = 8U;
  constexpr size_t batch_size = 1U;
  constexpr size_t layer_size_count = 3U;
  constexpr size_t io_size = 8U, middle_size = 2U;
  float input[io_size], middle[middle_size], output[io_size], answer[io_size],
      gradient[io_size], middle_gradient[middle_size];
  size_t sizes[layer_size_count] = {io_size, 4U, middle_size};
  size_t sizes_2[layer_size_count] = {middle_size, 4U, io_size};
  mlp_layer_size_data_t size_data = {layer_size_count, sizes};
  mlp_layer_size_data_t size_data_2 = {layer_size_count, sizes_2};

  float training_set[training_set_size][io_size] = {
      {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f},
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
  };
  float scalar = 1.0f / batch_size;
  TrainingConfig config = {training_set_size, batch_size, 10000};
  mlp_activation_t *activation_ptr = &activation;
  mlp_activation_t *activate_fns[layer_size_count] = {
      activation_ptr, activation_ptr, activation_ptr};

  // allocate
  model = mlp_create_model(&size_data, &status);
  TEST_ASSERT_NOT_NULL(model);
  backward_buffer = mlp_create_backward_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer);
  gradient_buffer = mlp_create_gradient_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer);
  msgd_buffer = mlp_create_msgd_buffer(&size_data, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer);
  model_2 = mlp_create_model(&size_data_2, &status);
  TEST_ASSERT_NOT_NULL(model_2);
  backward_buffer_2 = mlp_create_backward_buffer(&size_data_2, &status);
  TEST_ASSERT_NOT_NULL(backward_buffer_2);
  gradient_buffer_2 = mlp_create_gradient_buffer(&size_data_2, &status);
  TEST_ASSERT_NOT_NULL(gradient_buffer_2);
  msgd_buffer_2 = mlp_create_msgd_buffer(&size_data_2, &status);
  TEST_ASSERT_NOT_NULL(msgd_buffer_2);

  // setup
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model, 0.00390625f));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model, activate_fns));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_clear_msgd_buffer(msgd_buffer));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_set_momentum(msgd_buffer, 0.98f));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_learning_rate(model_2, 0.00390625f));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_weights(model_2, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_fill_biases(model_2, mlp_rand));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                    mlp_set_activation_fns_ref(model_2, activate_fns));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_clear_msgd_buffer(msgd_buffer_2));
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, mlp_set_momentum(msgd_buffer_2, 0.98f));

  TrainingPackEncoderDecoder pack = {io_size, scalar,   (float *)training_set,
                                     input,   middle,   output,
                                     answer,  gradient, middle_gradient};
  TrainingEvents<TrainingPackEncoderDecoder> events = {
      on_batch_begin_auto_encoder, on_integration_auto_encoder,
      on_update_msgd_auto_encoder};
  double init_loss = 0.0, tmp_loss = 0.0;
  for (auto &sample : training_set) {
    for (size_t i = 0; i < io_size; ++i) {
      input[i] = sample[i];
    }
    for (size_t i = 0; i < io_size; ++i) {
      answer[i] = sample[i];
    }
    TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                      mlp_forward(model, input, middle, nullptr));
    TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                      mlp_forward(model_2, middle, output, nullptr));
    for (size_t i = 0; i < io_size; ++i) {
      tmp_loss = output[i] - answer[i];
      init_loss += tmp_loss * tmp_loss;
    }
  }
  init_loss = sqrt(init_loss / (io_size * training_set_size));

  // act
  TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS, train(config, events, pack));

  // assert
  double trained_loss = 0.0;
  for (auto &sample : training_set) {
    for (size_t i = 0; i < io_size; ++i) {
      input[i] = sample[i];
    }
    for (size_t i = 0; i < io_size; ++i) {
      answer[i] = sample[i];
    }
    TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                      mlp_forward(model, input, middle, nullptr));
    TEST_ASSERT_EQUAL(MLP_STATUS_SUCCESS,
                      mlp_forward(model_2, middle, output, nullptr));
    for (size_t i = 0; i < io_size; ++i) {
      tmp_loss = output[i] - answer[i];
      trained_loss += tmp_loss * tmp_loss;
    }
    TEST_ASSERT_FLOAT_ARRAY_WITHIN(1e-1, answer, output, io_size);
  }
  trained_loss = sqrt(trained_loss / (io_size * training_set_size));
  TEST_ASSERT_LESS_THAN_FLOAT(init_loss * 0.1, trained_loss);
}

int main() {
  UNITY_BEGIN();

  RUN_TEST(test_mlp_xor);
  RUN_TEST(test_mlp_noisy_regression);
  RUN_TEST(test_mlp_spiral_ont_hot);
  RUN_TEST(test_mlp_auto_encoder);

  return UNITY_END();
}
