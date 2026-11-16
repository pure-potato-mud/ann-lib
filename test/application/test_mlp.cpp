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

static mlp_status_t on_update_msgd(TrainingPack &pack) {
  status = mlp_scale_gradients(gradient_buffer, pack.scalar);
  if (status != MLP_STATUS_SUCCESS) { return status; }
  status = mlp_update_parameters_msgd(model, msgd_buffer, gradient_buffer);
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
  TrainingEvents events = {on_batch_begin, on_integration, on_update_msgd};

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

int main() {
  UNITY_BEGIN();

  RUN_TEST(test_mlp_xor);
  RUN_TEST(test_mlp_noisy_regression);

  return UNITY_END();
}
