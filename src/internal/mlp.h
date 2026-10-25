#ifndef MLP_INTERNAL_H_
#define MLP_INTERNAL_H_

#include <stddef.h>

// NOTE: layers and its first pointer also need to free.
struct mlp_model {
  float learning_rate;
  size_t layer_size_count;               // N
  const size_t *layer_sizes_ref;         // n1, n2...
  float **buffers;                       // compute buffer(2)
  float **weight_layers;                 // weights(`N-1`)
  float **bias_layers;                   // bias (`N-1`)
  float (**activation_funcs_ref)(float); // activation functions(`N-1`)
};

struct mlp_learning_buffer {
  float **layers;            // layers(N)
  float **raw_output_layers; // raw output(N-1)
};

struct mlp_gradient_buffer {
  float **gradient_layers; // gradients(N-1)
};

#endif /* MLP_INTERNAL_H_ */