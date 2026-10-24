#ifndef MLP_INTERNAL_H_
#define MLP_INTERNAL_H_

#include <stddef.h>

// NOTE: layers and its first pointer also need to free.
struct mlp {
  float learning_rate;
  size_t layer_size_count;               // N
  const size_t *layer_sizes_ref;         // n1, n2...
  float **layers;                        // input & activated ouput (N)
  float **output_layers;                 // output (N-1)
  float **backward_layers;               // passes gradients (N-1)
  float **weights;                       // weights(N-1)
  float **biases;                        // bias (N-1)
  float (**activation_funcs_ref)(float); // activation functions(N-1)
};

#endif /* MLP_INTERNAL_H_ */