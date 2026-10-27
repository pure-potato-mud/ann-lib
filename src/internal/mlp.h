#ifndef INTERNAL_MLP_H_
#define INTERNAL_MLP_H_

#include <stddef.h>

#include <ann-lib/mlp.h>

struct mlp_layer_metadata {
  size_t layer_size_count;
  const size_t *layer_sizes_ref;
};

struct mlp_layer_data {
  float **element_ptrs;
  float *elements;
};

struct mlp_model {
  struct mlp_layer_metadata layer_metadata;
  struct mlp_layer_data layer_data;
  float learning_rate;
  float **buffers;                       // compute buffer(2)
  float **weight_layers;                 // weights(`N-1`)
  float **bias_layers;                   // bias (`N-1`)
  float (**activation_funcs_ref)(float); // activation functions(`N-1`)
};

struct mlp_backward_buffer {
  struct mlp_layer_metadata layer_metadata;
  struct mlp_layer_data layer_data;
  float **layers;            // layers(N)
  float **raw_output_layers; // raw output(N-1)
};

struct mlp_gradient_buffer {
  struct mlp_layer_metadata layer_metadata;
  struct mlp_layer_data layer_data;
  float **gradient_layers; // gradients(N-1)
};

#endif /* INTERNAL_MLP_H_ */