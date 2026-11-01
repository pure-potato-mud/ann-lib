#ifndef ANN_LIB_MLP_H_
#define ANN_LIB_MLP_H_

#include <stddef.h>

typedef struct mlp_layer_size_data mlp_layer_size_data_t;
typedef struct mlp_model mlp_model_t;
typedef struct mlp_backward_buffer mlp_backward_buffer_t;
typedef struct mlp_gradient_buffer mlp_gradient_buffer_t;
typedef enum mlp_status mlp_status_t;

struct mlp_layer_size_data {
  const size_t *layer_sizes_ref;
  size_t layer_size_count;
};

enum mlp_status {
  MLP_STATUS_SUCCESS = 0,
  MLP_STATUS_INVALID_ARGUMENT,
  MLP_STATUS_INT_OVERFLOW,
  MLP_STATUS_ALLOCATION_FAILED,
  MLP_STATUS_TOPOLOGY_MISMATCH,
  MLP_STATUS_INVALID_FIELD,
  MLP_STATUS_INVALID_LAYER,
  MLP_STATUS_INVALID_STATE,
  MLP_STATUS_INVALID_HARDCODE,
  MLP_STATUS_UNKNOWN_ERROR,
};

// create a mlp.
// layer_sizes_ref will keep and store in mlp. DO NOT free nor edit
// layer_sizes_ref before mlp been free.
// layer_sizes_ref cannot be NULL and have no zero in it.
// also, layer_size_count cannot be zero.
mlp_model_t *mlp_create_model(const mlp_layer_size_data_t *size_data,
                              mlp_status_t *status);

// free mlp_model safely.
// NOTE: not include `*_ref`.
void mlp_destroy_model(mlp_model_t *model);

// create a buffer for learning by reading the data of the mlp_model.
// NOTE: the buffer will store same layer_sizes_ref as model stores, so DO NOT
// free nor edit layer_sizes_ref before backward_buffer be freed.
mlp_backward_buffer_t *
mlp_create_backward_buffer(const mlp_layer_size_data_t *size_data,
                           mlp_status_t *status);

// free mlp_backward_buffer safely.
void mlp_destroy_backward_buffer(mlp_backward_buffer_t *buffer);

// create a buffer for store gradient compute result.
// NOTE: the buffer will store same layer_sizes_ref as model stores, so DO NOT
// free nor edit layer_sizes_ref before gradient_buffer be freed.
mlp_gradient_buffer_t *
mlp_create_gradient_buffer(const mlp_layer_size_data_t *size_data,
                           mlp_status_t *status);

// free mlp_gradient_buffer safely.
void mlp_destroy_gradient_buffer(mlp_gradient_buffer_t *buffer);

// set learning rate.
// the MLP cannot be NULL.
mlp_status_t mlp_set_learning_rate(mlp_model_t *model, float learning_rate);

// set activation functions reference.
// size must same as `layer_count - 1` of the MLP.
// the MLP and the activation functions reference cannot be NULL.
// activation_fns_ref will keep and store in mlp. DO NOT free nor edit
// activation_fns_ref before mlp been free.
mlp_status_t mlp_set_activation_fns_ref(mlp_model_t *model,
                                        float (**activation_fns_ref)(float));

// fill all bias with a value.
// MLP and it's biases cannot be NULL.
mlp_status_t mlp_fill_biases(mlp_model_t *model, float value);

// fill all weights with a value.
// MLP and it's weights field cannot be NULL.
mlp_status_t mlp_fill_weights(mlp_model_t *model, float value);

// do forward propagation.
// the MLP(and it's members) and the input cannot be NULL, but the output can.
// the output will be fill in values if it is not NULL.
// function stores data for learning if backward_buffer is not NULL.
mlp_status_t mlp_forward(mlp_model_t *model,
                         mlp_backward_buffer_t *backward_buffer,
                         float const *input, float *output);

// do backpropagation.
// the MLP(and it's members), backward_buffer and answer cannot be NULL.
mlp_status_t mlp_backward(mlp_model_t *model,
                          mlp_backward_buffer_t *backward_buffer,
                          mlp_gradient_buffer_t *gradient_output,
                          float const *answer);

#endif /* ann_lib_mlp_h_ */