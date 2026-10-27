#ifndef ANN_LIB_MLP_H_
#define ANN_LIB_MLP_H_

#include <stddef.h>

struct mlp_model;
struct mlp_backward_buffer;
struct mlp_gradient_buffer;

// create a mlp.
// layer_sizes_ref will keep and store in mlp. DO NOT free nor edit
// layer_sizes_ref before mlp been free.
// layer_sizes_ref cannot be NULL and have no zero in it.
// also, layer_size_count cannot be zero.
struct mlp_model *mlp_create_model(const size_t *layer_sizes_ref,
                                   size_t layer_size_count);

// free mlp_model safely.
// NOTE: not include `*_ref`.
void mlp_destroy_model(struct mlp_model *model);

// create a buffer for learning by reading the data of the mlp_model.
// NOTE: the buffer will store same layer_sizes_ref as model stores, so DO NOT
// free nor edit layer_sizes_ref before backward_buffer be freed.
struct mlp_backward_buffer *mlp_create_backward_buffer(struct mlp_model *model);

// free mlp_backward_buffer safely.
void mlp_destroy_backward_buffer(struct mlp_backward_buffer *buffer);

// create a buffer for store gradient compute result.
// NOTE: the buffer will store same layer_sizes_ref as model stores, so DO NOT
// free nor edit layer_sizes_ref before gradient_buffer be freed.
struct mlp_gradient_buffer *mlp_create_gradient_buffer(struct mlp_model *model);

// free mlp_gradient_buffer safely.
void mlp_destroy_gradient_buffer(struct mlp_gradient_buffer *buffer);

// set learning rate.
// the MLP cannot be NULL.
int mlp_set_learning_rate(struct mlp_model *model, float learning_rate);

// set activation functions reference.
// size must same as `layer_count - 1` of the MLP.
// the MLP and the activation functions reference cannot be NULL.
// activation_fns_ref will keep and store in mlp. DO NOT free nor edit
// activation_fns_ref before mlp been free.
int mlp_set_activation_fns_ref(struct mlp_model *model,
                               float (**activation_fns_ref)(float));

// fill all bias with a value.
// MLP and it's biases cannot be NULL.
int mlp_fill_biases(struct mlp_model *model, float value);

// fill all weights with a value.
// MLP and it's weights field cannot be NULL.
int mlp_fill_weights(struct mlp_model *model, float value);

// do forward propagation.
// the MLP(and it's members) and the input cannot be NULL, but the output can.
// the output will be fill in values if it is not NULL.
// function stores data for learning if backward_buffer is not NULL.
int mlp_forward(struct mlp_model *model,
                struct mlp_backward_buffer *backward_buffer, float const *input,
                float *output);

// do backpropagation.
// the MLP(and it's members), backward_buffer and answer cannot be NULL.
int mlp_backward(struct mlp_model *model,
                 struct mlp_backward_buffer *backward_buffer,
                 struct mlp_gradient_buffer *gradient_output,
                 float const *answer);

#endif /* ann_lib_mlp_h_ */