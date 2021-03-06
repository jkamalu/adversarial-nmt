import math

import torch


def transpose_for_scores(x, num_attention_heads, attention_head_size):
    '''
    Helper function to transpose the shape of an attention tensor.
    '''
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)

def calculate_attention_scores(q_layer, k_layer, num_attention_heads,
                               attention_head_size, mask=None):
    '''
    Given a set of query, andf matrices, calculates the output
    attention scores.
    '''

    query_layer = transpose_for_scores(q_layer, num_attention_heads,
                                                attention_head_size)
    key_layer = transpose_for_scores(k_layer, num_attention_heads,
                                                attention_head_size)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(attention_head_size)

    # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
    if mask is not None:
        mask = mask.to(dtype=torch.float)
        attention_scores = attention_scores + mask

    # Normalize the attention scores to probabilities.
    attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
    return attention_probs

def extract_attention_scores(hooks):
    ''' From a given set of hooks on BERT layers extract the attention scores'''
    self_attention_layer = 0
    num_attention_heads = 1
    attention_head_size = 768

    curr_attention_dict = {}
    for i, hook in enumerate(hooks):
        try:
            if hook.name == 'BertSelfAttention':
                q_linear = hooks[i+1].output
                k_linear = hooks[i+2].output
                attention_scores = calculate_attention_scores(q_linear,
                                                              k_linear,
                                                              num_attention_heads,
                                                              attention_head_size,
                                                              )

                curr_attention_dict[self_attention_layer] = attention_scores
                self_attention_layer += 1
        except AttributeError as e:
            continue

    return curr_attention_dict
