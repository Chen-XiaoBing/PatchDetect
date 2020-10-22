from functools import reduce
import math

def count_elt(inputs, outputs, params):
    return reduce(lambda x, y: x*y, outputs[0])

def count_bn(inputs, outputs, params):
    return 2*reduce(lambda x, y: x*y, outputs[0])

def count_conv(inputs, outputs, params):
    assert len(inputs[0]) == 4
    out_num = reduce(lambda x, y: x*y, outputs[0])
    k = params['kernel_shape']
    kernel = inputs[0][1] * k[0] * k[1]
    flops = 2 * out_num * kernel
    if len(inputs) == 3:
        flops += out_num
    return flops

def count_zeros(inputs, outputs, params):
    return 0

def count_linear(inputs, outputs, params):
    assert len(inputs[0]) == 2
    out_num = reduce(lambda x, y: x*y, outputs[0])
    flops = 2 * out_num * inputs[0][1]
    if len(inputs) == 3:
        flops += out_num
    return flops

def count_global_pool(inputs, outputs, params):
    assert len(inputs[0]) == 4
    in_num = reduce(lambda x, y: x*y, inputs[0])
    return in_num

def count_pool(inputs, outputs, params):
    k = params['kernel_shape']
    out_num = reduce(lambda x, y: x*y, outputs[0])
    return out_num*k[0]*k[1]

def count_relu(inputs, outputs, params):
    return reduce(lambda x, y: x*y, outputs[0])

def count_reduce_mean(inputs, outputs, params):
    return reduce(lambda x, y: x*y, outputs[0])

register_ops_count = {
    'Add'                : count_elt,
    'BatchNormalization' : count_bn,
    'Conv'               : count_conv,
    'Flatten'            : count_zeros,
    'Gemm'               : count_linear,
    'GlobalAveragePool'  : count_global_pool,
    'MaxPool'            : count_pool,
    'AveragePool'        : count_pool,
    'Relu'               : count_relu,
    'Concat'             : count_zeros,
    'Pad'                : count_zeros,
    'ReduceMean'         : count_reduce_mean,
}

def infer_out_elt(inputs, outputs, params, in_patch_size):
    return in_patch_size

def infer_out_bn(inputs, outputs, params, in_patch_size):
    return in_patch_size

def infer_out_conv(inputs, outputs, params, in_patch_size):
    k = params['kernel_shape']
    s = params['strides']
    out = int(in_patch_size + k[0] - 2) / s[0] + 1
    return min(out, outputs[0][2])

def infer_out_linear(inputs, outputs, params, in_patch_size):
    return outputs[0][1]

def infer_out_global_pool(inputs, outputs, params, in_patch_size):
    return outputs[0][1]

def infer_out_pool(inputs, outputs, params, in_patch_size):
    k = params['kernel_shape']
    s = params['strides']
    out = int(in_patch_size + k[0] - 2) / s[0] + 1
    return min(out, outputs[0][2])

def infer_out_relu(inputs, outputs, params, in_patch_size):
    return in_patch_size

def infer_out_reduce_mean(inputs, outputs, params, in_patch_size):
    return outputs[0][1]

def infer_out_flatten(inputs, outputs, params, in_patch_size):
    return outputs[0][1]

def infer_out_concat(inputs, outputs, params, in_patch_size):
    return in_patch_size

def infer_out_pad(inputs, outputs, params, in_patch_size):
    return in_patch_size

register_ops_output = {
    'Add'                : infer_out_elt,
    'BatchNormalization' : infer_out_bn,
    'Conv'               : infer_out_conv,
    'Flatten'            : infer_out_flatten,
    'Gemm'               : infer_out_linear,
    'GlobalAveragePool'  : infer_out_global_pool,
    'MaxPool'            : infer_out_pool,
    'AveragePool'        : infer_out_pool,
    'Relu'               : infer_out_relu,
    'Concat'             : infer_out_concat,
    'Pad'                : infer_out_pad,
    'ReduceMean'         : infer_out_reduce_mean,
}

def calc_total_flops(graph):
    total_flops = 0
    # print(graph)
    for layer in graph.layers:
        _type = layer['type']
        inputs  = [graph.tensor_shape[i] for i in layer['in']]
        outputs = [graph.tensor_shape[i] for i in layer['out']]
        params  = layer['params']
        total_flops += register_ops_count[_type](inputs, outputs, params)
    return total_flops

def calc_overlap_ratio(graph, patch_ratio):
    graph_in_size = graph.tensor_shape[0][2]
    patch_size = math.sqrt(graph_in_size * graph_in_size * patch_ratio) + 2
    total_flops = 0
    reused_flops = 0
    patch_size_map = {0:patch_size}

    # f = "{:>20}, {:>20}, {:>20} {:>20} {:>20} {:>20}"
    # print(f.format('type', 'total', 'patch_related', 'reuse ratio', 'out', 'patch'))
    for layer in graph.layers:
        _type = layer['type']
        inputs  = [graph.tensor_shape[i] for i in layer['in']]
        outputs = [graph.tensor_shape[i] for i in layer['out']]
        params  = layer['params']
        ttotal_flops = register_ops_count[_type](inputs, outputs, params)
        total_flops += ttotal_flops

        assert len(layer['out']) == 1
        patch_size_map[layer['out'][0]] = register_ops_output[_type](inputs, outputs, params, patch_size_map[layer['in'][0]])
        out = 0
        if len(outputs[0]) == 4:
            out = outputs[0][2]
            outputs[0][2] = outputs[0][3] = patch_size_map[layer['out'][0]]
        treused_flops = register_ops_count[_type](inputs, outputs, params)
        reused_flops += treused_flops
        patch = patch_size_map[layer['out'][0]]
        # print(f.format(_type, ttotal_flops, treused_flops, 1-treused_flops/max(1e-6,ttotal_flops), out, int(patch)))
    # print(f.format('sum', total_flops, reused_flops, 1-reused_flops*1./total_flops, 0, 0))

    return 1-reused_flops*1./total_flops
