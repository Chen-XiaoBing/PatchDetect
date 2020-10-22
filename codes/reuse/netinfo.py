from .graph import Graph
import re
import argparse
import numpy as np

import torch
from torchvision import models

def dump_pytorch_graph(graph):
    """List all the nodes in a PyTorch graph."""
    f = "{:25} {:40}   {} -> {}"
    print(f.format("kind", "scopeName", "inputs", "outputs"))

    for node in graph.nodes():
        print(f.format(node.kind(), node.scopeName(),
                       [i.unique() for i in node.inputs()],
                       [i.unique() for i in node.outputs()]
                       ))

def get_shape(graph):
    tensor_shape = dict()
    # get input shape
    input_str = str(next(graph.inputs()))
    input_str = input_str.split('defined in (')[-1].split('= prim')[0]
    items = input_str.split('%')[1:]
    for item in items:
        key, value = item.split(' : ')
        value = value.split('(')[-1].split(')')[0]
        if key.strip().isdigit(): key = int(key)
        else:             key = 0
        value = eval('[' + re.sub(':.*', '', re.sub(':.*?, ', ',', value)) + ']')
        tensor_shape[key] = value

    # print(str(graph))

    # get output shape
    pad_map = dict()
    for line in str(graph).split('):')[-1].split('\n'):
        if ': Tensor = onnx::Pad' in line:
            key, value = line.split(' : ')
            key = int(key.split('%')[-1])
            value = int(value.split('%')[-1].split(')')[0])
            pad_map[key] = value

    items = re.sub(' =.*', '',  str(graph).split('):')[-1]).split('return')[0].split('%')[1:]
    for item in items:
        # print('-----------------')
        # print(item)
        key, value = item.split(' : ')
        if key.strip().isdigit(): key = int(key)
        else:             key = 0
        value = value.split('(')[-1].split(')')[0]
        value = '[' + re.sub(':.*', '', re.sub(':.*?, ', ',', value)) + ']'
        if 'Tensor' in value:
            tensor_shape[key] = tensor_shape[pad_map[key]]
        else:
            value = eval(value)
            tensor_shape[key] = value
    return tensor_shape

def retrieve_net_info(model, input, verbose=False):
    trace, out = torch.jit._get_trace_graph(model, input)
    torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    if verbose:
        dump_pytorch_graph(torch_graph)

    tensor_shape = get_shape(torch_graph)
    # for key in tensor_shape:
    #     print(key, tensor_shape[key])

    layer2in = []
    layer2out = []
    layer_type = []
    layer2params = []
    for torch_node in torch_graph.nodes():
        # Op
        op = torch_node.kind().split('::')[-1]
        # Parameters
        params = {k: torch_node[k] for k in torch_node.attributeNames()}
        # Inputs/outputs
        inputs = [i.unique() for i in torch_node.inputs()]
        outputs = [o.unique() for o in torch_node.outputs()]

        layer2in.append(inputs)
        layer2out.append(outputs)
        layer_type.append(op)
        layer2params.append(params)

    graph = Graph()
    graph.tensor_shape = tensor_shape
    for i in range(len(layer2in)):
        layer = {'type':layer_type[i], 'in':layer2in[i], 'out':layer2out[i], 'params':layer2params[i]}
        graph.layers.append(layer)

    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50')
    args = parser.parse_args()

    model = getattr(models, args.model)()
    input = torch.zeros([1, 3, 224, 224])
    g = retrieve_net_info(model, input, verbose=False)
    print(g)
