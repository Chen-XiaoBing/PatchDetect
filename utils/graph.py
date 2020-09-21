class Graph:

    def __init__(self):
        self.tensor_shape = list()
        self.layers = list()

    def __str__(self):
        ret = ''
        ret += 'Graph Information:\n'

        ret += '1. Tensors\n'
        f = "{:<5} {:<30}\n"
        ret += f.format('Id', 'Shape')
        for idx, shape in enumerate(self.tensor_shape):
            ret += f.format(idx, str(shape))

        ret += '2. Layers\n'
        f = "{:<5} {:<20} {:<30} {:<30} {}\n"
        ret += f.format("Id", "Type", "Inputs", "Ouputs", "Params")
        for idx, layer in enumerate(self.layers):
            _type, inputs, outputs, params = layer['type'], layer['in'], layer['out'], layer['params']
            ret += f.format(idx, str(_type), str(inputs), str(outputs), str(params))

        return ret

if __name__ == '__main__':
    g = Graph()
    g.tensor_shape = [[1, 3, 224, 224], [], [1, 1000]]
    layer = {'type':'conv', 'in':[0,2], 'out':[1], 'params':{'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 7], 'pads': [3, 3, 3, 3], 'strides': [2, 2]}}
    g.layers = [('conv', [0, 2], [1], {'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 7], 'pads': [3, 3, 3, 3], 'strides': [2, 2]})]
    g.layers = [layer]
    print(g)
