import torch
import resnet
import mobilenet

class ReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # print('customized relu forward')
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        grad_input[grad_input<0] = 0
        # print('customized relu backward')
        # print("input: ", input)
        # print("output_grad: ", grad_output)
        # print("input_grad: ", grad_input)
        return grad_input

class ReLU6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # print('customized relu forward')
        return input.clamp(min=0, max=6)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] = 0
        grad_input[grad_input<0] = 0
        # print('customized relu backward')
        # print("input: ", input)
        # print("output_grad: ", grad_output)
        # print("input_grad: ", grad_input)
        return grad_input

class NNReLU6(torch.nn.Module):
    def __init__(self):
        super(NNReLU6, self).__init__()
    def forward(self, input):
        return ReLU6.apply(input)
    def extra_repr(self):
        return "Customized ReLU6."

def test():
    for i in range(30):
        dtype = torch.float
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        generator=torch.Generator(device).manual_seed(i)
        N, D_in, H, D_out = 1, 3, 3, 10
        x = torch.randn(N, D_in, device=device, dtype=dtype,generator=generator)
        y = torch.randn(N, D_out, device=device, dtype=dtype, generator=generator)
        w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True, generator=generator)
        w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True, generator=generator)

        relu = ReLU.apply
        y_pred = relu(x.mm(w1)).mm(w2)
        loss = (y_pred - y).pow(2).sum()
        loss.backward()

def resnet50(pretrained=False, progress=True):
    return resnet.resnet50(pretrained, progress)

def resnet152(pretrained=False, progress=True):
    return resnet.resnet152(pretrained, progress)

def mobilenet_v2(pretrained=False, progress=True):
    return mobilenet.mobilenet_v2(pretrained, progress)
