import torch

class AddSquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        return (input1 + input2) ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = 2 * (input1 + input2) * grad_output
        grad_input2 = 2 * (input1 + input2) * grad_output
        return grad_input1, grad_input2

    @staticmethod
    def symbolic(g, input1, input2):
        # 映射到自定义ONNX算子
        return g.op("MyDomain::AddSquare", input1, input2)

class AddSquare(torch.nn.Module):
    def forward(self, x, y):
        return AddSquareFunction.apply(x, y)

# 测试
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)
model = AddSquare()
out = model(x, y)
print(out)  # tensor([16., 36.])
out.sum().backward()
print(x.grad, y.grad)  # tensor([8., 12.]) tensor([8., 12.])