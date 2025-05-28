import torch
from sklearn.datasets import make_classification

N_FEATURES = 10
N_CLASSES = 2

N_EPOCHS = 10


class ComplexToRealEinsumFunction(torch.autograd.Function):
    """
    Custom autograd function for complex einsum operation: "i,kija,ta,j->tk"
    Returns only the real part of the complex result.
    """

    @staticmethod
    def forward(ctx, psi_real, psi_imag, A_real, A_imag, x_real, x_imag):
        # Save tensors for backward pass
        ctx.save_for_backward(psi_real, psi_imag, A_real, A_imag, x_real, x_imag)

        # Helper function for real einsum operations
        def real_einsum(psi_comp1, A_comp, x_comp, psi_comp2):
            return torch.einsum("i,kija,ta,j->tk", psi_comp1, A_comp, x_comp, psi_comp2)

        # Compute all 8 terms for complex multiplication
        # Forward: psi* @ A @ x @ psi, taking only real part
        term1 = real_einsum(psi_real, A_real, x_real, psi_real)  # + + + +
        term2 = real_einsum(psi_real, A_imag, x_imag, psi_real)  # + - - +
        term3 = real_einsum(psi_imag, A_real, x_imag, psi_real)  # - + - +
        term4 = real_einsum(psi_imag, A_imag, x_real, psi_real)  # - - + +
        term5 = real_einsum(psi_real, A_real, x_real, psi_imag)  # + + + -
        term6 = real_einsum(psi_real, A_imag, x_imag, psi_imag)  # + - - -
        term7 = real_einsum(psi_imag, A_real, x_imag, psi_imag)  # - + - -
        term8 = real_einsum(psi_imag, A_imag, x_real, psi_imag)  # - - + -

        # Real part of complex result
        result = term1 - term2 - term3 - term4 - term5 + term6 + term7 + term8
        return result

    @staticmethod
    def backward(ctx, grad_output):
        psi_real, psi_imag, A_real, A_imag, x_real, x_imag = ctx.saved_tensors

        # Helper functions for computing gradients via einsum
        def grad_psi_first(grad_out, psi1, A_comp, x_comp, psi2):
            return torch.einsum("tk,kija,ta,j->i", grad_out, A_comp, x_comp, psi2)

        def grad_A(grad_out, psi1, A_comp, x_comp, psi2):
            return torch.einsum("tk,i,ta,j->kija", grad_out, psi1, x_comp, psi2)

        def grad_x(grad_out, psi1, A_comp, x_comp, psi2):
            return torch.einsum("tk,i,kija,j->ta", grad_out, psi1, A_comp, psi2)

        def grad_psi_last(grad_out, psi1, A_comp, x_comp, psi2):
            return torch.einsum("tk,i,kija,ta->j", grad_out, psi1, A_comp, x_comp)

        # Initialize gradients
        grad_psi_real = torch.zeros_like(psi_real)
        grad_psi_imag = torch.zeros_like(psi_imag)
        grad_A_real = torch.zeros_like(A_real)
        grad_A_imag = torch.zeros_like(A_imag)
        grad_x_real = torch.zeros_like(x_real)
        grad_x_imag = torch.zeros_like(x_imag)

        # Compute gradients for each term (matching forward pass structure)

        # Term 1: +1 * einsum(psi_real, A_real, x_real, psi_real)
        grad_psi_real += grad_psi_first(grad_output, psi_real, A_real, x_real, psi_real)
        grad_psi_real += grad_psi_last(grad_output, psi_real, A_real, x_real, psi_real)
        grad_A_real += grad_A(grad_output, psi_real, A_real, x_real, psi_real)
        grad_x_real += grad_x(grad_output, psi_real, A_real, x_real, psi_real)

        # Term 2: -1 * einsum(psi_real, A_imag, x_imag, psi_real)
        grad_psi_real -= grad_psi_first(grad_output, psi_real, A_imag, x_imag, psi_real)
        grad_psi_real -= grad_psi_last(grad_output, psi_real, A_imag, x_imag, psi_real)
        grad_A_imag -= grad_A(grad_output, psi_real, A_imag, x_imag, psi_real)
        grad_x_imag -= grad_x(grad_output, psi_real, A_imag, x_imag, psi_real)

        # Term 3: -1 * einsum(psi_imag, A_real, x_imag, psi_real)
        grad_psi_imag -= grad_psi_first(grad_output, psi_imag, A_real, x_imag, psi_real)
        grad_psi_real -= grad_psi_last(grad_output, psi_imag, A_real, x_imag, psi_real)
        grad_A_real -= grad_A(grad_output, psi_imag, A_real, x_imag, psi_real)
        grad_x_imag -= grad_x(grad_output, psi_imag, A_real, x_imag, psi_real)

        # Term 4: -1 * einsum(psi_imag, A_imag, x_real, psi_real)
        grad_psi_imag -= grad_psi_first(grad_output, psi_imag, A_imag, x_real, psi_real)
        grad_psi_real -= grad_psi_last(grad_output, psi_imag, A_imag, x_real, psi_real)
        grad_A_imag -= grad_A(grad_output, psi_imag, A_imag, x_real, psi_real)
        grad_x_real -= grad_x(grad_output, psi_imag, A_imag, x_real, psi_real)

        # Term 5: -1 * einsum(psi_real, A_real, x_real, psi_imag)
        grad_psi_real -= grad_psi_first(grad_output, psi_real, A_real, x_real, psi_imag)
        grad_psi_imag -= grad_psi_last(grad_output, psi_real, A_real, x_real, psi_imag)
        grad_A_real -= grad_A(grad_output, psi_real, A_real, x_real, psi_imag)
        grad_x_real -= grad_x(grad_output, psi_real, A_real, x_real, psi_imag)

        # Term 6: +1 * einsum(psi_real, A_imag, x_imag, psi_imag)
        grad_psi_real += grad_psi_first(grad_output, psi_real, A_imag, x_imag, psi_imag)
        grad_psi_imag += grad_psi_last(grad_output, psi_real, A_imag, x_imag, psi_imag)
        grad_A_imag += grad_A(grad_output, psi_real, A_imag, x_imag, psi_imag)
        grad_x_imag += grad_x(grad_output, psi_real, A_imag, x_imag, psi_imag)

        # Term 7: +1 * einsum(psi_imag, A_real, x_imag, psi_imag)
        grad_psi_imag += grad_psi_first(grad_output, psi_imag, A_real, x_imag, psi_imag)
        grad_psi_imag += grad_psi_last(grad_output, psi_imag, A_real, x_imag, psi_imag)
        grad_A_real += grad_A(grad_output, psi_imag, A_real, x_imag, psi_imag)
        grad_x_imag += grad_x(grad_output, psi_imag, A_real, x_imag, psi_imag)

        # Term 8: +1 * einsum(psi_imag, A_imag, x_real, psi_imag)
        grad_psi_imag += grad_psi_first(grad_output, psi_imag, A_imag, x_real, psi_imag)
        grad_psi_imag += grad_psi_last(grad_output, psi_imag, A_imag, x_real, psi_imag)
        grad_A_imag += grad_A(grad_output, psi_imag, A_imag, x_real, psi_imag)
        grad_x_real += grad_x(grad_output, psi_imag, A_imag, x_real, psi_imag)

        return (
            grad_psi_real,
            grad_psi_imag,
            grad_A_real,
            grad_A_imag,
            grad_x_real,
            grad_x_imag,
        )


# Convenience function to call the custom autograd function
def complex_to_real_einsum(psi_real, psi_imag, A_real, A_imag, x_real, x_imag):
    """Wrapper function for the custom autograd operation"""
    return ComplexToRealEinsumFunction.apply(
        psi_real, psi_imag, A_real, A_imag, x_real, x_imag
    )


class ComplexNet(torch.nn.Module):
    """Original verbose implementation with manual complex arithmetic"""

    def __init__(self):
        super(ComplexNet, self).__init__()

        self.A_real = torch.nn.Parameter(torch.randn(N_CLASSES, 10, 10, N_FEATURES))
        self.A_imag = torch.nn.Parameter(torch.randn(N_CLASSES, 10, 10, N_FEATURES))

        self.psi_real = torch.nn.Parameter(torch.randn(10))
        self.psi_imag = torch.nn.Parameter(torch.randn(10))

    def forward(self, x):
        # x is already torch.float32 from the dataloader
        x_real = x
        x_imag = torch.zeros_like(x)

        x = complex_to_real_einsum(
            self.psi_real, self.psi_imag, self.A_real, self.A_imag, x_real, x_imag
        )
        return x


if __name__ == "__main__":
    X, y = make_classification(
        n_samples=10000, n_features=N_FEATURES, n_classes=N_CLASSES
    )
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X).float(), torch.from_numpy(y).long()
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

    net = ComplexNet().to("cuda")
    net = torch.compile(net)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)

    for epoch in range(N_EPOCHS):
        loss_sum = 0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to("cuda")
            batch_y = batch_y.to("cuda")
            optimizer.zero_grad()
            output = net(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        print(f"Epoch {epoch} loss: {loss_sum / len(dataloader)}")
