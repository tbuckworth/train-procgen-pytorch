import torch
from torchvision.models.video.resnet import model_urls


def generate_y(x, shp):
    y = torch.zeros(shp)
    y[..., 0] = x[..., 0] * x[..., 4] + torch.sin(x[..., 2])
    y[..., 1] = x[..., 1] + torch.cos(x[..., 4]) * torch.exp(x[..., 2])
    return y

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 10
    k = 2
    m = 5
    b = int(1000 / (m ** 2))
    lr = 1e-3
    epochs = 1000
    seed = 42

    x = torch.rand((b, n)).to(device)
    y = generate_y(x, (b, k)).to(device)
    x_test = x * 2
    y_test = generate_y(x_test, (b, k)).to(device)

    model
    loss_fn
    

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nan
    for epoch in range(epochs):
        y_hat = model(x)
        # loss = ((y_hat - y) ** 2).mean()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            print(f"loss: {loss.item():.4f}")
    # torch.save(model, logdir)
    return loss.item()


if __name__ == "__main__":
    main()
