from kan import KAN


def create_kan_model(
    input_dim,
    output_dim,
    hidden_dims,
    grid=3,
    k=3,
    seed=42,
    device="cpu",
):
    """
    Create a standard Kolmogorov-Arnold Network using pykan.

    Architecture and hyperparameters are supplied externally.
    No dataset-specific assumptions are made here.
    """
    width = [input_dim] + list(hidden_dims) + [output_dim]

    model = KAN(
        width=width,
        grid=grid,
        k=k,
        seed=seed,
        device=device,
    )

    return model
