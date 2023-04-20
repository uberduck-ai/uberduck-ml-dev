import torch

def save_checkpoint(model, optimizer, iteration, filepath):
    print(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath
        )
    )

    # NOTE (Sam): learning rate not accessible here
    torch.save(
        {
            "state_dict": model.state_dict(),
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
        },
        filepath,
    )
