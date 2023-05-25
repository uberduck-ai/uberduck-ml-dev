# TODO (Sam): combine with radtts save_checkpoint
import torch


def save_checkpoint(generator, generator_optimizer, discriminator, discriminator_optimizer, iteration, filepath):
    print(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath
        )
    )

    # TODO (Sam): figure out where to put learning rate.
    torch.save(
        {
            "generator_state_dict": generator.state_dict(),
            "iteration": iteration,
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
        },
        filepath,
    )
