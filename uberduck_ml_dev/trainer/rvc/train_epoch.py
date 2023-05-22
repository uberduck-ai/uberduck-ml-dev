# TODO (Sam): add config arguments to model / optimization / logging and remove.
from .train_step import _train_step


def train_epoch(
    dataloader,
    config,
    models,
    optimization_parameters,
    logging_parameters,
    iteration,
):
    for batch in dataloader:
        _train_step(
            batch,
            config,
            models
            optimization_parameters,
            logging_parameters,
            iteration,
        )
        iteration += 1

    return iteration
