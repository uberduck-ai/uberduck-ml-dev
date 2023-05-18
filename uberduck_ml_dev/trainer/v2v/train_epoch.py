from .train_step import _train_step


def train_epoch(
    train_dataloader,
    logging_parameters,
    model,
    optimization_parameters,
    iteration,
):
    for batch in train_dataloader:
        _train_step(
            batch,
            logging_parameters,
            model,
            optimization_parameters,
            iteration,
        )
        iteration += 1

    return iteration
