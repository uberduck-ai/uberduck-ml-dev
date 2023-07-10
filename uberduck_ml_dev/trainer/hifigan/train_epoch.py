# NOTE (Sam): we should pass functional arguments like this for all models


def train_epoch(
    _train_step,
    dataloader,
    config,
    models,
    optimization_parameters,
    logging_parameters,
    iteration,
):
    for batch in dataloader:
        print(iteration, "iteration")
        _train_step(
            batch,
            config,
            models,
            optimization_parameters,
            logging_parameters,
            iteration,
        )
        iteration += 1

    return iteration
