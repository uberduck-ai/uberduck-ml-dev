from .train_step import _train_step


# NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
def train_epoch(
    train_dataloader,
    log_decoder_samples,
    log_attribute_samples,
    model,
    optim,
    steps_per_sample,
    scaler,
    iters_per_checkpoint,
    output_directory,
    criterion,
    attention_kl_loss,
    kl_loss_start_iter,
    binarization_start_iter,
    iteration,
    vocoder,
    epoch=None,
    grad_clip_val=None,
):
    # def train_epoch(dataset_shard, batch_size, model, optim, steps_per_sample, scaler, scheduler, criterion, attention_kl_loss, kl_loss_start_iter, binarization_start_iter, epoch, iteration):
    # for batch_idx, ray_batch_df in enumerate(
    #     dataset_shard.iter_batches(batch_size=batch_size, prefetch_blocks=6)
    # ):
    # NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
    for batch in train_dataloader:
        _train_step(
            # ray_batch_df,
            # NOTE (Sam): uncomment to run with torch DataLoader rather than ray dataset
            batch,
            model,
            optim,
            iteration,
            log_decoder_samples,
            log_attribute_samples,
            steps_per_sample,
            scaler,
            iters_per_checkpoint,
            output_directory,
            criterion,
            attention_kl_loss,
            kl_loss_start_iter,
            binarization_start_iter,
            vocoder,
            epoch=epoch,
            grad_clip_val=grad_clip_val,
        )
        iteration += 1

    return iteration
