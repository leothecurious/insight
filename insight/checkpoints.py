"""Portable checkpoint naming."""


def filename(series, epoch, final_epoch):
    """Name a checkpoint from its run series and epoch."""
    if epoch == final_epoch and series.endswith(f"_ep{epoch}"):
        return f"{series}.pt"
    return f"{series}_epoch_{epoch}.pt"
