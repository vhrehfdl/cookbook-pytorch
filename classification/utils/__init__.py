import os
import torch


def save_model(model, model_dir, epoch, val_acc):
    if not os.path.isdir("model_save"):
        os.makedirs("model_save")

    torch.save(model.state_dict(), model_dir+"model-weights.{0:02d}-{1:.6f}.pt".format(epoch, val_acc))


    