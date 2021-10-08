import os
import torch


def save_model(best_val_loss, val_loss, model, model_dir):
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("model_save"):
            os.makedirs("model_save")
        torch.save(model.state_dict(), model_dir)


    