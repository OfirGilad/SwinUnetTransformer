import os

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
)

from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

from monai.data import (
    decollate_batch,
)

import torch


def define_model(device):
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)

    return model


def train_model(model, device, train_loader, val_loader):
    # define train parameters
    torch.backends.cudnn.benchmark = True
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    root_dir = "./"
    max_iterations = 30000
    eval_num = 500
    post_label = AsDiscrete(to_onehot=2)
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    # Decalre validation function
    def validation(epoch_iterator_val):
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)

                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description(
                    "Validate (%d / %d Steps)" % (global_step, 10.0)
                )
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val

    # Decalre train function
    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        step = 0
        epoch_iterator = tqdm(
            train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].to(device), batch["label"].to(device))
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)"
                % (global_step, max_iterations, loss)
            )
            if (
                    global_step % eval_num == 0 and global_step != 0
            ) or global_step == max_iterations:
                # if True:
                epoch_iterator_val = tqdm(
                    val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
                )
                dice_val = validation(epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                # print(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(
                        model.state_dict(), os.path.join(root_dir, "best_metric_model_swin.pth")
                    )
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
            global_step += 1

        return global_step, dice_val_best, global_step_best

    # Train the model
    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(
            global_step,
            train_loader,
            dice_val_best,
            global_step_best,

        )
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_swin.pth")))

    return model


def evaluate_model(model, device, val_loader):
    model.eval()
    with torch.no_grad():
        epoch_iterator_val = tqdm(
            val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        for step, batch in enumerate(epoch_iterator_val):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)

    return val_inputs, val_labels, val_outputs


def plot_results(val_inputs, val_labels, val_outputs):
    val_inputs = val_inputs.cpu().numpy()
    val_labels = val_labels.cpu().numpy()
    val_outputs = torch.argmax(val_outputs, dim=1).detach().cpu()

    plt.figure("check", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("image")
    plt.imshow(val_inputs[0, 0, :, :, 94], cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("label")
    plt.imshow(val_labels[0, 0, :, :, 94])
    plt.subplot(1, 3, 3)
    plt.title("output")
    plt.imshow(val_outputs[0, :, :, 94])
    plt.show()
