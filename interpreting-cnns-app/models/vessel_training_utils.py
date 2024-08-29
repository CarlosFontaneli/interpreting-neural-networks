import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Function to show images
def show_images(data_loader, num_sets=10):
    fig, axs = plt.subplots(3, num_sets, figsize=(20, 6))

    num_displayed = 0
    for images, masks, skeletons in data_loader:
        for i in range(images.size(0)):
            if num_displayed >= num_sets:
                break

            axs[0, num_displayed].imshow(images[i].permute(1, 2, 0), cmap="gray")
            axs[0, num_displayed].axis("off")
            axs[0, num_displayed].set_title(f"Image {num_displayed+1}")

            axs[1, num_displayed].imshow(masks[i].squeeze(), cmap="gray")
            axs[1, num_displayed].axis("off")
            axs[1, num_displayed].set_title(f"Mask {num_displayed+1}")

            axs[2, num_displayed].imshow(skeletons[i].squeeze(), cmap="gray")
            axs[2, num_displayed].axis("off")
            axs[2, num_displayed].set_title(f"Skeleton {num_displayed+1}")

            num_displayed += 1

        if num_displayed >= num_sets:
            break

    plt.tight_layout()
    plt.show()


# Function to calculate IoU metric
def iou_metric(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)


# Function to sum parameters
def sum_params(model, mode="l2"):
    s = 0
    for param in model.parameters():
        if mode == "l2":
            s += (param**2).sum()
        else:
            s += param.abs().sum()
    return s


# Training loop
def training_loop(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    regularization_mode,
    alpha_l1,
    alpha_l2,
    train_losses,
    train_aux_losses,
    test_losses,
    test_accuracies,
    test_ious,
):
    for epoch in range(num_epochs):
        model.train()
        aux_loss = 0.0
        running_aux_loss = 0.0
        running_loss = 0.0
        train_iou = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.cuda(), labels.cuda()
            labels = labels.squeeze(1).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            if regularization_mode == "l1":
                aux_loss = alpha_l1 * sum_params(model, mode="l1")
                running_aux_loss += aux_loss
                loss += aux_loss

            if regularization_mode == "l2":
                aux_loss = alpha_l2 * sum_params(model, mode="l2")
                running_aux_loss += aux_loss
                loss += aux_loss

            _, predicted = torch.max(outputs.data, 1)
            train_iou += iou_metric(labels.float(), predicted.float())
            loss.backward()
            optimizer.step()

        avg_train_loss = running_loss / len(train_loader)
        train_iou /= len(train_loader)
        train_losses.append(avg_train_loss)
        avg_train_aux_loss = running_aux_loss / len(train_loader)
        train_aux_losses.append(avg_train_aux_loss)

        model.eval()
        test_loss = 0.0
        total_iou = 0.0
        with torch.no_grad():
            for data in test_loader:
                images, labels, _ = data
                images, labels = images.cuda(), labels.cuda()
                labels = labels.squeeze(1).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_iou += iou_metric(labels.float(), predicted.float())

        avg_test_loss = test_loss / len(test_loader)
        avg_iou = total_iou / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_iou.cpu())
        test_ious.append(avg_iou.cpu())

        scheduler.step(avg_test_loss)
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} Aux Loss: {avg_train_aux_loss:.4f} Test Loss: {avg_test_loss:.4f} Train IoU: {train_iou:.4f} Test IoU: {avg_iou:.4f}"
        )


# Function to save the model
def save_model(model, path, regularization_mode):
    torch.save(model.state_dict(), path)


# Function to plot evaluation
def plot_evaluation(
    train_losses,
    train_aux_losses,
    test_losses,
    test_ious,
    regularization_mode,
    num_epochs,
    save_path,
):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    if regularization_mode != "none":
        plt.plot(
            [loss.detach().cpu().numpy() for loss in train_aux_losses],
            label="Training Aux Loss",
        )
    plt.plot(test_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(test_ious, label="IoU", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Test IoU")
    plt.legend()

    plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.tight_layout()
    plt.show()


# Function to output model predictions
def model_out(model, test_loader):
    model.cpu().eval()
    images, masks, _ = next(iter(test_loader))
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig, axs = plt.subplots(len(images), 3, figsize=(12, len(images) * 4))
    for idx, (img, mask, pred) in enumerate(zip(images, masks, preds)):
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = pred.squeeze().cpu().numpy()
        iou_score = iou_metric(mask_np, pred_np)

        axs[idx, 0].imshow(img.squeeze().cpu().numpy(), cmap="gray")
        axs[idx, 0].set_title("Original Image")

        axs[idx, 1].imshow(mask_np, cmap="gray")
        axs[idx, 1].set_title("Ground Truth Mask")

        axs[idx, 2].imshow(pred_np, cmap="gray")
        axs[idx, 2].set_title(f"Predicted Mask (IoU: {iou_score:.4f})")

    for ax in axs.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.show()
