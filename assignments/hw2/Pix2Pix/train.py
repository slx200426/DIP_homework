import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork


def tensor_to_image(tensor):
    """
    Convert tensor in [-1, 1] to uint8 image in [0, 255].
    Input shape : (C, H, W)
    Output shape: (H, W, C)
    """
    image = tensor.detach().cpu().float().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1.0) / 2.0
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255).astype(np.uint8)
    return image


def save_images(inputs, targets, outputs, folder_name, epoch, num_images=4):
    """
    Save input / target / output comparison images.
    """
    save_dir = os.path.join(folder_name, f'epoch_{epoch}')
    os.makedirs(save_dir, exist_ok=True)

    actual_num = min(num_images, inputs.shape[0])

    for i in range(actual_num):
        input_img = tensor_to_image(inputs[i])
        target_img = tensor_to_image(targets[i])
        output_img = tensor_to_image(outputs[i])

        # RGB -> BGR for cv2.imwrite
        input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        comparison = np.hstack([input_img, target_img, output_img])
        save_path = os.path.join(save_dir, f'result_{i + 1}.png')
        cv2.imwrite(save_path, comparison)


def train_one_epoch(model,
                    dataloader,
                    optimizer,
                    criterion,
                    device,
                    epoch,
                    num_epochs,
                    save_every=5):
    model.train()
    running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        optimizer.zero_grad()

        outputs = model(image_rgb)
        loss = criterion(outputs, image_semantic)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i == 0:
            print(
                f'[Epoch {epoch + 1}] output min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}'
            )
            print(
                f'[Epoch {epoch + 1}] target min: {image_semantic.min().item():.4f}, max: {image_semantic.max().item():.4f}'
            )

        if epoch % save_every == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, 'train_results',
                        epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}] '
              f'Step [{i + 1}/{len(dataloader)}] '
              f'Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(dataloader)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}] Average Train Loss: {avg_train_loss:.4f}'
    )
    return avg_train_loss


@torch.no_grad()
def validate(model,
             dataloader,
             criterion,
             device,
             epoch,
             num_epochs,
             save_every=5):
    model.eval()
    val_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        outputs = model(image_rgb)
        loss = criterion(outputs, image_semantic)
        val_loss += loss.item()

        if epoch % save_every == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, 'val_results',
                        epoch)

    avg_val_loss = val_loss / len(dataloader)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}'
    )
    return avg_val_loss


def main():
    # device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    # check files
    if not os.path.exists('train_list.txt'):
        raise FileNotFoundError('train_list.txt not found.')
    if not os.path.exists('val_list.txt'):
        raise FileNotFoundError('val_list.txt not found.')

    # dataset
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Val dataset size: {len(val_dataset)}')

    # dataloader
    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=0)

    val_loader = DataLoader(val_dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=0)

    # model
    model = FullyConvNetwork().to(device)

    # loss
    criterion = nn.L1Loss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    num_epochs = 50
    best_val_loss = float('inf')

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, epoch, num_epochs)

        val_loss = validate(model, val_loader, criterion, device, epoch,
                            num_epochs)

        # save latest
        torch.save(model.state_dict(), 'checkpoints/pix2pix_model_latest.pth')

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       'checkpoints/pix2pix_model_best.pth')
            print(
                f'Best model updated at epoch {epoch + 1}, val loss = {val_loss:.4f}'
            )

        # optional periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

    torch.save(model.state_dict(), 'checkpoints/pix2pix_model_final.pth')
    print('Training finished.')
    print('Saved: checkpoints/pix2pix_model_final.pth')


if __name__ == '__main__':
    main()
