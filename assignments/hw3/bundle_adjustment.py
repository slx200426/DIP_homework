import torch
import numpy as np
import matplotlib.pyplot as plt


def euler_to_rotation_matrix(euler):
    if euler.dim() == 1:
        euler = euler.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    rx, ry, rz = euler[:, 0], euler[:, 1], euler[:, 2]

    Rx = torch.zeros(euler.shape[0], 3, 3, device=euler.device)
    Rx[:, 0, 0] = 1
    Rx[:, 1, 1] = torch.cos(rx)
    Rx[:, 1, 2] = -torch.sin(rx)
    Rx[:, 2, 1] = torch.sin(rx)
    Rx[:, 2, 2] = torch.cos(rx)

    Ry = torch.zeros(euler.shape[0], 3, 3, device=euler.device)
    Ry[:, 0, 0] = torch.cos(ry)
    Ry[:, 0, 2] = torch.sin(ry)
    Ry[:, 1, 1] = 1
    Ry[:, 2, 0] = -torch.sin(ry)
    Ry[:, 2, 2] = torch.cos(ry)

    Rz = torch.zeros(euler.shape[0], 3, 3, device=euler.device)
    Rz[:, 0, 0] = torch.cos(rz)
    Rz[:, 0, 1] = -torch.sin(rz)
    Rz[:, 1, 0] = torch.sin(rz)
    Rz[:, 1, 1] = torch.cos(rz)
    Rz[:, 2, 2] = 1

    R = Rz @ Ry @ Rx

    if squeeze:
        R = R.squeeze(0)

    return R


def project_points(points3d, R, T, focal, cx, cy):
    Xc = (R @ points3d.T).T + T

    z = torch.clamp(Xc[:, 2], min=0.1)

    u = -focal * Xc[:, 0] / z + cx
    v = focal * Xc[:, 1] / z + cy

    return torch.stack([u, v], dim=1)


def main():
    print("Loading data...")
    data = np.load('data/points2d.npz')
    colors = np.load('data/points3d_colors.npy')

    num_views = 50
    num_points = 20000
    img_size = 1024
    cx, cy = img_size / 2, img_size / 2

    points2d_list = []
    visibility_list = []
    for i in range(num_views):
        view_data = data[f'view_{i:03d}']
        points2d_list.append(view_data[:, :2])
        visibility_list.append(view_data[:, 2])

    points2d = torch.tensor(np.stack(points2d_list), dtype=torch.float32)
    visibility = torch.tensor(np.stack(visibility_list), dtype=torch.float32)

    print(f"Loaded {num_views} views, {num_points} points")
    print(f"Total observations: {visibility.sum().item():.0f}")

    focal = torch.tensor([1000.0], requires_grad=True)

    euler_angles = torch.randn(num_views, 3) * 0.01
    euler_angles = euler_angles.requires_grad_(True)

    translations = torch.zeros(num_views, 3)
    translations[:, 2] = -2.5  
    translations = translations.requires_grad_(True)

    with torch.no_grad():
        u = (points2d[0, :, 0] - cx) / 1000.0
        v = (points2d[0, :, 1] - cy) / 1000.0
        depth = 2.5
        x = -u * depth
        y = v * depth
        z = torch.ones(num_points) * depth
        points3d_init = torch.stack([x, y, z], dim=1)

    points3d = points3d_init.requires_grad_(True)

    optimizer = torch.optim.Adam([{
        'params': [focal],
        'lr': 1.0
    }, {
        'params': [euler_angles],
        'lr': 0.001
    }, {
        'params': [translations],
        'lr': 0.001
    }, {
        'params': [points3d],
        'lr': 0.001
    }])

    print("\nStarting Bundle Adjustment...")
    losses = []

    for epoch in range(2000):
        optimizer.zero_grad()

        total_error = 0
        num_visible = 0

        for i in range(num_views):
            R = euler_to_rotation_matrix(euler_angles[i])
            T = translations[i]

            proj = project_points(points3d, R, T, focal, cx, cy)

            mask = visibility[i] > 0.5
            diff = (proj - points2d[i]) * mask.unsqueeze(1)
            total_error += (diff**2).sum()
            num_visible += mask.sum()

        loss = total_error / num_visible

        if torch.isnan(loss):
            print(f"\nNaN detected at epoch {epoch}!")
            print(f"Focal: {focal.item()}")
            print(f"Max euler: {euler_angles.abs().max().item()}")
            print(f"Max translation: {translations.abs().max().item()}")
            print(
                f"Points3D range: [{points3d.min().item():.2f}, {points3d.max().item():.2f}]"
            )
            break

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [focal, euler_angles, translations, points3d], max_norm=10.0)

        optimizer.step()

        with torch.no_grad():
            focal.clamp_(min=100.0, max=5000.0)

        losses.append(loss.item())

        if epoch % 100 == 0:
            rmse = torch.sqrt(loss).item()
            print(
                f"Epoch {epoch:4d} | RMSE: {rmse:7.2f} px | Focal: {focal.item():7.2f}"
            )

    if len(losses) > 0 and not np.isnan(losses[-1]):
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Bundle Adjustment Convergence')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
        print("\n✓ Loss curve saved to loss_curve.png")

        points3d_np = points3d.detach().cpu().numpy()
        colors_np = colors / 255.0

        with open('result.obj', 'w') as f:
            for i in range(num_points):
                x, y, z = points3d_np[i]
                r, g, b = colors_np[i]
                f.write(f'v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n')

        print("✓ 3D point cloud saved to result.obj")

        final_rmse = torch.sqrt(torch.tensor(losses[-1])).item()
        print(f"\n{'='*50}")
        print(f"Final Results:")
        print(f"  Focal length: {focal.item():.2f} pixels")
        print(f"  RMSE: {final_rmse:.2f} pixels")
        print(f"{'='*50}")
    else:
        print("\nOptimization failed - NaN encountered")


if __name__ == '__main__':
    main()
