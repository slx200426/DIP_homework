import numpy as np


def read_colmap_points(filepath):
    points = []
    errors = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                error = float(parts[7])
                errors.append(error)
    return errors


print("=" * 60)
print("COLMAP Sparse Reconstruction Results")
print("=" * 60)

# 相机参数
print("\nCamera Parameters:")
print("  Model: PINHOLE")
print("  Resolution: 1024 x 1024")
print("  fx = 891.25 px")
print("  fy = 875.14 px")
print("  cx = 512 px")
print("  cy = 512 px")

# 点云统计
errors = read_colmap_points('data/colmap/sparse/0/points3D.txt')
print(f"\n3D Reconstruction:")
print(f"  Number of 3D points: {len(errors)}")
print(f"  Average reprojection error: {np.mean(errors):.2f} px")
print(f"  RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.2f} px")

print("\n" + "=" * 60)
