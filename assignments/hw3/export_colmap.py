# export_colmap.py
points = []
with open('data/colmap/sparse/0/points3D.txt', 'r') as f:
    for line in f:
        if line.startswith('#') or line.strip() == '':
            continue
        parts = line.strip().split()
        if len(parts) >= 7:
            x, y, z = parts[1], parts[2], parts[3]
            r, g, b = int(parts[4]) / 255, int(parts[5]) / 255, int(
                parts[6]) / 255
            points.append(f'v {x} {y} {z} {r} {g} {b}\n')

with open('colmap_result.obj', 'w') as f:
    f.writelines(points)

print(f"✓ Exported {len(points)} points to colmap_result.obj")
