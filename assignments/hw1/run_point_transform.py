import cv2
import numpy as np
import gradio as gr

points_src = []
points_dst = []
image = None


def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img


def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 3, (255, 0, 0), -1)
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 3, (0, 0, 255), -1)

    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]),
                        tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image


def mls_affine_deform_point(v, p, q, alpha=1.0, eps=1e-8):
    dists = np.linalg.norm(p - v, axis=1)
    idx = np.argmin(dists)
    if dists[idx] < 1e-6:
        return q[idx]

    w = 1.0 / (np.power(dists, 2 * alpha) + eps)

    w_sum = np.sum(w)

    p_star = np.sum(p * w[:, None], axis=0) / w_sum
    q_star = np.sum(q * w[:, None], axis=0) / w_sum

    p_hat = p - p_star
    q_hat = q - q_star

    A = p_hat.T @ (w[:, None] * p_hat)
    B = p_hat.T @ (w[:, None] * q_hat)

    A += eps * np.eye(2)

    M = np.linalg.inv(A) @ B

    f_v = (v - p_star) @ M + q_star
    return f_v


def point_guided_deformation(image,
                             source_pts,
                             target_pts,
                             alpha=1.0,
                             eps=1e-8):
    if image is None:
        return None

    image = np.array(image)
    h, w = image.shape[:2]

    if len(source_pts) == 0 or len(target_pts) == 0:
        return image

    n = min(len(source_pts), len(target_pts))
    source_pts = source_pts[:n].astype(np.float32)
    target_pts = target_pts[:n].astype(np.float32)

    warped_image = np.zeros_like(image)

    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            v = np.array([x, y], dtype=np.float32)

            src_pos = mls_affine_deform_point(v,
                                              target_pts,
                                              source_pts,
                                              alpha=alpha,
                                              eps=eps)

            map_x[y, x] = src_pos[0]
            map_y[y, x] = src_pos[1]

    warped_image = cv2.remap(image,
                             map_x,
                             map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

    return warped_image


def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src),
                                            np.array(points_dst))

    return warped_image


def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image",
                                   interactive=True,
                                   width=800)
            point_select = gr.Image(
                label="Click to Select Source and Target Points",
                interactive=True,
                width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
