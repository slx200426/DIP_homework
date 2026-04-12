import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F


def init_polygon_state():
    """
    Create an empty polygon state.
    """
    return {"points": [], "closed": False}


def draw_polygon_points(base_img, pts, is_closed=False):
    """
    Draw points / polyline / polygon on a PIL image.
    """
    if base_img is None:
        return None

    canvas = base_img.copy()
    pen = ImageDraw.Draw(canvas)

    if len(pts) >= 2:
        if is_closed and len(pts) >= 3:
            pen.polygon(pts, outline="red")
        else:
            pen.line(pts, fill="red", width=2)

    for px, py in pts:
        pen.ellipse((px - 3, py - 3, px + 3, py + 3), fill="blue")

    return canvas


def append_polygon_vertex(img_original, poly_state, evt: gr.SelectData):
    """
    Add one clicked vertex into polygon state.
    """
    if img_original is None:
        return None, poly_state

    if poly_state["closed"]:
        return img_original, poly_state

    click_x, click_y = evt.index
    poly_state["points"].append((click_x, click_y))

    rendered = draw_polygon_points(img_original,
                                   poly_state["points"],
                                   is_closed=False)
    return rendered, poly_state


def finalize_polygon(img_original, poly_state):
    """
    Close polygon when enough points exist.
    """
    if img_original is None:
        return None, poly_state

    if (not poly_state["closed"]) and len(poly_state["points"]) >= 3:
        poly_state["closed"] = True
        rendered = draw_polygon_points(img_original,
                                       poly_state["points"],
                                       is_closed=True)
        return rendered, poly_state

    return img_original, poly_state


def render_shifted_polygon(bg_img, poly_state, offset_x, offset_y):
    """
    Show shifted polygon over background image.
    """
    if bg_img is None:
        return None

    if not poly_state["closed"]:
        return bg_img

    moved_pts = [(x + int(offset_x), y + int(offset_y))
                 for x, y in poly_state["points"]]

    preview = bg_img.copy()
    painter = ImageDraw.Draw(preview)
    painter.polygon(moved_pts, outline="red")
    return preview


def polygon_to_mask(vertices, height, width):
    """
    Convert polygon vertices to binary uint8 mask.
    """
    binary = np.zeros((height, width), dtype=np.uint8)

    if vertices is None or len(vertices) == 0:
        return binary

    mask_img = Image.new("L", (width, height), 0)
    drawer = ImageDraw.Draw(mask_img)
    drawer.polygon([tuple(p) for p in vertices], outline=255, fill=255)
    binary = np.asarray(mask_img, dtype=np.uint8)
    return binary


def masked_laplacian_distance(src_tensor, region_mask, current_tensor,
                              tgt_tensor):
    """
    Compute a diagnostic loss value for logging during Jacobi iterations.
    """
    out_h, out_w = current_tensor.shape[-2:]

    if src_tensor.shape[-2:] != (out_h, out_w):
        src_tensor = F.interpolate(src_tensor,
                                   size=(out_h, out_w),
                                   mode="bilinear",
                                   align_corners=False)
    if tgt_tensor.shape[-2:] != (out_h, out_w):
        tgt_tensor = F.interpolate(tgt_tensor,
                                   size=(out_h, out_w),
                                   mode="bilinear",
                                   align_corners=False)
    if region_mask.shape[-2:] != (out_h, out_w):
        region_mask = F.interpolate(region_mask,
                                    size=(out_h, out_w),
                                    mode="nearest")

    lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                       dtype=src_tensor.dtype,
                       device=src_tensor.device).view(1, 1, 3, 3)

    ch = src_tensor.shape[1]
    lap = lap.repeat(ch, 1, 1, 1)

    src_lap = F.conv2d(src_tensor, lap, padding=1, groups=ch)
    cur_lap = F.conv2d(current_tensor, lap, padding=1, groups=ch)

    expanded_mask = region_mask.expand_as(src_lap)
    ring_mask = torch.clamp(
        F.max_pool2d(expanded_mask, kernel_size=3, stride=1, padding=1) -
        expanded_mask,
        min=0.0,
        max=1.0)

    grad_term = torch.mean(torch.abs(src_lap - cur_lap) * expanded_mask)
    border_term = torch.mean(
        torch.abs(current_tensor - tgt_tensor) * ring_mask)
    source_term = torch.mean(
        torch.abs(current_tensor - src_tensor) * expanded_mask)

    return grad_term + 0.25 * border_term + 0.01 * source_term


def paste_foreground_on_canvas(fg_img, bg_shape, shift_x, shift_y):
    """
    Place the foreground image onto a background-sized canvas.
    """
    bg_h, bg_w = bg_shape[:2]
    fg_h, fg_w = fg_img.shape[:2]

    canvas = np.zeros((bg_h, bg_w, fg_img.shape[2]), dtype=fg_img.dtype)

    dst_x0 = max(0, int(shift_x))
    dst_y0 = max(0, int(shift_y))
    dst_x1 = min(bg_w, int(shift_x) + fg_w)
    dst_y1 = min(bg_h, int(shift_y) + fg_h)

    src_x0 = max(0, -int(shift_x))
    src_y0 = max(0, -int(shift_y))
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    if dst_x0 < dst_x1 and dst_y0 < dst_y1:
        canvas[dst_y0:dst_y1, dst_x0:dst_x1] = fg_img[src_y0:src_y1,
                                                      src_x0:src_x1]

    return canvas


def choose_device():
    """
    Pick available torch device.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def poisson_blend_with_jacobi(source_patch,
                              target_patch,
                              mask_patch,
                              steps=180):
    """
    Jacobi-style Poisson blending inside a cropped ROI.
    """
    device = choose_device()

    src = torch.from_numpy(source_patch).to(device).permute(
        2, 0, 1).unsqueeze(0).float() / 255.0
    tgt = torch.from_numpy(target_patch).to(device).permute(
        2, 0, 1).unsqueeze(0).float() / 255.0
    msk = torch.from_numpy(mask_patch).to(device).unsqueeze(0).unsqueeze(
        0).float() / 255.0

    # 4-neighbor kernel
    nb_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=src.dtype,
        device=device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    # guidance = Laplacian(source)
    guidance = 4.0 * src - F.conv2d(src, nb_kernel, padding=1, groups=3)
    guidance = guidance * msk

    current = tgt.clone()

    for it in range(steps):
        neighbor_avg_part = F.conv2d(current, nb_kernel, padding=1, groups=3)
        next_estimate = (neighbor_avg_part + guidance) / 4.0
        current = next_estimate * msk + tgt * (1.0 - msk)

        if it % 30 == 0:
            metric = masked_laplacian_distance(src, msk, current, tgt)
            print(f"Jacobi iteration {it}: loss = {metric.item():.6f}")

    out = torch.clamp(current.detach(), 0.0, 1.0)
    out = out.cpu().permute(0, 2, 3, 1).squeeze(0).numpy() * 255.0
    return out.astype(np.uint8)


def blend_images(foreground_image_original, background_image_original, dx, dy,
                 polygon_state):
    """
    Main blending routine.
    """
    if foreground_image_original is None or background_image_original is None:
        return background_image_original

    if not polygon_state["closed"]:
        return background_image_original

    fg_np = np.asarray(foreground_image_original)
    bg_np = np.asarray(background_image_original)

    src_poly = np.asarray(polygon_state["points"], dtype=np.int32)
    dst_poly = src_poly + np.array([int(dx), int(dy)], dtype=np.int32)

    src_mask = polygon_to_mask(src_poly, fg_np.shape[0], fg_np.shape[1])
    dst_mask = polygon_to_mask(dst_poly, bg_np.shape[0], bg_np.shape[1])

    yy, xx = np.where(dst_mask > 0)
    if len(xx) == 0 or len(yy) == 0:
        return background_image_original

    pad = 18
    left = max(int(xx.min()) - pad, 0)
    top = max(int(yy.min()) - pad, 0)
    right = min(int(xx.max()) + pad + 1, bg_np.shape[1])
    bottom = min(int(yy.max()) + pad + 1, bg_np.shape[0])

    src_canvas = paste_foreground_on_canvas(fg_np, bg_np.shape, int(dx),
                                            int(dy))

    src_roi = src_canvas[top:bottom, left:right].copy()
    tgt_roi = bg_np[top:bottom, left:right].copy()
    mask_roi = dst_mask[top:bottom, left:right].copy()

    if mask_roi.sum() == 0:
        return background_image_original

    mixed_roi = poisson_blend_with_jacobi(source_patch=src_roi,
                                          target_patch=tgt_roi,
                                          mask_patch=mask_roi,
                                          steps=180)

    final_img = bg_np.copy()
    final_img[top:bottom, left:right] = mixed_roi
    return final_img


def close_polygon_and_reset_dx(img_original, polygon_state, dx, dy,
                               background_image_original):
    """
    Close polygon, reset dx to 0, refresh overlay.
    """
    img_marked, new_state = finalize_polygon(img_original, polygon_state)
    reset_dx = gr.update(value=0)
    refreshed_bg = render_shifted_polygon(background_image_original, new_state,
                                          0, dy)
    return img_marked, new_state, refreshed_bg, reset_dx


with gr.Blocks(title="Poisson Image Blending",
               css="""
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .gr-button {
        font-size: 1em;
        padding: 0.75em 1.5em;
        border-radius: 8px;
        background-color: #6200ee;
        color: #ffffff;
        border: none;
    }
    .gr-button:hover {
        background-color: #3700b3;
    }
    .gr-slider input[type=range] {
        accent-color: #03dac6;
    }
    .gr-text, .gr-markdown {
        font-size: 1.1em;
    }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #bb86fc;
    }
    .gr-input, .gr-output {
        background-color: #2c2c2c;
        border: 1px solid #3c3c3c;
    }
""") as demo:
    polygon_state = gr.State(init_polygon_state())
    background_image_original = gr.State(value=None)

    gr.Markdown("<h1 style='text-align: center;'>Poisson Image Blending</h1>")
    gr.Markdown(
        "<p style='text-align: center; font-size: 1.2em;'>Blend a selected area from a foreground image onto a background image with adjustable positions.</p>"
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Foreground Image")
            foreground_image_original = gr.Image(label="",
                                                 type="pil",
                                                 interactive=True,
                                                 height=300)
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Upload the foreground image where the polygon will be selected.</p>"
            )

            gr.Markdown("### Foreground Image with Polygon")
            foreground_image_with_polygon = gr.Image(label="",
                                                     type="pil",
                                                     interactive=True,
                                                     height=300)
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Click on the image to define the polygon area. After selecting at least three points, click <strong>Close Polygon</strong>.</p>"
            )

            close_polygon_button = gr.Button("Close Polygon")

        with gr.Column():
            gr.Markdown("### Background Image")
            background_image = gr.Image(label="",
                                        type="pil",
                                        interactive=True,
                                        height=300)
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Upload the background image where the polygon will be placed.</p>"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Background Image with Polygon Overlay")
            background_image_with_polygon = gr.Image(label="",
                                                     type="pil",
                                                     height=500)
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Adjust the position of the polygon using the sliders below.</p>"
            )

        with gr.Column():
            gr.Markdown("### Blended Image")
            output_image = gr.Image(label="", type="pil", height=500)

    with gr.Row():
        with gr.Column():
            dx = gr.Slider(label="Horizontal Offset",
                           minimum=-500,
                           maximum=500,
                           step=1,
                           value=0)
        with gr.Column():
            dy = gr.Slider(label="Vertical Offset",
                           minimum=-500,
                           maximum=500,
                           step=1,
                           value=0)
        blend_button = gr.Button("Blend Images")

    foreground_image_original.change(
        fn=lambda img: img,
        inputs=foreground_image_original,
        outputs=foreground_image_with_polygon,
    )

    foreground_image_with_polygon.select(
        fn=append_polygon_vertex,
        inputs=[foreground_image_original, polygon_state],
        outputs=[foreground_image_with_polygon, polygon_state],
    )

    close_polygon_button.click(
        fn=close_polygon_and_reset_dx,
        inputs=[
            foreground_image_original, polygon_state, dx, dy,
            background_image_original
        ],
        outputs=[
            foreground_image_with_polygon, polygon_state,
            background_image_with_polygon, dx
        ],
    )

    background_image.change(
        fn=lambda img: img,
        inputs=background_image,
        outputs=background_image_original,
    )

    dx.change(
        fn=render_shifted_polygon,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )

    dy.change(
        fn=render_shifted_polygon,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )

    blend_button.click(
        fn=blend_images,
        inputs=[
            foreground_image_original, background_image_original, dx, dy,
            polygon_state
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
