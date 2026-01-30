import cv2
import time

import numpy as np
import matplotlib

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image



def process_frames(input_queue, output_queue, running):
    print("Loading model")
    import torch
    from transformers import Sam3Processor, Sam3Model
    model = Sam3Model.from_pretrained("facebook/sam3").to("cuda")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("Model loaded successfully")
    while running.value:
        if not input_queue.empty():
            frame = input_queue.get()
            inputs = processor(images=frame, text="person", return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            # Example computation: convert to grayscale
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output_queue.put(processed_frame)
        # time.sleep(0.03)  # Simulate processing rate

if __name__ == "__main__":
    pass  # This file is meant to be imported
