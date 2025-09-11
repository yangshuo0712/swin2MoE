import os
import sys

import matplotlib.pyplot as plt
import torch

# Add the project root directory to the Python path to ensure modules can be found.
# This script is assumed to be located in 'src/probe/'.
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, project_root)

# Import the probe generation functions from the local 'probe' package.
from src.probe.probe_data import make_checker_probe, make_stripe_probe


def tensor_to_img_for_display(tensor: torch.Tensor):
    """
    Converts a PyTorch tensor into a NumPy array suitable for image display.

    Args:
        tensor (torch.Tensor): The input tensor, expected in CHW format.

    Returns:
        np.ndarray: A displayable NumPy image array.
    """
    # If the tensor has 3 or 4 channels (like RGB or RGBA),
    # select the first channel for grayscale visualization.
    if tensor.dim() == 3 and tensor.shape[0] in [3, 4]:
        tensor = tensor[0, :, :]
    elif tensor.dim() != 2:
        raise ValueError(f"Unsupported tensor shape for display: {tensor.shape}")
    return tensor.cpu().numpy()


def main():
    """
    Main function to generate and save static probe images for visualization.
    """
    # Define the output directory in the project root.
    output_dir = os.path.join(project_root, "generated_probes_static_view")
    os.makedirs(output_dir, exist_ok=True)
    device = "cpu"

    # --- Generate and Save Stripe Probe ---
    print("Generating stripe probe...")
    stripe_img, stripe_ftrue, stripe_masks = make_stripe_probe(device=device)

    # Save the main probe image.
    plt.imsave(
        os.path.join(output_dir, "stripe_probe.png"),
        tensor_to_img_for_display(stripe_img.squeeze(0)),
        cmap="gray",
    )

    # Save the ground truth frequency map. Brighter colors indicate higher frequencies.
    plt.imsave(
        os.path.join(output_dir, "stripe_ftrue.png"),
        stripe_ftrue.cpu().numpy(),
        cmap="viridis",
    )

    # Create and save a combined visualization of the distortion masks.
    combined_masks_stripe = torch.zeros_like(stripe_masks["noise"]).float()
    combined_masks_stripe[stripe_masks["noise"]] = 0.33  # Dark gray for noise ROI
    combined_masks_stripe[stripe_masks["blur"]] = 0.66  # Light gray for blur ROI
    combined_masks_stripe[stripe_masks["clean"]] = 1.0  # White for the clean area
    plt.imsave(
        os.path.join(output_dir, "stripe_masks.png"),
        combined_masks_stripe.cpu().numpy(),
        cmap="gray",
    )

    # --- Generate and Save Checkerboard Probe ---
    print("Generating checkerboard probe...")
    checker_img, checker_ftrue, checker_masks = make_checker_probe(device=device)

    # Save the main checkerboard image.
    plt.imsave(
        os.path.join(output_dir, "checker_probe.png"),
        tensor_to_img_for_display(checker_img.squeeze(0)),
        cmap="gray",
    )

    # Save the ground truth frequency map (constant frequency for checkerboard).
    plt.imsave(
        os.path.join(output_dir, "checker_ftrue.png"),
        checker_ftrue.cpu().numpy(),
        cmap="viridis",
    )

    # Create and save a combined visualization of the distortion masks.
    combined_masks_checker = torch.zeros_like(checker_masks["noise"]).float()
    combined_masks_checker[checker_masks["noise"]] = 0.33
    combined_masks_checker[checker_masks["blur"]] = 0.66
    combined_masks_checker[checker_masks["clean"]] = 1.0
    plt.imsave(
        os.path.join(output_dir, "checker_masks.png"),
        combined_masks_checker.cpu().numpy(),
        cmap="gray",
    )

    print(f"\nSuccessfully generated images in the '{output_dir}' directory:")
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".png"):
            print(f"- {fname}")


if __name__ == "__main__":
    main()
