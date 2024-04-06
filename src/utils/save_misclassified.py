import os
from torchvision.utils import save_image


def save_misclassified_images(misclassified_examples, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    for i, example in enumerate(misclassified_examples):
        img_tensor = example["data"]
        true_label = example["true_label"]
        predicted_label = example["predicted_label"]

        img_tensor = img_tensor.cpu().clone()
        img_tensor = img_tensor.squeeze(0)  # Remove batch dimension if present

        filename = f"{i}_True_{true_label}_Pred_{predicted_label}.png"
        filepath = os.path.join(save_dir, filename)

        save_image(img_tensor, filepath)
        print(f"Saved misclassified image: {filepath}")
