import os
import shutil
import random

def distribute_images_unique(folder1, folder2, output_root, n, x, y):
    assert n % 2 == 0, "n must be an even number"
    assert y < x, "y must be smaller than x"

    num_set1 = n // 2  # Normal sets
    num_set2 = n // 2  # Ambnormal sets

    total_needed_from_folder1 = (num_set1 * x) + (num_set2 * (x - y))
    total_needed_from_folder2 = num_set2 * y

    images1 = os.listdir(folder1)
    images2 = os.listdir(folder2)

    assert len(images1) >= total_needed_from_folder1, \
        f"Not enough images in folder1. Needed: {total_needed_from_folder1}, Available: {len(images1)}"
    assert len(images2) >= total_needed_from_folder2, \
        f"Not enough images in folder2. Needed: {total_needed_from_folder2}, Available: {len(images2)}"

    random.shuffle(images1)
    random.shuffle(images2)

    os.makedirs(output_root, exist_ok=True)

    idx1 = 0
    idx2 = 0

    for i in range(n):
        if i < num_set1:
            folder_name = f"normal_{i + 1}"
        else:
            folder_name = f"abnormal_{i - num_set1 + 1}"

        target_dir = os.path.join(output_root, folder_name)
        os.makedirs(target_dir, exist_ok=True)

        if i < num_set1:
            # Normal folder: x images from folder1
            selected_1 = images1[idx1:idx1 + x]
            idx1 += x
            for img in selected_1:
                shutil.copy(os.path.join(folder1, img), os.path.join(target_dir, img))
        else:
            # Abnormal folder: y from folder2, x-y from folder1
            selected_2 = images2[idx2:idx2 + y]
            idx2 += y
            selected_1 = images1[idx1:idx1 + (x - y)]
            idx1 += (x - y)
            for img in selected_2:
                shutil.copy(os.path.join(folder2, img), os.path.join(target_dir, img))
            for img in selected_1:
                shutil.copy(os.path.join(folder1, img), os.path.join(target_dir, img))

    print(f"\n✅ Images uniquely distributed into {n} folders under '{output_root}'.")


if __name__ == "__main__":
    folder1 = "path/to/normal/patches/dir"
    folder2 = "path/to/abnormal/patches/dir"
    output_root = "save/dir"
    n = 6 # Total number of bags
    x = 1000 # Total number of images in a bag
    y = 10 # Number of positive patches in a positive bag

    distribute_images_unique(folder1, folder2, output_root, n, x, y)
