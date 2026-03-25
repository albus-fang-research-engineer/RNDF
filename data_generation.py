from data import DataSampler
import numpy as np
import os


if __name__ == "__main__":

    robo = DataSampler(dataset_path="dataset_new_scheme/")

    os.makedirs(robo.dataset_path, exist_ok=True)

    batch_size = 3000
    uniform_base_num = 500

    print("Starting dataset generation...")

    data = robo.batch_sample_mixed(
        batch_size=batch_size,
        base_num=uniform_base_num
    )

    print("Dataset shape:", data.shape)

    save_path = os.path.join(robo.dataset_path, "mixed_dataset.npy")
    np.save(save_path, data)

    print(f"Saved dataset to: {save_path}")