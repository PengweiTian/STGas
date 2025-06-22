import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import numpy as np

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])


# 显示原图和输入图
def show_batch_img(batch_data):
    # k=5,bs=8
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    # 显示原图
    frame_path = "D:\\tpw\\tpw_graduate\\dataset\\IOD-Video\\Frames"
    raw_img_list = []
    for name in batch_data["img_info"]["file_name"]:
        image_path = str(os.path.join(frame_path, name))
        image_tensor = transform(Image.open(image_path).convert('RGB'))
        raw_img_list.append(image_tensor.numpy().transpose((1, 2, 0)))
    raw_img_np = np.array(raw_img_list)

    # 显示输入图
    result_img = np.concatenate((raw_img_np, batch_data["img"][2].detach().cpu().numpy().transpose((0, 2, 3, 1))),
                                axis=0)
    for i in range(result_img.shape[0]):
        row = i // 4
        col = i % 4
        axes[row, col].imshow(result_img[i])
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()
