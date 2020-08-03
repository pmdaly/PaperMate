import matplotlib.pyplot as plt
from torchvision import transforms


def save_tensor_png(img, filename):
    pil_img = transforms.ToPILImage()(img)
    plt.imsave(fname=filename, arr=pil_img)
