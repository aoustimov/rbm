import imageio
import numpy as np
import matplotlib.pyplot as pyplot

def save_merged_images(images, size, path):
    h, w = images.shape[1], images.shape[2]

    merge_img = np.zeros((h*size[0], w*size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = int(idx/size[1])
        merge_img[j * h:j * h + h, i * w:i * w + w] = image

    imageio.imwrite(path, merge_img)


def convert_to_onehot(datain, nb_classes):

    def indices_to_one_hot(data, nb_classes):
        targets = np.array(data).reshape(-1).astype(int)
        return np.transpose(np.eye(nb_classes)[targets])

    oh_list = []
    for i in range(datain.shape[0]):
        obs = datain[i,:]
        oh = indices_to_one_hot(data=obs, nb_classes=nb_classes)
        oh_list.append(oh)

    return oh_list