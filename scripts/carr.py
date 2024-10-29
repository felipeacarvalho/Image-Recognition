import os
from PIL import Image
import numpy as np

#função para carregar imagens e nomes
def carregarImgNomes(img_pasta, txt_nomes, img_size=(640, 480)):
    imgs = []
    nomes = []

    #abrir .txt com lista de nomes
    with open(txt_nomes, 'r') as f:
        lista_nomes = [int(line.strip()) for line in f.readlines()]

    img_lista = sorted(os.listdir(img_pasta))
    assert len(img_lista) == len(lista_nomes), "Número de imagens e nomes deve ser o mesmo"

    #assimilar imagem a nome
    for arq_img, nome in zip(img_lista, lista_nomes):
        img_dir = os.path.join(img_pasta, arq_img)
        img = Image.open(img_dir)
        img = img.resize(img_size)
        img_array = np.array(img, dtype=np.float32) / 255
        imgs.append(img_array)
        nomes.append(nome)

    return np.array(imgs), np.array(nomes)