import io
import os
import zipfile
import numpy as np
from PIL import Image

ZIP = "datos/anime_faces.zip"
TAM = 64

z = zipfile.ZipFile(ZIP)
nombres = sorted(n for n in z.namelist() if n.endswith(".jpg"))
imgs = np.zeros((len(nombres), TAM, TAM, 3), dtype=np.uint8)
años = np.zeros(len(nombres), dtype=np.int16)
for i, n in enumerate(nombres):
    im = Image.open(io.BytesIO(z.read(n))).convert("RGB").resize((TAM, TAM), Image.LANCZOS)
    imgs[i] = np.asarray(im)
    años[i] = int(n.split("_")[-1][:4])

os.makedirs("datos", exist_ok=True)
np.savez("datos/anime64.npz", imgs=imgs, años=años)
print(imgs.shape, años.min(), años.max())
