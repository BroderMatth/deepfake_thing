import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML
import warnings
import os
import face_recognition
from PIL import Image
warnings.filterwarnings("ignore")

video = input('file path to video: (if possible, make 256x256 - the program can do it but not very well) ')
picture = 'C:\\Users\\TheEpicMatt14\\Desktop\\drive-download-20200824T135944Z-001\\first-order-model\\02.png'
#input('file path to picture: (if possible, make 256x256 for same reason) ')

image = face_recognition.load_image_file(picture)
face_locations = face_recognition.face_locations(image)
loop = 0

print(face_locations)
        
for coords in face_locations:
    coords = list(coords)
    coords[0], coords[1], coords[2], coords[3] = coords[3], coords[0], coords[1], coords[2]

print(face_locations)

'''
    im = Image.open(picture)
    im_crop = im.crop()
    im_crop.save(f'face{loop}.png')
    loop+=1
'''
'''
source_image = imageio.imread(picture)
driving_video = imageio.mimread(video, memtest=False)

#Resize image and video to 256x256

source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.show()

from demo import load_checkpoints
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', 
                                        checkpoint_path='vox-cpk.pth.tar')

from demo import make_animation
from skimage import img_as_ubyte

predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

#save resulting video
imageio.mimsave('../generated.mp4', [img_as_ubyte(frame) for frame in predictions])
#video can be downloaded from /content folder

display(source_image, driving_video, predictions)
'''