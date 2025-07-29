import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio  #helps to convert numpy array to a GIF
import gdown  #to grab data out of google drive




physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass


# # GRID DATASET
# 2 datasets - TO LOAD VIDEOS AND PRPROCESS ANNOTATIONS


def load_video(path:str) -> List[float]:

    cap = cv2.VideoCapture(path)
    frames= []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()

    

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames-mean) , tf.float32)/std

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789"]

char_to_num = tf.keras.layers.StringLookup(vocabulary= vocab , oov_token= "")
num_to_char = tf.keras.layers.StringLookup(
vocabulary= char_to_num.get_vocabulary(), oov_token="", invert = True
     )

# print(
#         f"The vocabulary is : {char_to_num.get_vocabulary()}"
#         f"(size= {char_to_num.vocabulary_size()})"
#      )


def load_alignments(path:str)-> List[str]:
    with open(path, 'r') as f:
            lines = f.readlines()
    tokens = [] 
    for line in lines:
        line = line.split()
        if(line[2] != 'sil'):
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding= 'UTF-8'), (-1)))[1:]


def load_data(path: str): 
    path = bytes.decode(path.numpy())
    #file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments



test_path = '.\\data\\alignments\\s1\\bbal6n.mpg'
tf.convert_to_tensor(test_path).numpy().decode('utf-8').split('\\')[-1].split('.')[0]


frames, alignments = load_data(tf.convert_to_tensor(test_path))
frame = frames[0]  # Tensor of shape (H, W, 1)
plt.imshow(tf.squeeze(frame), cmap='gray')  # Remove extra channel dim
plt.axis('off')
plt.title("First Frame")
plt.show()


print([bytes.decode(x) for x in num_to_char(alignments.numpy()).numpy()])


def mappable_function(path: str) ->List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result

#CREATE DATA PIPELINE
data= tf.data.Dataset.list_files('.\\data\\s1\\*.mpg')
data = data.shuffle(500, reshuffle_each_iteration= False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes = ([75, None , None, None], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

train = data.take(450)
test = data.skip(450)

len(test)

frames , alignments = data.as_numpy_iterator().next()

len(frames)

sample = data.as_numpy_iterator()

val = sample.next(); val[0]

frames = val[0][1]  # list of arrays

processed_frames = []
for i, frame in enumerate(frames):
    frame = np.squeeze(frame)  # (1,1,1) -> (1,1) or (H,W,C)
    
    # Ensure dtype is uint8
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)

    # Resize to a usable dimension (e.g. 128x128)
    if frame.ndim == 2:
        frame = cv2.resize(frame, (128, 128))  # grayscale
    elif frame.ndim == 3 and frame.shape[2] in [1, 3]:
        frame = cv2.resize(frame, (128, 128))
        if frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)  # convert to RGB
    else:
        print(f"Skipping invalid frame {i} with shape: {frame.shape}")
        continue

    processed_frames.append(frame)

# Save as gif
imageio.mimsave('./animation.gif', processed_frames, fps=10)


