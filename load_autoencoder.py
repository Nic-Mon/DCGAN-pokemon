from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras.models import Model
from sklearn.cross_validation import train_test_split
import numpy as np
from keras import backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
from keras.models import load_model
autoencoder = load_model('poke_autoencoder.h5')

# Load in images
img_rows, img_cols = 64, 64
path2 = "/home/ubuntu/stuff_to_keep/pokemon"
imlist = os.listdir(path2)
imlist = np.sort(imlist).tolist()
immatrix = np.array(
    [np.array(Image.open(path2 + '/' + im2)).flatten()
              for im2 in imlist],
    'f')

num_samples = np.size(imlist) # 100
label=np.ones((num_samples,),dtype = int)
from sklearn.utils import shuffle
data,label = shuffle(immatrix,label, random_state=2)
train_data = [data,label]
# Separate data into images and labels
(X, y) = (train_data[0],train_data[1])
# Separate data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
# Resize X
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print "X_test shape:", X_test.shape
#(x_train, _), (x_test, _) = mnist.load_data()

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
x_test = np.reshape(X_test, (len(X_test), img_rows,img_cols, 3))  # adapt this if using `channels_first` image data format
x_train = np.reshape(X_train, (len(X_train), img_rows,img_cols, 3))


X = X.reshape(X.shape[0], img_rows, img_cols, 3).astype('float32')
X /= 255
print "X shape:{}".format(X.shape)
decoded_imgs = autoencoder.predict(X)
np.save('decoded_imgs', decoded_imgs)
print "decoded"
print "shape of decoded images: {}".format(decoded_imgs.shape)
n = 1
plt.figure(figsize=(10,10))
print "plt.figure"
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(X[4].reshape(64,64,3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[4].reshape(64,64,3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#plt.show()
plt.suptitle('Reconstructed Images with Convolutional Autoencoder')
plt.savefig('im4.png')
plt.close('all')

#x2 = autoencoder.get_layer(index=0)
input_img = Input(shape=(img_rows, img_cols, 3))
x2 = autoencoder.get_layer(index=1)(input_img)
x2 = autoencoder.get_layer(index=2)(x2)
x2 = autoencoder.get_layer(index=3)(x2)
x2 = autoencoder.get_layer(index=4)(x2)
x2 = autoencoder.get_layer(index=5)(x2)
encoded = autoencoder.get_layer(index=6)(x2)
encoder = Model(input_img, encoded)
print encoder.summary()

encoded_imgs = encoder.predict(X)
print "Predicting  encoded representations"
# now representation is (8, 8, 8) (512-dimensional)
print "size of encoded images: {}".format(encoded_imgs.shape)
np.save('encoded_imgs',encoded_imgs)
n = 5
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(8, 8 * 8).T)
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.suptitle('Visualization of encoded representations')
plt.savefig('encoded1.png')
plt.close('all')


# Test averaging 
im1 = np.array(encoded_imgs[0,:,:,:])
print "im1 shape:", im1.shape
print "shape encoded", encoded_imgs[0].shape
print np.array(im1).shape
im2 = np.array(encoded_imgs[1,:,:,:])
#new_im = np.mean(im1,im2)
#print "new_im shape: {}".format(new_im.shape)







# Save encoded images
n = encoded_imgs.shape[0]
for i in range(n):
    
    plt.imshow(encoded_imgs[i].reshape(8, 8 * 8).T)
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('encoded' + str(i) + '.png')
plt.close('all')

