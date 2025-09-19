from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import LeakyReLU, Dense,Conv2DTranspose,Conv2D,MaxPooling2D,BatchNormalization,add,AveragePooling2D
from tensorflow.keras.layers import Layer,concatenate,GlobalMaxPooling2D,GlobalAveragePooling2D,Lambda,Permute,Reshape,Dense,Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import random
import os


SEED = 0
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


low_res_dirs=["/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_LowRes/5by5/X_train",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_LowRes/7by7/X_train",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_LowRes/10by10/X_train",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_RandomSampled/25points/X_train_0",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_RandomSampled/25points/X_train_1",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_RandomSampled/25points/X_train_2",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_RandomSampled/49points/X_train_0",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_RandomSampled/49points/X_train_1",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_RandomSampled/49points/X_train_2",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_RandomSampled/100points/X_train_0",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_RandomSampled/100points/X_train_1",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_RandomSampled/100points/X_train_2",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_pb_RandomSampled/100points/X_train_0",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_pb_RandomSampled/100points/X_train_1",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_pb_RandomSampled/100points/X_train_2",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_pb_RandomSampled/25points/X_train_0",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_pb_RandomSampled/25points/X_train_1",
              "/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_pb_RandomSampled/25points/X_train_2",              
              ]
HighRes_train_directory="/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_highres/X_train"


#NOW LETS LOAD THE train DATASET progressively

n_input=17320
n_valid=3640
batch_size=10
# n_input=120
# n_valid=30


def training_data_generator(batch_size, low_res_dirs=low_res_dirs, high_res_dir=HighRes_train_directory):
    while True:
        for low_res_dir in low_res_dirs:
            # List all files in the current low resolution directory
            low_res_files = os.listdir(low_res_dir)
            low_res_files = [f for f in low_res_files if f.endswith('.npy')]

            # Shuffle the list of low resolution files
            np.random.shuffle(low_res_files)
            
            for i in range(0, len(low_res_files), batch_size):
                batch_files = low_res_files[i:i+batch_size]

                lowres_batch = np.array([np.load(os.path.join(low_res_dir, f)) for f in batch_files])
                lowres_batch[:, 3] = lowres_batch[:, 3] / 4  # Normalising T40

                highres_batch = np.array([np.load(os.path.join(high_res_dir, f)) for f in batch_files])
                terrain_batch = highres_batch[:, 4:8]
                terrain_batch[:, 0] = terrain_batch[:, 0] / 1500
                terrain_batch[:, 1] = terrain_batch[:, 1] / 90
                terrain_batch[:, 2] = terrain_batch[:, 2] / 2

                highres_batch = highres_batch[:, 0:4]

                yield (lowres_batch / 45, terrain_batch), highres_batch
train_data = tf.data.Dataset.from_generator(training_data_generator, ((tf.float32, tf.float32), tf.float32), args=[batch_size],
                                             output_shapes=(([batch_size, 4, 13, 13], [batch_size, 4, 416, 416]), [batch_size, 4, 416, 416]))
train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
train_data = train_data.repeat()
print(train_data.element_spec)




#NOW LETS LOAD THE Validation DATASET progressively
LowRes_directory="/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_LowRes/10by10"
HighRes_directory="/home/availab-dl2/Sandboxes/Mohammad/FireWind-Dataset/FireWind_highres/X_train"
def validation_data_generator(batch_size=batch_size,n_target=n_valid):
    for i in range(0, n_target, batch_size):
        lowres_batch = np.array([np.load(f"{LowRes_directory}/X_valid/%d.npy"%j) for j in range(i, i+batch_size)])       #lowres images for input
        lowres_batch[:,3]=lowres_batch[:,3]/4  #normalising T40


        Loaded_highres_batch = np.array([np.load(f"{HighRes_directory}/X_valid/%d.npy"%j) for j in range(i, i+batch_size)])     #highres images as target
        terrain_batch=Loaded_highres_batch[:,4:8]
        terrain_batch[:,0]=terrain_batch[:,0]/1500
        terrain_batch[:,1]=terrain_batch[:,1]/90
        terrain_batch[:,2]=terrain_batch[:,2]/2


        highres_batch=Loaded_highres_batch[:,0:4]


        yield (lowres_batch/45, terrain_batch), highres_batch
valid_data = tf.data.Dataset.from_generator(validation_data_generator, ((tf.float32,tf.float32), tf.float32), args=[batch_size],
                                             output_shapes=(([batch_size, 4, 13, 13], [batch_size, 4, 416, 416]), [batch_size, 4, 416, 416]))
valid_data = valid_data.repeat()
print(valid_data.element_spec)




@tf.keras.utils.register_keras_serializable(package="MyLayers")
class CBAM(Layer):
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.channel = input_shape[1]
        self.shared_layer_one = Dense(self.channel // self.ratio,
                                      activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.shared_layer_two = Dense(self.channel,
                                      kernel_initializer='he_normal',
                                      use_bias=True,
                                      bias_initializer='zeros')
        self.conv_layer = Conv2D(filters=1, kernel_size=self.kernel_size, strides=1, padding='same',
                                 activation='sigmoid', kernel_initializer='he_normal',
                                 use_bias=False, data_format='channels_first')
        super(CBAM, self).build(input_shape)

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D(data_format='channels_first')(inputs)
        avg_pool = Reshape((1, 1, self.channel))(avg_pool)
        avg_pool = self.shared_layer_one(avg_pool)
        avg_pool = self.shared_layer_two(avg_pool)

        max_pool = GlobalMaxPooling2D(data_format='channels_first')(inputs)
        max_pool = Reshape((1, 1, self.channel))(max_pool)
        max_pool = self.shared_layer_one(max_pool)
        max_pool = self.shared_layer_two(max_pool)

        cbam_channel = add([avg_pool, max_pool])
        cbam_channel = Activation('sigmoid')(cbam_channel)
        cbam_channel = Permute((3, 1, 2))(cbam_channel)

        avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(inputs)
        max_pool = Lambda(lambda x: tf.reduce_max(x, axis=1, keepdims=True))(inputs)
        concat = concatenate([avg_pool, max_pool], axis=1)

        cbam_spatial = self.conv_layer(concat)

        cbam_feature = tf.multiply(inputs, cbam_channel)
        cbam_feature = tf.multiply(cbam_feature, cbam_spatial)

        return cbam_feature

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({'ratio': self.ratio, 'kernel_size': self.kernel_size})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    


def Residual_Module(layer_in,n_filters):
    
    merge_input=layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    if layer_in.shape[1]!= n_filters:
        merge_input=Conv2D(n_filters,(1,1),padding='same',activation='linear',kernel_initializer='he_normal',data_format='channels_first')(layer_in)
    
    #conv1
    BN=BatchNormalization()(layer_in,training=True)
    conv1=Conv2D(n_filters,(3,3),padding='same',activation='tanh',kernel_initializer='he_normal',data_format='channels_first')(BN)
    conv1=BatchNormalization()(conv1,training=True)
    #conv2
    conv1=CBAM()(conv1)
    conv2=Conv2D(n_filters,(3,3),padding='same',activation='tanh',kernel_initializer='he_normal',data_format='channels_first')(conv1)
    conv2=BatchNormalization()(conv2,training=True)
    #skip connection
    layer_out=add([conv2,merge_input])
    #activation_function
    
    return layer_out

def encoder_block(inputs,n_filters):
    
    output=Residual_Module(inputs,n_filters)
    output=Dropout(0.05)(output,training=True)
    output=MaxPooling2D(pool_size=(2,2), data_format='channels_first')(output)
    return output
    
def decoder_block(input_tensor,skip_tensor,n_filters):
    BN=BatchNormalization()(input_tensor,training=True)
    output=Conv2DTranspose(n_filters,(2,2),strides=(2,2),padding='same',activation='linear',data_format='channels_first')(BN)
    output=concatenate([output,skip_tensor],axis=1)
    output=Dropout(0.05)(output,training=True)
    output=Residual_Module(output,n_filters)
    return output




def Architecture1():
    #input shape
    input1=Input(shape=[4,13,13])
    input2=Input(shape=[4,416,416])
    
    a=Residual_Module(input1,32)
    b=Residual_Module(input2,32)


    #encoder branch for high res temp and terrain
    e4=encoder_block(b,16)
    e3=encoder_block(e4,32)
    e2=encoder_block(e3,64)
    e1=encoder_block(e2,128)
    e0=encoder_block(e1,256)

    #decoder branch and concatenations with temp and terrain
    conc=concatenate([a,e0],axis=1)
    d1=decoder_block(conc,e1,256)
    d2=decoder_block(d1,e2,128)
    d3=decoder_block(d2,e3,64)
    d4=decoder_block(d3,e4,32)
    d5=decoder_block(d4,b,16)

    
    output=Residual_Module(d5,16)
    output=Conv2D(4,(1,1),activation='linear', kernel_initializer="he_uniform",padding="valid", data_format='channels_first')(output)#

    #wrapping model
    model=Model(inputs=[input1,input2],outputs=output)

    #compiling model 
    opt = Adam(learning_rate=0.0007,beta_1=0.9)
    model.compile( optimizer=opt, loss="mae", metrics=['mse'])   
    return model


#model plot and summary
model=Architecture1()
#model.summary()
#plot_model(model,show_shapes=True)



cp = tf.keras.callbacks.ModelCheckpoint(f"/home/availab-dl1/Desktop/trained_models_SR/SR-10by10-noCBAM.keras",
                                         monitor="val_loss", mode = 'auto', save_best_only=True )

history=model.fit(train_data, steps_per_epoch=(n_input//batch_size), validation_data=valid_data, 
                  validation_steps=(n_valid//batch_size), epochs=150,callbacks=[ cp],verbose=2)


from matplotlib import pyplot as plt
#plotting validation and training curves
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.savefig(f'/home/availab-dl1/Desktop/trained_models_SR/SR-10by10-noCBAM.keras.png')
plt.show()