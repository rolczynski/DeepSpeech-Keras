import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class Small_block(tf.keras.Model):
    def __init__(self, kernel_size, filters,residual=False):
        super(Small_block, self).__init__(name='small_block')
        self.conv = layers.SeparableConv1D(filters,kernel_size,padding='same', use_bias = False)
        self.bn = layers.BatchNormalization()
        self.residual = residual
        self.relu = layers.ReLU()
    def call(self, input_tensor,residual_value, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x,training=training)
        if self.residual:
            x += residual_value
        x = self.relu(x)
        return x
    
class B_block(tf.keras.Model):
    def __init__(self, kernel_size, filters,n_small_blocks , name ):
        super(B_block, self).__init__(name= name)
        self.small_blocks = []
        for i in range(n_small_blocks - 1):
            self.small_blocks.append(Small_block(kernel_size, filters  ))
        self.res_block = Small_block(kernel_size, filters,residual=True)
        self.conv = layers.Conv1D(filters,1,padding='same' , use_bias = False )
        self.bn = layers.BatchNormalization()
    def call(self,x,training=False):
        residual_value = self.conv(x)
        residual_value = self.bn(residual_value,training=training)
        for i in range(len(self.small_blocks)):
            x = self.small_blocks[i](x, None, training=training)
        x = self.res_block(x, residual_value, training=training)
        return x
 
def get_QuartzNet(tflite=True,num_b_blocks=3):#num_b_blocks^ 1 is 5x5 quartznet, 2 is 10x5, 3 is 15x5
    if tflite:
        input_tensor = layers.Input([100, 64], name='X')
    else:
        input_tensor = layers.Input([None, 64], name='X')
    x = layers.SeparableConv1D(256,33,padding='same',strides=2,name='conv_1', use_bias = False)(input_tensor)
    x = layers.BatchNormalization(name = 'BN-1')(x)
    x = layers.ReLU(name = 'RELU-1')(x)

    x = B_block(33,256,5 , 'B-1')(x)
    if num_b_blocks > 1:
        x = B_block(33,256,5 , 'B-2')(x)
    if num_b_blocks > 2:
        x = B_block(33,256,5 , 'B-3')(x)

    x = B_block(39,256,5, 'B-4')(x)
    if num_b_blocks > 1:
        x = B_block(39,256,5, 'B-5')(x)
    if num_b_blocks > 2:
        x = B_block(39,256,5, 'B-6')(x)


    x = B_block(51,512,5, 'B-7')(x)
    if num_b_blocks > 1:
        x = B_block(51,512,5, 'B-8')(x)
    if num_b_blocks > 2:
        x = B_block(51,512,5, 'B-9')(x)

    x = B_block(63,512,5, 'B-10')(x)
    if num_b_blocks > 1:
        x = B_block(63,512,5, 'B-11')(x)
    if num_b_blocks > 2:
        x = B_block(63,512,5, 'B-12')(x)

    x = B_block(75,512,5, 'B-13')(x)
    if num_b_blocks > 1:
        x = B_block(75,512,5, 'B-14')(x)
    if num_b_blocks > 2:
        x = B_block(75,512,5, 'B-15')(x)

    x = layers.SeparableConv1D(512,87,padding='same',name='conv_2' , use_bias = False)(x)
    x = layers.BatchNormalization(name = 'BN-2')(x)
    x = layers.ReLU(name = 'RELU-2')(x)
    x = layers.Conv1D(1024,1,padding='same',name='conv_3' , use_bias = False)(x)
    x = layers.BatchNormalization(name = 'BN-3')(x)
    x = layers.ReLU(name = 'RELU-3')(x)
    if tflite: #correct dilation is 2, but tf cant convert it correctli to tflite
        x = layers.Conv1D(29,1,padding='same',dilation_rate=1,name='conv_4')(x)
    else:
        x = layers.Conv1D(29,1,padding='same',dilation_rate=2,name='conv_4')(x)
    model = tf.keras.Model([input_tensor], x, name='QuartzNet')
    return model