import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import torch

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
# to use pretrained model, we need to load weights from two following links:
#curl -LO https://api.ngc.nvidia.com/v2/models/nvidia/quartznet15x5/versions/2/files/quartznet15x5/JasperDecoderForCTC-STEP-247400.pt
#curl -LO https://api.ngc.nvidia.com/v2/models/nvidia/quartznet15x5/versions/2/files/quartznet15x5/JasperEncoder-STEP-247400.pt
#and pass these files as decoder encoder
def load_pretrained_quartznet(enc_path, dec_path):
    model = get_QuartzNet(tflite=False)
    l = torch.load(enc_path)
    l_1 = torch.load(dec_path)
    conv_1 = model.get_layer(name = 'conv_1')
    conv_1.set_weights([l['encoder.0.mconv.0.conv.weight'].cpu().permute(2,0,1).numpy() ,
                    l['encoder.0.mconv.1.conv.weight'].cpu().permute(2,1,0).numpy() ])
    BN_1 = model.get_layer(name = 'BN-1')
    BN_1.set_weights([
    l['encoder.0.mconv.2.weight'].cpu().numpy(),   
    l['encoder.0.mconv.2.bias'].cpu().numpy(),      
    l['encoder.0.mconv.2.running_mean'].cpu().numpy(),           
    l['encoder.0.mconv.2.running_var'].cpu().numpy() 
    ])

    for i in range(1,16):
        n = 'B-'+str(i)
        b_bl = model.get_layer(name = n)
        #print(n)
        b_bl.set_weights( [
            l[('encoder.' + str(i) + '.mconv.0.conv.weight')].cpu().permute(2,0,1).numpy(),
            l[('encoder.' + str(i) + '.mconv.1.conv.weight')].cpu().permute(2,1,0).numpy(),
            l[('encoder.' + str(i) + '.mconv.2.weight')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.2.bias')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.5.conv.weight')].cpu().permute(2,0,1).numpy(),
            l[('encoder.' + str(i) + '.mconv.6.conv.weight')].cpu().permute(2,1,0).numpy(),
            l[('encoder.' + str(i) + '.mconv.7.weight')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.7.bias')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.10.conv.weight')].cpu().permute(2,0,1).numpy(),
            l[('encoder.' + str(i) + '.mconv.11.conv.weight')].cpu().permute(2,1,0).numpy(),
            l[('encoder.' + str(i) + '.mconv.12.weight')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.12.bias')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.15.conv.weight')].cpu().permute(2,0,1).numpy(),
            l[('encoder.' + str(i) + '.mconv.16.conv.weight')].cpu().permute(2,1,0).numpy(),
            l[('encoder.' + str(i) + '.mconv.17.weight')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.17.bias')].cpu().numpy(),


            l[('encoder.' + str(i) + '.mconv.2.running_mean')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.2.running_var')].cpu().numpy(),

            l[('encoder.' + str(i) + '.mconv.7.running_mean')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.7.running_var')].cpu().numpy(),

            l[('encoder.' + str(i) + '.mconv.12.running_mean')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.12.running_var')].cpu().numpy(),

            l[('encoder.' + str(i) + '.mconv.17.running_mean')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.17.running_var')].cpu().numpy(),


            l[('encoder.' + str(i) + '.mconv.20.conv.weight')].cpu().permute(2,0,1).numpy(),
            l[('encoder.' + str(i) + '.mconv.21.conv.weight')].cpu().permute(2,1,0).numpy(),
            l[('encoder.' + str(i) + '.mconv.22.weight')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.22.bias')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.22.running_mean')].cpu().numpy(),
            l[('encoder.' + str(i) + '.mconv.22.running_var')].cpu().numpy(),


            l[('encoder.' + str(i) + '.res.0.0.conv.weight')].cpu().permute(2,1,0).numpy(),
            l[('encoder.' + str(i) + '.res.0.1.weight')].cpu().numpy(),
            l[('encoder.' + str(i) + '.res.0.1.bias')].cpu().numpy(),
            l[('encoder.' + str(i) + '.res.0.1.running_mean')].cpu().numpy(),
            l[('encoder.' + str(i) + '.res.0.1.running_var')].cpu().numpy()
        ])
        
    conv_2 = model.get_layer(name = 'conv_2')
    conv_2.set_weights([l['encoder.16.mconv.0.conv.weight'].cpu().permute(2,0,1).numpy() ,
                      l['encoder.16.mconv.1.conv.weight'].cpu().permute(2,1,0).numpy() ])
                      
    BN_2 = model.get_layer(name = 'BN-2')
    BN_2.set_weights([
    l['encoder.16.mconv.2.weight'].cpu().numpy(),   
    l['encoder.16.mconv.2.bias'].cpu().numpy(),      
    l['encoder.16.mconv.2.running_mean'].cpu().numpy(),           
    l['encoder.16.mconv.2.running_var'].cpu().numpy() 
    ])
    conv_3 = model.get_layer(name = 'conv_3')
    conv_3.set_weights([l['encoder.17.mconv.0.conv.weight'].cpu().permute(2,1,0).numpy()  
                      ] )
    BN_3 = model.get_layer(name = 'BN-3')
    BN_3.set_weights([
    l['encoder.17.mconv.1.weight'].cpu().numpy(),   
    l['encoder.17.mconv.1.bias'].cpu().numpy(),      
    l['encoder.17.mconv.1.running_mean'].cpu().numpy(),           
    l['encoder.17.mconv.1.running_var'].cpu().numpy() 
    ])

    conv_4 = model.get_layer(name = 'conv_4')
    conv_4.set_weights([l_1['decoder_layers.0.weight'].cpu().permute(2,1,0).numpy() ,
                      l_1['decoder_layers.0.bias'].cpu().numpy() ])
    return model