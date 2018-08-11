from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation,Add,Reshape,BatchNormalization
from keras.regularizers import l2
from keras.models import Model

weight_decay = 0.005

def sptial_filter(x,filters):
    x = Conv3D(filters,(1,1,1),strides=(1,1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters,(3,3,1),strides=(1,1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def temporal_filter(x,filters):
    x = Conv3D(filters,(1,1,3),strides=(1,1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters,(1,1,1),strides=(1,1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x 

def convblock1_1(x,filters):
    x = Conv3D(filters,(1,1,1),strides=(1,1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def p3d(inputs,filters):
    x1 = sptial_filter(inputs,filters)
    x1 = temporal_filter(x1,filters)  
    x = Add()([inputs, x1])    
    return x
def c3d_model():
    input_shape = (90,120,16,3)
    
    nb_classes = 45
    inputs = Input(input_shape)

    x = sptial_filter(inputs,32)
    x = temporal_filter(x,32)

    x = p3d(x,32)
    x = p3d(x,32)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)			 #45*60*16*64

    x = convblock1_1(x,64)
    x = p3d(x,64)
    x = p3d(x,64)       							 
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)                     #23*30*8*128 

    x = convblock1_1(x,128)
    x = p3d(x,128)  
    x = p3d(x,128)       							 
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)                     #12*15*4*128

    x = convblock1_1(x,256)
    x = p3d(x,256)
    x = p3d(x,256)          							 
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)                     #6*8*2*128

    x = p3d(x,256)
    x = p3d(x,256)        							 
    x = MaxPool3D((2,2,2),strides=(2,2,2), padding='same')(x)                    #3*4*1*128
    
    x = Conv3D(512,(3,3,1),strides=(1,1,1),padding='valid',                      
               activation='relu')(x)         
    x = Conv3D(nb_classes,(1,2,1),strides=(1,1,1),padding='valid',                      
               activation='softmax')(x)   
    x = Reshape([nb_classes])(x)


    model = Model(inputs, x)
    return model
