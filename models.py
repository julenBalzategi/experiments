from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def unet(pretrained_weights="None", input_size=(256, 256, 1), final_activation="sigmoid", classes=1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv1')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv2')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    ##200
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv1')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv2')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    ##100
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv1')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv2')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    ##50
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv1')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv2')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    ##25


    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    if pretrained_weights != "None":
        vgg16 = Model(inputs, conv5)
        vgg16.load_weights(pretrained_weights, by_name=True)
        # vgg16.trainable = False

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation=final_activation)(conv9)

    model = Model(input=inputs, output=conv10)


    return model

def unet_reduced(pretrained_weights="None", input_size=(256, 256, 1), final_activation="sigmoid", classes=1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv1')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv2')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv1')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv2')(conv2)
    conv2 = BatchNormalization()(conv2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)


    if pretrained_weights != "None":
        vgg16 = Model(inputs, conv2)
        vgg16.load_weights(pretrained_weights, by_name=True)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(pool2))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation=final_activation)(conv9)

    model = Model(input=inputs, output=conv10)


    return model

def unet_deconv(pretrained_weights="None", input_size=(256, 256, 1), final_activation="sigmoid", classes=1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv1')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv2')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding="same")(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv1')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv2')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv1')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv2')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv1')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv2')(conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)


    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)
    # 25
    if pretrained_weights != "None":
        vgg16 = Model(inputs, conv5)
        vgg16.load_weights(pretrained_weights, by_name=True)
        # vgg16.trainable = False

    up6 = Conv2DTranspose(512, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation=final_activation)(conv9)

    model = Model(input=inputs, output=conv10)


    return model

def multi_class_unet(pretrained_weights="None", input_size=(256, 256, 1), final_activation="softmax", classes=1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv1')(
        inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv2')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    ##200
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv1')(
        pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv2')(
        conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    ##100
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv1')(
        pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv2')(
        conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    ##50
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv1')(
        pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv2')(
        conv4)
    conv4 = BatchNormalization()(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    ##25

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    if pretrained_weights != "None":
        vgg16 = Model(inputs, conv5)
        vgg16.load_weights(pretrained_weights, by_name=True)
        # vgg16.trainable = False

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(classes, 1, activation=final_activation)(conv9)

    model = Model(input=inputs, output=conv10)

    return model

def unet_dropout(pretrained_weights="None", input_size=(256, 256, 1), final_activation="sigmoid", classes=1):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv1')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv2')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    ##200
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv1')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv2')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)
    ##100
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv1')(drop2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv2')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.5)(pool3)
    ##50
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv1')(drop3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block4_conv2')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.5)(pool4)
    ##25


    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    if pretrained_weights != "None":
        vgg16 = Model(inputs, conv5)
        vgg16.load_weights(pretrained_weights, by_name=True)
        # vgg16.trainable = False

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    drop6 = Dropout(0.5)(conv6)
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    drop7 = Dropout(0.5)(conv7)
    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    drop8 = Dropout(0.5)(conv8)
    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation=final_activation)(conv9)

    model = Model(inputs=inputs, outputs=conv10)


    return model

def unet_attention(pretrained_weights="None", input_size=(256, 256, 1), final_activation="sigmoid", classes=1):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', name='block1_conv1')(inputs)
    conv1 = LeakyReLU()(conv1)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal', name='block1_conv2')(conv1)
    conv1 = LeakyReLU()(conv1)
    conv1 = attention_module(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    ##200
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='block2_conv1')(pool1)
    conv2 = LeakyReLU()(conv2)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal', name='block2_conv2')(conv2)
    conv2 = LeakyReLU()(conv2)
    conv2 = attention_module(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    ##100
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='block3_conv1')(pool2)
    conv3 = LeakyReLU()(conv3)
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal', name='block3_conv2')(conv3)
    conv3 = LeakyReLU()(conv3)
    conv3 = attention_module(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    ##50
    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='block4_conv1')(pool3)
    conv4 = LeakyReLU()(conv4)
    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal', name='block4_conv2')(conv4)
    conv4 = LeakyReLU()(conv4)
    conv4 = attention_module(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    ##25


    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    drop5 = Dropout(0.5)(conv5)

    if pretrained_weights != "None":
        vgg16 = Model(inputs, conv5)
        vgg16.load_weights(pretrained_weights, by_name=True)
        # vgg16.trainable = False

    up6 = Conv2D(256, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = LeakyReLU()(conv6)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU()(conv6)
    conv6 = BatchNormalization()(conv6)
    up7 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = LeakyReLU()(conv7)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU()(conv7)
    conv7 = BatchNormalization()(conv7)
    up8 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = LeakyReLU()(conv8)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = LeakyReLU()(conv8)
    conv8 = BatchNormalization()(conv8)
    up9 = Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = LeakyReLU()(conv9)
    conv9 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv10 = Conv2D(1, 1, activation=final_activation)(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

def attention_module(input):

    b, w, h, channels = input.get_shape().as_list()
    #channel attention
    ch_a_1 = GlobalAveragePooling2D()(input)
    ch_a_1 = Dense(channels)(ch_a_1)
    ch_a_1 = LeakyReLU()(ch_a_1)
    ch_a_1 = Dense(channels, activation="sigmoid")(ch_a_1)
    ch_a_2 = GlobalAveragePooling2D()(input)
    ch_a_2 = Dense(channels)(ch_a_2)
    ch_a_2 = LeakyReLU()(ch_a_2)
    ch_a_2 = Dense(channels, activation="sigmoid")(ch_a_2)

    ch_a_added = Add()([ch_a_1, ch_a_2])

    #spatial attention
    sp_a_1 = Conv2D(filters=channels//2, kernel_size=(1, 5), padding='same')(input)
    sp_a_1 = BatchNormalization()(sp_a_1)
    sp_a_1 = LeakyReLU()(sp_a_1)
    sp_a_1 = Conv2D(1, kernel_size=(1, 5), padding='same')(sp_a_1)
    sp_a_1 = BatchNormalization()(sp_a_1)
    sp_a_1 = LeakyReLU()(sp_a_1)

    sp_a_2 = Conv2D(filters=channels//2, kernel_size=(5, 1), padding='same')(input)
    sp_a_2 = BatchNormalization()(sp_a_2)
    sp_a_2 = LeakyReLU()(sp_a_2)
    sp_a_2 = Conv2D(1, kernel_size=(5, 1), padding='same')(sp_a_2)
    sp_a_2 = BatchNormalization()(sp_a_2)
    sp_a_2 = LeakyReLU()(sp_a_2)
    sp_a_added = Add()([sp_a_1, sp_a_2])
    sp_a_added = Dense(channels, activation="sigmoid")(sp_a_added)

    ch_a_added = Multiply()([input, ch_a_added])
    sp_a_added = Multiply()([input, sp_a_added])

    return Add()([ch_a_added, sp_a_added])