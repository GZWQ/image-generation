from keras.models import Model, Input, Sequential,model_from_json
from keras import layers
# from keras_contrib.layers.normalization import InstanceNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
from keras.backend import tf as ktf
from utils.stn import SpatialTransformer
import numpy as np
from tqdm import tqdm
from dataloader import Dataloader
import os,keras
from keras.models import load_model
from keras.optimizers import Adam
from utils.pose_transform import AffineTransformLayer
from utils.layer_utils import content_features_model
import cv2

class PoseGAN():
    def __init__(self,cfg):

        ########## Loss Setting #########
        self._l1_penalty_weight = cfg.l1_penalty_weight
        self._content_loss_layer = cfg.content_loss_layer
        self._gan_penalty_weight = cfg.gan_penalty_weight
        self._tv_penalty_weight = cfg.tv_penalty_weight
        self._nn_loss_area_size = cfg.nn_loss_area_size
        self._lstruct_penalty_weight = cfg.lstruct_penalty_weight
        self._mae_weight = cfg.mae_weight
        self._pose_estimator = load_model(cfg.pose_estimator)

        ##########General Setting########
        self.im_size = cfg.im_size
        self.use_warp = cfg.use_warp
        self.warp_agg = cfg.warp_agg
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size
        self.dataset_name = cfg.dataset_name
        self.display_ratio = cfg.display_ratio
        
        common_path = '{}/l1_{}/tv_{}/struct_{}/mae_{}/'.format(self.dataset_name,self._l1_penalty_weight,self._tv_penalty_weight,self._lstruct_penalty_weight,self._mae_weight)
        self.checkpoint_ratio = cfg.checkpoint_ratio
        self.checkpoints_dir = cfg.checkpoints_dir+'{}/'.format(self.dataset_name)
        self.output_dir = cfg.output_dir+'{}/l1_{}/tv_{}/struct_{}/mae_{}'.format(self.dataset_name,
                                                                                        self._l1_penalty_weight,
                                                                                        #self._tv_penalty_weight,
                                                                                        self._nn_loss_area_size,
                                                                                        self._lstruct_penalty_weight,
                                                                                        self._mae_weight)

        self.checkpoints_dir = cfg.checkpoints_dir+common_path
        self.output_dir = cfg.output_dir+common_path
        print(self.checkpoints_dir)
        os.makedirs(self.checkpoints_dir,exist_ok=True)
        os.makedirs(self.output_dir,exist_ok=True)

        ########### #############

        self.nfilters_decoder = (512, 512, 512, 256, 128, 3)
        self.nfilters_encoder = (64, 128, 256, 512, 512, 512)

        self.dataset = Dataloader(cfg)
        opt_g = Adam(2e-4,0.5,0.999)
        opt_d = Adam(2e-4,0.5,0.999)

        ############# Train Discriminator ###########

        self.discriminator = self.make_discriminator()
        self._generator = self.make_generator()

        self._set_trainable(self._generator, False)
        self._set_trainable(self.discriminator, True)
        self.discriminator.compile(loss=['binary_crossentropy'],optimizer=opt_d,metrics=['accuracy'])

        ############# Train Generator ###########

        self._set_trainable(self._generator, True)
        self._set_trainable(self.discriminator, False)

        input_img = Input([self.im_size[0], self.im_size[1], 3])
        input_pose = Input([self.im_size[0], self.im_size[1], 18])
        target_img = Input([self.im_size[0], self.im_size[1], 3])
        target_pose = Input([self.im_size[0], self.im_size[1], 18])

        if self.use_warp == 'full':
            warp = [Input((1, 8))]
        elif self.use_warp == 'mask':
            warp = [Input((10, 8)), Input((10, self.im_size[0], self.im_size[1]))]
        elif self.use_warp == 'stn':
            warp = [Input((72,))]
        else:
            warp = []

        fake_imgs = self._generator([input_img, input_pose, target_pose]+warp)
        pred = self.discriminator([fake_imgs,input_pose,target_img,target_pose])

        self.generator = Model([input_img, input_pose, target_img,target_pose]+warp,[pred,fake_imgs])
        # self.generator.compile(loss=['binary_crossentropy',self.gene_loss],loss_weights=[1,1],optimizer=opt_g)
        self.generator.compile(loss=['binary_crossentropy', self.gene_loss], loss_weights=[1, 1], optimizer=opt_g)


    def _set_trainable(self, net, trainable):
        for layer in net.layers:
            layer.trainable = trainable
        net.trainable = trainable

    def block(self,x,f,down=True,bn=True,dropout=False,leaky=True):
        if leaky:
            x = LeakyReLU(0.2)(x)
        else:
            x = layers.Activation('relu')(x)
        if down:
            x = layers.ZeroPadding2D()(x)
            x = layers.Conv2D(f,kernel_size=4,strides=2,use_bias=False)(x)
        else:
            x = layers.Conv2DTranspose(f,kernel_size=4,strides=2,use_bias=False)(x)
            x = layers.Cropping2D((1,1))(x)
        if bn:
            x = InstanceNormalization()(x)
        if dropout:
            x = layers.Dropout(0.5)(x)
        return x

    def encoder(self,ins,nfilters=(64,128,256,512,512,512)):
        _layers = []
        if len(ins) != 1:
            x = layers.Concatenate(axis=-1)(ins)
        else:
            x = ins[0]
        for i,nf in enumerate(nfilters):
            if i==0:
                x = layers.Conv2D(nf,kernel_size=3,padding='same')(x)
            elif i==len(nfilters)-1:
                x = self.block(x,nf,bn=False)
            else:
                x = self.block(x,nf)
            _layers.append(x)
        return _layers

    def decoder(self,skips,nfilters=(64,128,256,512,512,512)):
        x = None
        for i,(skip,nf) in enumerate(zip(skips,nfilters)):
            if 0<i<3:
                x = layers.Concatenate(axis=-1)([x,skip])
                x = self.block(x,nf,down=False,leaky=False,dropout=True)
            elif i==0:
                x = self.block(skip,nf,down=False,leaky=False,dropout=True)
            elif i== len(nfilters)-1:
                x = layers.Concatenate(axis=-1)([x,skip])
                x = layers.Activation('relu')(x)
                x = layers.Conv2D(nf,kernel_size=3,use_bias=True,padding='same')(x)
            else:
                x = layers.Concatenate(axis=-1)([x,skip])
                x = self.block(x,nf,down=False,leaky=False)
        x = layers.Activation('tanh')(x)
        return x

    def concatenate_skips(self,skips_app,skips_pose,warp):
        skips = []
        if self.use_warp == 'stn':
            b = np.zeros((2, 3), dtype='float32')
            b[0, 0] = 1
            b[1, 1] = 1
            W = np.zeros((32, 6), dtype='float32')
            weights = [W, b.flatten()]

            locnet = Sequential()
            locnet.add(layers.Dense(64, input_shape=(72,)))
            locnet.add(LeakyReLU(0.2))
            locnet.add(layers.Dense(32))
            locnet.add(LeakyReLU(0.2))
            locnet.add(layers.Dense(6, weights=weights))

        for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
            if i < 4:
                if self.use_warp != 'stn':
                    out = AffineTransformLayer(10 if self.use_warp == 'mask' else 1, self.warp_agg, (self.im_size[0],self.im_size[1]))([sk_app] + warp)
                else:
                    out = SpatialTransformer(locnet, K.int_shape(sk_app)[1:3])(warp + [sk_app])
                out = layers.Concatenate(axis=-1)([out, sk_pose])
            else:
                out = layers.Concatenate(axis=-1)([sk_app, sk_pose])
            skips.append(out)
        return skips

    def make_generator(self):
        use_warp_skip = self.use_warp != 'none'
        input_img = Input([self.im_size[0], self.im_size[1], 3])
        input_pose = Input([self.im_size[0], self.im_size[1], 18])
        target_pose = Input([self.im_size[0], self.im_size[1], 18])

        if self.use_warp == 'full':
            warp = [Input((1, 8))]
        elif self.use_warp == 'mask':
            warp = [Input((10, 8)), Input((10, self.im_size[0], self.im_size[1]))]
        elif self.use_warp == 'stn':
            warp = [Input((72,))]
        else:
            warp = []

        if use_warp_skip:
            enc_app_layers = self.encoder([input_img] + [input_pose], self.nfilters_encoder)
            enc_tg_layers = self.encoder([target_pose] , self.nfilters_encoder)
            enc_layers = self.concatenate_skips(enc_app_layers, enc_tg_layers, warp)
        else:
            enc_layers = self.encoder([input_img] + [input_pose] + [target_pose], self.nfilters_encoder)

        out = self.decoder(enc_layers[::-1],self.nfilters_decoder)

        model = Model([input_img,input_pose,target_pose]+warp,[out])
        # model.summary()
        return model

    def make_discriminator(self):
        input_img = Input([self.im_size[0],self.im_size[1],3])
        input_pose = Input([self.im_size[0],self.im_size[1],18])
        target_img = Input([self.im_size[0],self.im_size[1],3])
        target_pose = Input([self.im_size[0],self.im_size[1],18])

        '''
        out = layers.Concatenate(axis=-1)([input_img,input_pose,target_img,target_pose])
        out = layers.Conv2D(64,kernel_size=4,strides=2)(out)
        out = self.block(out,128)
        out = self.block(out, 256)
        out = self.block(out, 512)
        out = self.block(out, 1, bn=False)
        out = layers.Activation('sigmoid')(out)
        out = layers.Flatten()(out)
        model = Model([input_img,input_pose,target_img,target_pose],out)
        model.summary()
        '''

        out = layers.Concatenate(axis=-1)([input_img,input_pose])
        out = layers.Conv2D(64,kernel_size=4,strides=2)(out)
        out = self.block(out,128)
        out = self.block(out, 256)
        out = self.block(out, 512)
        m_share = Model([input_img,input_pose],[out])

        output_feat = m_share([target_img,target_pose])
        input_feat = m_share([input_img,input_pose])

        out = layers.Concatenate(axis=-1)([output_feat,input_feat])
        out = LeakyReLU(0.2)(out)
        out = layers.Flatten()(out)
        out = layers.Dense(1)(out)
        out = layers.Activation('sigmoid')(out)

        model = Model([input_img, input_pose, target_img, target_pose], out)
        # model.summary()

        return model

    def nn_loss(self,reference,target,neighborhood_size=(3,3)):
        v_pad = neighborhood_size[0] // 2
        h_pad = neighborhood_size[1] // 2
        val_pad = ktf.pad(reference, [[0, 0], [v_pad, v_pad], [h_pad, h_pad], [0, 0]],
                          mode='CONSTANT', constant_values=-10000)

        reference_tensors = []
        for i_begin in range(0, neighborhood_size[0]):
            i_end = i_begin - neighborhood_size[0] + 1
            i_end = None if i_end == 0 else i_end
            for j_begin in range(0, neighborhood_size[1]):
                j_end = j_begin - neighborhood_size[0] + 1
                j_end = None if j_end == 0 else j_end
                sub_tensor = val_pad[:, i_begin:i_end, j_begin:j_end, :]
                reference_tensors.append(ktf.expand_dims(sub_tensor, -1))
        reference = ktf.concat(reference_tensors, axis=-1)
        target = ktf.expand_dims(target, axis=-1)

        abs = ktf.abs(reference - target)
        norms = ktf.reduce_sum(abs, reduction_indices=[-2])
        loss = ktf.reduce_min(norms, reduction_indices=[-1])

        return loss

    def total_variation_loss(self,x):
        img_nrows, img_ncols = self.im_size[0],self.im_size[1]
        assert K.ndim(x) == 4
        if K.image_data_format() == 'channels_first':
            a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
            b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
        else:
            a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
            b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    def gan_loss(self,y_true,y_pred):
        return -K.mean(K.log(y_pred+1e-7))

    def struct_loss(self,y_true,y_pred):
        target_struct = self._pose_estimator(y_true[...,::-1]/2)[1][...,:18]
        struct = self._pose_estimator(y_pred[...,::-1]/2)[1][...,:18]
        return K.mean(target_struct-struct)**2

    def l1_loss(self,y_true,y_pred):
        return keras.losses.mean_absolute_error(y_true, y_pred)

    def gene_loss(self,y_true,y_pred):
        return self._l1_penalty_weight*self.nn_loss(y_pred,y_true)+\
               self._tv_penalty_weight*self.total_variation_loss(y_pred)+\
               self._lstruct_penalty_weight*self.struct_loss(y_true,y_pred)+self.l1_loss(y_true,y_pred)*100



    def train(self):

        valid = np.ones((self.batch_size,1))
        fake = np.zeros((self.batch_size,1))

        for epoch in tqdm(range(self.epochs)):
            for ite in tqdm(range(self.dataset.number_of_batches_per_epoch())):

                self.discriminator.save(self.checkpoints_dir + '{}_{}.h5'.format('discriminator', epoch + 1))
                from_imgs, to_imgs, from_pose, to_pose, warp = self.dataset.next_text_sample()
                self.sample_images(epoch, ite, from_imgs, to_imgs, from_pose, to_pose, warp)
                from_imgs, to_imgs, from_pose, to_pose, warp = self.dataset.next_sample()
                pred,fake_imgs = self.generator.predict([from_imgs, from_pose, to_imgs, to_pose]+warp)
                d_loss_real = self.discriminator.train_on_batch([from_imgs,from_pose,to_imgs,to_pose], valid)

                d_loss_fake = self.discriminator.train_on_batch([from_imgs,from_pose,fake_imgs,to_pose], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                g_loss = self.generator.train_on_batch([from_imgs, from_pose, to_imgs,to_pose]+warp, [valid, to_imgs])


            if (epoch+1)%self.checkpoint_ratio == 0:
                name = self.checkpoints_dir+'{}_{}.h5'.format('discriminator',epoch+1)
                self.discriminator.save(self.checkpoints_dir+'{}_{}.h5'.format('discriminator',epoch+1))

            if (epoch+1)%self.display_ratio == 0:
                from_imgs, to_imgs, from_pose, to_pose, warp = self.dataset.next_text_sample()
                self.sample_images(epoch,ite,from_imgs, to_imgs, from_pose, to_pose,warp)

    def sample_images(self, epoch,iter, from_imgs, to_imgs, from_pose, to_pose, warp,testing=True):
        pred, gen_iamges = self.generator.predict([from_imgs, from_pose, to_imgs, to_pose]+warp)

        size = gen_iamges.shape[0]
        size = size if size < 10 else 10
        pose = to_pose
        for i in range(size):
            result_imgs = []

            if testing:
                result_imgs.append(self.transfrom(from_imgs[i]))
                #result_imgs.append(pose[i])
                result_imgs.append(self.transfrom(gen_iamges[i]))
                result_imgs.append(self.transfrom(to_imgs[i]))
            else:
                # result_imgs.append(self.transfrom(gen_iamges[i]))
                # result_imgs.append(self.transfrom(conditional_image[i]))
                result_imgs.append(self.transfrom(to_imgs[i]))
                result_imgs.append(pose[i])
                result_imgs.append(self.transfrom(gen_iamges[i]))
            # result_imgs.append(self.transfrom(gen_iamges[i]))
            result_imgs = np.hstack(result_imgs)
            result_imgs = result_imgs.astype(np.uint8, copy=False)

            cv2.imwrite(self.output_dir + '{}_{}_{}.png'.format(epoch,iter, i), result_imgs)
    def transfrom(self,img):
        scale = 127.5
        img = scale * img + scale
        return img


if __name__ == '__main__':
    from easydict import EasyDict as edict
    cfg = edict({'batch_size': 16,
                 'im_size': (128, 64, 3),
                 'data_path': '',
                 'use_warp': 'none',
                 'dataset_name': 'cad60',
                 'disc_type': '',
                 'l1_penalty_weight':100,
                 'content_loss_layer':'none',
                 'gan_penalty_weight':1,
                 'tv_penalty_weight':0,
                 'nn_loss_area_size':1,
                 'lstruct_penalty_weight':0})
    model = PoseGAN(cfg)
