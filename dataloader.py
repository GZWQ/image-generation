import numpy as np
from utils import pose_utils
from utils import pose_transform
import pandas as pd
import cv2
import random
from easydict import EasyDict as edict

class Dataloader():
    def __init__(self,cfg):

        ######## Outside Setting ########

        self._batch_size = cfg.batch_size
        self._im_size = cfg.im_size
        self._data_path = cfg.data_path
        self._use_warp = cfg.use_warp
        self._dataset_name = cfg.dataset_name
        # self._disc_type = cfg.disc_type

        ####### Dataset ######

        self.read_datatxt()
        self._shuffle_data()

        self.common_path = '{}/{}/original_{}/'.format(self._data_path, self._dataset_name, self._dataset_name)

        self._annotations_file_test = pd.read_csv('data/{}-annotation-test.csv'.format(self._dataset_name), sep=':')
        self._annotations_file_train = pd.read_csv('data/{}-annotation-train.csv'.format(self._dataset_name), sep=':')

        self._annotations_file = pd.concat([self._annotations_file_test, self._annotations_file_train],
                                           axis=0, ignore_index=True)

        self._annotations_file = self._annotations_file.set_index('name')

        ######## Inside Setting ########
        self._batches_before_shuffle = len(self.datatxt)//self._batch_size
        self._current_batch = 0

    def number_of_batches_per_epoch(self):
        return self._batches_before_shuffle

    def _shuffle_data(self):
        random.shuffle(self.datatxt)

    def _next_data_index(self):
        self._current_batch %= self._batches_before_shuffle
        if self._current_batch == 0:
            self._shuffle_data()
        index = np.arange(self._current_batch * self._batch_size, (self._current_batch + 1) * self._batch_size)
        self._current_batch += 1
        return index

    def read_datatxt(self):
        self.datatxt = []
        with open('data/dataset_{}.txt'.format(self._dataset_name), 'r') as f:
            lines = f.readlines()
            for line in lines:
                l = line.strip().replace('\n', '')
                self.datatxt.append(l)

    def _preprocess_image(self,image):
        return (image/255-0.5)*2

    def _deprocess_image(self,image):
        return (255*(image+1)/2).astype('uint8')

    def load_image_batch(self,pair_data):
        _from = []
        _to = []
        for pair in pair_data:
            pairs = pair.split(',')
            from_im = cv2.imread(self.common_path+pairs[0])
            _from.append(cv2.resize(from_im,(self._im_size[1],self._im_size[0])))

            to_im = cv2.imread(self.common_path+pairs[1])
            _to.append(cv2.resize(to_im,(self._im_size[1],self._im_size[0])))
        _from = np.array(_from)
        _to = np.array(_to)
        return self._preprocess_image(_from),self._preprocess_image(_to)

    def load_batch(self,index):
        pair_data = [self.datatxt[i] for i in index]
        from_imgs,to_imgs = self.load_image_batch(pair_data)
        from_pose,to_pose = self.compute_pose_batch(pair_data)
        # if self._use_warp != 'none' and (not for_discriminator):
        warp = self.compute_cord_warp_batch(pair_data)

        return from_imgs,to_imgs,from_pose,to_pose,warp

    def next_sample(self):
        index = self._next_data_index()
        return self.load_batch(index)

    def next_text_sample(self):
        index1 = self._next_data_index()
        index2 = self._next_data_index()

        from_imgs1, to_imgs1, from_pose1, to_pose1, _ = self.load_batch(index1)
        from_imgs2, to_imgs2, from_pose2, to_pose2, _ = self.load_batch(index2)

        p1 = [self.datatxt[i] for i in index1]
        p2 = [self.datatxt[i] for i in index2]
        p = []
        for i,e in enumerate(p1):
            l1 = p1[i].split(',')
            l2 = p2[i].split(',')
            p.append(l1[0]+','+l2[0])
        warp = self.compute_cord_warp_batch(p)

        return from_imgs1,from_imgs2,from_pose1,from_pose2,warp

    def compute_pose_batch(self,pair_data):
        _size = [self._batch_size,self._im_size[0],self._im_size[1],18]
        _from = np.empty(_size)
        _to = np.empty(_size)
        for i,pair in enumerate(pair_data):
            pairs = pair.split(',')
            from_d = '_'.join(pairs[0].split('/'))
            to_d = '_'.join(pairs[1].split('/'))

            from_row = self._annotations_file.loc[from_d]
            to_row = self._annotations_file.loc[to_d]

            from_pose_corordinates = pose_utils.load_pose_cords_from_strings(from_row['keypoints_y'], from_row['keypoints_x'])
            from_pose = pose_utils.cords_to_map(from_pose_corordinates,(self._im_size[0],self._im_size[1]))
            _from[i] = from_pose

            to_pose_corordinates = pose_utils.load_pose_cords_from_strings(to_row['keypoints_y'], to_row['keypoints_x'])
            to_pose = pose_utils.cords_to_map(to_pose_corordinates, (self._im_size[0],self._im_size[1]))
            _to[i] = to_pose

        return _from,_to

    def compute_cord_warp_batch(self,pair_data):
        if self._use_warp == 'full':
            warp = [np.empty([self._batch_size] + [1, 8])]
        elif self._use_warp == 'mask':
            warp = [np.empty([self._batch_size] + [10, 8]),
                     np.empty([self._batch_size, 10] + list(self._im_size))]
        elif self._use_warp == 'stn':
            warp = [np.empty([self._batch_size]+[72])]
        else:
            return []

        for i,p in enumerate(pair_data):

            pairs = p.split(',')
            from_d = '_'.join(pairs[0].split('/'))
            to_d = '_'.join(pairs[1].split('/'))

            from_row = self._annotations_file.loc[from_d]
            to_row = self._annotations_file.loc[to_d]

            from_pose_corordinates = pose_utils.load_pose_cords_from_strings(from_row['keypoints_y'], from_row['keypoints_x'])
            to_pose_corordinates = pose_utils.load_pose_cords_from_strings(to_row['keypoints_y'], to_row['keypoints_x'])

            if self._use_warp == 'mask':
                warp[0][i] = pose_transform.affine_transforms(from_pose_corordinates, to_pose_corordinates)
                warp[1][i] = pose_transform.pose_masks(to_pose_corordinates, self._im_size)
            elif self._use_warp == 'full':
                warp[0][i] = pose_transform.estimate_uniform_transform(from_pose_corordinates, to_pose_corordinates)
            else:  # sel._use_warp == 'stn'
                warp[0][i][:36] = from_pose_corordinates.reshape((-1,))
                warp[0][i][36:] = to_pose_corordinates.reshape((-1,))
        return warp


if __name__ == '__main__':
    cfg = edict({'batch_size':16,
            'im_size':(128,64,3),
            'data_path':'/Users/daniel/Documents/JupiterGit/mydata/differentialPoseGan/data',
            'use_warp':False,
            'dataset_name':'cad60',
            'disc_type':''})


    dataloader = Dataloader(cfg)
    a,b,c,d,e,f = dataloader.next_generator_sample()

    g,h,i,j,k,l = dataloader.next_discriminator_sample()