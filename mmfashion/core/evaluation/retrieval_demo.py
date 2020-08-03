import os

import numpy as np
from scipy.spatial.distance import cosine as cosine
import numpy as np
import cv2

class ClothesRetriever(object):

    def __init__(self,
                 gallery_im_fn,
                 data_dir,
                 img_path,
                 topks=[15],
                 extract_feature=False):
        self.topks = topks
        self.data_dir = data_dir
        self.img_path = img_path
        self.gallery_idx2im = {}
        gallery_imgs = open(gallery_im_fn).readlines()
        for i, img in enumerate(gallery_imgs):
            self.gallery_idx2im[i] = img.strip('\n')

    def show_topk_retrieved_images(self, retrieved_idxes):
        path_images = []
        for idx in retrieved_idxes:
            retrieved_img = self.gallery_idx2im[idx]
            path_image = os.path.join(self.img_path, retrieved_img)
            path_images.append(path_image)
            print(path_image)
        return path_images

    def show_retrieved_images(self, query_feat, gallery_embeds):
        topk_path = []
        query_dist = []
        for i, feat in enumerate(gallery_embeds):
            cosine_dist = cosine(
                feat.reshape(1, -1), query_feat.reshape(1, -1))
            query_dist.append(cosine_dist)

        query_dist = np.array(query_dist)
        order = np.argsort(query_dist)
       # self.show_topk_retrieved_images(order[:5])

        for topk in self.topks:
            print('topk:',topk)
            print('Retrieved Top%d Results' % topk)
            topk_path=self.show_topk_retrieved_images(order[:topk])
        topk_path = list(map(lambda x: '/'.join(x.split('/')[4:]), topk_path))
        return topk_path
            