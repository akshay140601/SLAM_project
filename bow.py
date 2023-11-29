import sys
import os, os.path as osp

from tqdm import tqdm

import random
import numpy as np
import cv2 as cv

class BoW(object) :
    def __init__(self, 
        desc_name='sift',
        k=1000, 
        n_iter=1,
    ) :

        assert desc_name in ['sift'];
        if desc_name == 'sift' :
            self.desc_obj = cv.SIFT_create();

        self.k = k; # num of clusters
        self.kmeans_iter = n_iter;

        FLANN_INDEX_KDTREE = 1;
        flann_params = {"algorithm": FLANN_INDEX_KDTREE, "trees": 5};
        matcher = cv.FlannBasedMatcher(flann_params, {});

        self.extractor = cv.BOWImgDescriptorExtractor(self.desc_obj, matcher);
        self.trainer = cv.BOWKMeansTrainer(k);  


    def build_and_save(self, 
        data_dir, 
        save_path,
        image_subdir='image_0',
        step_size=100,
    ) :

        for i, sdir in enumerate(sorted(os.listdir(data_dir))) :
            dir_ = osp.join(data_dir, sdir, image_subdir);
            assert osp.isdir(dir_);
            
            file_list = sorted(os.listdir(dir_))[::step_size];
            for fname in tqdm(file_list, desc=f"{i+1}") :
                fpath = osp.join(dir_, fname);
                im = cv.imread(fpath, cv.IMREAD_GRAYSCALE);

                kp = self.desc_obj.detect(im, None);
                kp, des = self.desc_obj.compute(im, kp);
                self.trainer.add(des);

        vocab = self.trainer.cluster();
        print("Vocabulary shape = ", vocab.shape);
        np.save(save_path, vocab);

    def load_vocab(self, vocab_path) :
        vocab = np.load(vocab_path);
        self.extractor.setVocabulary(vocab);

    def get_image_features(self, fpath) :
        im = cv.imread(fpath, cv.IMREAD_GRAYSCALE);
        kp = self.desc_obj.detect(im, None);        
        feat = self.extractor.compute(im, kp);
        return feat;


    def get_hist_similarity(self, x1, x2) :
        return cv.compareHist(x1, x2, cv.HISTCMP_CORREL);


    def sample_test(self, data_dir, image_subdir='image_0') :
        for i, sdir in enumerate(sorted(os.listdir(data_dir))) :
            dir_ = osp.join(data_dir, sdir, image_subdir);
            assert osp.isdir(dir_);
            
            file_list = sorted(os.listdir(dir_))
            # random.shuffle(file_list);
            file_list = file_list[:50];
            print("Extracting features ...");
            features = [];
            for fname in tqdm(file_list, desc=f"{i+1}") :
                fpath = osp.join(dir_, fname);
                features.append( self.get_image_features(fpath) );

            break;

        print("Print similarity ...");
        n = len(features);
        for i in range(n) :
            for j in range(n) :
                score = self.get_hist_similarity(features[i], features[j]);
                print(f"Similarity ({i}, {j}) = {score:.4f}");

            break;

data_dir = "../../dataset/sequences";        
vocab_path = '../vocab.npy';
bow = BoW();
# bow.build_and_save(data_dir, vocab_path);
bow.load_vocab(vocab_path);
bow.sample_test(data_dir);