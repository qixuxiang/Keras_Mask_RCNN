"""BDD Dataset Classes

Original author: Liang Yi

Updated by: Ellis Brown, Max deGroot
"""

import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import uuid
import csv
import pandas as pd
# import bdd_val
# from bdd_eval import evaluate_detection

from collections import defaultdict
import time

import random

filename = 'anno.csv'

abs_path = '/home/JDMAC/keras_frcnn/data/train/restricted/'

class BDD:
            
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft BDD helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.cats =  []
        self.cats_dict = {}
        self.dataset,self.imgs,self.imgs_info = list(),list(), list()
        self.attributes,self.labels,self.bboxes = dict(),dict(),dict()
        self.imgToLabs, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            print(dataset['info'])
            print(dataset['licenses'])
            print(dataset['categories'])
            print(type(dataset['images']))
            print(len(dataset['images']))
            print((dataset['images'][0]))
            print((dataset['images'][1]))
            print((dataset['images'][2]))
            # print(type(dataset['annotations']))
            # print(len(dataset['annotations']))
            # print(dataset['annotations'][0])
            # print(dataset['annotations'][1])
            # print(dataset['annotations'][2])
            # print(dataset['annotations'][3])

            # assert type(dataset)==list, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
            
    def createIndex(self):
        # create index
        print('creating index...')
        self.cats_dict = self.dataset['categories']
        # print(self.cats_dict)
        for cat in self.cats_dict:
            # print(cat)
            self.cats.append(cat['name'])
        print(self.cats)
        
        for img_info in self.dataset['images']:
            # print(img_info['file_name'],"   ", img_info['height'],"  ", img_info['width'])
            img_info_dict = {'id':img_info['id'], 'file_name': img_info['file_name'], 'height': img_info['height'], 'width': img_info['width'] }
            self.imgs_info.append(img_info_dict)
            img = img_info['file_name'][:-4:]
            self.imgs.append(img)
        # print(img)
        # print(len(self.imgs))
        # print(len(self.imgs_info))

        bboxes = {}
        boxes = list()

        data_lst = list()
        mapper = {1: 'tieke', 2: 'heiding',
            3: 'daoju', 4: 'dianchi', 5: 'jiandao'}
        i = 0
        anno_len = len(self.dataset['annotations']) 
        # print(anno_len)
        j = 0
        for img_info in self.imgs_info:
            annotation = self.dataset['annotations'][i]
            img = self.imgs[j]
            #print(img_info['file_name'])
            while(annotation['image_id'] == img_info['id']):
                xmin = annotation['bbox'][0]
                ymin = annotation['bbox'][1]
                xmax = annotation['bbox'][0] + annotation['bbox'][2]
                ymax = annotation['bbox'][1] + annotation['bbox'][3]
                #print(xmin)
                if (xmax > xmin and ymax > ymin):
                    box = {'category_id': annotation['category_id'], 'bbox': [xmin, ymin, xmax, ymax]}
                    boxes.append(box)
                i += 1
                if (i < anno_len):
                    annotation = self.dataset['annotations'][i]
                else:
                    break
                
            temp_boxes = boxes.copy()
            with open('example.csv', "a", newline='') as f:
                writer = csv.writer(f)
                for _, annot in enumerate(temp_boxes):
                    #print(annot['bbox'])
                    #print(annot['category_id'])
                    data_lst.append(abs_path+img_info['file_name'])
                    data_lst.extend(annot['bbox'])
                    data_lst.append(mapper.get(annot['category_id']))
                    writer.writerow(data_lst)
                    data_lst.clear()
            #if(len(temp_boxes)==0):
            #    print(img_info['file_name'])
            bboxes[img] = temp_boxes
            boxes.clear()
            j += 1
        
        # # print(len(bboxes))
        # for img, bbox in bboxes.items():
        #     print(img)
        #     # if (len(bbox) == 0):
        #     #     print(img)
        #     print(len(bbox)) 
        
        
        # # create class members
        # self.imgs = imgs
        # self.attributes = attrs
        # self.labels = labs
        self.bboxes = bboxes
        print('-------------------------------------')
        print(len(self.bboxes))
        

    def loadCats(self):
        """
        Load cats with the specified ids.
        :return: cats (object array) : loaded cat objects
        """
        return self.cats

    def getImgIds(self):
        """
        Load cats with the specified ids.
        :return: imgs (object array) : loaded cat objects
        """
        return self.imgs

    def loadBboxes(self, index):
        """
        Load cats with the specified ids.
        :return: bbox (object array) : loaded cat objects
        """
        # print(self.bboxes.get(index))
        return self.bboxes.get(index)

    # _BDD.loadBboxes(index)
    
    def loadAttributes(self, index):
        """
        Load cats with the specified ids.
        :return: bbox (object array) : loaded cat objects
        """
        # print(self.bboxes.get(index))
        return self.attributes.get(index)

  
class BDDDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    

    def __init__(self, root, image_sets, preproc=None, target_transform=None,
                 dataset_name='BDD'):

        self.i = 0
        self.root = root
        self.cache_path = os.path.join('data/cache')
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        self.annotations = list()
        self._view_map = {
            'train' : 'train',          # 5k val2014 subset
            'val' : 'val',  # val2014 \setminus minival2014
            'test' : 'test',
        }
        # image_sets = ['val']
        for (image_set) in image_sets:
            bdd_name = image_set
            data_name = (self._view_map[bdd_name]
                        if bdd_name in self._view_map
                        else bdd_name)
            # print(data_name)
            annofile = self._get_ann_file(bdd_name)
            print("annofile:  ", annofile)
            _BDD = BDD(annofile)
            self._BDD = _BDD
            self.bdd_name = bdd_name
            cats = _BDD.loadCats()
            self._classes = ['__background__']  + cats
            self.num_classes = len(self._classes)
            self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
            self._ind_to_class =  dict(zip(range(self.num_classes), self._classes))
            indexes = _BDD.getImgIds()
            print('indexes: ', len(indexes))
            self.image_indexes = indexes
            self.ids.extend([self.image_path_from_index(data_name, index) for index in indexes ])
            if image_set.find('test') != -1:
                print('test set will not load annotations!')
            else:
                self.annotations.extend(self._load_bdd_annotations(bdd_name, indexes,_BDD))


    def getImgIds(self):
        """
        Load cats with the specified ids.
        :return: imgs (object array) : loaded cat objects
        """
        return self.image_indexes

    def image_path_from_index(self, name, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = (index +'.jpg')
        image_path = os.path.join(self.root,'images/100k',
                              name, file_name)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path


    def _get_ann_file(self, name):
        prefix = 'bdd100k_labels_images' if name.find('test') == -1 \
                else 'image_info'
        return os.path.join(self.root, 'labels',
                        prefix + '_' + name + '.json')


    def _load_bdd_annotations(self, bdd_name, indexes, _BDD):
        cache_file=os.path.join(self.cache_path,bdd_name+'_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(bdd_name,cache_file))
            return roidb

        gt_roidb = [self._annotation_from_index(index, _BDD)
                    for index in indexes]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb,fid,pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb


    def _annotation_from_index(self, index, _BDD):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        objs = _BDD.loadBboxes(index)
        attr = _BDD.loadAttributes(index)
        # Sanitize bboxes -- some are invalid]
        valid_objs = []    
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.max((0, obj['bbox'][2])) 
            y2 = np.max((0, obj['bbox'][3])) 
            # print(x1, ' ', y1, ' ', x2, ' ', y2)
            if x2 >= x1 and y2 >= y1:
                obj['bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)    
        res = np.zeros((num_objs, 5))
        bdd_cat_id_to_class_ind = self._class_to_ind
        for ix, obj in enumerate(objs):
            cls = bdd_cat_id_to_class_ind[obj['category']]
            res[ix, 0:4] = obj['bbox']
            res[ix, 4] = cls
        return res

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = self.annotations[index]
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        # print('---------------img-------------target-------------------')
        # print(img)
        # print(target)
        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        # print('img_id: ', img_id)
        return cv2.imread(img_id, cv2.IMREAD_COLOR)


    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def _bdd_results_one_category(self, boxes, cat):
        results = []
        i = 0 
        for im_ind, index in enumerate(self.image_indexes):
            # i = i + 1 
            # if(i == 40):
            #     break
                
            # print('im_ind: ', im_ind)
            # print('index: ', index)
            img_name = index + '.jpg'
            dets = boxes[im_ind].astype(np.float)
            # print(dets)
            if dets == []:
                continue
            scores = dets[:, -1]
            x1s = dets[:, 0]
            y1s = dets[:, 1]
            ws = dets[:, 2] - x1s + 1
            hs = dets[:, 3] - y1s + 1
            x2s = x1s + ws
            y2s = y1s + hs
            results.extend(
              [{'name' : img_name,
                'timestamp': 1000,
                'category' : cat,
                'bbox' : [x1s[k], y1s[k], x2s[k], y2s[k]],
                'score' : scores[k]} for k in range(dets.shape[0])])

            # break
        # print(results)
        return results


def csv2txt(csv_file):
    data = pd.read_csv(csv_file) 
    with open('example.txt','a') as f:
        for line in data.values:
            f.write((str(line[0])+','+str(line[1])+','+ str(line[2])+','\
                +str(line[3])+','+str(line[4])+','+str(line[5])+'\n'))


if __name__ == '__main__':
    root = 'train_no_poly.json'
    # out = json.load(open(root, 'r'))
    _BDD = BDD(root)
    csv2txt('example.csv')



