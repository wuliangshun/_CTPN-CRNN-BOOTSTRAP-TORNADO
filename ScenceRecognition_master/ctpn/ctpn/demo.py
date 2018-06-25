from __future__ import print_function
import tensorflow as tf
import os, sys, cv2
import numpy as np

from ctpn.lib.networks.factory import get_network
from ctpn.lib.fast_rcnn.config import cfg, cfg_from_file
from ctpn.lib.fast_rcnn.test import test_ctpn
from ctpn.lib.fast_rcnn.nms_wrapper import nms
from ctpn.lib.utils.timer import Timer
from ctpn.ctpn.text_proposal_connector import TextProposalConnector
from ctpn.ctpn.other import draw_boxes

CLASSES = ('__background__',
           'text')

#tf.AUTO_REUSE = 3

           
class CTPN(object):
    #load the model
    def __init__(self,first = True):
        if not first:
            tf.get_variable_scope().reuse_variables()
        # cfg_from_file(os.getcwd() + '/ctpn/ctpn/text.yml')
        cfg_from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),'text.yml'))
        # init session
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.config)
        # load network
        self.net = get_network("VGGnet_test")
        # load model
        print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
        saver = tf.train.Saver()

        try:
            #ckpt = tf.train.get_checkpoint_state("ctpn/checkpoints/")
            ckpt = tf.train.get_checkpoint_state("G:/DeepLearningProjects/Web_SceneRecognition/ScenceRecognition_master/ctpn/checkpoints/")            
            # ckpt=tf.train.get_checkpoint_state("output/ctpn_end2end/voc_2007_trainval/")
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('done', end=' ')
        except:
            raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
        print(' done.')

    def connect_proposal(self,text_proposals, scores, im_size):
        cp = TextProposalConnector()
        line = cp.get_text_lines(text_proposals, scores, im_size)
        return line


    def show_results(self,im, im_scale, line, thresh):
        inds = np.where(line[:, -1] >= thresh)[0]
        if len(inds) == 0:
            # im = cv2.resize(im, None, None, fx=1.0 / im_scale, fy=1.0 / im_scale, interpolation=cv2.INTER_LINEAR)
            return

        for i in inds:
            bbox = line[i, :4]
            score = line[i, -1]
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 255, 0), thickness=2)
        im = cv2.resize(im, None, None, fx=1.0 / im_scale, fy=1.0 / im_scale, interpolation=cv2.INTER_LINEAR)
        cv2.waitKey(0)


    # 得到每一个文本框
    def get_text_lines(self,line, thresh):
        inds = np.where(line[:, -1] >= thresh)[0]
        if len(inds) == 0:
            # im = cv2.resize(im, None, None, fx=1.0/im_scale, fy=1.0/im_scale, interpolation=cv2.INTER_LINEAR)
            return
        lines = []
        for i in inds:
            bbox = line[i,:4]
            lines.append(bbox)
        return lines


    def check_img(self,img):
        img_size = img.shape
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])

        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 1200:
            im_scale = float(1200) / float(im_size_max)
        re_im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        return re_im, im_scale


    def ctpn(self,sess, net, img):
        im, im_scale = self.check_img(img)
        timer = Timer()
        timer.tic()
        scores, boxes = test_ctpn(sess, net, im)
        timer.toc()
        print(('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

        # Visualize detections for each class
        CONF_THRESH = 0.9
        NMS_THRESH = 0.3
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep]

        keep = np.where(dets[:, 4] >= 0.7)[0]
        dets = dets[keep, :]
        text_lines = self.connect_proposal(dets[:, :], dets[:, 4], im.shape[:2])
        tmp = im.copy()
        text_recs = draw_boxes(tmp,text_lines,caption="im_name",wait=True,is_display=False)
        self.show_results(tmp,im_scale, text_recs, thresh=0.9)
        
        return tmp,text_recs
        # return get_textbox(im, im_scale, line, thresh=0.9)
        # save_results(im,im_scale, line,thresh=0.9)


    def get_text_box(self,img):
        # saver.restore(sess, os.path.join(os.getcwd(),"checkpoints/model_final_tf13.ckpt"))
        # Warmup on a dummy image
        im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
        
        for i in range(2):
            _, _ = test_ctpn(self.sess, self.net, im)
        
        return self.ctpn(self.sess, self.net, img)