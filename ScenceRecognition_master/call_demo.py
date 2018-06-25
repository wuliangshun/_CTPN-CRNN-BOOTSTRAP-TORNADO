import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ctpn.ctpn.demo import CTPN
from crnn.demo import *
import sys,os
import glob
sys.path.append(os.getcwd())

class CTPN_CRNN(object):

    # 文本检测
    def text_detection(self,ctpn,im):
        img,text_recs = ctpn.get_text_box(im)
        return img, text_recs

    # 文本识别
    def text_recognition(self,img, text_recs):
        model, converter = crnnSource()
        raw_preds, sim_preds = crnnRec(model=model, converter=converter, im=img, text_recs=text_recs)
        return raw_preds, sim_preds

    def do(self,ctpn,img_name):
        print("---------------------------------------------------------------")
        print("start to recognize : %s"%img_name)
        # 读取图片
        im = cv2.imread(img_name)
        # 利用CTPN检测文本块
        img, text_recs = self.text_detection(ctpn,im)      
        new_img_path = '.'.join(img_name.split('.')[:-1])+'ctpn.' + img_name.split('.')[-1]
        cv2.imwrite(new_img_path, img)   #使用上传文件的文件名+'.ctpn'     
        # 使用CRNN识别文本
        raw_preds, sim_preds = self.text_recognition(img, text_recs)

        # 输出识别结果
        # for i in range(len(raw_preds)):
            # print("%s" % (sim_preds[i]))
        # print("---------------------------------------------------------------")
        return raw_preds,sim_preds

        
def demo(file_name,first = True):
    ctpn_crnn = CTPN_CRNN()
    ctpn = CTPN(first=first)
    raw_preds,sim_preds = ctpn_crnn.do(ctpn,os.path.join(os.path.dirname(os.path.abspath(__file__)),"imgs",file_name))
    ctpn.sess.close()
    return raw_preds,sim_preds
    
        

