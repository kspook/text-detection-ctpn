from __future__ import print_function

import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

sys.path.append(os.getcwd())
from lib.fast_rcnn.config import cfg, cfg_from_file
#from lib.networks.factory import get_network
#from lib.fast_rcnn.test import test_ctpn
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer
from PIL import Image 

from scipy import ndimage

import pytesseract   as ggt
#import pytesseract.src.pytesseract   as ggt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--detect_images_path', type=str, default='data/demo_2/', help='the path to your images')
#parser.add_argument('--detect_images_path', type=str, default='detect_data/detect_images', help='the path to your images')
opt = parser.parse_args()

from word_job.before_segmentation import image_for_detection

# ctpn params
ctpn_model_path= 'trained_models/trained_ctpn_models'
save_coordinates_path = 'coordinates_results/'
#save_coordinates_path = 'coordinates_results/'
detect_data_path = opt.detect_images_path
#cropped_images_path = 'detect_data/cropped_images/' 
cropped_images_path = 'data/crpd_img/' 

 

import  word_job.wocr as wocr
import  rec_char.tools.classify_hangul as rchar
import  cropping as crp 




def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

#!/usr/bin/env python

#''
#Python-tesseract. For more information: https://github.com/madmaze/pytesseract
  
 
def get_errors(error_string):
    return u' '.join(
        line for line in error_string.decode('utf-8').splitlines()
    ).strip()


def cleanup(temp_name):
    #Tries to remove temp files by filename wildcard path. 
    for filename in iglob(temp_name + '*' if temp_name else temp_name):
        try:
            os.remove(filename)
        except OSError:
            pass



def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name.split('/')[-1]

    img_2 = Image.open(image_name)
    #img_2 = img_2.rotate(90)
    img_saved = cv2.imread(image_name)
    cv2.imwrite(image_name, img_saved)

    base_name = format(format(base_name.split('.')[0]).split('\\')[-1])
    with open('data/results/' + '{}.txt'.format(base_name.split('.')[0]), 'w') as f:
#    with open('data/results/' + '{}.txt'.format(format(base_name.split('.')[0]).split('\\')[-1]), 'w') as f:
        i=len(boxes) -1
        j=0
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
            

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

            line = ',x'.join([str(int(box[0] / scale)), str(int(box[2] / scale)), str(int(box[4] / scale)), str(int(box[6] / scale))]) + '\r\n'
            f.write(line)
            line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
            f.write(line)

            #print("box[0] ", int(box[0]/scale), int(box[0]), scale, 8/scale, 1.5/scale, 0/scale)
            #print("box[1] ", int(box[1]/scale), int(box[1]), scale)
            #print("box[6] ", int(box[6]/scale), int(box[6]), scale)
            #print("box[7] ", int(box[7]/scale), int(box[7]), scale)

#            if not os.path.exists('data/crpd_img/'+base_name+'/'):
#                os.makedirs('data/crpd_img/'+base_name+'/')
            if not os.path.exists('data/crpd_img/'+format(format(base_name.split('.')[0]).split('\\')[-1])+'/'):
                os.makedirs('data/crpd_img/'+format(format(base_name.split('.')[0]).split('\\')[-1])+'/')

            cropImage = img_2.crop ((min_x, min_y, max_x, max_y ))
            #cropImage = img_2.crop (( int(box[0]/scale), int(box[1]/scale), int(box[6]/scale), int(box[7]/scale) ))
            #cropImage = img_2.crop (( int(box[3]/scale), int(box[2]/scale), int(box[5]/scale), int(box[4]/scale)  ))
            #cropImage = img_2.crop (( int(box[1]/scale), int(box[6]/scale), int(box[7]/scale), int(box[0]/scale)  ))
            #cropImage = img_2.crop (( int(box[7]/scale), int(box[6]/scale), int(box[1]/scale), int(box[0]/scale)  ))  # out of bound
            #cropImage = img_2.crop (( int(box[1]/scale), int(box[0]/scale)+2200, int(box[7]/scale), int(box[6]/scale)+2200 ))
            #cropImage = cropImage.rotate(270)

            cropImage.save ('data/crpd_img/'+base_name+'/'+base_name+str(j)+'.png')
			
            img_saved = cv2.imread('data/crpd_img/'+base_name+'/'+base_name+str(j)+'.png')

            height, width = img_saved.shape[:2]
       
            max_height = 150
            if max_height < height:
                 scaling_factor = max_height/float(height)
                 img_saved = cv2.resize(img_saved, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            elif max_height >height and height >10 :
                #print("resize gray complete: up")
                 scaling_factor = max_height/float(height)
                 img_saved = cv2.resize(img_saved,None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)

            cv2.imwrite('data/crpd_img/'+base_name+'/'+base_name+str(j)+'.png', img_saved)

            img_bw = image_for_detection(img_saved)	
            cv2.imwrite('data/crpd_img/'+base_name+'/'+base_name+str(j)+'bw.png', img_bw)

            syn_base_name = base_name + '/'+ base_name+str(j)+".png"
            filename = os.path.join("data/crpd_img/", syn_base_name)
            #print("filename before call init_word_boxes", filename)
            #init_word_boxes(i, filename, syn_base_name)
            #word_det_rec(i, filename, syn_base_name)
            #if j==0 :
            #wocr.perform_ocr(j, filename, syn_base_name)

    #        print("filename : ", filename)

            #img_test = cv2.imread(filename)
            #height, width = img_test.shape[:2]

            #crp.crop_word(i, filename)

            j += 1
            i -= 1			

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    #cv2.imwrite(os.path.join("data/results/", base_name), img)
 
    #print("draw box : base_name - "+base_name)
    #print(base_name.split('.')[0])


# 对获得的坐标由上到下排序
def sort_list(min_y_sort_list):
    cnt=0
    for i in range(len(min_y_sort_list)):
        for j in range(1, len(min_y_sort_list)-i):
            if min_y_sort_list[i][1] > min_y_sort_list[i+j][1]:
                temp = min_y_sort_list[i]
                min_y_sort_list[i] = min_y_sort_list[i+j]
                min_y_sort_list[i+j] = temp
            j+=1
        i+=1

    return min_y_sort_list


#def crop_images(coordinates, base_name, num_of_boxes, model, img):
def crop_images(coordinates, base_name, num_of_boxes, img):
    global txt_reco
    #txt_reco.write(base_name+'\n')

    img_2 = Image.open(detect_data_path+'/'+base_name)

    #test cropped images

    # new_file = './test_cropped_images/'+base_name+'/'
    #new_file = './data/test_cropped_images/'+base_name+'/'
    new_file = cropped_images_path+base_name+'/'
    if os.path.exists(new_file):
         shutil.rmtree(new_file)
    os.makedirs(new_file)

    for i in range(3):
    #for i in range(num_of_boxes):
        cropped_image = img_2.crop((coordinates[i][0]-8 if coordinates[i][0]!=0 else coordinates[i][0],
                                    coordinates[i][1]-1.5,coordinates[i][2]+8,coordinates[i][3]-0.5))
        # cropped_image = img_2.crop((coordinates[i][0],
        #                     coordinates[i][1],coordinates[i][2],coordinates[i][3]))
        #crnn_recognition(cropped_image, model)


        #test cropped images
        syn_base_name = base_name+str(i)+".png"
        #nm_cropped_image = new_file+base_name+str(i)+".png"
        nm_cropped_image = new_file+syn_base_name
        cropped_image.save(nm_cropped_image)

        img_saved = cv2.imread(nm_cropped_image )
          
        height, width = img_saved.shape[:2]
        #print("height - " +  str(height)+"width -"+ str(width)+'\n')


        max_height = 200
        if max_height < height:
              scaling_factor = max_height/float(height)
              img_saved = cv2.resize(img_saved, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        elif max_height >height and height >10 :
              #print("resize gray complete: up")
              scaling_factor = max_height/float(height)
              img_saved = cv2.resize(img_saved,None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)


        cv2.imwrite(nm_cropped_image,img_saved)



        # If you don't have tesseract executable in your PATH, include the following:
        #pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
        # Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

        # Simple image to string
        #print(pytesseract.image_to_string(Image.open('test.png')))

        # Korean text image to string
        #print(pytesseract.image_to_string(Image.open(new_file+str(i)+".png"), lang='kor'))
        # French text image to string
        #print(pytesseract.image_to_string(Image.open('test-european.jpg'), lang='fra'))


        #print("crop_word & classify------------------")
        crp.crop_word(i, nm_cropped_image)
        '''
        # cropped words -->  characters
        wjob_file = nm_cropped_image+'_/'
        if os.path.exists(wjob_file):
              shutil.rmtree(wjob_file)
        os.makedirs(wjob_file)

        wjob_file2 = wjob_file+'words/'
        if os.path.exists(wjob_file2):
              shutil.rmtree(wjob_file2)
        os.makedirs(wjob_file2)
        '''


        #wocr.perform_ocr(i, nm_cropped_image, syn_base_name)
        '''
        #google tesseract 
        
        try:
           print("(google tesseract - nm_cropped_image", nm_cropped_image)
           #print(ggt.image_to_string(Image.open(nm_cropped_image),  config="-c preserve_interword_spaces=1 " +"\n"))
           #print(ggt.image_to_string(Image.open(nm_cropped_image),  config="-c preserve_interword_spaces=1 -psm 8" +"\n"))
           print(ggt.image_to_string(Image.open(nm_cropped_image), lang='kor+eng', config="-c preserve_interword_spaces=1 " +"\n"))
           #1 = Automatic page segmentation with OSD.
           #2 = Automatic page segmentation, but no OSD, or OCR
           #3 = Fully automatic page segmentation, but no OSD. (Default)
           #4 = Assume a single column of text of variable sizes.
           #5 = Assume a single uniform block of vertically aligned text.
           #6 = Assume a single uniform block of text.
           #7 = Treat the image as a single text line.
           #8 = Treat the image as a single word.
           #9 = Treat the image as a single word in a circle.
           #10 = Treat the image as a single character.


        except IOError:
           sys.stderr.write('ERROR: Could not open file \n')
           #sys.stderr.write('ERROR: Could not open file "%s"\n' % filename)
           #exit(1)
        '''
        #hangul 
        
#        try:
#           print("(google tesseract - ")
#           print(crop2(Image.open(new_file+str(i)+".png"))
#        except IOError:
#           sys.stderr.write('ERROR: Could not open file "%s"\n' % filename)
 #       '''


# 获取文本区域的坐标信息
#def get_coordinates(img, image_name, boxes, scale, model):
def get_coordinates(img, image_name, boxes, scale):

    global txt_reco

	# 获取需要检测的图片名称
    base_name = image_name.split('/')[-1]
    # to save detected text area's coordinates
    min_y_sort_list = []
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
        max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
        line = [min_x, min_y, max_x, max_y]
        min_y_sort_list.append(line)
        # to sort coordinates' y 
    min_y_sort_list = sort_list(min_y_sort_list)
    #crop_images(min_y_sort_list, base_name, len(min_y_sort_list), model, img)
    crop_images(min_y_sort_list, base_name, len(min_y_sort_list), img)


# ctpn检测文本区域
def ctpn(sess, image_name, model):
#def ctpn(sess, net, image_name, model):
    #img = cv2.imread(image_name)

    #r = image_to_binary(img)
    #noise = np.ones(img.shape[:2],dtype="uint8") * 125
    #img = cv2.merge((r+noise, r, noise))
    im_names = image_name    

    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #print(('Demo for {:s}'.format(im_name)))
        img = cv2.imread(im_name)
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]

#   img, scale = resize_im(img, scale=600, max_scale=1000) # 参考ctpn论文
#    print('ctpn', img.shape)
#    scores, boxes = test_ctpn(sess, net, img)
    # ctpn识别实例
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, im_name, boxes, scale)
    get_coordinates(img, im_name, boxes, scale)
    #get_coordinates(img, im_name, boxes, scale, model)




# crnn文本信息识别
def crnn_recognition(cropped_image, model):

    global txt_reco

    converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')
    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)

    #print(image)

    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print("recognition : " + sim_pred  + "\n")

    #print(sim_pred)
    txt_reco.write(sim_pred+'\n')




def endtoend_det_rec():
    global txt_reco
    global detect_data_path
    global txt_results_path

#    if os.path.exists("data/results/"):
#        shutil.rmtree("data/results/")
#    os.makedirs("data/results/")

    cfg_from_file('ctpn/text_drive.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)

#   # tensorflow GPU内存分配0.75,按需分配
    config=tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75

#    sess = tf.Session(config=config)
    sess = tf.Session(config=config)
    with gfile.FastGFile('data/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

    # load network
    #net = get_network("VGGnet_test")
    # load model
    #saver = tf.train.Saver()
    # crnn network
#    model = crnn.CRNN(32, 1, nclass, 256)
#    if torch.cuda.is_available():
#        model = model.cuda()
#    print('loading pretrained model from %s' % crnn_model_path)
    # 导入已经训练好的crnn模型
#    model.load_state_dict(torch.load(crnn_model_path))

#   try:
        # 导入已训练好的模型
#       ckpt = tf.train.get_checkpoint_state(ctpn_model_path)
#       saver.restore(sess, ckpt.model_checkpoint_path)
#   except:
#       raise 'import error, please check the path!'
#

#    # focus on png or jpg
#    print(glob.glob(os.path.join(detect_data_path, '*.png')))
#    img_names = glob.glob(os.path.join(detect_data_path, '*.png')) + \
#               glob.glob(os.path.join(detect_data_path, '*.jpg'))

    # focus on png or jpg
    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo_2', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo_2', '*.jpg'))

    # txt_reco 保存识别的文本信息
    #txt_reco = open(txt_results_path, 'w')
    #for img_name in img_names:
    for im_name in im_names:
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #print('Recognizing...[{0}]'.format(img_name))
        #print('Recognizing...[{0}]'.format(im_name))
        #ctpn(sess, net, img_name, model)
        #ctpn(sess, im_name, model)


#        for im_name in im_names:
        print(('Demo for {:s}'.format(im_name)))
        img = cv2.imread(im_name)
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
             im_blob = blobs['data']
             blobs['im_info'] = np.array(
                  [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                  dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        draw_boxes(img, im_name, boxes, scale)
        #get_coordinates(img, im_name, boxes, scale)
        #get_coordinates(img, im_name, boxes, scale, model) ## crnn model appy
        #get_coordinates(img, image_name, boxes, scale, model)

#   txt_reco.close()

    # txt_reco 保存识别的文本信息
#    txt_reco = open(txt_results_path, 'w')
#    for img_name in img_names:
#        print('Recognizing...[{0}]'.format(img_name))
#        ctpn(sess, net, img_name, model)
#    txt_reco.close()



if __name__ == '__main__':

#    global txt_reco

#    if os.path.exists("data/results/"):
#        shutil.rmtree("data/results/")
#   os.makedirs("data/results/")

#    cfg_from_file('ctpn/text.yml')

     endtoend_det_rec()
