import time
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
import cv2
import sys
from PyQt5.QtWidgets import *
from yolo import YOLO
from PIL import Image
import numpy as np
from skimage import exposure
#from tuxiangchuli import *
class picture(QWidget):

    def __init__(self):
        super(picture, self).__init__()

        self.str_name = '0'

        #self.my_model = my_lodelmodel()
        self.resize(1600, 900)
        self.setWindowIcon(QIcon(os.getcwd() + '\\icons\\1.jpg'))#设置检测主页面图标
        self.setWindowTitle("《垃圾智能分类平台》")

        window_pale = QtGui.QPalette()
        window_pale.setBrush(self.backgroundRole(), QtGui.QBrush(
            #QtGui.QPixmap(os.getcwd() )))#设置检测主画面背景
            QtGui.QPixmap(os.getcwd() + '\\icons\\R-C.jfif')))#设置检测主画面背景
        self.setPalette(window_pale)

        camera_or_video_save_path = 'data\\test'
        if not os.path.exists(camera_or_video_save_path):
            os.makedirs(camera_or_video_save_path)

        self.label1 = QLabel(self)
        #self.label1.setText("待检测图片")
        self.label1.setFixedSize(611, 500)
        self.label1.move(110, 220)

        self.label1.setStyleSheet("QLabel{background:#7A6969;}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                  )
        self.label2 = QLabel(self)
        #self.label2.setText("检测结果")
        self.label2.setFixedSize(611, 500)
        self.label2.move(850, 220)

        self.label2.setStyleSheet("QLabel{background:#7A6969;}"
                                  "QLabel{color:rgb(300,300,300,120);font-size:20px;font-weight:bold;font-family:宋体;}"
                                  )

        self.label3 = QLabel(self)
        self.label3.setText("《垃圾智能分类平台》")
        self.label3.move(330, 50)
        self.label3.setStyleSheet("color:rgb(0, 0 0);\n"
        "font: 40pt \"Times New Roman\";")
        self.label3.adjustSize()

        self.label4 = QLabel(self)
       #self.label4.setText("110：110kV套管\n1000：1000kV套管\n500：500kV套管")
        self.label4.setText("厨余垃圾\n可回收垃圾\n有害垃圾\n其他垃圾")
        self.label4.move(10, 735)
        self.label4.setStyleSheet("color:rgb(204,232,207,255);\n"
                                  "font: 14pt \"Times New Roman\";")
        self.label4.adjustSize()

        self.label5 = QLabel(self)
        self.label5.setText("待检测图片")
        self.label5.move(335, 190)
        self.label5.setStyleSheet("color:rgb(300,300,300,120);\n"
                                  "font: 16pt \"Times New Roman\";"
                                  )



        self.label5.adjustSize()

        self.label6 = QLabel(self)
        self.label6.setText("检测结果")
        self.label6.move(1115, 190)
        self.label6.setStyleSheet("color:rgb(300,300,300,120);\n"
                                  "font: 16pt \"Times New Roman\";"
                                  )
        self.label6.adjustSize()

        

        btn = QPushButton(self)
        btn.setText("选择图片")
        btn.setStyleSheet(''' 
                                                     QPushButton
                                                     {text-align : center;
                                                     background-color : white;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 3px;
                                                     border-radius: 5px;
                                                     padding: 11px;
                                                     height : 25px;
                                                     border-style: outset;
                                                     font : 18pt \黑体\;}
                                                     QPushButton:pressed
                                                     {text-align : center;
                                                     background-color : light gray;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 2px;
                                                     border-radius: 10px;
                                                     padding: 6px;
                                                     height : 14px;
                                                     border-style: outset;
                                                     font : 14px;}
                                                     ''')
        btn.move(580, 760)
        btn.clicked.connect(self.openimage)

        btn1 = QPushButton(self)
        btn1.setText("检测图片")
        btn1.setStyleSheet(''' 
                                                     QPushButton
                                                     {text-align : center;
                                                     background-color : white;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 3px;
                                                     border-radius: 5px;
                                                     padding: 11px;
                                                     height : 25px;
                                                     border-style: outset;
                                                     font : 18pt \黑体\;}
                                                     QPushButton:pressed
                                                     {text-align : center;
                                                     background-color : light gray;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 2px;
                                                     border-radius: 10px;
                                                     padding: 6px;
                                                     height : 14px;
                                                     border-style: outset;
                                                     font : 14px;}
                                                     ''')
        btn1.move(850, 760)
        # print("QPushButton构建")
        btn1.clicked.connect(self.button1_test)

        btn3 = QPushButton(self)
        btn3.setText("选择文件夹")
        btn3.setStyleSheet(''' 
                                                     QPushButton
                                                     {text-align : center;
                                                     background-color : white;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 3px;
                                                     border-radius: 5px;
                                                     padding: 11px;
                                                     height : 25px;
                                                     border-style: outset;
                                                     font : 18pt \"Times New Roman\";}
                                                     QPushButton:pressed
                                                     {text-align : center;
                                                     background-color : light gray;
                                                     font: bold;
                                                     border-color: gray;
                                                     border-width: 2px;
                                                     border-radius: 10px;
                                                     padding: 6px;
                                                     height : 14px;
                                                     border-style: outset;
                                                     font : 14px;}
                                                     ''')
        btn3.move(555, 830)
        btn3.clicked.connect(self.openimagefile)

        btn4 = QPushButton(self)
        btn4.setText("检测文件夹图片")
        btn4.setStyleSheet(''' 
                                                           QPushButton
                                                           {text-align : center;
                                                           background-color : white;
                                                           font: bold;
                                                           border-color: gray;
                                                           border-width: 3px;
                                                           border-radius: 5px;
                                                           padding: 11px;
                                                           height : 25px;
                                                           border-style: outset;
                                                           font : 18pt \"Times New Roman\";}
                                                           QPushButton:pressed
                                                           {text-align : center;
                                                           background-color : light gray;
                                                           font: bold;
                                                           border-color: gray;
                                                           border-width: 2px;
                                                           border-radius: 10px;
                                                           padding: 6px;
                                                           height : 14px;
                                                           border-style: outset;
                                                           font : 14px;}
                                                           ''')
        btn4.move(850, 830)
        btn4.clicked.connect(self.button2_test)

        #**
        btn5 = QPushButton(self)
        btn5.setText("图像处理")
        btn5.setStyleSheet(''' 
                                                                  QPushButton
                                                                  {text-align : center;
                                                                  background-color : white;
                                                                  font: bold;
                                                                  border-color: gray;
                                                                  border-width: 3px;
                                                                  border-radius: 5px;
                                                                  padding: 11px;
                                                                  height : 25px;
                                                                  border-style: outset;
                                                                  font : 18pt \"Times New Roman\";}
                                                                  QPushButton:pressed
                                                                  {text-align : center;
                                                                  background-color : light gray;
                                                                  font: bold;
                                                                  border-color: gray;
                                                                  border-width: 2px;
                                                                  border-radius: 10px;
                                                                  padding: 6px;
                                                                  height : 14px;
                                                                  border-style: outset;
                                                                  font : 14px;}
                                                                  ''')
        btn5.move(400, 760)
        btn5.clicked.connect(self.button3_chuli)
        btn6 = QPushButton(self)
        btn6.setText("文件夹图像处理")
        btn6.setStyleSheet(''' 
                                                                          QPushButton
                                                                          {text-align : center;
                                                                          background-color : white;
                                                                          font: bold;
                                                                          border-color: gray;
                                                                          border-width: 3px;
                                                                          border-radius: 5px;
                                                                          padding: 11px;
                                                                          height : 25px;
                                                                          border-style: outset;
                                                                          font : 18pt \"Times New Roman\";}
                                                                          QPushButton:pressed
                                                                          {text-align : center;
                                                                          background-color : light gray;
                                                                          font: bold;
                                                                          border-color: gray;
                                                                          border-width: 2px;
                                                                          border-radius: 10px;
                                                                          padding: 6px;
                                                                          height : 14px;
                                                                          border-style: outset;
                                                                          font : 14px;}
                                                                          ''')
        btn6.move(300, 830)
        btn6.clicked.connect(self.button4_chuli)



    '''
        self.label7 = QtWidgets.QLineEdit(self)
        self.label7.move(1200, 735)
        #self.label7.setStyleSheet("color:rgb(0, 0, 255);\n"
        self.label7.setText("CYLJ\n1000：1000kV套管\n500：500kV套管")
        self.label7.setStyleSheet("color:rgb(255, 255, 255);\n"
                                  "font:16pt \"Times New Roman\";\n")
        self.label7.adjustSize()

        self.imgname1 = '0'
        self.fileName1 = '0'
        self.a=0
    #def camera_find(self):
        #ui_p = picture()
        #ui_p.close()
        #cam_t = Ui_MainWindow()
        #cam_t.show()
    ##图像处理部分程序


    '''

    def openimage(self):

        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;*.gif;;*.tif;;All Files(*)")

        if imgName != '':
            self.imgname1 = imgName
            # print("imgName",imgName,type(imgName))
            im3=Image.open(imgName)
            if im3.format =='GIF':
                im3.convert('RGB').save('image.jpg')
                im0=Image.open('image.jpg')
                width = im0.size[0]
                height = im0.size[1]
                width_new = 700
                height_new = 500
                # 判断图片的长宽比率
                if width / height >= width_new / height_new:

                    show1 = im0.resize((width_new, int(height * width_new / width)))
                else:

                    show1 = im0.resize((int(width * height_new / height), height_new))
                    im0 = np.array(show1)
            else:
                im0=cv2.imread(imgName)
                cv2.imwrite('image.jpg',im0)
                # cv2.imshow('image.jpg',im0)
                self.im = im0
                #im0 = cv2.imread('image.jpg')
                self.width = self.im.shape[1]
                self.height =  self.im.shape[0]
                width_new = 700
                height_new = 500
                # 判断图片的长宽比率
                if self.width / self.height >= width_new / height_new:

                    show1 = cv2.resize(im0, (width_new, int(self.height * width_new / self.width)))
                else:

                    show1 = cv2.resize(im0, (int(self.width * height_new / self.height), height_new))
                im0 = cv2.cvtColor(show1, cv2.COLOR_RGB2BGR)
               
                


            # 设置新的图片分辨率框架
            showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
            self.label1.setPixmap(QtGui.QPixmap.fromImage(showImage))
           

    def button3_chuli(self):
        if self.imgname1 != '0':
                    p2, p98 = np.percentile(self.im, (2, 98))
                    img_rescale = exposure.rescale_intensity(self.im, in_range=(p2, p98))
                    img_rescale = exposure.adjust_gamma(img_rescale, 0.8)
                   # image_lap0 = laplacian(img_rescale)
                    #image_clahe0 = clahe(image_lap0)
                   # im0 = cv2.GaussianBlur(image_clahe0, (5, 5), 1.3)
                    self.im=im0

                    width_new = 700
                    height_new = 500
                    # 判断图片的长宽比率
                    if self.width / self.height >= width_new / height_new:

                        show1 = cv2.resize(im0, (width_new, int(self.height * width_new / self.width)))
                    else:

                        show1 = cv2.resize(im0, (int(self.width * height_new / self.height), height_new))
                    im0 = cv2.cvtColor(show1, cv2.COLOR_RGB2BGR)
                    showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1],
                                             QtGui.QImage.Format_RGB888)
                    self.label1.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            QMessageBox.information(self, '错误', '请先选择一个图片文件', QMessageBox.Yes, QMessageBox.Yes)

            # jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label1.height())
            # self.label1.setPixmap(jpg)





    def openimagefile(self):

        fileName = QFileDialog.getExistingDirectory(self, "打开文件夹", "./")

        if fileName != '':
            self.fileName1 = fileName
            self.files=os.listdir(self.fileName1)
            for imgfile in self.files:
              if not os.path.isdir(imgfile):
                print(imgfile)
                pass
              imgpath=self.fileName1+'/'+imgfile
              im3 = Image.open(imgpath)
              if im3.format == 'GIF':
                  im3.convert('RGB').save('image.jpg')
                  im0 = Image.open('image.jpg')
                  width = im0.size[0]
                  height = im0.size[1]
                  width_new = 700
                  height_new = 500
                  # 判断图片的长宽比率
                  if width / height >= width_new / height_new:

                      show1 = im0.resize((width_new, int(height * width_new / width)))
                  else:

                      show1 = im0.resize((int(width * height_new / height), height_new))
                      im0 = np.array(show1)
              else:
                  im0 = cv2.imread(imgpath)
                  width = im0.shape[1]
                  height = im0.shape[0]
                  width_new = 700
                  height_new = 500
                  # 判断图片的长宽比率
                  if width / height >= width_new / height_new:

                      show1 = cv2.resize(im0, (width_new, int(height * width_new / width)))
                  else:

                      show1 = cv2.resize(im0, (int(width * height_new / height), height_new))
                  im0 = cv2.cvtColor(show1, cv2.COLOR_RGB2BGR)
              showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
              self.label1.setPixmap(QtGui.QPixmap.fromImage(showImage))
            pass

            # jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label1.height())
            # self.label1.setPixmap(jpg)

    def button4_chuli(self):
        self.a=1
        if self.fileName1 != '0':
            QApplication.processEvents()
            for imgfile in self.files:
              if not os.path.isdir(imgfile):
                print(imgfile)
                pass
              imgpath=self.fileName1+'/'+imgfile
              im3 = Image.open(imgpath)
              if im3.format == 'GIF':
                  im3.convert('RGB').save('image.jpg')
                  image = Image.open('image.jpg')
              elif im3.format == 'TIFF':
                  im3 = cv2.imread(imgpath)
                  cv2.imwrite('image.jpg', im3)
                  image = Image.open('image.jpg')
              else:
                  image = im3
                  width = image.size[0]
                  height = image.size[1]
                  im0 = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                  p2, p98 = np.percentile(im0, (2, 98))
                  img_rescale = exposure.rescale_intensity(im0, in_range=(p2, p98))
                  img_rescale = exposure.adjust_gamma(img_rescale, 0.8)
                  #image_lap0 = laplacian(img_rescale)
                  #image_clahe0 = clahe(image_lap0)
                  #im0 = cv2.GaussianBlur(image_clahe0, (5, 5), 1.3)
                  width_new = 700
                  height_new = 500
                  # 判断图片的长宽比率
                  if width / height >= width_new / height_new:

                     show1 = cv2.resize(im0, (width_new, int(height * width_new / width)))
                  else:

                     show1 = cv2.resize(im0, (int(width * height_new / height), height_new))
                  im0 = cv2.cvtColor(show1, cv2.COLOR_RGB2BGR)
                  showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1],
                                     QtGui.QImage.Format_RGB888)
                  self.label1.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
          QMessageBox.information(self, '错误', '请先选择文件夹', QMessageBox.Yes, QMessageBox.Yes)
    def button1_test(self):

        if self.imgname1 != '0':
            QApplication.processEvents()
            im3 = Image.fromarray(cv2.cvtColor(self.im,cv2.COLOR_BGR2RGB))
            if im3.format == 'GIF':
                im3.convert('RGB').save('image.jpg')
                image = Image.open('image.jpg')
            elif im3.format == 'TIFF':
                im3=cv2.imread(self.imgname1)
                cv2.imwrite('image.jpg',im3)
                # cv2.imshow('image.jpg',im3)
                image = Image.open('image.jpg')
            else:
                image = im3
            #image = cv2.imread(self.imgname1)

            yolo = YOLO()
            im0 = yolo.detect_image(image)
            QApplication.processEvents()

            im0.save(".\\predict-result\\im0.jpg")
            width = im0.size[0]
            height = im0.size[1]

            # 设置新的图片分辨率框架
            width_new = 700
            height_new = 500

            # 判断图片的长宽比率
            if width / height >= width_new / height_new:

                show2 = im0.resize((width_new, int(height * width_new / width)))
            else:

                show2 = im0.resize((int(width * height_new / height), height_new))

            #im0 = cv2.cvtColor(show2, cv2.COLOR_RGB2BGR)
            #show2.show()
           
            im0 = np.array(show2)
            im0 = cv2.cvtColor(np.asarray(im0), cv2.COLOR_RGB2BGR)
            
            # image_name = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
            
            # self.label2.setPixmap(QtGui.QPixmap.fromImage(image_name))
            image = QtGui.QPixmap(".\\predict-result\\im0.jpg").scaled(611, 500)
            self.label2.setPixmap(image)


            




            # cv2.imshow('dd',image_name )
            #image_name.show()
            # label=label.split(' ')[0]    #label 59 0.96   分割字符串  取前一个
            

            # showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
            # self.label1.setPixmap(QtGui.QPixmap.fromImage(showImage))
            # jpg = QtGui.QPixmap(image_name).scaled(self.label1.width(), self.label1.height())
            # self.label2.setPixmap(jpg)
        else:
            QMessageBox.information(self, '错误', '请先选择一个图片文件', QMessageBox.Yes, QMessageBox.Yes)

    def button2_test(self):
      yolo = YOLO()
      if self.a==1:
        if self.fileName1 != '0':
            QApplication.processEvents()
            #im0, label = YOLO.detect_image(self.imgname1)

            file3 = os.listdir(self.fileName1)
            for imgfile in file3:
              if not os.path.isdir(imgfile):
                print(imgfile)
                pass
              self.label7.setText("正在检测：%s"%imgfile)
              time.sleep(0.6)
              imgpath=self.fileName1+'/'+imgfile
              im3= Image.open(imgpath)
              if im3.format == 'GIF':
                  im3.convert('RGB').save('image.jpg')
                  image = Image.open('image.jpg')
              elif im3.format == 'TIFF':
                  im3 = cv2.imread(imgpath)
                  cv2.imwrite('image.jpg', im3)
                  image = Image.open('image.jpg')
              else:
                  image = im3
              im0 = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
              p2, p98 = np.percentile(im0, (2, 98))
              img_rescale = exposure.rescale_intensity(im0, in_range=(p2, p98))
              img_rescale = exposure.adjust_gamma(img_rescale, 0.8)
              #image_lap0 = laplacian(img_rescale)
              #image_clahe0 = clahe(image_lap0)
              #im0 = cv2.GaussianBlur(image_clahe0, (5, 5), 1.3)
              image = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
              im0 = yolo.detect_image(image)
              QApplication.processEvents()
              imgname=imgfile+'detection'
              im0.save(".\\predict-result\\%s.jpg"%imgname)
              width = im0.size[0]
              height = im0.size[1]

              # 设置新的图片分辨率框架
              width_new = 700
              height_new = 500

               # 判断图片的长宽比率
              if width / height >= width_new / height_new:

                show2 = im0.resize((width_new, int(height * width_new / width)))
              else:

                show2 = im0.resize((int(width * height_new / height), height_new))

              im0 = np.array(show2)
              im0 = cv2.cvtColor(np.asarray(im0), cv2.COLOR_RGB2BGR)
              image_name = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
              #image_name.show()
              # label=label.split(' ')[0]    #label 59 0.96   分割字符串  取前一个
              self.label2.setPixmap(QtGui.QPixmap.fromImage(image_name))
              # jpg = QtGui.QPixmap(image_name).scaled(self.label1.width(), self.label1.height())
              # self.label2.setPixmap(jpg)
        else:
            QMessageBox.information(self, '错误', '请先选择文件夹', QMessageBox.Yes, QMessageBox.Yes)
      else:
          if self.fileName1 != '0':
              QApplication.processEvents()
              # im0, label = YOLO.detect_image(self.imgname1)

              file3 = os.listdir(self.fileName1)
              for imgfile in file3:
                  if not os.path.isdir(imgfile):
                      print(imgfile)
                      pass
                  self.label7.setText("正在检测：%s" % imgfile)
                  time.sleep(0.6)
                  imgpath = self.fileName1 + '/' + imgfile
                  im3 = Image.open(imgpath)
                  if im3.format == 'GIF':
                      im3.convert('RGB').save('image.jpg')
                      image = Image.open('image.jpg')
                  elif im3.format == 'TIFF':
                      im3 = cv2.imread(imgpath)
                      cv2.imwrite('image.jpg', im3)
                      image = Image.open('image.jpg')
                  else:
                      image = im3

                  # image = cv2.imread(self.imgname1)
                  im0 = yolo.detect_image(image)
                  QApplication.processEvents()
                  imgname = imgfile + 'detection'
                  im0.save(".\\predict-result\\%s.jpg" % imgname)
                  width = im0.size[0]
                  height = im0.size[1]

                  # 设置新的图片分辨率框架
                  width_new = 700
                  height_new = 500

                  # 判断图片的长宽比率
                  if width / height >= width_new / height_new:

                      show2 = im0.resize((width_new, int(height * width_new / width)))
                  else:

                      show2 = im0.resize((int(width * height_new / height), height_new))

                  # im0 = cv2.cvtColor(show2, cv2.COLOR_RGB2BGR)
                  # show2.show()
                  im0 = np.array(show2)

                  image_name = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1],
                                            QtGui.QImage.Format_RGB888)
                  # image_name.show()
                  # label=label.split(' ')[0]    #label 59 0.96   分割字符串  取前一个
                  self.label2.setPixmap(QtGui.QPixmap.fromImage(image_name))
                  # jpg = QtGui.QPixmap(image_name).scaled(self.label1.width(), self.label1.height())
                  # self.label2.setPixmap(jpg)
          else:
              QMessageBox.information(self, '错误', '请先选择文件夹', QMessageBox.Yes, QMessageBox.Yes)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui_p = picture()
    ui_p.show()

    sys.exit(app.exec_())