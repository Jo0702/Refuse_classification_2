#-----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
#-----------------------------------------------------------------------#
import time

# import cv2
import io
import numpy as np
import tensorflow as tf
from PIL import Image

from yolo import YOLO

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import tempfile

authors_css = """
        style='
        display: block;
        margin-bottom: 0px;
        margin-top: 0px;
        padding-top: 0px;
        font-weight: 400;
        font-size:1.1em;
        color:#DBBD8A;
        filter: brightness(85%);
        text-align: center;
        text-decoration: none;
        '
"""

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #   
    #   test_interval和fps_image_path仅在mode='fps'有效
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #   
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #   
    #   heatmap_save_path仅在mode='heatmap'有效
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    #-------------------------------------------------------------------------#
    # 网页侧边栏
    #-------------------------------------------------------------------------#
    with st.sidebar:
        choose = option_menu("垃圾智能分类", ["智能识物", "视频识物", "环保小游戏", "环保小故事", "环保论坛", "上传图片"],
                            icons=['card-image', 'file-play', 'egg-fried', 'chat', 'bug', 'box-arrow-in-left'],
                            menu_icon="brightness-high", default_index=0)
    st.sidebar.markdown(f"\n\n\n")
    st.sidebar.markdown(
        '<p ' + authors_css + '>' + 'By </p>',
        unsafe_allow_html=True)
    st.sidebar.markdown(
        '<a ' + authors_css + ' target="_blank" href="http://ies.ncu.edu.cn/">' + '南昌大学能源与电气工程系</a>',
        unsafe_allow_html=True,
    )  

    if choose == "智能识物":
        #-------------------------------------------------------------------------#
        # 1、如果想要进行检测完的图片的保存，利用 r_image.save("img.jpg")即可保存，直接在predict.py里进行修改即可。 
        # 2、如果想要获得预测框的坐标，可以进入yolo.detect_image函数，在绘图部分读取top，left，bottom，right这四个值。
        # 3、如果想要利用预测框截取下目标，可以进入yolo.detect_image函数，在绘图部分利用获取到的top，left，bottom，right这四个值
        # 在原图上利用矩阵的方式进行截取。
        # 4、如果想要在预测图上写额外的字，比如检测到的特定目标的数量，可以进入yolo.detect_image函数，在绘图部分对predicted_class进行判断，
        # 比如判断 if predicted_class == 'car': 即可判断当前目标是否为车，然后记录数量即可。利用draw.text即可写字。
        #-------------------------------------------------------------------------#
        # 根据侧边栏选择识别模式
        #-------------------------------------------------------------------------#
        mode = "predict" 
        #-------------------------------------------------------------------------#
        # 标题文本编辑
        #-------------------------------------------------------------------------#        
        st.title(':star:智能识别') 
        st.info('为了构建咱们秀美江西，和我一起进入这有趣的垃圾分类吧') 
        #-------------------------------------------------------------------------#
        # 图片文件加载处
        #-------------------------------------------------------------------------#        
        img = st.file_uploader("Choose a file（请选择图片格式文件）", key = "<uniquevalueofsomesort>")      
        #-------------------------------------------------------------------------#
        # 图片检测及反馈
        #-------------------------------------------------------------------------#       
        detection_result = ['','','','','','','']
        if img:
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count) 
                #---------------------------------------------------------#
                # 显示检测结果
                #---------------------------------------------------------#                 
                st.balloons()                       # 显示成功放气球提示
                cols = st.columns(2)
                with cols[0]:
                    st.image(r_image)               # 显示检测后带先验框的图片
                with cols[1]:
                    link_Baidu = 'https://baike.baidu.com/item/' + \
                        yolo.Predicted_Trabbish_name.replace(' ', '_') 
                    st.markdown(yolo.Predicted_Trabbish_name + '（附百度百科链接）') 
                    st.info(link_Baidu) 
                    st.markdown("可能危害:") 
                    st.info(yolo.trabbish_info1[yolo.Predicted_Trabbish_name]) 
                    st.markdown("潜在作用:") 
                    st.info(yolo.trabbish_info2[yolo.Predicted_Trabbish_name])
                    # yolo.show_result_table( result_2=detection_result[1], result_3=detection_result[2], result_4=detection_result[3], result_5=detection_result[4], result_6=detection_result[5], result_7=detection_result[6], Video=1)        # 用表格显示检测结果
        else:
            st.title(":exclamation:您还未选择图片")

    elif choose == "环保小游戏":
        st.title(':star:环保小游戏') 

    elif choose == "环保小故事":
        st.title(':star:环保小故事') 

    elif choose == "环保论坛":
        st.title(':star:环保论坛')    

    elif choose == "上传图片":
        #-------------------------------------------------------------------------#
        # 标题文本编辑
        #-------------------------------------------------------------------------#        
        st.title(':star:上传图片') 
        st.info('该模块是为了让垃圾智能分类结果更加精准，因此希望可以使用到用户的图像数据。\n注：我们将严格保护您上传的数据，该数据仅用于检测模型的训练') 
        #-------------------------------------------------------------------------#
        # 图片文件加载处
        #-------------------------------------------------------------------------#        
        img2 = st.file_uploader("请选择需要上传的图片文件", key = "<uniquevalueofsomesort>")  
        if img2:
            #---------------------------------------------------------#
            # 显示检测结果
            #---------------------------------------------------------#                 
            st.balloons()                       # 显示成功放气球提示
            st.info('上传成功，感谢您为垃圾智能分类做出贡献，谢谢你！') 




    elif choose == "视频识物":
        #-------------------------------------------------------------------------#
        # 根据侧边栏选择识别模式
        #-------------------------------------------------------------------------#        
        mode = "video"
#         #-------------------------------------------------------------------------#
#         # 标题文本编辑
#         #-------------------------------------------------------------------------#  
#         st.title(':camera:视频识别') #网页上的文本
#         st.info('info3') 
#         #-------------------------------------------------------------------------#
#         # 进度条
#         #-------------------------------------------------------------------------# 
#         frame_frequency = st.slider('请选择您需要的检测频度（注：多次选择将重新开始检测）：', 0, 50, 1)      
#         st.write("每 ", frame_frequency, '帧检测一次，大概需要等待', int(250/frame_frequency), '秒')
#         #-------------------------------------------------------------------------#
#         # 视频文件加载处
#         #-------------------------------------------------------------------------# 
#         video_path = st.file_uploader('视频加载处', type=['mp4']) 
#         #-------------------------------------------------------------------------#
#         # 检测结果初始化
#         #-------------------------------------------------------------------------# 
#         detection_result = ['','','','','','','']
#         #-------------------------------------------------------------------------#
#         # 视频检测及反馈
#         #-------------------------------------------------------------------------#
#         if video_path:
#             st.video(video_path)                    # 播放加载的原始视频
#             #---------------------------------------------------------#
#             # 转格式（tfile为转格式后的视频源）
#             #---------------------------------------------------------#  
#             tfile = tempfile.NamedTemporaryFile(delete=False)
#             tfile.write(video_path.read())
#             capture = cv2.VideoCapture(tfile.name)
#             if video_save_path!="":
#                 fourcc  = cv2.VideoWriter_fourcc(*'avc1')
#                 size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#                 out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
#             fps = 0.0
#             #---------------------------------------------------------#
#             # Streamlit：Error Message
#             #---------------------------------------------------------#  
#             if (capture.isOpened() == False):
#                     st.write("Error opening video stream or file")        
#             fps = int(round(capture.get(cv2.CAP_PROP_FPS)))
#             frame_counter = 0   
#             #---------------------------------------------------------#
#             # 逐帧检测
#             #---------------------------------------------------------# 
#             while(capture.isOpened()):
#                 t1 = time.time()
#                 # 读取某一帧
#                 ref, frame = capture.read()
#                 if not ref:
#                     break
#                 # 读取该帧后，帧数+1
#                 frame_counter += 1
#                 # 帧数为检测频度时才检测
#                 if frame_counter == frame_frequency:                # 表示每 frame_frequency帧检测一次，后期可将其调整为用户可调节参数（可推拉进度条等）
#                     frame_counter = 0 
#                     # 格式转变，BGRtoRGB
#                     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#                     # 转变成Image
#                     frame = Image.fromarray(np.uint8(frame))
#                     # 进行检测
#                     frame = np.array(yolo.detect_image(frame))
#                     # RGBtoBGR满足opencv显示格式
#                     frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                    
#                     fps  = ( fps + (1./(time.time()-t1)) ) / 2
#                     print("fps= %.2f"%(fps))
#                     frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#                     # 保存视频
#                     if video_save_path!="":
#                         out.write(frame)
#                     # Streamlit转格式显示重要步骤（往内存中写入estimate数据）
#                     frame = io.BytesIO(frame)  
#             #---------------------------------------------------------#
#             # 显示检测结果
#             #---------------------------------------------------------#              
#             st.balloons()                   # 显示成功放气球提示
#             cols = st.columns(2)
#             st.info('垃圾种类')
#             st.info(yolo.Predicted_Trabbish_name)
#             print("Video Detection Done!")
#             capture.release()
#             if video_save_path!="":
#                 print("Save processed video to the path :" + video_save_path)
#                 out.release()
#             cv2.destroyAllWindows()
        
