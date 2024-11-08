# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, request, render_template    #記得要import render_template
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
# 從參數讀取圖檔路徑
@app.route('/',methods=['GET'])
def index():
    return render_template('Predict.html')

@app.route('/predict',methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        #取得路徑並儲存在upload
        file_path = os.path.join(
        'uploads',secure_filename(f.filename))
        f.save(file_path)
         
        print('Begin')
        files = "'"+ file_path +"'"
#        files = 'E:\RestnetBanana\valid\Healthys\90_Healthy00039.jpeg'
        # # 載入訓練好的模型
##        net = load_model('model-resnet50-final6.h5')
#        net = load_model('catdog.h5')
#        # # preds = model_predict(file_path,net)
#        cls_list = ['cat(貓)', 'dog(狗)']
#        
#        img = image.load_img(file_path, target_size=(224, 224))
#        
#        x = image.img_to_array(img)
#        x = np.expand_dims(x, axis = 0)
#        pred = net.predict(x)[0]
#        print(pred)
#        print('End')
#        top_inds = pred.argsort()[::-1][:6]
#        print(top_inds)
#        for i in top_inds:
#            print(i)
#            print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
#            return  cls_list[i]
    return None
    
if __name__ == '__main__':
    app.run(
        host = '127.0.0.1',
        port = '5000',
        )
    
# def model_predict(img_path,net):
#     # 從參數讀取圖檔路徑
#     cls_list = ['cats', 'dogs']
    
#     img = image.load_img(img_path, target_size=(224, 224))

    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis = 0)
    # pred = net.predict(x)[0]
    # top_inds = pred.argsort()[::-1][:5]
    # for i in top_inds:
    #     print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
    # result =  cls_list[0]   
    # return '1'
    

