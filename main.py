from fastapi import FastAPI, File,UploadFile,Request,Form
from typing import Optional
import io
from pydantic import BaseModel
import argparse
import numpy as np
import uvicorn
# from fakeantispoffing import get_predict, getmodel
from starlette.responses import Response
from fastapi.templating import Jinja2Templates
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles #new
import pyvips

# image = pyvips.Image.new_from_file('photo_2021-07-30_15-38-37.jpg', access='sequential')
# # image *= [1, 2, 1]
# # mask = pyvips.Image.new_from_array([[-1, -1, -1],
# #                                     [-1, 16, -1],
# #                                     [-1, -1, -1]
# #                                    ], scale=8)
# # out = image.resize([600,600]
# # out = image.crop(77,137,600,600)

# out = image
# # out = image.resize(1, kernel = "linear")
# # image = image.conv(mask, precision='integer')
# out.write_to_file('2/pyvipsx100.jpg')
# out.jpegsave("asd.jpeg")
import requests
import io
import json
import urllib
import cv2
from typing import Optional
# model ,face_detector = getmodel()
templates = Jinja2Templates(directory='templates/')
app = FastAPI()

def configure_static(app):  #new
    app.mount("/static", StaticFiles(directory="static"), name="static")
configure_static(app)

# model,device = getmodel()
class Item(BaseModel):
    source: str
    # description: Optional[str] = None
    # price: float
    # tax: Optional[float] = None
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
def predict(image_path,x,y,w,h):
    image = pyvips.Image.new_from_file(image_path, access='sequential')
# image *= [1, 2, 1]
# mask = pyvips.Image.new_from_array([[-1, -1, -1],
#                                     [-1, 16, -1],
#                                     [-1, -1, -1]
#                                    ], scale=8)
# out = image.resize([600,600]
    x,y,w,h=int(x),int(y),int(w),int(h)
    out = image.crop(x,y,w,h)
    print(out)
    # out = image
    # out = image.resize(1, kernel = "linear")
    # image = image.conv(mask, precision='integer')
    out.write_to_file('static/pyvipsx100.jpg')
   
def get_predict(image,x,y,w,h):
    input_image = Image.open(io.BytesIO(image)).convert('RGB')
    input_image = np.array(input_image)
    x,y,w,h=int(x),int(y),int(w),int(h)
    cropped_ocvImage = input_image[y:y+h,x:x+w]
    # im0 = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(np.uint8(cropped_ocvImage)).convert('RGB')
    return input_image
UPLOAD_FOLDER= "static"
import os
from PIL import Image
@app.get('/')
def form_post(request: Request):
    result = 'Type a number'
    return templates.TemplateResponse('index.html', context={'request': request})

@app.post("/mlbigdata/detect_person/file")
async def home_page(request: Request,width: str = Form(...),height: str = Form(...),x: str = Form(...),y: str = Form(...),file: UploadFile = File(...) ):
    try:
    # Lấy file gửi lên
        image = await file.read()
        if image:
            print(width,height,x,y)
            input_image = Image.open(io.BytesIO(image))
            input_image1=input_image.convert('RGB')
            input_image1 = np.array(input_image1)
            input_image1 = Image.fromarray(np.uint8(input_image1)).convert('RGB')
            try:
                image_predict = get_predict(image,x,y,width,height)
            except:
                return "Sai dinh dang,x,y,width,height"
            path1=file.filename[:-4]+".jpg"
            im1 = input_image.save("static/"+path1,icc_profile=input_image.info.get('icc_profile'))
            path=file.filename[:-4]+"predict.jpg"
            # print(file.filename[:-4])
            # print("text",text)
            im1 = image_predict.save("static/"+path,icc_profile=input_image.info.get('icc_profile'))
            # path1=file.filename[:-4]+".png"
            # path=file.filename[:-4]+"predict.png"
            print("image_predict",image_predict)
            # bytes_io = io.BytesIO()
            # image_predict.save(bytes_io, format="PNG")
            # result=bytes_io.getvalue()
            print("path1",path1)
            print("path",path)
            return templates.TemplateResponse('index.html', context={'request': request, 'result': path,'path123' : path1,'path1234' : path })
        else:
            # Nếu không có file thì yêu cầu tải file
            templates.TemplateResponse('index.html', context={'request': request})
            # return render_template('index.html', msg='Hãy chọn file để tải lên')

    except Exception as ex:
    # Nếu lỗi thì thông báo
        print(ex)
        return templates.TemplateResponse('index.html', context={'request': request})
@app.post("/mlbigdata/detect_person/file1")
def getperson(request: Request,width: str = Form(...),height: str = Form(...),x: str = Form(...),y: str = Form(...), url_image: str = Form(...)) :
    # try:
    # print("ASdsa")
    # print(url_image)
    image_url=url_image
    img_data = requests.get(image_url).content
    img_data_path='static/pic1.jpg'
    with open('static/pic1.jpg', 'wb') as handle:
        handle.write(img_data)
    # img_data.save("static/1.jpg")
    predict(img_data_path,x,y,width,height)
    
    path='pic1.jpg'
    path1="pyvipsx100.jpg"
    # path="static/"+"URLpredict.jpg"
    # # print(file.filename[:-4])
    # image_predict.save(path)
    # path="URLpredict.jpg"
    # bytes_io = io.BytesIO()
    # image_predict.save(bytes_io, format="PNG")
    # result=bytes_io.getvalue()
    return templates.TemplateResponse('index.html', context={'request': request, 'result': path,'path123' : path, 'path1234' : path1})
    # except:
    #     return "loi"
if __name__=="__main__":
    uvicorn.run(app, port = 4000, host = "0.0.0.0" )
