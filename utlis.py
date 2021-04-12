import serial
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import numpy as np
# from PIL import Image


# 标签对应关系
class_label = {0: '亿派薄饼', 1: '可可椰子糖', 2: '喜之郎吸吸果冻', 3: '果冻', 4: '沙琪玛', 5: '阿尔卑斯棒棒糖'}

# 价格对应关系  xx元/kg
price = {'亿派薄饼': 10, '可可椰子糖': 14, '喜之郎吸吸果冻':15, '果冻': 18, '沙琪玛': 20, '阿尔卑斯棒棒糖':25}


size = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_transformer_ImageNet = transforms.Compose([
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    normalize
])


def get_model(use_gpu = False,model_path ='model_out/bestmodel.pth' ):
    torch.cuda.empty_cache()
    model_ft = models.mobilenet_v2(pretrained=False)
    model_ft.classifier = nn.Sequential(nn.Dropout(0.4),nn.Linear(1280, 6))
    if use_gpu:
        model_ft.load_state_dict(torch.load(model_path))
    else:
        model_ft.load_state_dict(torch.load(model_path,'cpu'))
    if use_gpu:
        # device = torch.cuda.is_available()
        model_ft = model_ft.cuda()
    model_ft.eval()
    print('model loaded success!')
    return model_ft

def infer_one_img(model,img_path,use_gpu = False):
    # while True:
    # img = input('input filename:')
    # inputs = inputs.to(device)
    device = torch.device("cuda" if use_gpu else "cpu")
    star = time.time()
    try:
        # image_PIL = Image.open(img_path)
        image_tensor = val_transformer_ImageNet(img_path)
    except:
        print('open err')
    else:
        # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor.unsqueeze_(0)
        # 没有这句话会报错
        image_tensor = image_tensor.to(device)

        out = model(image_tensor)
        # # 得到预测结果，并且从大到小排序
        # _, indices = torch.sort(out, descending=True)
        # 返回每个预测值的百分数
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
        # print(percentage)
        pre = (percentage.detach().cpu().numpy())
        pre = np.argmax(pre)
        # print(pre)
        end = time.time()
        # print('label:',class_label[pre],'p:',percentage[pre].detach().cpu().numpy(),'infer time:',(end-star)*1000,'ms')
        pre_class = class_label[pre]
        p = percentage[pre].detach().cpu().numpy()
        # if img_show:
        #     image_PIL.show()
        return pre_class,p

# 获取串口
def get_ser(portx="COM3"):
    try:

        bps=9600
        ser=serial.Serial(portx,bps,timeout=0.1)
        return ser
    except:
        print('open serial err')

        return 0


# 获取串口返回数据
def get_data_from_seiral(ser):
    # while True:
      #十六进制的发送
    # A3 00 A2 A4 A5
    try:
        ser.write(b'\xA3\x00\xA2\xA4\xA5')#写数据
        #十六进制的读取
        res = (ser.readline())#读一个字节.hex()
        res = (res.hex())
        # print(res)
        lenth = (len(res))

        if lenth!=20 or res[:2]!='aa' or res[-2:]!='ff':
            return False
        res = (res[8:-6])
        res = eval('0x'+res)
        log_time =  (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        # print(log_time,res,'g')
        # time.sleep(0.1)
        return (log_time,res)

    except Exception as e:
        # print('err:',res)
        return False

# 获取重量
def get_kg(ser):
    res_list = []
    post_res = 0
    # 数组中的元素个数
    num_res_list = 0
    # 读取次数
    num_loop = 0
    # 重量为0的次数
    zero_num = 0
    while True:
        res = get_data_from_seiral(ser)
        if res!=False:
            # print(res)
            if res[1]==0:
                # print('0')
                # 20次0g，就退出程序
                if zero_num>20:
                    return False,res[0]
                zero_num+=1

            else:
                # res[1]!=0:
                decy = res[1]-post_res
                if decy==0:
                    temp_num_loop = num_loop
                    num_res_list+=1
                    # print(res)
                    res_list.append(res[1])
                if num_res_list>6:
                    # print('res_list',res_list)
                    return res[0],max(set(res_list), key=res_list.count)
                post_res = res[1]
                # 如果20次内还没有取到数据，就清零重新计数
                if num_res_list!=0 and (num_loop-temp_num_loop)>10:
                    res_list = []
            num_loop+=1
        else:
            continue



# 获取当前时间。日期作为文件夹，时间作为图片名
def get_save_name(path):
    x = (time.localtime(time.time()))
    # print(x)
    day = '{}_{}_{}'.format(x.tm_year, x.tm_mon, x.tm_mday)
    # print(day)
    img_name = '{}_{}_{}.png'.format(x.tm_hour, x.tm_min, x.tm_sec)
    # print(img_name)
    img_path_name = os.path.join(path,day)
    if not os.path.exists(img_path_name):
        os.mkdir(img_path_name)
    return os.path.join(img_path_name,img_name)

# 获取不放物体时候的重量，用于调零
def get_init_g(ser):
    x = get_kg(ser)
    return x[1]
