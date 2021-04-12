
from tkinter import Tk,Canvas,Button,PhotoImage,FLAT,NW
from PIL import Image
import cv2
import time
from utlis import get_ser,get_data_from_seiral,infer_one_img,get_model,get_kg,get_save_name,get_init_g
# use gpu or not
use_gpu = False

# 价格对应关系  xx元/kg
price = {'亿派薄饼': 10, '可可椰子糖': 14, '喜之郎吸吸果冻':15, '果冻': 18, '沙琪玛': 20, '阿尔卑斯棒棒糖':25}

# 商品的保存目录，每识别一个商品会保存在该目录下
image_save_dir = 'img_save'

ser = get_ser(portx="COM5")
# 端口信息
print(ser)
if ser==0:
    print('串口打开错误，请检查')
else:
    # 加载模型
    model = get_model(use_gpu=use_gpu,model_path ='model_out/bestmodel.pth')

    # 打开摄像头
    capture  = cv2.VideoCapture(1)
    print('open camre succse')

    # 获取不放物体时候的重量，用于调零
    # global err_g
    err_g = get_init_g(ser)
    print('errG = ',err_g)

    global data
    data=[time.time()]

    root = Tk()
    root.title = "55555"
    root.resizable(0,0)
    # root.wm_attributes("-topmost", 1)


    # 按钮的点击功能
    def commit(canvas,text_id2):
        global data,photo

        # canvas.itemconfig(text_id1, text='1111111')
        if len(data)>1:
            canvas.itemconfig(text_id2, text='**** 操 作 成 功， 谢 谢 惠 顾 ！****')
            print(data)
            data = [time.time()]
            photo = PhotoImage(file="index.png")
            canvas.itemconfig(img_1,image=photo)
        else:
            canvas.itemconfig(text_id2, text='**** 操 作 失 败， 请 先 称 重 哦 ！****')
    # 置零程序
    def init_kg(ser):
        global err_g
        err_g = get_init_g(ser)
        print('errG = ',err_g)
        canvas.itemconfig(text_id2, text='**** 清 零 成 功 ！****')


    canvas = Canvas(root, width=800, height=500, bd=0, highlightthickness=0)
    # 图片显示
    filename = PhotoImage(file = "index.png")
    img_1 = canvas.create_image(10, 50, anchor=NW, image=filename)

    # 文本框
    text_id0 = canvas.create_text(10, 20, anchor=NW)
    canvas.itemconfig(text_id0, text='商品图片：')

    text_id1 = canvas.create_text(550, 50, anchor=NW)
    canvas.itemconfig(text_id1, text=' ')
    text_id2 = canvas.create_text(550, 150, anchor=NW)
    canvas.itemconfig(text_id2, text=' ')

    # 按钮
    button1 = Button(canvas, text=' 确 定 ', command=lambda :commit(canvas,text_id2), width=10, height=2)
    button1.configure(width = 10, activebackground = "#33B5E5", relief = FLAT)
    button1_window = canvas.create_window(600, 350, anchor=NW, window=button1)

    # 按钮
    button1 = Button(canvas, text=' 清 零 ', command=lambda :init_kg(ser), width=10, height=2)
    button1.configure(width = 10, activebackground = "#33B5E5", relief = FLAT)
    button1_window = canvas.create_window(700, 10, anchor=NW, window=button1)
    # 打包控件
    canvas.pack()

    class process():
        def __init__(self, canvas,):
            self.canvas = canvas

        def get_data(self):
            # 从串口读取数据，
            res = get_data_from_seiral(ser)
            if res!=False:
                self.canvas.itemconfig(
                    text_id1,
                    text="{}  {} g ".format(res[0],res[1]-err_g)
                )

                global data
                crrunt_time = time.time()
                # 4秒内的数据不做处理，认为3秒内还是同一个商品，时间可修改
                # global err_g
                if crrunt_time-data[-1]>3 and len(data)<3 and res!=False and res[1]-err_g>3 :
                    # 获取重量
                    re_kg = get_kg(ser)

                    # 5g以上的重量才做处理，这里认为3g以下为波动，不算做有商品，可修改
                    if res[1]-err_g>3 and re_kg[0]!=False:
                        # 从摄像头读取数据
                        ret, frame = capture.read()
                        # cv2.imshow('01', frame)
                        # 转换成PIL的格式，为了使用网络预测
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        # 保存在本地

                        img_save_name = get_save_name(image_save_dir)
                        image = image.resize((513, 392), Image.ANTIALIAS)
                        image.save(img_save_name)
                        # 预测
                        pre_resulet = infer_one_img(model, image, use_gpu=use_gpu)
                        # 预测的类别
                        pre_class_name = pre_resulet[0]
                        # 预测的置信度
                        # pre_class_p = pre_resulet[1]
                        # print(pre_class_name, pre_class_p)
                        # print('re_kg[1],err_g,res[1]-err_g :',re_kg[1],err_g,res[1]-err_g)
                        disp = "称重成功：\n\n商品重量： {}  g\n\n商品名称：{}\n\n商品价格：{}元/kg\n\n金额：{}元".format(re_kg[1]-err_g,
                                                                                                    pre_class_name,
                                                                                                 price[pre_class_name],
                                                                                                 price[pre_class_name]*(re_kg[1]-err_g)/1000
                                                                                                 )
                        self.canvas.itemconfig(
                            text_id2,
                            text=disp
                        )
                        # 改变图片需要全局变量
                        global img_name
                        img_name = PhotoImage(file=img_save_name)
                        # print(filename)

                        self.canvas.itemconfig(
                            img_1,
                            image=img_name
                        )
                        # print(re_kg)
                        result = [re_kg[0],res[1]-err_g,pre_class_name,price[pre_class_name],price[pre_class_name]*(re_kg[1]-err_g)/1000,crrunt_time]
                        data = result

            self.canvas.after(100, self.get_data)


    ball = process(canvas)
    ball.get_data()

    root.mainloop()
