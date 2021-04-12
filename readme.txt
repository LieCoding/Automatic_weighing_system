各代码块及文件夹的功能：

***********
main.py   主程序，运行即可打开软件界面
utlis.py    定义了主程序中所需要的功能，如串口，模型推理
**********
**********
model.py   模型训练文件，运行即可训练模型
data_load.py    定义了训练模型时加载本地文件相关操作
LabelSmoothing.py    定义了标签平滑相关操作
*********
*********
install   安装文件以及安装脚本，运行install.py即可安装python环境下所依赖的文件
checkpiont    存放训练时每个epoch产生的模型参数
data  存放训练的数据，
img_save   软件识别成功后保存的目录，会以每一天的日期作为文件夹存放
model_out   训练时候产生的验证集准确率最高的模型
premodel   预训练模型存放位置，在ImageNet训练的预训练模型
*********
*********
index.png     首页图片