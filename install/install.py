import os
print('*******开始安装*********')

cmd = ['pip install -r requirement.txt -i https://pypi.douban.com/simple/',
       'pip install torch-1.0.1-cp36-cp36m-win_amd64.whl',
       'pip install torchvision-0.4.1+cpu-cp36-cp36m-win_amd64.whl',
       'cd Ranger-Deep-Learning-Optimizer&&pip install -e .',
       ]
for i in cmd:
    shell = os.popen(i)
    print(shell.readlines())

print('*******安装成功*********')
