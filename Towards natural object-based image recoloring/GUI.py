import numpy as np
import cv2
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageTk
from tkinter import Tk,Label,Frame
from tkinter import messagebox
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# 创建画布需要的库
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from argparse import ArgumentParser
from detect import de
from mattes import inference
from recolor import Recolor, Extract
from util import LABtoRGB, RegularLAB

#显示数组
num=[ 0,0,4,4,0,0,0,0,4,0,
      4,3,0,4,4,4,4,4,4,4,
      4,4,4,4,0,4,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,4,5,0,4,
      5,4,0,0,0,0,0,0,0,0,
      0,0,0,4,4,4,4,0,0,0,
      0,0,0,0,0,0,0,0,0,0]
# 读入图片
image_path = 'img.png'
show_path = 'result.png'
im = Image.open(image_path)
im.save(show_path)
img = cv2.imread(show_path)
blue,green,red = cv2.split(img)
img = cv2.merge((red,green,blue))
img = cv2.resize(img,(600,600))
# 检测物体
o = de(image_path)
#o=46
# 生成alpha mattes
inference(image_path,'trimap.png')
# 提取原始调色板
ori_pal = Extract(image_path,'mattes.png',o)
# 初始化窗口
root = Tk()
root.title("Recoloring")
root.geometry('1200x700')

imm = Image.fromarray(img)
imgtk = ImageTk.PhotoImage(image=imm) 

#添加label
Label(root, text='Original palette: ').place(x=0, y=0, width=150, height=50)
Label(root, image=imgtk).place(x=0,y=50,width=600,height= 600)
Label(root, text='Target palette: ').place(x=600, y=0, width=150, height=50)




# 显示原始调色板
for i in range(num[o]):
    r, g, b = LABtoRGB(RegularLAB(ori_pal[i]))
    srv='%02x'%int((r))
    sgv = '%02x' % int((g))
    sbv = '%02x' % int((b))
    bgstr="#"+srv+sgv+sbv
    Label(root, bg = bgstr).place(x= 150+40*i, y=10,width=30, height=30)

# 加载目标调色板
fl = open("offline/GUI/{}_l.txt".format(o),"r")
fa = open("offline/GUI/{}_a.txt".format(o),"r")
fb = open("offline/GUI/{}_b.txt".format(o),"r")
L = fl.readlines()
A = fa.readlines()
B = fb.readlines()
l = []
a = []
b = []
show_r = []
show_g = []
show_b = []
rgb = []
for j in range(len(L)):
	ll = [float(k) for k in L[j].split()]
	aa = [float(k) for k in A[j].split()]
	bb = [float(k) for k in B[j].split()]
	RGB = LABtoRGB(RegularLAB([ll[0],aa[0],bb[0]]))
	if ((RGB[0]>255) | (RGB[1]>255) | (RGB[2]>255)|(RGB[0]<0)|(RGB[1]<0)|(RGB[2]<0)):
		continue
	l.append(ll)
	a.append(aa)   
	b.append(bb)
	show_r.append(RGB[0]/255)
	show_g.append(RGB[1]/255)
	show_b.append(RGB[2]/255)
	rgb.append([RGB[0]/255,RGB[1]/255,RGB[2]/255])
# 创建一个容器, 没有画布时的背景
#Label(root).place(x=600, y=50, width=600, height=600)
frame1 = Frame(root, bg="#ffffff")
frame1.place(x=600, y=50, width=600, height=600)
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
fig = plt.figure()#figsize=(6.5, 7), edgecolor='blue')
ax = Axes3D(fig)
# 定义刻度
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.scatter(show_r, show_g, show_b,c=rgb)
canvas = FigureCanvasTkAgg(fig, master=frame1)
canvas.draw()
# 显示画布
canvas.get_tk_widget().place(x=0, y=0)
#frame1.bind("<Double-Button-1>",func)
def on_press(event):
    if event.button==3: #鼠标右键点击
        dist = 10000
        index = -1        
        for i in range(len(show_r)):
           x, y, _ = proj3d.proj_transform(show_r[i], show_g[i], show_b[i], ax.get_proj())
           d = (event.xdata-x)*(event.xdata-x) + (event.ydata-y)*(event.ydata-y)
           if d < dist:
              dist = d
              index = i
        #显示所选调色板
        for i in range(num[o]):
           rgb = LABtoRGB(RegularLAB([l[index][i],a[index][i],b[index][i]]))
           srv='%02x'%int((rgb[0]))
           sgv = '%02x' % int((rgb[1]))
           sbv = '%02x' % int((rgb[2]))
           bgstr="#"+srv+sgv+sbv
           Label(root, bg = bgstr).place(x= 750+40*i, y=10,width=30, height=30)
        #进行重新着色
        tar_pal = np.zeros((num[o],3))
        tar_pal[:,0] = l[index]
        tar_pal[:,1] = a[index]
        tar_pal[:,2] = b[index]
        Recolor(image_path,'mattes.png',ori_pal,tar_pal,o)
        img2 = cv2.imread(show_path)
        blue2,green2,red2 = cv2.split(img2)
        img2 = cv2.merge((red2,green2,blue2))
        img2 = cv2.resize(img2,(600,600))
        imm2 = Image.fromarray(img2)
        global imgtk2
        imgtk2 = ImageTk.PhotoImage(image=imm2)
        Label(root, image=imgtk2).place(x=0,y=50,width=600,height= 600) 

fig.canvas.mpl_connect('button_press_event', on_press)

root.mainloop()

