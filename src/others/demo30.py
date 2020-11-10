from matplotlib import pyplot as plt
from matplotlib import font_manager

myfont = font_manager.FontProperties(fname="C:\Windows\Fonts\msyh.ttc")


y_1 = [1,2,3,2,2,2,1,1,0,3,1,1,3,4,1,1,5,2,4,6]
y_2 = [2,1,0,3,2,2,2,1,1,5,2,1,1,3,4,1,2,3,4,3]
x = range(11,31)

#设置图形大小
plt.figure(figsize=(15,8),dpi=80)

plt.plot(x,y_1,label="自己",color='red',linestyle=':')
plt.plot(x,y_2,label="同桌",color='blue',linestyle='-.')
#label="自己"  图例

#设置x轴刻度
_xtick_labels = ["{}岁".format(i) for i in x]
plt.xticks(x,_xtick_labels,fontproperties=myfont)
plt.yticks(range(1,9))
#绘制网格
plt.grid(alpha=0.2,linestyle=':')
#alpha=0.4透明度

#添加图例
plt.legend(prop=myfont,loc="upper left")

plt.show()
