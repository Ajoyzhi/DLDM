""" 绘制折线图 """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd


def plot_broken_line(x, y, x_name="dos", y_name="dos", labels=[], y_scale=()):
    """ 画折线 """
    colors = ['indianred', 'green', 'dodgerblue']
    for i in range(len(labels)):
        plt.plot(x, y[i], marker='.', ms=8, label=labels[i], color=colors[i])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.ylim(y_scale)
    plt.legend(loc="lower right")
    plt.savefig(x_name+" -- "+y_name)
    plt.show()


# 针对六种dos攻击，绘制四个折线图
dos_index = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']

aek_auc = [0.93408604145235, 0.9703822275677875, 0.973388116976704,
           0.9684820516433579, 0.9724568658808942, 0.9728733915481647]

rbm_auc = [0.9738730458539218, 0.9826502998036684, 0.9780043202024081,
           0.98306619227111, 0.976782606857779, 0.973144622407896]

join_auc = [0.9938673081612923, 0.9919437856206004, 0.9984109074150131,
            0.9895768891974264, 0.9947739940109909, 0.998136034161182]

aek_time = [2.7195000648498535, 6.482999801635742, 6.661999702453613,
            6.773999929428101, 6.447999954223633, 6.882999658584595]

rbm_time = [1.434000015258789, 3.5965001583099365, 3.3610000610351562,
            3.500999927520752, 3.5169999599456787, 3.548999786376953]

join_time = [2.4845000505447388, 6.000499963760376, 6.769000053405762,
             5.690000057220459, 5.7769999504089355, 5.761000156402588]

aek_mcc = [0.8749954267582014, 0.9112964713470075, 0.9200139401109736,
           0.9047142369437241, 0.9171571994174942, 0.9184736763101093]

rbm_mcc = [0.9488539171953191, 0.9481248924063427, 0.9350857452996049,
           0.9495465894359449, 0.930398237268225, 0.9188832709707578]

join_mcc = [0.9721217051160606, 0.9381898994288667, 0.9699222876109876,
            0.897866868295231, 0.9367627415794659, 0.9743723166456922]

aek_f_score = [0.9365640223071572, 0.9814989016288858, 0.9833456829328535,
               0.9802825716621746, 0.9827735489485453, 0.983036137147137]

rbm_f_score = [0.9739433788752858, 0.9890661104794871, 0.9859765632020705,
               0.9892371520881984, 0.985283774641472, 0.9831181024251121]

join_f_score = [0.9857960284685068, 0.9867928504518491, 0.9934615834300156,
                0.9776373950696833, 0.9864802199008474, 0.9943561331745456]

plot_broken_line(x=dos_index, y=[aek_auc, rbm_auc, join_auc], x_name="dos", y_name="auc",
                 labels=["ae+kmeans", "rbm+svm", "join"], y_scale=(0.8, 1))
plot_broken_line(x=dos_index, y=[aek_time, rbm_time, join_time], y_name="time", x_name="dos",
                 labels=["ae+kmeans", "rbm+svm", "join"], y_scale=(1, 10))
plot_broken_line(x=dos_index, y=[aek_mcc, rbm_mcc, join_mcc], y_name="mcc", x_name="dos",
                 labels=["ae+kmeans", "rbm+svm", "join"], y_scale=(0.8, 1))
plot_broken_line(x=dos_index, y=[aek_f_score, rbm_f_score, join_f_score], y_name="f-score", x_name="dos",
                 labels=["ae+kmeans", "rbm+svm", "join"], y_scale=(0.8, 1))

# 针对三种非dos攻击，绘制四个折线图
no_dos_index = ['D7', 'D8', 'D9']
aek_no_auc = [0.9731819254489932, 0.960651103094214, 0.960651103094214]
rbm_no_auc = [0.9729036711950492, 0.9630029832107674, 0.9630029832107674]
join_no_auc = [0.9894922797527295, 0.9844742737709047, 0.9844742737709047]

aek_no_time = [6.550000190734863, 6.699000000953674, 6.699000000953674]
rbm_no_time = [3.490999937057495, 3.7044999599456787, 3.7044999599456787]
join_no_time = [9.381999731063843, 6.162999987602234, 6.162999987602234]

aek_no_mcc = [0.9194531739575976, 0.8794225940846363, 0.8794225940846363]
rbm_no_mcc = [0.9181304063709337, 0.8885000383359243, 0.8885000383359243]
join_no_mcc = [0.8754831809875304, 0.926979239532955, 0.926979239532955]

aek_no_f_score = [0.9832315864584159, 0.9752960808621401, 0.9752960808621401]
rbm_no_f_score = [0.9829540292792743, 0.9765842977989949, 0.9765842977989949]
join_no_f_score = [0.9710953947668219, 0.9832934684147658, 0.9832934684147658]

plot_broken_line(x=no_dos_index, y=[aek_no_auc, rbm_no_auc, join_no_auc], y_name="auc", x_name="other attack",
                 labels=["ae+kmeans", "rbm+svm", "join"], y_scale=(0.8, 1))
plot_broken_line(x=no_dos_index, y=[aek_no_time, rbm_no_time, join_no_time], y_name="time", x_name="other attack",
                 labels=["ae+kmeans", "rbm+svm", "join"], y_scale=(1, 10))
plot_broken_line(x=no_dos_index, y=[aek_no_mcc, rbm_no_mcc, join_no_mcc], y_name="mcc", x_name="other attack",
                 labels=["ae+kmeans", "rbm+svm", "join"], y_scale=(0.8, 1))
plot_broken_line(x=no_dos_index, y=[aek_no_f_score, rbm_no_f_score, join_no_f_score], y_name="f_score",
                 x_name="other attack", labels=["ae+kmeans", "rbm+svm", "join"], y_scale=(0.8, 1))
