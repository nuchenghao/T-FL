import random
import csv

import matplotlib.pyplot as plt


# 假设你的有序对列表如下
def savePairToCsv(pairs):
    with open("result.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(pairs)


def openCsvToPair():
    pairs = []
    with open('result.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            a, b = row
            pairs.append((int(a), float(b)))  # 类型转换
    return pairs


def drawResults(stateInServer, path, title):
    # 解包列表中的有序对到两个不同的列表
    pairs = stateInServer.resultRecord
    a_values, b_values = zip(*pairs)
    # 使用matplotlib绘制散点图和线图
    plt.figure(figsize=(10, 5))  # 设置图表大小
    plt.plot(a_values, b_values, '-o')  # '-o'表示点之间用线连接
    plt.xlim(min(a_values), max(a_values) + 1)  # x轴范围稍微宽一点，以便起始点位于边框上

    plt.title(title)  # 设置图表标题
    plt.xlabel('time(minutes)')  # 设置x轴标签
    plt.ylabel('accuracy')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    # plt.show()  # 显示图表
    plt.savefig(path)

# drawResults("noniid.png")
