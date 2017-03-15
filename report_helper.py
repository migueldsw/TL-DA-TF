import matplotlib.pyplot as plt
import numpy as np
import os

import sys

reload(sys)
sys.setdefaultencoding("utf-8")

# pretty printing
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


def checkDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# graph plotting
def plotLine(x, y, path, fileName):
    line, = plt.plot(x, y, '-', linewidth=2, color='blue', label='J')

    plt.plot(x, y, 'bo')

    # tick no eixo x
    # plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    # tick no eixo y
    # plt.yticks(np.arange(min(x), max(x)+1, .05))

    # defines fixed x y  range
    #	plt.axis([0,5,1,2])

    # # draw vertical line from [xfinal,xinicial][yfinal,yinicial]
    # for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
    # 	plt.plot([i, i], [2, 1], 'k--')

    # plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.legend(loc=1, borderaxespad=0.)

    # dashes = [10, 5, 100, 5] # 10 points on, 5 off, 100 on, 5 off
    # line.set_dashes(dashes)
    # plt.show()

    # stub
    for i, j in zip(x, y):
        # plt.annotate(str(j),xy=(i,j))
        plt.annotate(str("%.3f" % j), xy=(i, j), xytext=(5, 5), textcoords='offset points')
        # end stub

    checkDir(path)
    plt.savefig(path + "/" + fileName)
    plt.clf()
    plt.close()


def plotValuesLine(y, path, fileName):
    plotLine(range(len(y)), y, path, fileName)


def plotDots(y, path, fileName):
    x = range(len(y))
    plt.plot(x, y, 'bo', label='classes')
    plt.legend(loc=1, borderaxespad=0.)
    # defines fixed x y  range
    plt.axis([0, len(y), min(y) - 1, max(y) + 1])
    # for i,j in zip(x,y):
    # 	plt.annotate(str(j),xy=(i,j))
    checkDir(path)
    plt.savefig(path + "/" + fileName)
    plt.clf()
    plt.close()


# file writing
def writeFile(fileName, lines):
    f = open(fileName, 'w')
    for l in lines:
        f.write(l + "\n")
    f.close()


def appendFile(fileName, lines):
    f = open(fileName, 'a')
    for l in lines:
        f.write(str(l) + "\n")
    f.close()


def strArray(l):
    return ''.join(str(i) + ' ' for i in l)


def strMat(m):
    out = []
    for i in m:
        out.append(strArray(i))
    return out


def writeMat(fileName, m):
    writeFile(fileName, strMat(m))

