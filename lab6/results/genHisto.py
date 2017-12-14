import sys
import os
import math

filepath = "inter_issue_dists.txt"

def printToFile(histo, labels, filepath):
    filepath = filepath[:-3]+"tsv"
    F = open(filepath, "w")
    F.write("Bins \t Frequency \n")
    for i in range(0, len(histo)):
        F.write(labels[i] + "\t" + str(histo[i]) + "\n")
    return

def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    F = open(filepath, "r")
    data = []
    for line in F:
        data.append(int(line.strip()))

    F.close()

    mindist = min(data)
    maxdist = max(data)

    histogram = [0]*10
    bins = []


    if (maxdist - mindist < 10):
        stride = 1
        bins = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        #populate histogram
        for i in data:
            histogram[i] += 1

    else:
        stride = int(math.ceil((maxdist - mindist)/10.0))
        #populate histogram
        for i in data:
            histogram[i/stride] += 1
        #populate bin labels
        startval = 0
        endval = stride-1
        for i in range(0, 10):
            binlabel = str(startval) + "-" + str(endval)
            bins.append(binlabel)
            startval += stride
            endval += stride


    printToFile(histogram, bins, filepath)
    return


if __name__ == "__main__":
    main()
