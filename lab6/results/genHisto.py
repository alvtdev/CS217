import sys
import os

def main():
    filepath = "inter_issue_dists.txt"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]

    F = open(filepath, "r")
    data = []
    for line in F:
        data.append(int(line.strip()))

    print min(data) 
    print max(data)

    mindist = min(data)
    maxdist = max(data)

    histogram = [0]*10
    bins = []


    if (maxdist - mindist < 10):
        stride = 1
        bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for i in data:
            histogram[i] += 1

    else:
        stride = (maxdist - mindist)/10
        for i in data:
            histogram[i/stride] += 1

    print("stride = " + str(stride))
    print histogram


if __name__ == "__main__":
    main()
