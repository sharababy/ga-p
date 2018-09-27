import csv

X = []

with open('train-num.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	# next(readCSV)  # Skip header line
	for row in readCSV:
		if int(row[-1]) > 0:
			row[-1] = "1"
		X.append(row)


with open("train-01.csv", 'w') as f:
	csv.writer(f).writerows(X)