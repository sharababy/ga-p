import csv

f = 0

with open('train-01.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	# next(readCSV)  # Skip header line
	for row in readCSV:
		if int(row[-1]) == 1:
			f+=1


print(f)