import csv
import json


continent = {}
subcontinent = {}
country = {}
city = {}
source = {}
medium = {}
channel = {}


with open('alldata.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
		channel[row[0]] = 1
		continent[row[3]] = 1
		subcontinent[row[4]] = 1
		country[row[5]] = 1
		city[row[6]] = 1
		source[row[12]] = 1
		medium[row[13]] = 1

alldicts = [continent,subcontinent,country,city,source,medium,channel]

for dic in alldicts:
	x = 5
	for key in dic:
		dic[key] = x
		x+=3


with open('dicts.txt', 'w') as file:
	file.write(json.dumps(continent))
	file.write("\n")
	file.write(json.dumps(subcontinent))
	file.write("\n")
	file.write(json.dumps(country))
	file.write("\n")
	file.write(json.dumps(city))
	file.write("\n")
	file.write(json.dumps(source))
	file.write("\n")
	file.write(json.dumps(medium))
	file.write("\n")
	file.write(json.dumps(channel))
	file.write("\n")



