import csv
import numpy as np
import json

X = []

def flatten(csvf):
	with open(csvf) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		next(readCSV)  # Skip header line
		for row in readCSV:
			k = []
			k.append(row[0]) # channel grouping
			k.append(row[1]) # date
			k.append(row[3]) # visitor id
			geo = json.loads(row[4])
			if "continent" in geo:
				k.append(geo["continent"])
			else:
				k.append("N/A")
			if "subContinent" in geo:
				k.append(geo["subContinent"])
			else:
				k.append("N/A")
			# if "region" in geo:
			# 	k.append(geo["region"])
			# else:
			# 	k.append("N/A")
			if "country" in geo:
				k.append(geo["country"])
			else:
				k.append("N/A")
			if "city" in geo:
				k.append(geo["city"])
			else:
				k.append("N/A")

			total = json.loads(row[7])
			if "visits" in total:
				k.append(total["visits"])
			else:
				k.append("0")
			if "hits" in total:
				k.append(total["hits"])
			else:
				k.append("0")
			if "pageviews" in total:
				k.append(total["pageviews"])
			else:
				k.append("0")
			if "bounces" in total:
				k.append(total["bounces"])
			else:
				k.append("0")
			if "newVisits" in total:
				k.append(total["newVisits"])
			else:
				k.append("0")


			ts = json.loads(row[8])
			if "source" in ts:
				k.append(ts["source"])
			else:
				k.append("N/A")
			if "medium" in ts:
				k.append(ts["medium"])
			else:
				k.append("N/A")

			k.append(row[9]) # visitnumber
			# k.append(row[10])

			if "transactionRevenue" in total:
				k.append(total["transactionRevenue"])
			else:
				k.append("0")

			X.append(k)

			# print(k)
			# exit()



flatten("train.csv")
flatten("test.csv")


# X = np.asarray(X)

# channelGrouping = {
# 					'Organic Search': 1,
# 					'Referral': 2,
# 					'Paid Search': 3,
# 					'Affiliates': 4,
# 					'Direct': 5,
# 					'Display': 6,
# 					'Social': 7,
# 					'(Other)': 8}



# for x in X:

# 	channelGrouping[x[0]] = 1


# print(channelGrouping)


# np.savetxt("processed-train.csv", X, delimiter=",")

with open("alldata.csv", 'w') as f:
	csv.writer(f).writerows(X)