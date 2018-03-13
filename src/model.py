import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pylab import plot, show, savefig, xlim, figure, \
                hold, ylim, legend, boxplot, setp, axes
import numpy as np
from sklearn import tree

INPUT_FILE = '../data/LoanStats3a.csv'
COLUMNS_TO_KEEP = [
	'loan_amnt',
	'term',
	'int_rate',
	'installment',
	'emp_length',
	'home_ownership',
	'annual_inc',
	'purpose',
	'loan_status',
	'addr_state',
	'dti',
	'delinq_2yrs',
	'mths_since_last_delinq',
	'mths_since_last_record',
	'mths_since_last_major_derog'
]

def main():

	print 'loading data...'
	data = pd.read_csv(INPUT_FILE, low_memory=False)

	print list(data.columns)

	data = data[:35000]
	data = data[COLUMNS_TO_KEEP]

	data['charged_off'] = [1 if x == 'Charged Off' else 0 for x in data['loan_status']]

	print data[['loan_status', 'charged_off']]
	print list(data.columns)

	fig = figure()
	ax = axes()
	hold(True)

	amt_charged_off = list(data.loc[data['loan_status'] == 'Charged Off']['annual_inc'])
	amt_fully_paid = list(data.loc[data['loan_status'] == 'Fully Paid']['annual_inc'])

	# print amt_charged_off
	bp = boxplot(amt_charged_off, positions = [1], widths = 0.6)
	bp = boxplot(amt_fully_paid, positions = [2], widths = 0.6)

	ax.set_xticklabels(['Charged off', 'Fully paid'])
	ax.set_xticks([1, 2])

	xlim(0,3)
	ylim(0,300000)

	show()

	p_by_purpose = data[['purpose','charged_off']].groupby(['purpose'])
	# p_charged_off = p_charged_off.sort_values('charged_off')

	print p_by_purpose.describe()

	p_by_purpose.boxplot()

	data['int_rate'] = [float(per.strip('%'))/100.0 for per in data['int_rate']]
	X = data[['int_rate']]
	Y = data['charged_off']
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X, Y)

	import graphviz 
	dot_data = tree.export_graphviz(clf, out_file=None) 
	graph = graphviz.Source(dot_data) 
	graph.render("loan") 

	# print p_charged_off
	# print type(p_charged_off)

	# p = list(p_charged_off['charged_off'])
	# y_pos = np.arange(len(p))
 
	# plt.bar(y_pos, p, align='center', alpha=0.5)
	# plt.xticks(y_pos, p_charged_off.groups)

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# cax = ax.matshow(data.corr())
	# fig.colorbar(cax)

	# ax.set_xticklabels(['']+list(data.columns))
	# ax.set_yticklabels(['']+list(data.columns))

	# plt.boxplot(data.loc[data['loan_status'] == 'Charged Off'])
	# plt.boxplot(data.loc[data['loan_status'] == 'Charged Off'])
	# plt.show()

if __name__ == "__main__":
	main()