import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h2o
data = pd.read_csv("/home/python/Desktop/data/student-mat.csv")

data = data.drop('reason',axis =1)
def absence_heatmap(): 
    high_absence = data.loc[data['absences'] > 8]
    corr = high_absence.corr()
    test = sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns, annot = False)
    return test
absence_heatmap()

#bar plot to show weekly alc consumption
list = []
for i in range(11):
    list.append(len(data[data.Dalc == i]))
ax = sns.barplot(x = [0,1,2,3,4,5,6,7,8,9,10], y = list)
plt.ylabel('Number of Students')
plt.xlabel('Weekly alcohol consumption')

# pie chart to show grade distribution compared to weekly alc. consumption
labels = ['2','3','4','5','6','7','8','9','10']
colors = ['lime','blue','orange','cyan','grey','purple','brown','red','darksalmon']
explode = [0,0,0,0,0,0,0,0,0]
sizes = []
for i in range(2,11):
    sizes.append(sum(data[data.Dalc == i].G3))
total_grade = sum(sizes)
average = total_grade/float(len(data))
plt.pie(sizes,explode=explode,colors=colors,labels=labels,autopct = '%1.1f%%')
plt.axis('equal')
plt.title('Total grade : '+str(total_grade))
plt.xlabel('Students grade distribution according to weekly alcohol consumption')


#swam plot to show avg. test score per daily alc intake
ave = sum(data.G3)/float(len(data))
data['ave_line'] = ave
data['average'] = ['above average' if i > ave else 'under average' for i in data.G3]
sns.swarmplot(x='Dalc', y = 'G3', hue = 'average',data= data,palette={'above average':'lime', 'under average': 'red'})

#auto ML
import h2o
h2o.init(max_mem_size = '50g')
h2o_data = h2o.import_file("C:/Users/Daniel Droder/Desktop/alc anal/student-alcohol-consumption/student-mat.csv")
train,test,valid = h2o.splitFrame(h2o_data,ratios = [.4,.3])
from h2o.estimators.gbm import H2OGradientBoostingEstimator
model = H2OGradientBoostingEstimator(ntrees = 10, max_depth = 5)
model.train(x = h2o_data[['sex','age','Dalc','Walc']], y = h2o_data['G1'])


len(h2o_data.columns)

h2o_data = h2o_data.drop(['G1', 'G2'], axis=1)

x = h2o_data.columns
del x[30]
print(x)
x2 = ['school',
 'sex',
 'age',
 'address',
 'famsize',
 'Pstatus',
 'Medu',
 'Fedu',
 'Mjob',
 'Fjob',
 'reason',
 'guardian',
 'traveltime',
 'studytime',
 'failures',
 'schoolsup',
 'famsup',
 'paid',
 'activities',
 'nursery',
 'higher',
 'internet',
 'romantic',
 'famrel',
 'freetime',
 'goout',
 'Dalc',
 'Walc',
 'health',
 'absences']
train,test,valid = h2o_data.split_frame(ratios = [.7,.2])
from h2o.automl import H2OAutoML
aml = H2OAutoML(max_models=20)
aml.train(x=x, y=y, training_frame=train)
aml.leader
lb = aml.leaderboard
lb.head(rows=lb.nrows)

preds = aml.predict(test)

# or:
#preds = aml.leader.predict(test)
preds.head()
test['G3'].head()

m = h2o.get_model(lb[2,"model_id"])
m.varimp(True)
aml.std_coef_plot()




