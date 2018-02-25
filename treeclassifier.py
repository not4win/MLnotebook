#simple example of tree classifier of classsifying a male and a female 
from sklearn import tree
#[height,weight,shoe size]
X=[[181,80,44],[177,70,43],[160,60,38],[160,60,38],[154,54,37],[165,65,40],
   [190,90,47],[175,64,39],[177,70,40],[181,85,43]]
#Outputs
Y=['male','female','female','female','male','male','male','female','male','female']
#fitting values into the tree
clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,Y)
#prediction
prediction=clf.predict([[190,70,43]])
#predicts female
print(prediction)
