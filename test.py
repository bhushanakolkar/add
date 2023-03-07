import pickle

pickle_rf=open("RFClassifier",'rb')
classifier_rf=pickle.load(pickle_rf)

probability = classifier_rf.predict_proba([[30,2,1,1,0,1500,1,0,2,20,4,200,0,10,1,0]])[0][1]
    #prediction = classifier.predict([[age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome]])
prediction = classifier_rf.predict([[30,2,1,1,0,1500,1,0,2,20,4,200,0,10,1,0]])

print(probability, prediction)


    