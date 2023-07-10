from lib2to3.pgen2.pgen import DFAState
from flask import *
import pandas as pd
import numpy as np
import pickle
from xgboost import  XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


app= Flask(__name__)

def prediction(input_df):
    ct=pickle.load(open("coltranform.pkl","rb"))
    xg=pickle.load(open("model.pkl","rb"))
    x=ct.fit_transform(input_df)
    output=xg.predict(x)[0]
    if output==1:
      
      return "Customer will continue the services of the bank"
    else:
      return "Customer will not continue the services of the bank"


@app.route("/")
def displayform():
    return render_template("home.html")

@app.route("/predict",methods=["POST"])

def getinputdata():
    CreditScore= int(request.form["CreditScore"])
    Geography= request.form["Geography"]
    Gender= request.form["Gender"]
    Age=int(request.form["Age"])
    Tenure=int(request.form["Tenure"])
    Balance= float(request.form["Balance"])
    NumOfProducts=int( request.form["NumOfProducts"])
    HasCrCard= request.form["HasCrCard"]
    if HasCrCard== "Yes":
        HasCrCard= 1
    else:
        HasCrCard= 0

    IsActiveMember=  request.form["HasCrCard"]
    if IsActiveMember== "Yes":
        IsActiveMember= 1
    else:
        IsActiveMember= 0
    
    EstimatedSalary= float(request.form["EstimatedSalary"])

    df= pd.DataFrame(data= [[CreditScore,Geography, Gender,Age , Tenure ,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]],
                     columns=["CreditScore","Geography", "Gender","Age" , "Tenure" ,"Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"])
    df["Age"]= np.log(df["Age"])
    output=prediction(df)
    return render_template("home.html",output=output)
   

if(__name__=="__main__"):
    app.run(debug=True)