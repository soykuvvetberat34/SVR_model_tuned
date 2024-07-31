from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\Hitters.csv")
df=df.dropna()
dms=pd.get_dummies(df[["League","Division","NewLeague"]])
y=df["Salary"]
x_=df.drop(["Salary","League","Division","NewLeague"],axis=1).astype("float64")
x=pd.concat([x_,dms],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=99)

svr_model=SVR(kernel="linear")
svr_model.fit(x_train,y_train)
predict_svr=svr_model.predict(x_test)
#modelin değişkenlerini bulma
intercept=svr_model.intercept_
coefs=svr_model.coef_
#RMSE bulma
RMSE=np.sqrt(mean_squared_error(y_test,predict_svr))
print(RMSE)


#Model tuning
svr_params={
    "C":[0.1,0.5,1,3]#svr mdoelinde kullanmak için ceza katsayıları ayarladık
}
svr_cv_model=GridSearchCV(svr_model,svr_params,cv=5,verbose=2,n_jobs=-1)
#uzun sürecek bir işlemdir
#verbose işlem sırasında açıklamalar yapmaya yarar
#njobs ise işlemciyi tam performans kullanmaya yarar
svr_cv_model.fit(x_train,y_train)
best_params_of_c=svr_cv_model.best_params_
svr_model_tuned=SVR(kernel="linear",C=best_params_of_c["C"])
svr_model_tuned.fit(x_train,y_train)
predict_svr_tuned=svr_model_tuned.predict(x_test)
RMSE2=np.sqrt(mean_squared_error(y_test,predict_svr_tuned))
print(RMSE2)


