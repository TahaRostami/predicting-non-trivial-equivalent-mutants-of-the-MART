import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from feature_engine.encoding import MeanEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score,matthews_corrcoef,balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

X=pd.read_parquet("../data/Codeflaws_features_Eq_4.parquet")
y=X['Eq']
projectID=X['projectID']
mutantID=X['mutantID']
X=X.drop(['Eq','mutantID','projectID'],axis=1)

cv=StratifiedGroupKFold(n_splits=10,shuffle=True,random_state=42)


y_preds = []
y_trues = []
roundx=0

for train_idxs, test_idxs in cv.split(X, y, projectID):

    roundx+=1

    X_train, y_train = X.loc[train_idxs, :], y.loc[train_idxs]
    X_test, y_test = X.loc[test_idxs, :], y.loc[test_idxs]

    cf = MeanEncoder().fit(X_train,y_train)
    X_train,X_test=cf.transform(X_train),cf.transform(X_test)

    # for DecisionTree-based models MinMaxScaler is optional
    scaler=MinMaxScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    model = DecisionTreeClassifier(criterion='log_loss', max_depth=8).fit(X_train, y_train)
    #model = xgb.XGBClassifier(n_estimators=50, max_depth=8, n_jobs=-1, learning_rate=0.1, scale_pos_weight=0.8).fit(X_train, y_train)

    y_preds += list(model.predict_proba(X_test)[:, 1])
    y_trues += list(y_test)




thr=0.50
y_preds_thr=[1 if pr>=thr else 0 for pr in y_preds]
print("AUC:", roc_auc_score(y_trues, y_preds))
print("MCC:",matthews_corrcoef(y_trues,y_preds_thr))
print("ABA:",balanced_accuracy_score(y_trues,y_preds_thr,adjusted=True))
