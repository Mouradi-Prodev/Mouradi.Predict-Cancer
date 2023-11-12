import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import pickle as pickle
def create_model(data):
    """Create a model for the data."""
    X = data.iloc[:, 1:31].values
    Y = data.iloc[:, 0].values
    # Scaling features to have mean=0 and std=1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    

    X_train,X_test,Y_train,Y_test = train_test_split(
        X,Y,test_size=0.2,random_state=42
    )
    #train the model
    model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    model.fit(X_train, Y_train)
    #test the model
    y_predict = model.predict(X_test)
    print("Accuracy of our model: ",accuracy_score(Y_test,y_predict))
    print("Classification report: \n",classification_report(Y_test,y_predict))

    return model,scaler

    

def get_clean_data():
    data = pd.read_csv('data/data.csv')
    data = data.drop(['Unnamed: 32','id'],axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

    return data

def main():
    data = get_clean_data()
    
    model,scaler = create_model(data)
    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)

   

if __name__ == '__main__':
    main()