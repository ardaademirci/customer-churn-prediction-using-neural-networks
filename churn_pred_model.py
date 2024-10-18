import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#   In my case importing directly from keras does not work, but it might work for you so try that first.
#   However, if that's not the case you can then try the commented imports. If nothing works, go along with my imports.
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Dropout, Input#InputLayer
#from tensorflow.python.keras.optimizers import adam_v2
#from tensorflow.python.keras.callbacks import EarlyStopping

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, Input#InputLayer
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping

## Load Dataset

dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:,3:-1].values
y= dataset.iloc[:, -1].values       # target variable 'Exited'

#   print(dataset.head())

#   print("\nMissing Values:")
#   print(dataset.isnull().sum())       # There should be no missing values


## Visualize Data --- Historgrams

plt.figure(figsize=(16,10))

# Geography distribution
plt.subplot(2,3,1)
sns.histplot(dataset['Geography'], bins=15, kde=True)
plt.title('Geography Distribution')

# Credit score distribution
plt.subplot(2,3,2)
sns.histplot(dataset['CreditScore'], bins=15, kde=True)
plt.title('Credit Score Distribution')

# Age distribution
plt.subplot(2,3,3)
sns.histplot(dataset['Age'], bins=15, kde=True)
plt.title('Age Distribution')

plt.tight_layout()
#plt.show()

## Visualize Data --- Box plots and Count plots

plt.figure(figsize=(16,10))

# Balance vs Has Credit Card
plt.subplot(2,3,1)
sns.boxplot(x='HasCrCard', y='Balance', data=dataset)
plt.title('Balance vs Has Credit Card')

# Tenure vs Number of Products
plt.subplot(2,3,2)
sns.boxplot(x='NumOfProducts', y='Tenure', data=dataset)
plt.title('Tenure vs Number of Products')

# Geography vs Tenure
plt.subplot(2,3,3)
sns.boxplot(x='Tenure', y='Geography', data=dataset)
plt.title('Geography vs Tenure')

# Credit Score vs Numebr of Products
plt.subplot(2,3,4)
sns.countplot(x='NumOfProducts', hue='CreditScore', data=dataset)
plt.title('Credit Score vs Numebr of Products')

# Number of Products vs Has Credit Card
plt.subplot(2,3,5)
sns.countplot(x='HasCrCard', hue='NumOfProducts', data=dataset)
plt.title('Number of Prodcuts vs Has Credit Card')

# Tenure vs Is Active Member
plt.subplot(2,3,6)
sns.countplot(x='IsActiveMember', hue='Tenure', data=dataset)
plt.title('Tenure vs Is Active Member')

plt.tight_layout()
#plt.show()


## Encoding Categorical Data

le = LabelEncoder()

# Label Encoding the 'Gender' Column
X[:,2] = le.fit_transform(X[:,2])

# One Hot Encoding the 'Geography' column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), [1])], remainder='passthrough')

X = ct.fit_transform(X)
print("Encoded Data:")
print(X)
print(y)


## Splitting the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Scaling the Features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


## Model Building

def build_model(input_shape):
    model = Sequential([
        #Input(shape=input_shape),       # Input Layer
        Dense(64, activation='relu', input_shape=input_shape),   # 1st Hidden Layer with 64 neurons and ReLU activation function
        Dropout(0.5),
        Dense(32, activation='relu'),   # 2nd Hidden Layer with 32 neurons and ReLU activation function
        Dropout(0.5),
        Dense(1, activation='relu')     # 3rd Hidden Layer with 1 neurons and ReLU activation function
    ])
    return model

input_shape = (X_train.shape[1],)
model = build_model(input_shape)


## Compiling the model
#optimizer = adam_v2.Adam(learning_rate=0.001)      #   If in your case you have to use adam_v2 library, use this line
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.summary()    # Model summary


## Training the model


history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=early_stopping,
    verbose=1
)

# Plotting training history

def plot_history(history, metric):
    plt.figure(figsize=(10,6))
    plt.plot(history.history[metric], label=f'Training {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.title(f'Training and Validation {metric.capitalize()}')
    plt.legend()
    plt.show()

plot_history(history, 'accuracy')
plot_history(history, 'loss')


## Evaluating the Model

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')


## Testing the Artificial Neural Netwrok to predict specific customer churn
p1 = model.predict(scaler.transform([[0, 0, 1, 608, 0, 41, 1, 83807.86, 1, 0, 1, 112542.58]]))
p2 = model.predict(scaler.transform([[1, 0, 0, 653, 1, 41, 8, 102768.42, 1, 1, 0, 55663.85]]))
p3 = model.predict(scaler.transform([[0, 1, 0, 635, 1, 53, 1, 117005.55, 1, 0, 0, 123646.57]]))
p4 = model.predict(scaler.transform([[1, 0, 0, 711, 0, 28, 8, 0, 2, 0, 0, 105159.89]]))

print(p1, p2, p3, p4)

# A prediction less than 0.5 means that the model predicts the customer will stay. 
print(p1 < 0.5)
print(p2 < 0.5)
print(p3 < 0.5)
print(p4 < 0.5)
