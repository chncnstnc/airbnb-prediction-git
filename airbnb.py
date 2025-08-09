# source venv/bin/activate
# python airbnb.py
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# get data from db
conn = sqlite3.connect("airbnb.db")
cur = conn.cursor()

query = "SELECT room_type, neighbourhood, number_of_reviews, availability_365, price FROM Listing"
cur.execute(query)
rows = cur.fetchall()  # get all rows

# get col names
cols = [desc[0] for desc in cur.description]

# conv to dataframe pandas
df = pd.DataFrame(rows, columns=cols)

conn.close()

# remove rows w/ missing vals
df = df.dropna()
df = df[df['price'] > 0]
df = df[df['price'] < 9000]  # remove that one outlier at 9750

# convert room type and neighborhood(categorical var) into numbers (dummy var)
# room type and neighborhood are the two features most likely contributing to price
# availability and number of reviews are already numbers -> can use directly for training
df_dummy = pd.get_dummies(df, columns=['room_type', 'neighbourhood'], drop_first=True)  # drop_first = True avoids redundancy

# setup input and output
x = df_dummy.drop(columns=['price'])  # everything we use to predict price(room type, neighborhood, review ct) + remove price col
y = df_dummy['price']  # actual price col, what we are trying to predict

# train/test (70% train, 30% test)
# x_train, x_test: feature data for training and testing
# y_train, y_test: actual price data for training and testing (target vals)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30)  # random number

# train model !!!
model = LinearRegression()  # using linear reg as model :P
model.fit(x_train, y_train)  # fit the model to training data (learning from data)
y_train_predicted = model.predict(x_train)  # predict on training data

# predict on testing data
y_predicted = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))  # root mean squared error: how far the predictions are on avg
r2 = r2_score(y_test, y_predicted)  # r^2: how well the model explains difference in price (close to 1 -> better at explaining)

# plotting training data
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_predicted, alpha=0.5)
plt.xlabel("Actual Prices ($US)")
plt.ylabel("Predicted Prices ($US)")
plt.title("Actual vs Predicted Airbnb Prices in Boston (Training)")
# predicted price = actual price for every point that lies on this line
min_val_train = min(y_train.min(), y_train_predicted.min())
max_val_train = max(y_train.max(), y_train_predicted.max())
plt.plot([min_val_train, max_val_train], [min_val_train, max_val_train], 'k:')  # ref line that shows what perfect predictions would look like

# display RMSE and R^2
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_predicted))
train_r2 = r2_score(y_train, y_train_predicted)
plt.text(0.025, 0.9, f'RMSE: {train_rmse:.2f}\n R^2: {train_r2:.2f}', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show(block=False)  # setting to false lets both windows appear simultaneously

# plotting testing data
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_predicted, alpha=0.5)
plt.xlabel("Actual Prices ($US)")
plt.ylabel("Predicted Prices ($US)")
plt.title("Actual vs Predicted Airbnb Prices in Boston (Testing)")
min_val = min(y_test.min(), y_predicted.min())
max_val = max(y_test.max(), y_predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k:')


# display RMSE and R^2
plt.text(0.025, 0.9, f'RMSE: {rmse:.2f}\n R^2: {r2:.2f}', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()  # displays window for graph

