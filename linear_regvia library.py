import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# x represents classes and y depicts the population of respective  classes 
x_data = np.array([6,7,8,9,10,11,12]).reshape(-1,1)
y_data = np.array([500,590,650,690,700,850,1000])

# linear regression and fit the value of x in model
model = LinearRegression()
model.fit(x_data,y_data)

# predicted value of y via model
y_pred = model.predict(x_data)

# plotting predicted value graph
plt.figure(figsize=(15, 9))
plt.scatter(x_data, y_data, label="Actual Data")
plt.plot(x_data, y_pred, color='red', label="Predicted Line")
plt.xlabel("classes")
plt.ylabel("students")
plt.title("Linear Regression: Original Scale")
plt.legend()
plt.grid(True)
plt.show()
