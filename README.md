# **Basketball Analytics with Linear Regression**

## **Overview**
This project uses **linear regression** to analyze basketball team and player performance metrics. By applying statistical modeling, it demonstrates how to predict outcomes such as **win percentages** or **player impacts** based on offensive and defensive ratings, usage rates, minutes played, and other performance metrics. The project also provides insights into how these predictors influence success, offering actionable data for analysts, coaches, and teams.

---

## **Libraries Used**
The following Python libraries are required:
1. **scikit-learn**:
   - `LinearRegression`: To build and train the regression model.
   - `mean_squared_error` & `r2_score`: For evaluating model performance.
   - `train_test_split`: For splitting the dataset into training and testing subsets.
   - `StandardScaler`: For scaling features to improve model performance.
2. **matplotlib**:
   - To visualize actual vs. predicted values with scatter plots and trend lines.
3. **pandas**:
   - For reading, cleaning, and processing datasets.

---

## **Data Requirements**
### **Dataset Features**
The analysis requires datasets with team or player performance statistics. Below are the required columns based on use cases:
- **Team Data**:
  - `Team`: Team name.
  - `ORtg`: Offensive rating.
  - `DRtg`: Defensive rating.
  - `W/L%`: Win percentage.
  
- **Player Data**:
  - `Player`: Player name.
  - `MP`: Minutes played.
  - `USG%`: Usage rate.
  - `PTS`, `AST`, `REB`, `STL`, `BLK`: Key performance metrics.

If your dataset does not include these columns, you may need to preprocess or enrich the data.

---

## **Usage Instructions**
### **1. Installation**
Ensure you have Python 3 installed. Install the required dependencies:
```bash
pip install scikit-learn matplotlib pandas
```

### **2. Load the Dataset**
Prepare your dataset as a CSV file and load it using pandas:
```python
import pandas as pd
df = pd.read_csv("your_dataset.csv")
```

### **3. Preprocess the Data**
- **Normalize Features**: Use `StandardScaler` to scale numeric columns:
  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(df[['ORtg', 'DRtg']])
  ```
- **Handle Missing Values**: Fill missing values with column means:
  ```python
  df.fillna(df.mean(), inplace=True)
  ```

### **4. Train the Model**
Split the data into training and testing sets, and fit a regression model:
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X_scaled, df['W/L%'], test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
```

### **5. Evaluate the Model**
Evaluate the model's performance using MSE and R-squared metrics:
```python
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
```

### **6. Visualize Results**
Plot the actual vs. predicted values:
```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Fit')
plt.xlabel('Actual Win Rate')
plt.ylabel('Predicted Win Rate')
plt.title('Actual vs Predicted Win Rate')
plt.legend()
plt.show()
```

---

## **Outputs**
1. **Coefficients**:
   - Quantify the impact of each feature on the target variable.
   - Example: Coefficients for `ORtg` and `DRtg` represent their contribution to win percentage (`W/L%`).
2. **Performance Metrics**:
   - **Mean Squared Error (MSE)**: Measures average prediction error.
   - **R-squared (R²)**: Explains the proportion of variance captured by the model.
3. **Visualization**:
   - Scatter plot of actual vs. predicted values with a "perfect fit" line.

---

## **Extending the Project**
### **Additional Features**:
- Include additional metrics such as turnovers, pace, and free throw rates to improve accuracy.
- Analyze non-linear relationships using polynomial regression or tree-based models.

### **Broader Applications**:
- Optimize game strategies by identifying high-impact players and team strengths.
- Predict player salaries or future performance based on historical data.

---

## **Need Additional Data?**
If your dataset is missing key columns or you need help preparing data, let me know! I can assist with:
- Preprocessing data,
- Handling missing values or inconsistencies,
- Adding new metrics for deeper insights.

This project is highly adaptable and can be extended for other sports or analytical tasks. Let me know how I can support your objectives!
