
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


data = {
    "Income": [50000, 60000, 20000, 30000, 80000,
               25000, 70000, 40000, 90000, 35000,
               65000, 22000, 75000, 45000, 55000],

    "Credit": [1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
               1, 0, 1, 1, 1],

    "Loan": [200000, 250000, 300000, 100000, 300000,
             250000, 200000, 150000, 350000, 250000,
             200000, 300000, 250000, 180000, 200000],

    "Debt": [5000, 8000, 15000, 4000, 10000,
             12000, 7000, 10000, 9000, 14000,
             6000, 16000, 8000, 5000, 7000],

    "Status": [1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
               1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)


X = df[["Income", "Credit", "Loan", "Debt"]]
y = df["Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

accuracy1 = accuracy_score(y_test, pred1) * 100

print("Logistic Regression Accuracy:",
      round(accuracy1, 2), "%")


model2 = DecisionTreeClassifier(random_state=42)
model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

accuracy2 = accuracy_score(y_test, pred2) * 100

print("Decision Tree Accuracy:",
      round(accuracy2, 2), "%")


income = float(input("Enter income: "))
credit = int(input("Credit history (1=Good, 0=Poor): "))
loan = float(input("Enter loan amount: "))
debt = float(input("Enter existing debt: "))

applicant = pd.DataFrame(
    [[income, credit, loan, debt]],
    columns=["Income", "Credit", "Loan", "Debt"]
)

result1 = model1.predict(applicant)[0]
result2 = model2.predict(applicant)[0]

print("\n--- LOAN ELIGIBILITY RESULT ---")

print("Logistic Regression:",
      "Eligible" if result1 == 1 else "Not Eligible")

print("Decision Tree:",
      "Eligible" if result2 == 1 else "Not Eligible")


plt.figure(figsize=(6, 4))

models = ["Logistic Regression", "Decision Tree"]
accuracies = [accuracy1, accuracy2]

bars = plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)
plt.bar_label(bars, fmt="%.2f%%")
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 4))

plt.plot(df["Income"], df["Loan"], marker="o")

plt.title("Income vs Loan Amount")
plt.xlabel("Income")
plt.ylabel("Loan Amount")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 4))

plt.scatter(df["Income"], df["Debt"])

plt.title("Income vs Existing Debt")
plt.xlabel("Income")
plt.ylabel("Existing Debt")
plt.grid(True)
plt.tight_layout()
plt.show()


status_count = df["Status"].value_counts().reindex(
    [1, 0], fill_value=0
)

plt.figure(figsize=(5, 5))

plt.pie(
    status_count.values,
    labels=["Eligible", "Not Eligible"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Loan Eligibility Distribution")
plt.tight_layout()
plt.show()


cm = confusion_matrix(y_test, pred1, labels=[0, 1])

plt.figure(figsize=(5, 4))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Logistic Regression")
plt.colorbar()

plt.xticks([0, 1], ["Not Eligible", "Eligible"])
plt.yticks([0, 1], ["Not Eligible", "Eligible"])

plt.xlabel("Predicted Status")
plt.ylabel("Actual Status")

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center")

plt.tight_layout()
plt.show()
