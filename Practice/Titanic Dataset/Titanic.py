import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv')
print(data.head())

print(data[['Sex', 'Survived']].groupby('Sex').mean())

# Ubah 'Sex' jadi numerik dulu
data['Sex_num'] = data['Sex'].map({'male': 0, 'female': 1})

# Lihat korelasi dengan target
print(data[['Sex_num', 'Fare', 'Survived']].corr())

# plt.figure(figsize=(8, 4))
# data.boxplot(column='Fare', by='Survived')
# plt.title('Distribusi Fare berdasarkan Kelangsungan Hidup')
# plt.suptitle('')
# plt.xlabel('Survived')
# plt.ylabel('Fare')
# plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/refs/heads/master/titanic.csv'
data = pd.read_csv(url)

# 2. Pilih fitur dan label
data['Sex_num'] = data['Sex'].map({'male': 0, 'female': 1})
X = data[['Sex_num', 'Fare']]
y = data['Survived']

# 3. Tangani missing value
X['Fare'].fillna(X['Fare'].median(), inplace=True)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Buat dan latih model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Prediksi
y_pred = model.predict(X_test)

# 7. Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


