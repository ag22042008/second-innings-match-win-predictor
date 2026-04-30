# 🏏 IPL Win Predictor

A Machine Learning powered web application that predicts the **win probability of an IPL match in real-time** during the second innings.

Built using **Python, Streamlit, and Scikit-learn**, this project provides an interactive UI along with visual insights like probability gauges, team stats, and head-to-head analysis.

---

## 🚀 Features

* 🔮 Predict win probability based on live match situation
* 📊 Visualizations:

  * Win probability gauge
  * Team performance stats
  * Head-to-head comparison
* 🧠 Machine Learning model (Logistic Regression)
* 🎯 Real-time inputs (score, overs, wickets, etc.)

---

## 📁 Project Structure

```
.
├── app.py                 # Main Streamlit application
├── matches.csv           # Match-level data
├── deliveries.csv / zip  # Ball-by-ball data
├── pipe.pkl              # Trained ML model (optional)
└── README.md             # Project documentation
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Data Processing**

   * Uses IPL ball-by-ball dataset
   * Calculates:

     * Runs left
     * Balls left
     * Wickets remaining
     * Current Run Rate (CRR)
     * Required Run Rate (RRR)

2. **Model**

   * Logistic Regression
   * Encodes categorical features using OneHotEncoder
   * Predicts:

     * Win probability
     * Loss probability

3. **Prediction Input Features**

   * Batting Team
   * Bowling Team
   * City
   * Runs Left
   * Balls Left
   * Wickets Left
   * CRR & RRR

---

## 📊 Example Input

| Feature | Value |
| ------- | ----- |
| Target  | 180   |
| Score   | 100   |
| Overs   | 12.0  |
| Wickets | 3     |

👉 Output:

* Win Probability: 62.5%
* Loss Probability: 37.5%

---

## 📈 Model Performance

* Algorithm: Logistic Regression
* Accuracy: ~84%
* Dataset: IPL 2008–2019

---

## ⚠️ Notes

* Ensure dataset column names match:

  * `city`, `batting_team`, `bowling_team`
  * `runs_left`, `BallsLeft`, `wickets_left`
* Model retrains automatically if `pipe.pkl` is not found
* Data cleaning handles team name changes

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas & NumPy
* Scikit-learn
* Plotly

---

## 🎯 Future Improvements

* Add live match API integration
* Improve model accuracy with advanced algorithms
* Deploy on cloud (Streamlit Cloud / AWS)
* Add first innings prediction

---

##  Contributing

Feel free to fork this repo and improve it. Pull requests are welcome!

---

## 📄 License

This project is open-source and available under the MIT License.

---

##  Acknowledgment

Dataset sourced from IPL historical data.

---

## 👨‍💻 Author

Aditya Gupta

---

⭐ If you like this project, consider giving it a star!
