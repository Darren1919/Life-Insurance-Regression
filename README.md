# 🏦 Life Insurance Claim Prediction  

This project applies **Gradient Boosting Regression** to predict **insurance claim amounts**, helping insurers assess risk and improve claim management.  

## 📂 Dataset  
The dataset contains various policyholder details and claim-related information. The target variable is `total_claim_amount`, and key features include `injury_claim`, `property_claim`, `months_as_customer`, `incident_severity`, `policy_annual_premium`, and more.  

## 📊 Model Performance  
The trained **Gradient Boosting Regressor** achieved:  
✅ **R² Score:** 0.9567 (indicating a strong fit)  
✅ **MAE:** 3922.47  
✅ **MSE:** 29,283,235  
✅ **RMSE:** 5411.40  

### 🔥 Feature Importance  
Below is a visualization of the most important features affecting claim amounts:  

![Feature Importance](download.png)  

## ⚡ How to Run  
Clone the repository, install dependencies, and run the script using the commands below:  
```bash
git clone https://github.com/Darren1919/Life-Insurance-Regression.git
cd Life-Insurance-Regression
pip install -r requirements.txt
python life_insurance_regression.py
