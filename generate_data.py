"""
Generates a realistic synthetic telecom/subscription customer churn dataset.
This mimics the structure of the well-known Telco Customer Churn dataset
so it's a defensible, standard-looking ML problem.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 3000

tenure = np.random.randint(1, 72, n)
monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20])
tech_support = np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6])
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.20])
payment_method = np.random.choice(
    ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n
)
num_support_calls = np.random.poisson(1.5, n)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
paperless_billing = np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4])
total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.9, 1.0, n), 2)

# Build churn probability from a realistic logic (not pure random) so the model
# has real signal to learn -- this matters for a defensible, explainable project.
churn_prob = (
    0.35 * (contract_type == 'Month-to-month').astype(int)
    + 0.20 * (tech_support == 'No').astype(int)
    + 0.15 * (internet_service == 'Fiber optic').astype(int)
    + 0.10 * (payment_method == 'Electronic check').astype(int)
    + 0.03 * num_support_calls
    - 0.15 * (tenure / 72)
    + 0.05 * senior_citizen
    + np.random.normal(0, 0.1, n)
)
churn_prob = np.clip(churn_prob, 0, 1)
churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

df = pd.DataFrame({
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'contract_type': contract_type,
    'tech_support': tech_support,
    'internet_service': internet_service,
    'payment_method': payment_method,
    'num_support_calls': num_support_calls,
    'senior_citizen': senior_citizen,
    'paperless_billing': paperless_billing,
    'churn': churn
})

df.to_csv('/home/claude/churn_project/churn_data.csv', index=False)
print(f"Generated {len(df)} rows. Churn rate: {df['churn'].mean():.2%}")
print(df.head())
