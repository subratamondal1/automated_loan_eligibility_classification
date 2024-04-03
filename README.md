<h1 align="center">Classify Loan Eligibility</h1>

<h2 align="left">Problem Statement</h2>

**`Automate`** the loan eligibility process based on customer detail provided while filling online application form. It is a **`classification problem`** where we have to predict whether a loan would be approved or not.

<h2 align="left">Data</h2>

The data corresponds to a set of financial requests associated with individuals.

| **`Variables`** | **`Description`** |
| --- | --- |
| **Loan\_ID** | Unique Loan ID |
| **Gender** | Male/ Female |
| **Married** | Applicant married (Y/N) |
| **Dependents** | Number of dependents |
| **Education** | Applicant Education (Graduate/ Under Graduate) |
| **Self\_Employed** | Self employed (Y/N) |
| **ApplicantIncome** | Applicant income |
| **CoapplicantIncome** | Coapplicant income |
| **LoanAmount** | Loan amount in thousands |
| **Loan\_Amount\_Term** | Term of loan in months |
| **Credit\_History** | credit history meets guidelines |
| **Property\_Area** | Urban/ Semi Urban/ Rural |
| **Loan\_Status** | Loan approved (Y/N) |

<h2 align="left">MLflow Local Setup</h2>

<img src="https://mlflow.org/docs/latest/_static/MLflow-logo-final-black.png" width=200/>

Run the below commands in the terminal one by one.

```bash
# To install the 'mlflow' package
1. pip install mlflow
```

Run the below commands to launch the server @ [`http://127.0.0.1:5000`](http://127.0.0.1:5000)
```bash
# To launch the 'mlflow ui' @ http://127.0.0.1:5000
2. mlflow ui
```
