# regression-project
The repsitory contains the files for Jared Godar's Codeup project on regression and modeling of zillo real estate data.

## About the Project

### Project Goals

The main goal of this project is to be able to accurately predict the values of single unit properties.

This will be accomplished by using past using property data from transactions between May and August 2017 to build various regression models, rating the effectiveness of each model, and testing the best model on new data is has never seen.

I will also determine the distribution of tax rates by state and county.

The ability to accurately value a home is essential for both buyers and sellers. Having an accurate model will allow us to determine which houses are over and under-valued and make apporptiate decisions accordingly. 

### Project Description

This project provides the opportunity to create and evaluate multiple predictive models as well as implement other essential parts of the data science pipeline.

It will incolve pulling relavant data from a SQL database; cleaning that data; splitting the data into training, validation, and test sets; scaling data; feature engineering; exploratory data analysis; modeling; model evaluation; model testing; and effectively communicating findings in written and oral formats.

A home is often the most expensiver purchase one makes in their lifetime. Having a good handle on pricing is essential for both buyers and sellers. An accurate pricing model factoring in the properties of similar homes will allow for appropriate prices to be set as well as the alility to identify under and overvalued homes.

### Initial Questions

- What are the main drivers of home price?
- What are the relative importances of the assorted drivers?
- What factors don't matter?
- Are there any other potentially useful features that can be engineered from the current data available?
- Are the relationships suggested by initial visualizations statistically significant?
- Is the data balanced or unbalanced?
- Are there null values or missing data that must be addressed?
- Are there any duplicates in the dataset?
- Which model feature is most important for this data and business case?
- Which model evaluation metrics are most sensitive to this primary feature?

### Data Dictionary

## Data Dictionary

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| churn_encoded | int64 | 0: Customer retained 1: Customer Churned|

Int64Index: 15947 entries

Data columns (total 10 columns):
____
| variable  |     Dtype    |
|:----------|:-----------------------|
|bedrooms    | float64 |
|bathrooms   | float64 |
|square_feet |  int64 |
|taxes       | float64 |
|home_value   |  float64|
|propertylandusedesc  |  object|
|fips_number   |   int64 |
|zip_code      |   float64 |
|tax_rate (calculated) |  float64 |
|county_name  (engineered) |  object|
|state_name  (engineered) |  object|

</br>
</br>

</br>


|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| senior_citizen       | int64 |    0: Not Senior 1: Senior |
| monthly_charges        | float64 |    month;y charges in dollars |
| tenure       | 
 int64 |    tenure in months |
| paperless_billing_encoded        | int64 |    i0: no paperless billing 1:paperless billing |
| internet_service_type_Fiber optic        | uint8 |    0: No fiber 1: fiber |
| payment_type_Electronic check        | uint8 |    0: no e-check 1:e-check |

### Steps to Reproduce

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md.
- [ ] Download the `zillo_aquire.py`, `zillo_prepare.py`, and `zillo_project_report.ipynb` files into your working directory.
- [ ] Add your own `env` file to your directory. (user, password, host).
- [ ] Run the `zillo_project_report.ipynb` workbook.




### The Plan

![story map](story_map.jpg)

1. **Acquire, clean, prepare, and split the data:**
    - Pull from zillo database.
    - Eliminate any unnecessary or redundant fields.
    - Engineer new, potentially informative features.
    - Search for null values and respond appropriately (delete, impute, etc.).
    - Deal with outliers.
    - Scale data appropriately.
    - Correlate IDs to states and counties.
    - Distribution of tax rates by ocunty.
        - Calculate tax rate using home value and taxes paid.
    - Divide the data in to training, validation, and testing sets (~50-30-20 splits)
2. **Exploratory data analysis:**
    - Visualize pairwaise relationships looking for correlation with home value.
    - Note any interesting correlations or other findings.
    - Test presumptive relationships for statistical significance.
    -  Think of what features would be most useful for model.
    - Record any other interesng observations or findings.
    *NOTE: This data analysis will be limited to the training dataset*
3. **Model generation, assessment, and optimization:**
    - Establish baseline performance (median home price).
    - Generate a basic regression model using only home area, number of bedrroms, number of bathrooms.
    - Calculate evaluation metrics to assess quality of models (RMSE, R^2, and p as primary metrics).
    - Generate additional models incorporating other existing fields.
    - Engineer additional features to use in other models.
    - Evaluate ensemble of better models on validation data to look for overfitting.
    - Select the highest performing model.
    - Test that model with the previously unused and unseen test data once and only once.
4. **Streamline presentation**
    - Take only the most relative information from the working along and create a succinct report that walks through the rationale, steps, code, and observations for the entire data science pipeline of acquiring, cleaning, preparing, modeling, evaluating, and testing our model.
    - Outline next steps for this project:
        - Potential specific changes designed to retain customers
        - Strategy to tests and evaluate implementation of those changes
        - Potential revenu and savings for success

### Key Findings

- Most important factors 
- Factors that don't matter
- Model performance
- Improvement over baseline
- Counties