import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib

# Preprocessing & Feature Engineering
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel # For feature selection
import category_encoders as ce

# Models
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Lasso # Used within SelectFromModel if desired

# Metrics & Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import partial_dependence, PartialDependenceDisplay

# Parameter Tuning Distributions
from scipy.stats import uniform, randint

# Ignore potential warnings
warnings.filterwarnings('ignore', category=UserWarning) # Specifically suppress UserWarning if needed, though fix should work
warnings.filterwarnings('ignore', category=FutureWarning)


# --- Configuration ---
TARGET_COLUMN = 'Premium_Amount'
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ITER_SEARCH = 30 # Number of iterations for RandomizedSearchCV
CV_FOLDS_TUNING = 3
CV_FOLDS_EVAL = 5
FEATURE_SELECTION_THRESHOLD = 'median' # Threshold for SelectFromModel

# --- 1. Load Data ---
try:
    df = pd.read_csv('train_sampled2.csv')
    print(f"Dataset loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'train_sampled2.csv' not found.")
    exit()

# --- 2. Basic Data Preprocessing ---
if 'id' in df.columns:
    df = df.drop(columns=['id'])
    print("Dropped 'id' column.")
df.columns = [col.replace(' ', '_') for col in df.columns]
print("Cleaned column names.")

# --- 3. Enhanced Date Feature Processing ---
try:
    if 'Policy_Start_Date' in df.columns:
        df['Policy_Start_Date'] = pd.to_datetime(df['Policy_Start_Date'], errors='coerce')
        orig_count = len(df)
        df.dropna(subset=['Policy_Start_Date'], inplace=True)
        if orig_count > len(df): print(f"Dropped {orig_count - len(df)} rows with invalid dates.")
        df['Policy_Start_Year'] = df['Policy_Start_Date'].dt.year
        df['Policy_Start_Month'] = df['Policy_Start_Date'].dt.month
        df['Policy_Start_Day'] = df['Policy_Start_Date'].dt.day
        df['Policy_Start_DayOfWeek'] = df['Policy_Start_Date'].dt.dayofweek
        df['Policy_Start_Quarter'] = df['Policy_Start_Date'].dt.quarter
        df['Policy_Start_IsWeekend'] = df['Policy_Start_Date'].dt.dayofweek.isin([5, 6]).astype(int)
        reference_date = pd.Timestamp('2020-01-01')
        df['Days_Since_Reference'] = (df['Policy_Start_Date'] - reference_date).dt.days
        df = df.drop(columns=['Policy_Start_Date'])
        print("Created enhanced date features.")
    else:
        print("Policy_Start_Date column not found, skipping date features.")
except Exception as e:
    print(f"Error processing date features: {e}")

# --- 4. Handle Target Variable ---
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found.")
    exit()
df.dropna(subset=[TARGET_COLUMN], inplace=True)
df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
df.dropna(subset=[TARGET_COLUMN], inplace=True)
p99 = df[TARGET_COLUMN].quantile(0.99)
outliers_count = sum(df[TARGET_COLUMN] > p99)
if outliers_count > 0:
    print(f"Capping {outliers_count} outliers at 99th percentile ({p99:.2f}).")
    df[TARGET_COLUMN] = np.where(df[TARGET_COLUMN] > p99, p99, df[TARGET_COLUMN])

# --- 5. Feature Engineering ---
print("\nPerforming feature engineering...")
temp_imputer_median = SimpleImputer(strategy='median')
num_cols_for_eng = df.select_dtypes(include=np.number).drop(columns=[TARGET_COLUMN]).columns
if len(num_cols_for_eng) > 0:
    df[num_cols_for_eng] = temp_imputer_median.fit_transform(df[num_cols_for_eng])
else:
    print("Warning: No numeric columns found for temporary imputation during FE.")
# Create features
if 'Previous_Claims' in df.columns and 'Insurance_Duration' in df.columns:
    df['Claims_per_Year'] = df['Previous_Claims'] / df['Insurance_Duration'].clip(1)
    df['Has_Claims'] = (df['Previous_Claims'] > 0).astype(int)
if 'Vehicle_Age' in df.columns: df['Vehicle_Age_Risk'] = np.exp(df['Vehicle_Age'] / 10)
if 'Annual_Income' in df.columns:
    df['Log_Income'] = np.log1p(df['Annual_Income'])
    if 'Number_of_Dependents' in df.columns: df['Income_per_Dependent'] = df['Annual_Income'] / df['Number_of_Dependents'].replace(0, 1).fillna(1)
if 'Credit_Score' in df.columns: df['Credit_Factor'] = np.exp((df['Credit_Score'] - 300) / 100)
# Interactions
key_interactions = [('Age', 'Health_Score'), ('Credit_Score', 'Annual_Income'), ('Previous_Claims', 'Vehicle_Age'), ('Age', 'Vehicle_Age'), ('Credit_Score', 'Insurance_Duration'), ('Log_Income', 'Credit_Factor')]
for col1, col2 in key_interactions:
    if col1 in df.columns and col2 in df.columns:
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        # print(f"Created interaction: {col1}_x_{col2}") # Less verbose

# --- 6. Identify Feature Types ---
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
# Ensure no overlap and select final features
numerical_features = [f for f in numerical_features if f in X.columns]
categorical_features = [f for f in categorical_features if f in X.columns]
final_features = numerical_features + categorical_features
X = X[final_features]
print(f"\nIdentified {len(categorical_features)} categorical features.")
print(f"Identified {len(numerical_features)} numerical features.")
print(f"Total features selected: {X.shape[1]}")

# --- 7. Split Data ---
print("\nSplitting data...")
try:
    target_bins = pd.qcut(y, 4, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target_bins)
except ValueError as e:
    print(f"Warning: Stratified split failed ('{e}'). Using non-stratified.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# --- 8. Create Preprocessing Pipelines ---
numerical_transformer = Pipeline(steps=[('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', ce.CatBoostEncoder(handle_unknown='value', handle_missing='value'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_features), ('cat', categorical_transformer, categorical_features)], remainder='passthrough', verbose_feature_names_out=False)
preprocessor.set_output(transform="pandas")

# --- 9. Define 5 Segments ---
print("\nDefining 5 premium segments...")
p25, p50, p75, p95 = y_train.quantile([0.25, 0.50, 0.75, 0.95])
premium_thresholds = {'p25': p25, 'p50': p50, 'p75': p75, 'p95': p95}
print(f"Thresholds: p25={p25:.2f}, p50={p50:.2f}, p75={p75:.2f}, p95={p95:.2f}")
# Train masks
very_low_mask_train = y_train <= p25
low_mask_train = (y_train > p25) & (y_train <= p50)
medium_mask_train = (y_train > p50) & (y_train <= p75)
high_mask_train = (y_train > p75) & (y_train <= p95)
very_high_mask_train = y_train > p95
# Train segment data dictionary
segment_data_train = {'very_low': (X_train[very_low_mask_train], y_train[very_low_mask_train]), 'low': (X_train[low_mask_train], y_train[low_mask_train]), 'medium': (X_train[medium_mask_train], y_train[medium_mask_train]), 'high': (X_train[high_mask_train], y_train[high_mask_train]), 'very_high': (X_train[very_high_mask_train], y_train[very_high_mask_train])}
for segment, (X_seg, _) in segment_data_train.items(): print(f"{segment.capitalize()} train segment: {len(X_seg)} samples")

# --- 10. Define Models, Feature Selection, and Tuning Params per Segment ---
# ADDED feature_name='auto' to LGBM base model
lgbm_model_base = lgb.LGBMRegressor(objective='regression', random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, feature_name='auto')
xgb_model_base = xgb.XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1)
selector_lgbm = SelectFromModel(lgbm_model_base, threshold=FEATURE_SELECTION_THRESHOLD, prefit=False)
selector_xgb = SelectFromModel(xgb_model_base, threshold=FEATURE_SELECTION_THRESHOLD, prefit=False)
# Parameter distributions (adjusted prefixes)
lgbm_param_dist = {'model__n_estimators': randint(100, 800),'model__learning_rate': uniform(0.005, 0.05),'model__num_leaves': randint(15, 60),'model__max_depth': randint(3, 10),'model__min_child_samples': randint(10, 50),'model__subsample': uniform(0.6, 0.4),'model__colsample_bytree': uniform(0.6, 0.4),'model__reg_alpha': uniform(0.0, 1.0),'model__reg_lambda': uniform(0.0, 1.0),}
xgb_param_dist = {'model__n_estimators': randint(100, 800),'model__learning_rate': uniform(0.005, 0.05),'model__max_depth': randint(3, 9),'model__min_child_weight': randint(1, 10),'model__subsample': uniform(0.6, 0.4),'model__colsample_bytree': uniform(0.6, 0.4),'model__gamma': uniform(0, 0.5),'model__reg_alpha': uniform(0.0, 1.0),'model__reg_lambda': uniform(0.0, 1.0),}
# Segment configs
segment_model_configs = {
    'very_low': {'model': lgbm_model_base, 'selector': selector_lgbm, 'params': lgbm_param_dist, 'use_log_transform': False},
    'low':      {'model': xgb_model_base,  'selector': selector_xgb,  'params': xgb_param_dist,  'use_log_transform': True},
    'medium':   {'model': lgbm_model_base, 'selector': selector_lgbm, 'params': lgbm_param_dist, 'use_log_transform': True},
    'high':     {'model': lgbm_model_base, 'selector': selector_lgbm, 'params': lgbm_param_dist, 'use_log_transform': True},
    'very_high':{'model': lgbm_model_base, 'selector': selector_lgbm,
                 'params': {**lgbm_param_dist, 'model__n_estimators': randint(400, 1200)}, # More estimators
                 'use_log_transform': True}
}

# --- 11. Train and Tune Segment Models ---
print("\nTraining and tuning segment-specific models...")
trained_models = {}
for segment, config in segment_model_configs.items():
    X_seg, y_seg = segment_data_train[segment]
    if len(X_seg) < CV_FOLDS_TUNING * 2 :
        print(f"Skipping segment '{segment}' due to insufficient data ({len(X_seg)} samples) for CV={CV_FOLDS_TUNING}.")
        trained_models[segment] = None; continue
    print(f"--- Tuning {segment.capitalize()} segment model ---")
    pipeline_to_tune = Pipeline(steps=[('preprocessor', preprocessor), ('selector', config['selector']), ('model', config['model'])])
    if config['use_log_transform']:
        final_estimator = TransformedTargetRegressor(regressor=pipeline_to_tune, func=np.log1p, inverse_func=np.expm1)
        search_param_dist = {f'regressor__{k}': v for k, v in config['params'].items()}
    else:
        final_estimator = pipeline_to_tune
        search_param_dist = config['params']
    random_search = RandomizedSearchCV(estimator=final_estimator, param_distributions=search_param_dist, n_iter=N_ITER_SEARCH, cv=CV_FOLDS_TUNING, scoring='neg_root_mean_squared_error', random_state=RANDOM_STATE, n_jobs=-1, verbose=1, error_score='raise')
    try:
        random_search.fit(X_seg, y_seg)
        trained_models[segment] = random_search.best_estimator_
        print(f"Best score (neg RMSE) for {segment}: {random_search.best_score_:.4f}")
    except Exception as e:
        print(f"ERROR during tuning for segment {segment}: {e}")
        trained_models[segment] = None

# --- 12. Predict on Test Set using Segment Models ---
print("\nMaking predictions on the test set...")
# Test masks
very_low_mask_test = y_test <= premium_thresholds['p25']
low_mask_test = (y_test > premium_thresholds['p25']) & (y_test <= premium_thresholds['p50'])
medium_mask_test = (y_test > premium_thresholds['p50']) & (y_test <= premium_thresholds['p75'])
high_mask_test = (y_test > premium_thresholds['p75']) & (y_test <= premium_thresholds['p95'])
very_high_mask_test = y_test > premium_thresholds['p95']
test_masks = {'very_low': very_low_mask_test, 'low': low_mask_test, 'medium': medium_mask_test, 'high': high_mask_test, 'very_high': very_high_mask_test}
# Predict
y_pred_combined = np.zeros_like(y_test, dtype=float)
prediction_successful = True
for segment, mask in test_masks.items():
    X_seg_test = X_test[mask]
    model = trained_models.get(segment)
    if model is not None and len(X_seg_test) > 0:
        try:
            # print(f"Predicting for {segment} segment ({len(X_seg_test)} samples)...") # Less verbose
            y_pred_segment = model.predict(X_seg_test)
            y_pred_segment = np.maximum(0, y_pred_segment)
            y_pred_combined[mask] = y_pred_segment
        except Exception as e:
            print(f"ERROR predicting for segment {segment}: {e}"); prediction_successful = False
    elif len(X_seg_test) > 0:
        print(f"WARNING: Model for segment '{segment}' not trained. Predictions zero."); prediction_successful = False

# --- 13. Evaluate Combined Model Performance ---
if prediction_successful:
    print("\n--- Combined Segmented Model Performance ---")
    combined_mae = mean_absolute_error(y_test, y_pred_combined)
    combined_rmse = np.sqrt(mean_squared_error(y_test, y_pred_combined))
    combined_r2 = r2_score(y_test, y_pred_combined)
    print(f"Overall MAE:  {combined_mae:.2f}")
    print(f"Overall RMSE: {combined_rmse:.2f}")
    print(f"Overall R²:   {combined_r2:.4f}")
    print("\n--- Performance within Test Segments ---")
    segment_metrics = {}
    for segment, mask in test_masks.items():
        y_seg_test = y_test[mask]; y_seg_pred = y_pred_combined[mask]; count = len(y_seg_test)
        if count > 0:
            mae = mean_absolute_error(y_seg_test, y_seg_pred); rmse = np.sqrt(mean_squared_error(y_seg_test, y_seg_pred))
            r2 = r2_score(y_seg_test, y_seg_pred) if np.var(y_seg_test) > 1e-9 else np.nan
            segment_metrics[segment] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'count': count}
            r2_str = f"{metrics['r2']:.4f}" if not np.isnan(metrics['r2']) else "N/A"
            print(f"{segment.capitalize():<10} (N={count:<5}): MAE={mae:<7.2f} RMSE={rmse:<7.2f} R²={r2_str:<7}")
        else:
             segment_metrics[segment] = {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'count': 0}
             print(f"{segment.capitalize():<10} (N={count:<5}): No test samples")
else:
    print("\nEvaluation skipped due to prediction errors.")

# --- 14. Feature Importance Analysis per Segment ---
print("\n--- Feature Importance Analysis ---")
feature_importance_data = {}
for segment, model in trained_models.items():
    if model is None: continue
    try:
        if isinstance(model, TransformedTargetRegressor): pipeline = model.regressor_
        else: pipeline = model
        model_step = pipeline.named_steps['model']
        selector_step = pipeline.named_steps['selector']
        preprocessor_step = pipeline.named_steps['preprocessor']

        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            X_seg_train_fi, y_seg_train_fi = segment_data_train[segment]
            # Fit preprocessor and selector on segment training data to get names/mask correctly
            if not hasattr(preprocessor_step, 'transformers_') or not hasattr(selector_step, 'estimator_'):
                 print(f"Fitting preprocessor/selector for FI on {segment}...")
                 X_processed_fi = preprocessor_step.fit_transform(X_seg_train_fi, y_seg_train_fi) # Pass y
                 # Need to fit the base model instance used inside selector
                 temp_selector_model = segment_model_configs[segment]['model'] # Get the base model instance
                 temp_selector_model.fit(X_processed_fi, y_seg_train_fi)
                 selector_step.estimator_ = temp_selector_model # Assign fitted base model
                 selector_step.fit(X_processed_fi, y_seg_train_fi) # Fit selector itself

            feature_names_processed = preprocessor_step.get_feature_names_out()
            support_mask = selector_step.get_support()

            if len(support_mask) != len(feature_names_processed):
                 print(f"Warning: FI mask length ({len(support_mask)}) != processed names length ({len(feature_names_processed)}) for {segment}. Skipping.")
                 continue
            selected_feature_names = np.array(feature_names_processed)[support_mask]

            if len(importances) != len(selected_feature_names):
                # Special case for XGBoost: it might include gain for features it didn't split on (0 importance)
                # If using XGB, allow this mismatch if importances > selected_features and extra importances are 0
                is_xgb = isinstance(model_step, xgb.XGBRegressor)
                if is_xgb and len(importances) > len(selected_feature_names):
                     # Check if extra importances are zero
                     if np.all(importances[len(selected_feature_names):] == 0):
                          importances = importances[:len(selected_feature_names)] # Trim zero importances
                     else:
                          print(f"Warning: FI Mismatch (XGB?): Importances ({len(importances)}) vs Selected Features ({len(selected_feature_names)}) for {segment}. Skipping.")
                          continue
                else:
                     print(f"Warning: FI Mismatch: Importances ({len(importances)}) vs Selected Features ({len(selected_feature_names)}) for {segment}. Skipping.")
                     continue


            importance_df = pd.DataFrame({'Feature': selected_feature_names,'Importance': importances}).sort_values(by='Importance', ascending=False)
            feature_importance_data[segment] = importance_df
            print(f"\n--- Top 10 Features for {segment.capitalize()} (Post-Selection) ---")
            print(importance_df.head(10))
            plt.figure(figsize=(10, 6)); sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), palette='viridis')
            plt.title(f'Top 15 FI (Post-Selection) - {segment.capitalize()}'); plt.tight_layout(); plt.savefig(f'feature_importances_{segment}.png'); plt.close()
            # print(f"FI plot saved: 'feature_importances_{segment}.png'") # Less verbose

        else: print(f"Model for {segment} lacks feature_importances_.")
    except Exception as e: print(f"ERROR analyzing FI for {segment}: {e}")

# --- 15. Prediction Visualization and Error Analysis ---
if prediction_successful:
    print("\n--- Generating Visualizations ---")
    # Actual vs Predicted
    plt.figure(figsize=(8, 8)); plt.scatter(y_test, y_pred_combined, alpha=0.3, s=10, label=f'Overall R²: {combined_r2:.3f}')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect'); plt.xlabel('Actual Premium'); plt.ylabel('Predicted Premium')
    plt.title('Actual vs Predicted Premiums (5 Segments + Selection)'); plt.grid(True, alpha=0.3); plt.legend(); plt.savefig('combined_actual_vs_predicted_5seg_sel.png'); plt.close()
    print("Actual vs predicted plot saved.")
    # Error Analysis DF
    errors = y_test - y_pred_combined; abs_errors = np.abs(errors); rel_errors = abs_errors / y_test.clip(lower=1)
    error_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_combined, 'Error': errors,'Abs_Error': abs_errors, 'Relative_Error': rel_errors, 'Segment': 'Unknown'})
    for segment, mask in test_masks.items(): error_df.loc[mask, 'Segment'] = segment.capitalize()
    # Error distribution
    plt.figure(figsize=(10, 5)); sns.histplot(errors, bins=50, kde=True); plt.xlabel('Prediction Error (Actual - Predicted)')
    plt.title('Distribution of Prediction Errors (5 Segments + Selection)'); plt.grid(True, alpha=0.3); plt.savefig('combined_error_distribution_5seg_sel.png'); plt.close()
    print("Error distribution plot saved.")
    # Error boxplot
    plt.figure(figsize=(10, 6)); segment_order = ['Very_low', 'Low', 'Medium', 'High', 'Very_high']
    sns.boxplot(data=error_df, x='Segment', y='Abs_Error', showfliers=False, order=segment_order); plt.title('Absolute Error Distribution by Segment (5 Segments + Selection)')
    plt.ylabel('Absolute Error'); plt.grid(True, axis='y', alpha=0.3); plt.savefig('error_boxplot_by_segment_5seg_sel.png'); plt.close()
    print("Error boxplot saved.")
    # Relative Error plot
    plt.figure(figsize=(10, 6)); sns.scatterplot(data=error_df, x='Actual', y='Relative_Error', hue='Segment', alpha=0.3, s=10, hue_order=segment_order)
    plt.axhline(y=0.25, color='r', linestyle='--', label='25% Error Threshold'); plt.ylim(0, max(1.5, error_df['Relative_Error'].quantile(0.98)))
    plt.xlabel('Actual Premium'); plt.ylabel('Relative Error (|Actual-Pred| / Actual)'); plt.title('Relative Error vs Actual Premium (5 Segments + Selection)')
    plt.grid(True, alpha=0.3); plt.legend(title='Segment'); plt.savefig('combined_relative_error_5seg_sel.png'); plt.close()
    print("Relative error plot saved.")

# --- 16. Prediction Reliability Diagram ---
    def plot_reliability_diagram(y_true, y_pred, n_bins=15):
        # (Function remains the same)
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        idx = np.argsort(y_pred)
        y_true_sorted, y_pred_sorted = y_true[idx], y_pred[idx]
        bin_size = len(y_pred) // n_bins
        bin_true_means, bin_pred_means = [], []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_pred)
            if start_idx >= end_idx: continue
            bin_true_means.append(np.mean(y_true_sorted[start_idx:end_idx]))
            bin_pred_means.append(np.mean(y_pred_sorted[start_idx:end_idx]))
        plt.figure(figsize=(8, 8)); plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Perfect')
        plt.scatter(bin_pred_means, bin_true_means, marker='o', s=50, edgecolor='k', label='Model Bins')
        plt.xlabel('Mean Predicted Premium'); plt.ylabel('Mean Actual Premium'); plt.title('Regression Reliability Diagram'); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        return plt
    try:
        reliability_plt = plot_reliability_diagram(y_test, y_pred_combined)
        reliability_plt.savefig('reliability_diagram_5seg_sel.png'); plt.close()
        print("Reliability diagram saved.")
    except Exception as e: print(f"Error generating reliability diagram: {e}")

# --- 17. Partial Dependence Plots ---
    print("\n--- Generating Partial Dependence Plots ---")
    pdp_segment = 'medium' # Choose segment
    pdp_features = ['Annual_Income', 'Credit_Score', 'Days_Since_Reference', 'Vehicle_Age'] # Choose features
    if pdp_segment in trained_models and trained_models[pdp_segment] is not None:
        model_for_pdp = trained_models[pdp_segment]
        X_seg_train_pdp, y_seg_train_pdp = segment_data_train[pdp_segment]
        try:
            if isinstance(model_for_pdp, TransformedTargetRegressor): pipeline = model_for_pdp.regressor_
            else: pipeline = model_for_pdp
            preprocessor_step = pipeline.named_steps['preprocessor']
            selector_step = pipeline.named_steps['selector']
            model_step = pipeline.named_steps['model']
            # Fit preprocessor & selector WITH y
            # print(f"Fitting preprocessor/selector for PDP on {pdp_segment}...")
            X_processed_pdp = preprocessor_step.fit_transform(X_seg_train_pdp, y_seg_train_pdp)
            selector_step.estimator_ = model_step # Use the fitted model from tuning
            selector_step.fit(X_processed_pdp, y_seg_train_pdp)
            support_mask = selector_step.get_support()
            selected_feature_names = preprocessor_step.get_feature_names_out()[support_mask]
            X_selected_pdp = X_processed_pdp.loc[:, support_mask]

            pdp_feature_indices_in_selected = []
            pdp_feature_names_found = []
            for feature in pdp_features:
                 matches = [i for i, name in enumerate(selected_feature_names) if name.split('__')[-1] == feature]
                 if matches: pdp_feature_indices_in_selected.append(matches[0]); pdp_feature_names_found.append(selected_feature_names[matches[0]])

            if pdp_feature_indices_in_selected:
                # print(f"Generating PDP for selected features: {pdp_feature_names_found}")
                pdp_result = partial_dependence(model_step, X_selected_pdp, features=pdp_feature_indices_in_selected, kind='average', grid_resolution=50)
                display = PartialDependenceDisplay([pdp_result], features=[(idx,) for idx in pdp_feature_indices_in_selected], feature_names=list(selected_feature_names), target_idx=0, deciles=np.linspace(0.05, 0.95, 9))
                fig, ax = plt.subplots(figsize=(min(15, 4 * len(pdp_feature_indices_in_selected)), 4))
                display.plot(ax=ax, line_kw={"color": "green", "linewidth": 2}); plt.suptitle(f'PDPs (Post-Selection) - {pdp_segment.capitalize()}')
                plt.subplots_adjust(top=0.88); plt.savefig(f'partial_dependence_plots_{pdp_segment}_5seg_sel.png'); plt.close()
                print(f"Partial dependence plots saved for {pdp_segment}.")
            else: print("No valid features found for PDP after selection.")
        except Exception as e: print(f"Error generating PDP: {e}")
    else: print(f"Skipping PDP for {pdp_segment} (model not trained).")

# --- 18. Final Summary ---
print("\n========== FINAL MODEL SUMMARY (5 Segments + Selection) ==========")
if prediction_successful:
    print(f"Overall R²:   {combined_r2:.4f}"); print(f"Overall MAE:  {combined_mae:.2f}"); print(f"Overall RMSE: {combined_rmse:.2f}")
    print("\n--- Segment Performance (Test Set) ---")
    for segment, metrics in segment_metrics.items():
        count = metrics['count']
        if count > 0:
             r2_str = f"{metrics['r2']:.4f}" if not np.isnan(metrics['r2']) else "N/A"
             print(f"  {segment.capitalize():<10} (N={count:<5}): R²={r2_str:<8} MAE={metrics['mae']:.2f}")
        else: print(f"  {segment.capitalize():<10} (N={count:<5}): No test samples")
    very_high_actual_mask = y_test > premium_thresholds['p95']
    if np.sum(very_high_actual_mask) > 0:
         vh_mae = mean_absolute_error(y_test[very_high_actual_mask], y_pred_combined[very_high_actual_mask])
         vh_r2 = r2_score(y_test[very_high_actual_mask], y_pred_combined[very_high_actual_mask])
         print(f"\n--- Performance on Top 5% Actual Premiums (> {premium_thresholds['p95']:.2f}) ---"); print(f"  Samples: {np.sum(very_high_actual_mask)}"); print(f"  MAE: {vh_mae:.2f}"); print(f"  R²: {vh_r2:.4f}")
else: print("Overall performance could not be calculated.")
print("\n--- Recommendations ---"); print("1. Very High Segment R²: If still low/negative, try Quantile Regression or Huber loss."); print("2. Feature Selection Threshold: Adjust 'median' threshold if too many/few features selected."); print("3. Tuning: Increase n_iter or use Optuna for potentially better hyperparameters."); print("4. Review FI: Ensure sensible features are being selected per segment.")
print("\nScript finished.")