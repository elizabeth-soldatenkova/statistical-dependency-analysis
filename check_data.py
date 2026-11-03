# from src.npv_prediction.data import (
#     load_datasets,
#     validate_datasets,
# )

# df_train, df_control = load_datasets()
# validate_datasets(df_train, df_control)

# print("Training shape:", df_train.shape)
# print("Control shape:", df_control.shape)

# print("\nTraining columns:")
# print(df_train.columns.tolist())

# print("\nControl columns:")
# print(df_control.columns.tolist())

# print("\nTraining sample:")
# print(df_train.head())

# print("\nControl sample:")
# print(df_control.head())


from src.npv_prediction.data import (
    load_datasets,
    validate_datasets,
)
from src.npv_prediction.preprocessing import (
    preprocess_dataset,
    split_features_target,
)

df_train, df_control = load_datasets()
validate_datasets(df_train, df_control)

train_processed = preprocess_dataset(df_train)
control_processed = preprocess_dataset(df_control)

X_train, y_train = split_features_target(train_processed)
X_control, y_control = split_features_target(control_processed)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_control:", X_control.shape)
print("y_control:", y_control.shape)

print("\nRemaining string values:")
print(
    X_train.select_dtypes(include="object").columns.tolist()
)

print("\nMissing values:")
print(X_train.isna().sum())
