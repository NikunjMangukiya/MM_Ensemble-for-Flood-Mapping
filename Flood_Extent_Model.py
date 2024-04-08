"""
@author: Nikunj Mangukiya; Shashwat Kushwaha; Ashutosh Sharma
"""

#Import required packages
import numpy as np
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, roc_auc_score


#Input Predictor: Flood conditioning factor dataset
elevation_raster = rasterio.open("__file_path__")
dist_river_raster = rasterio.open("__file_path__")
slope_raster = rasterio.open("__file_path__")
flow_dir_raster = rasterio.open("__file_path__")


#Data-preprocessing: removing Nodata Pixels
elevation_data = elevation_raster.read(1).astype(float)
distance_from_river_data=dist_river_raster.read(1).astype(float)
slope_data = slope_raster.read(1).astype(float)
flow_dirn_data=flow_dir_raster.read(1).astype(float)

elevation_data[elevation_data == elevation_raster.nodata] = np.nan
distance_from_river_data[distance_from_river_data==dist_river_raster.nodata]=np.nan
slope_data[slope_data == slope_raster.nodata] = np.nan
flow_dirn_data[flow_dirn_data==flow_dir_raster.nodata]=np.nan

valid_indices = ~np.isnan(elevation_data) & ~np.isnan(slope_data) & ~np.isnan(distance_from_river_data) & ~np.isnan(flow_dirn_data)


#If flood depth map is provided as input for training flood extent model, the depth map will be converted to binary map
def convert_to_binary_map(flood_depth_raster, valid_indices):
    """
    Function to covert Flood Depth Map to Binary Flood Map where 1= Flood Area, 0= Non-Flood Area
    :param flood_depth_rast:
    :param valid_indices:
    :return:
    """
    flood_depth_map = flood_depth_raster.read(1).astype(float)
    flood_extent = flood_depth_map[valid_indices]
    flood_extent_array = np.where(flood_extent == flood_depth_raster.nodata, 0, 1)
    flood_map = np.empty_like(elevation_data)
    flood_map[:] = np.nan
    flood_map[valid_indices] = flood_extent_array
    return (flood_extent_array, flood_map)

def get_binary_map(binary_map):
    """
    Returns Binary Flood Map of the Basin
    :param binary_map:
    :return:
    """
    elevation_array = elevation_raster.read(1).astype(float)
    flood_map = np.empty_like(elevation_array)
    flood_map[:] = np.nan
    flood_map[valid_indices] = binary_map
    return flood_map


#Input Predictand: Observed flood extent/depth maps
no_event = 12 #number of flood exntent maps available for training
streamflow_values = ["___, ____, ____, streamflow_range, ____, ____"]
train_extent_maps = []
train_streamflow_values = []
for v in streamflow_values:
    # Converting flood raster to binary maps
    depth_raster = rasterio.open("__file_path__")
    train_extent_maps.append(convert_to_binary_map(depth_raster, valid_indices)[0])
    train_streamflow_values.append(np.random.uniform(v - 400, v + 400, size=1859772))


#Data-preprocessing: Stacking all inputs for model training
train_elevation_maps = np.array([elevation_data[valid_indices]] * no_event)
train_distance_maps = np.array([distance_from_river_data[valid_indices]] * no_event)
train_slope_maps = np.array([slope_data[valid_indices]] * no_event)
train_flow_direction_maps = np.array([flow_dirn_data[valid_indices]] * no_event)
train_streamflow_values = np.array(train_streamflow_values)
train_extent_maps = np.array(train_extent_maps)

for i in range(0, no_event):
    train_feature_matrix = np.column_stack((train_elevation_maps[i],
                                 train_distance_maps[i],
                                 train_slope_maps[i],
                                 train_flow_direction_maps[i],
                                 train_streamflow_values[i]))

target_vector = np.concatenate([train_extent_maps[i] for i in range(0, no_event)])
y_train = target_vector.astype(int)


#Data-preprocessing: Normalizing input/target features
scaler = StandardScaler()
scaler.fit(train_feature_matrix)
normalized_features = scaler.transform(train_feature_matrix)
X_train = normalized_features


#Test flood event dataset
no_test_event = 2 
test_event_1 = rasterio.open("__file_path__")
test_event_2 = rasterio.open("__file_path__")
test_flood_maps = [convert_to_binary_map(test_event_1, valid_indices)[0],
                   convert_to_binary_map(test_event_2, valid_indices)[0]]
test_flood_maps = np.array(test_flood_maps)

# y_test
test_vector = np.concatenate([test_flood_maps[i] for i in range(2)])
y_test = test_vector.astype(int)

# Preparing Test Feature Data
test_elevation_maps = np.array([elevation_data[valid_indices]] * 2)
test_distance_maps = np.array([distance_from_river_data[valid_indices]] * 2)
test_slope_maps = np.array([slope_data[valid_indices]] * 2)
test_flow_direction_maps = np.array([flow_dirn_data[valid_indices]] * 2)
test_streamflow_values = np.array([np.random.uniform(test_event_1 - 400, test_event_1 + 400, size=1859772),
                                   np.random.uniform(test_event_2 - 400, test_event_2 + 400, size=1859772)])

for i in range(0, no_test_event):
    test_feature_matrix = np.column_stack((test_elevation_maps[i],
                                            test_distance_maps[i],
                                            test_slope_maps[i],
                                            test_flow_direction_maps[i],
                                            test_streamflow_values[0]))

X_test = scaler.transform(test_feature_matrix)


# Training the model
def train_model(model, name):
    """
    Function to train the model and display metrics
    :param model: classifier instance
    :param name: str
    :return:
    """
    print(f'Training model : {name}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    acc_score = model.score(X_test, y_test)
    print(f'Test: Accuracy: {acc_score} \n Train Accuracy: {model.score(X_train, y_train)}')
    print(f'ROC_AUC: {roc_auc_score(y_test, y_pred)}')
    print('Classification Report: Test')
    print(classification_report(y_test, y_pred))
    print('Classification Report: Train')
    print(classification_report(y_train, y_pred_train))
    confusion_matrix1 = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix1, display_labels=["Dry", "Wet"])
    cm_display.plot(cmap='Blues')
    plt.grid(False)
    plt.savefig("__file_path__", dpi=1200)
    plt.show()
    # Save the Model
    joblib.dump(model, "__file_path__" + str(name) + "_model.pkl")
    return model


#models
logistic_model = LogisticRegression(random_state=55)
tree_model = DecisionTreeClassifier(random_state=55, max_depth=25)
xgb_model = XGBClassifier(random_state=55, max_depth=16, n_estimators=350)
# Training The Voting Classifier
ensemble_clf2 = VotingClassifier(estimators=[('logistic', logistic_model),
                                             ('tree', tree_model),
                                             ('XGB', xgb_model)],
                                 voting='soft', # 'hard' for majority voting, 'soft' for weighted voting based on probabilities
                                 weights=[0.20, 0.45, 0.35])

m = train_model(ensemble_clf2, 'EnsembleCLF')


# Generate raster for predicted extent for a discharge
def get_pred_raster_for_a_discharge(elevation_vector,
                                    distance_vector,
                                    slope_vector,
                                    flow_direction_vector,
                                    streamflow_value,
                                    scaler,
                                    trained_model,
                                    name):
    """
    Convert the predicted binary map for a given discharge to Raster
    :param elevation_vector: Elevation Array for the basin
    :param distance_vector: Distance from River Array for the basin
    :param slope_vector: Slope Array for the basin
    :param flow_direction_vector: Flow Direction Array for the basin
    :param streamflow_value: Value of the streamFlow
    :param scaler: scaler instance
    :param trained_model: saved model instance
    :param name: str (Classifier Name)

    """
    # Size here for current Narmada River Basin (Total Number of Valid Pixels in DEM)
    discharge_flow_vector = np.random.uniform(streamflow_value - 400, streamflow_value + 400, size=1859772)
    feature_matrix = np.column_stack((elevation_vector,
                                      distance_vector,
                                      slope_vector,
                                      flow_direction_vector,
                                      discharge_flow_vector))
    normalized_features = scaler.transform(feature_matrix)
    y_pred = trained_model.predict(normalized_features)

    y_plot = np.empty_like(elevation_data)
    y_plot[:] = 0
    y_plot[valid_indices] = y_pred
    original_rast_path = "__file_path__"

    with rasterio.open(original_rast_path) as src:
        metadata = src.meta
        crs = src.crs
        height, width = y_plot.shape
        metadata.update({'count': 1,
                         'dtype': 'float32',
                         'nodata': 0})

        output_path = "__file_path__"
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(y_plot.astype('float32'), 1)
            dst.crs = crs
    dst.close()
    src.close()
    print('Binary Flood Map for streamflow: {streamflow_value} created successfully!')


def pred_for_single_event(elevation_vector,
                          distance_vector,
                          slope_vector,
                          flow_direction_vector,
                          streamflow_value,
                          scaler,
                          trained_model,
                          name):
    """
    Predict Binary Map for given Discharge Flow
    :param elevation_vector: Elevation Array for the basin
    :param distance_vector: Distance from River Array for the basin
    :param slope_vector: Slope Array for the basin
    :param flow_direction_vector: Flow Direction Array for the basin
    :param streamflow_value: Value of the StreamFlow
    :param scaler: Scaler instance
    :param trained_model: Trained Model INstance
    :param name: str (Classifier Name)
    """
    discharge_flow_vector = np.random.uniform(streamflow_value - 400, streamflow_value + 400, size=1859772)
    feature_matrix = np.column_stack((elevation_vector,
                                      distance_vector,
                                      slope_vector,
                                      flow_direction_vector,
                                      discharge_flow_vector))
    normalized_features = scaler.transform(feature_matrix)
    y_pred = trained_model.predict(normalized_features)
    y_pred_train = trained_model.predict(X_train)

    original_map = rasterio.open("__file_path__")
    acc_score = accuracy_score(convert_to_binary_map(original_map, valid_indices)[0], y_pred)
    prec_score = precision_score(convert_to_binary_map(original_map, valid_indices)[0], y_pred)
    rec_score = recall_score(convert_to_binary_map(original_map, valid_indices)[0], y_pred)

    print(f'Test: Accuracy: {acc_score} \t Train Accuracy: {accuracy_score(y_pred_train, y_train)} \n'
          f'Test: Precision: {prec_score} \t Train Precision: {precision_score(y_pred_train, y_train)} \n'
          f'Test: Recall: {rec_score} \t Train Recall: {recall_score(y_pred_train, y_train)}')

    y_plot = np.empty_like(elevation_data)
    y_plot[:] = np.nan
    y_plot[valid_indices] = y_pred

    # PLotting the distribution
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    ax[0].imshow(convert_to_binary_map(original_map, valid_indices)[1], cmap='Blues')
    ax[0].set_title(f'Original Inundation Map at streamflow = {streamflow_value}')
    ax[0].grid(False)

    ax[1].imshow(y_plot, cmap='Blues')
    ax[1].set_title(f'Predicted Inundation Map at streamflow = {streamflow_value}')
    ax[1].grid(False)

    plt.tight_layout()
    plt.savefig("__file_path__", dpi=1200)
    plt.show()
