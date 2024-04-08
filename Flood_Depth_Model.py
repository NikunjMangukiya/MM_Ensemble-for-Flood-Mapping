"""
@author: Nikunj Mangukiya; Shashwat Kushwaha; Ashutosh Sharma
"""

#Import required packages
import numpy as np
import rasterio
import joblib
from pyproj import Transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import contextily as ctx
import rasterio.plot as rioplot
import matplotlib.pyplot as plt


#Input Predictor: Flood conditioning factor dataset
elevation_raster = rasterio.open("__file_path__")
elevation_array = elevation_raster.read(1)
dist_river_raster = rasterio.open("__file_path__")
dfr_array = dist_river_raster.read(1)
slope_raster = rasterio.open("__file_path__")
slope_array = slope_raster.read(1)
flow_dir_raster = rasterio.open("__file_path__")
fd_array = flow_dir_raster.read(1)


#Get Latitude/Longitude from any input features. For ex. Elevation
src_crs = 'EPSG:32643'
target_crs = 'EPSG:4326'
transformer = Transformer.from_crs(src_crs, target_crs)
with rasterio.open(elevation_raster) as dataset:
    geotransform = dataset.transform
    num_rows = dataset.height
    num_cols = dataset.width
    lat_lon_array = np.empty((num_rows, num_cols, 2))
    for row in range(num_rows):
        for col in range(num_cols):
            x, y = rasterio.transform.xy(geotransform, row, col, offset='center')
            lon, lat = transformer.transform(x, y)
            lat_lon_array[row, col, 0] = lat
            lat_lon_array[row, col, 1] = lon
Latitude_array = lat_lon_array[:, :, 0]
Longitude_array = lat_lon_array[:, :, 1]


#Data-preprocessing: Fill/remove Nodata Values after Masking
def fill_nodata_with_average(arr, nodata):
    #Fill the nodata vlaues with average of three surrounding non-nodata values
    n = len(arr)
    result = []
    for i in range(n):
        if arr[i] != nodata:
            result.append(arr[i])
        else:
            valid_values = [arr[j] for j in range(max(i - 2, 0), min(i + 3, n)) if arr[j] != nodata]
            if valid_values:
                result.append(sum(valid_values) / len(valid_values))
            else:
                result.append(nodata)
    return result

def fill_closest_not_nodata(arr, nodata):
    #Fill the nodata values with closest value
    n = len(arr)
    def find_closest_non_nodata(index):
        left = index - 1
        right = index + 1
        while left >= 0 and arr[left] == nodata and right < n and arr[right] == nodata:
            left -= 1
            right += 1
        if left >= 0 and arr[left] != nodata:
            return arr[left]
        elif right < n and arr[right] != nodata:
            return arr[right]
        else:
            return arr[index]
    return [arr[i] if arr[i] != nodata else find_closest_non_nodata(i) for i in range(n)]

def remove_nodata_pixels(depth_map, corresponding_map, no_data_value):
    #Remove Nodata Pixels #Use to extract only flood depth pixels and features for model training
    mask = depth_map == no_data_value
    depth_map_masked = np.ma.masked_array(depth_map, mask).compressed()
    corresponding_map_masked = np.ma.masked_array(corresponding_map, mask).compressed()
    return (np.array(depth_map_masked), np.array(corresponding_map_masked))


#Input Predictand: Observed flood depth maps
no_event = 12 #number of flood depth maps available for training
streamflow_values = ["___, ____, ____, streamflow_range, ____, ____"]
flood_depth_maps = []
train_streamflow_values = []
feature_maps = []

for v in streamflow_values:
    #Converting flood raster to binary maps
    inun_rast = rasterio.open(f"__file_path__{v}.inundation_depth.tif")
    inun_array = inun_rast.read(1)
    flood_depth_maps.append(remove_nodata_pixels(inun_array, elevation_array, inun_rast.nodata)[0])
    feature_maps.append(np.column_stack((remove_nodata_pixels(inun_array,Latitude_array,inun_rast.nodata)[1],
                                         remove_nodata_pixels(inun_array,Longitude_array,inun_rast.nodata)[1],
                                         fill_nodata_with_average(remove_nodata_pixels(inun_array, elevation_array, inun_rast.nodata)[1],elevation_raster.nodata),
                                         fill_nodata_with_average(remove_nodata_pixels(inun_array, dfr_array, inun_rast.nodata)[1],dist_river_raster.nodata),
                                         fill_nodata_with_average(remove_nodata_pixels(inun_array, slope_array, inun_rast.nodata)[1],slope_raster.nodata),
                                         remove_nodata_pixels(inun_array, np.random.uniform(v - 400, v + 400, size=(1684, 3207)), inun_rast.nodata)[1])))

y_train = np.concatenate(flood_depth_maps)
X_train = np.concatenate(feature_maps)


#Data-preprocessing: Normalizing input/target features
scaler = MinMaxScaler()
scaler.fit(X_train)
normalized_features = scaler.transform(X_train)
X_train=normalized_features


#Test flood event dataset (depth maps)
test_event_1 = rasterio.open("__file_path__")
test_event_1_array = test_event_1.read(1)
test_event_2 = rasterio.open("__file_path__")
test_event_2_array = test_event_2.read(1)

# y_test
test_vector = np.concatenate([remove_nodata_pixels(test_event_1_array, elevation_array, test_event_1.nodata)[0],
                              remove_nodata_pixels(test_event_2_array, elevation_array, test_event_2.nodata)[0]])
y_test = test_vector


# Preparing Test Feature Data
feature_matrix1 = np.column_stack((remove_nodata_pixels(test_event_1_array,Latitude_array,test_event_1.nodata)[1], #latitude
                                   remove_nodata_pixels(test_event_1_array,Longitude_array,test_event_1.nodata)[1], #longitude
                                   remove_nodata_pixels(test_event_1_array, elevation_array, test_event_1.nodata)[1], #elevation
                                   fill_nodata_with_average(remove_nodata_pixels(test_event_1_array, dfr_array, test_event_1.nodata)[1],dist_river_raster.nodata), #DFR
                                   remove_nodata_pixels(test_event_1_array, slope_array, test_event_1.nodata)[1], #Slope
                                   remove_nodata_pixels(test_event_1_array, np.random.uniform(test_event_1 - 400, test_event_1 + 400, size=(1684, 3207)), test_event_1.nodata)[1])) #streamflow

feature_matrix2 = np.column_stack((remove_nodata_pixels(test_event_2_array,Latitude_array,test_event_2.nodata)[1], #latitude
                                   remove_nodata_pixels(test_event_2_array,Longitude_array,test_event_2.nodata)[1], #longitude
                                   remove_nodata_pixels(test_event_2_array, elevation_array, test_event_2.nodata)[1], #elevation
                                   fill_nodata_with_average(remove_nodata_pixels(test_event_2_array, dfr_array, test_event_2.nodata)[1],dist_river_raster.nodata), #DFR
                                   remove_nodata_pixels(test_event_2_array, slope_array, test_event_2.nodata)[1], #Slope
                                   remove_nodata_pixels(test_event_2_array, np.random.uniform(test_event_2 - 400, test_event_2 + 400, size=(1684, 3207)), test_event_2.nodata)[1])) #streamflow

test_feature_matrix = np.concatenate([feature_matrix1, feature_matrix2])
X_test = test_feature_matrix
X_test = scaler.transform(test_feature_matrix)


# Training the model
d = {}  #Dictionary for Results
def train_model(model, name):
    print(f'Training model : {name}')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    r2_Score_test = r2_score(y_test, y_pred)
    train_mse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_mse = np.sqrt(mean_squared_error(y_test, y_pred))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred)
    r2_Score_train = r2_score(y_train, y_pred_train)
    print(f'R2_score Test:{r2_Score_test}  Train:{r2_Score_train}')
    print(f'Test: RMSE: {test_mse} \n Train RMSE: {train_mse}')
    print(f'Test: MAE: {test_mae} \n Train MAE: {train_mae}')
    d[name] = {'R2_Score_train': r2_Score_train, 'R2_Score_test': r2_Score_test,
               'RMSE_train': train_mse, 'RMSE_test': test_mse,
               'MAE_train': train_mae, 'MAE_test': test_mae}

    joblib.dump(model, "__file_path__" + "model_name.pkl")
    return model

#Kfold validation
kfold = KFold(n_splits=10, shuffle=True, random_state=55)
def cros_v(model):
    scores = cross_val_score(model, X_train, y_train, cv = kfold, scoring='neg_mean_absolute_error')
    positive_scores = -scores
    for fold_idx,score in enumerate(positive_scores):
        print(f"Fold{fold_idx+1}: {score:.4f}")
    mean_mae=positive_scores.mean()
    std_mae=positive_scores.std()
    print(f"\nMen Absolute Error: {mean_mae:.4f}")
    print(f"Standard Deviation: {std_mae:.4f}")

#model hyperparameter tunning
def hpt(model, params):
    grid_search = GridSearchCV(model,
                               param_grid=params,
                               n_jobs=-1,
                               cv = kfold,
                               scoring='neg_mean_squared_error',
                               return_train_score=True)
    grid_result = grid_search.fit(X_train, y_train)
    print(f'Best Score:{grid_result.best_score_}\nParams: {grid_result.best_params_} ')
    rcsore = grid_result.score(X_test, y_test)
    print(f'Test Score: {rcsore}')

#models
linear_model = Ridge(random_state=55)
xgb_model = XGBRegressor(random_state=55, max_depth=3, n_estimators=150, learning_rate=0.1)
tree_model=DecisionTreeRegressor(random_state=55,max_depth=11)
ensemble_reg = VotingRegressor(estimators=[('Linear', linear_model),
                                           ('XGB', xgb_model),
                                           ('tree',tree_model)],
                               weights=[0.25, 0.45,0.30])
m = train_model(ensemble_reg, 'MultiEnsembleRegressor')



#Function to predict depth for specific discharge
def pred_for_an_event(elevation_rast,
                      dfr_rast,
                      slope_rast,
                      discharge,
                      trained_model,
                      name):
    #Features
    elevation_data = elevation_rast.read(1).astype(float)
    distance_from_river_data = dfr_rast.read(1).astype(float)
    slope_data = slope_rast.read(1).astype(float)
    discharge_data = np.random.uniform(discharge - 400, discharge + 400, size=(1684, 3207))

    #loading the predicted/observed extent raster
    predicted_extent = rasterio.open(f"__file_path__\extent_raster_{discharge}.tif").read(1).astype(bool)

    #Regression Features
    elevation_values = fill_nodata_with_average(elevation_data[predicted_extent],elevation_rast.nodata)
    slope_values = fill_nodata_with_average(slope_data[predicted_extent],slope_rast.nodata)
    distance_from_river_values = fill_nodata_with_average(distance_from_river_data[predicted_extent],dfr_rast.nodata)
    discharge_values = discharge_data[predicted_extent]
    lat_values=Latitude_array[predicted_extent]
    lon_values=Longitude_array[predicted_extent]

    #Feature Matrix for Regression
    input_features = np.column_stack((lat_values,
                                      lon_values,
                                      elevation_values,
                                      distance_from_river_values,
                                      slope_values,
                                      discharge_values))
    
    #normalized_feature=scaler.fit_transform(input_features)
    predicted_depths = trained_model.predict(input_features)
    return predicted_depths


#Generating Rasters for Predicted Depth
def get_pred_Depth_raster_for_a_discharge(elevation_rast,
                                          dfr_rast,
                                          slope_rast,
                                          discharge,
                                          trained_model,
                                          name):

    elevation_data = elevation_rast.read(1).astype(float)
    distance_from_river_data = dfr_rast.read(1).astype(float)
    slope_data = slope_rast.read(1).astype(float)
    discharge_data = np.random.uniform(discharge - 400, discharge + 400, size=(1684, 3207))

    d1 = min(streamflow_values, key=lambda x: abs(x - discharge))
    if discharge in streamflow_values:
        actual_depth_rast = rasterio.open(f"__file_path__\Depth_{discharge}.tif")
        actual_depth = rasterio.open(f"__file_path__\Depth_{discharge}.tif").read(1).astype(float)
    else:
        actual_depth_rast = rasterio.open(f"__file_path__\Depth_{d1}.tif")
        actual_depth = rasterio.open(f"__file_path__\Depth_{d1}.tif").read(1).astype(float)
    
    #loading the predicted/observed extent raster
    predicted_extent = rasterio.open(f"__file_path__\extent_raster_{discharge}.tif").read(1).astype(bool)

    #Regression Features
    elevation_values = fill_nodata_with_average(elevation_data[predicted_extent],elevation_rast.nodata)
    slope_values = fill_nodata_with_average(slope_data[predicted_extent],slope_rast.nodata)
    distance_from_river_values = fill_nodata_with_average(distance_from_river_data[predicted_extent],dfr_rast.nodata)
    discharge_values = discharge_data[predicted_extent]
    lat_values=Latitude_array[predicted_extent]
    lon_values=Longitude_array[predicted_extent]

    actual_depth[actual_depth == actual_depth_rast.nodata] = np.nan
    actual_depth_values = actual_depth[predicted_extent]

    #Feature Matrix for Regression
    input_features = np.column_stack((lat_values, lon_values, elevation_values, distance_from_river_values, slope_values, discharge_values))
    predicted_depths = trained_model.predict(input_features)

    #Making Actual Depth Map
    actual_map = np.empty_like(elevation_data)
    actual_map[:] = np.nan
    actual_map[predicted_extent] = actual_depth_values

    #Making Depth Map from Predicted data
    depth_map = np.empty_like(elevation_data)
    depth_map[:] = np.nan
    depth_map[predicted_extent] = predicted_depths
    threshold_value = 0
    depth_map[depth_map < threshold_value] = np.nan

    with rasterio.open(f"__file_path__\Depth_{d1}.tif") as src:
        metadata=src.meta
        crs=src.crs
        height,width=depth_map.shape
        metadata.update({'count':1,
                         'dtype':'float32',
                         'nodata':np.nan})

        output_path_pred="__file_path__"+f"{name}_depth_{discharge}.tif"
        output_path_act="__file_path__"+f"{name}_depth_{discharge}.tif"
        with rasterio.open(output_path_act,'w',**metadata) as dst1:
            dst1.write(actual_map.astype('float32'),1)
            dst1.crs=crs
        with rasterio.open(output_path_pred,'w',**metadata) as dst2:
            dst2.write(depth_map.astype('float32'),1)
            dst2.crs=crs
    dst1.close()
    dst2.close()
    src.close()


#Get number of pixels with specific depths
def get_area_with_different_inundation_depths(y_true, y_pred):
    true_area_dict = {'0-1 m': 0, '1-2 m': 0, '2-3 m': 0, '3-4 m': 0, '4-8 m': 0, '8-12 m': 0, '12+': 0}
    pred_area_dict = {'0-1 m': 0, '1-2 m': 0, '2-3 m': 0, '3-4 m': 0, '4-8 m': 0, '8-12 m': 0, '12+': 0}
    for i in range(len(y_true)):
        if y_true[i] >= 0 and y_true[i] < 1:
            true_area_dict['0-1 m'] += 1
        elif y_true[i] >= 1 and y_true[i] < 2:
            true_area_dict['1-2 m'] += 1
        elif y_true[i] >= 2 and y_true[i] < 3:
            true_area_dict['2-3 m'] += 1
        elif y_true[i] >= 3 and y_true[i] < 4:
            true_area_dict['3-4 m'] += 1
        elif y_true[i] >= 4 and y_true[i] < 8:
            true_area_dict['4-8 m'] += 1
        elif y_true[i] >= 8 and y_true[i] < 12:
            true_area_dict['8-12 m'] += 1
        elif y_true[i] >= 12:
            true_area_dict['12+'] += 1

        if y_pred[i] >= 0 and y_pred[i] < 1:
            pred_area_dict['0-1 m'] += 1
        elif y_pred[i] >= 1 and y_pred[i] < 2:
            pred_area_dict['1-2 m'] += 1
        elif y_pred[i] >= 2 and y_pred[i] < 3:
            pred_area_dict['2-3 m'] += 1
        elif y_pred[i] >= 3 and y_pred[i] < 4:
            pred_area_dict['3-4 m'] += 1
        elif y_pred[i] >= 4 and y_pred[i] < 8:
            pred_area_dict['4-8 m'] += 1
        elif y_pred[i] >= 8 and y_pred[i] < 12:
            pred_area_dict['8-12 m'] += 1
        elif y_pred[i] >= 12:
            pred_area_dict['12+'] += 1
    return (true_area_dict, pred_area_dict)


#Generating Comparison Plots with visualization on Real Map
def visualize_original_map_in_real_map(elevation_rast,discharge, name, model):
    get_pred_Depth_raster_for_a_discharge(elevation_raster, dist_river_raster, slope_raster, discharge, model, name)
    original_rast_path = f"__file_path__\{name}_depth_{discharge}.tif"
    elevation_rast_path = "__file_path__\elevation.tif"
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    with rasterio.open(elevation_rast_path) as elevation_rast:
        elevation_data = elevation_rast.read(1)
        elevation_extent = rioplot.plotting_extent(elevation_rast)
        ax[0].imshow(elevation_data, extent=elevation_extent, cmap='gray')
        ctx.add_basemap(ax[0], source=ctx.providers.Stamen.Terrain, crs=elevation_rast.crs)
        with rasterio.open(original_rast_path) as orig_flood_map:
            orig_flood_data = orig_flood_map.read(1, masked=True)
        im1 = ax[0].imshow(orig_flood_data, extent=elevation_extent, cmap='jet', alpha=0.8, vmin=0, vmax=8)

    ax[0].set_title(f'Original Depth Map at Discharge: {discharge} cubic meters/second')
    ax[0].grid(False)
    cbar1 = fig.colorbar(im1, ax=ax[0], shrink=0.6)
    cbar1.set_label('Water Depth(m)')

    with rasterio.open(elevation_rast_path) as elevation_rast:
        elevation_data = elevation_rast.read(1)
        elevation_extent = rioplot.plotting_extent(elevation_rast)
        ax[1].imshow(elevation_data, extent=elevation_extent, cmap='gray')
        ctx.add_basemap(ax[1], source=ctx.providers.Stamen.Terrain, crs=elevation_rast.crs)

        with rasterio.open(f"__file_path__\Predicted\{name}_depth_{discharge}.tif") as flood_map:
            flood_map_data = flood_map.read(1, masked=True)
        im2 = ax[1].imshow(flood_map_data, extent=elevation_extent, cmap='jet', alpha=0.8, vmin=0, vmax=8)

    ax[1].set_title(f'Predicted Depth Map at Discharge: {discharge} cubic meters/second')
    ax[1].grid(False)
    cbar2 = fig.colorbar(im2, ax=ax[1], shrink=0.6)
    cbar2.set_label('Water Depth(m)')
    plt.tight_layout()
    plt.savefig("__file_path__"+"FloodDepth_comparison_"+str(name)+f"_{discharge}.png", dpi=1200)
    plt.show()
