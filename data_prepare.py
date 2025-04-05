import os
import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC
from pyproj import Proj, Transformer
from scipy.ndimage import zoom
import rasterio
from rasterio.merge import merge

# -------------------- Yardımcı Fonksiyon --------------------
def read_hdf_layer(hdf_path, layer_name, scale_factor=1.0, offset=0.0, fill_value=0):
    hdf = SD(hdf_path, SDC.READ)
    layer = hdf.select(layer_name)
    data = layer.get()
    attrs = layer.attributes()
    fill = attrs.get('_FillValue', fill_value)
    scaled_data = np.ma.masked_equal(data, fill) * scale_factor + offset
    hdf.end()
    return scaled_data.filled(np.nan)

# -------------------- Dosya Yolları --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
lst_path = os.path.join(script_dir, "MOD11A1.A2024211.h20v04.061.2024212094859.hdf")
ndvi_path = os.path.join(script_dir, "MOD13A2.A2024209.h20v04.061.2024228024044.hdf")
albedo_path = os.path.join(script_dir, "MCD43A3.A2024206.h20v04.061.2024221213147.hdf")

# -------------------- LST ve Emissivity --------------------
lst_day = read_hdf_layer(lst_path, 'LST_Day_1km', scale_factor=0.02)
lst_night = read_hdf_layer(lst_path, 'LST_Night_1km', scale_factor=0.02)
qa = SD(lst_path, SDC.READ).select('QC_Day').get()
mask = (qa & 0b11) == 0
for arr in [lst_day, lst_night]:
    arr[~mask] = np.nan

emis31 = read_hdf_layer(lst_path, 'Emis_31', scale_factor=0.002, offset=0.49)
emis32 = read_hdf_layer(lst_path, 'Emis_32', scale_factor=0.002, offset=0.49)
emis31[~mask] = np.nan
emis32[~mask] = np.nan

# -------------------- Koordinatlar --------------------
tile_h, tile_v = 20, 4
tile_size = 1200
pixel_size = 926.62543306
x0, y0 = -20015109.354, 10007554.677
x_offset = tile_h * tile_size * pixel_size
y_offset = tile_v * tile_size * pixel_size
x_coords = x0 + x_offset + np.arange(tile_size) * pixel_size
y_coords = y0 - y_offset - np.arange(tile_size) * pixel_size
xv, yv = np.meshgrid(x_coords, y_coords)
transformer = Transformer.from_proj(
    Proj('+proj=sinu +R=6371007.181 +nadgrids=@null +wktext'),
    Proj('epsg:4326'), always_xy=True)
lon, lat = transformer.transform(xv, yv)

# -------------------- NDVI, EVI, NDWI --------------------
ndvi = read_hdf_layer(ndvi_path, '1 km 16 days NDVI', scale_factor=0.0001)
evi = read_hdf_layer(ndvi_path, '1 km 16 days EVI', scale_factor=0.0001)
nir = read_hdf_layer(ndvi_path, '1 km 16 days NIR reflectance', scale_factor=0.0001)
mir = read_hdf_layer(ndvi_path, '1 km 16 days MIR reflectance', scale_factor=0.0001)
ndwi = (nir - mir) / (nir + mir)

# -------------------- Güneş Açısı --------------------
sun_angle = read_hdf_layer(ndvi_path, '1 km 16 days sun zenith angle', scale_factor=0.01)

# -------------------- Albedo --------------------
bsa = read_hdf_layer(albedo_path, 'Albedo_BSA_shortwave', scale_factor=0.001)
wsa = read_hdf_layer(albedo_path, 'Albedo_WSA_shortwave', scale_factor=0.001)
bsa = zoom(bsa, 0.5, order=1)
wsa = zoom(wsa, 0.5, order=1)
albedo_diff = bsa - wsa

# -------------------- 4. Ek Açı Katmanları --------------------
view_angle = read_hdf_layer(ndvi_path, '1 km 16 days view zenith angle', scale_factor=0.01)
rel_azimuth = read_hdf_layer(ndvi_path, '1 km 16 days relative azimuth angle', scale_factor=0.01)


with rasterio.open(dem_files[0]) as src:
    print("DEM NoData değeri:", src.nodata)

# -------------------- DEM --------------------
dem_files = [os.path.join(script_dir, f"N{n}E0{e}.tif") for n in [40, 41] for e in [27, 28]]
dem_datasets = [rasterio.open(f) for f in dem_files]
dem_mosaic, dem_transform = merge(dem_datasets)
with rasterio.open(dem_files[0]) as src:
    dem_crs = src.crs.to_string()
transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
dem_coords = np.array([transformer.transform(lon_, lat_) for lon_, lat_ in zip(lon.flatten(), lat.flatten())])
temp_dem = os.path.join(script_dir, "temp_dem.tif")
with rasterio.open(temp_dem, 'w', driver='GTiff',
                   height=dem_mosaic.shape[1], width=dem_mosaic.shape[2],
                   count=1, dtype=dem_mosaic.dtype,
                   crs=dem_crs, transform=dem_transform) as dest:
    dest.write(dem_mosaic[0], 1)
with rasterio.open(temp_dem) as dem_raster:
    dem_values = [val[0] for val in dem_raster.sample(dem_coords)]
for ds in dem_datasets:
    ds.close()
os.remove(temp_dem)

# -------------------- LST Farkı --------------------
lst_diff = lst_day - lst_night

# -------------------- Maskeleri Eşle --------------------
for arr in [ndvi, evi, ndwi, sun_angle, albedo_diff, lst_diff, view_angle, rel_azimuth]:
    arr[np.isnan(lst_day)] = np.nan

# -------------------- DataFrame --------------------
df = pd.DataFrame({
    'Latitude': lat.flatten(),
    'Longitude': lon.flatten(),
    'LST_Day': lst_day.flatten(),
    'LST_Night': lst_night.flatten(),
    'LST_Diff': lst_diff.flatten(),
    'Emis_31': emis31.flatten(),
    'Emis_32': emis32.flatten(),
    'NDVI': ndvi.flatten(),
    'EVI': evi.flatten(),
    'NDWI': ndwi.flatten(),
    'Sun_Angle': sun_angle.flatten(),
    'Albedo_Diff': albedo_diff.flatten(),
    'View_Angle': view_angle.flatten(),
    'Rel_Azimuth': rel_azimuth.flatten(),
    'DEM': dem_values
})

# -------------------- Marmara Filtresi ve Kaydetme --------------------
df = df[
    (df['Latitude'] >= 39.5) & (df['Latitude'] <= 42.5) &
    (df['Longitude'] >= 26.0) & (df['Longitude'] <= 31.0)
]

df = df.dropna()
output_path = os.path.join(script_dir, "final_features_dataset.csv")
df.to_csv(output_path, index=False)
print("Veri başarıyla kaydedildi:", output_path)
