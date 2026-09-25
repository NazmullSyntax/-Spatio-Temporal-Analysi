/**
 * DHAKA GREEN SPACE / URBAN HEAT PROJECT
 * Google Earth Engine script: extract NDVI, NDBI, daytime LST (Landsat)
 * and nighttime LST (MODIS) for the DNCC+DSCC study area, for each
 * analysis year: 2000, 2005, 2010, 2015, 2020, 2025.
 *
 * HOW TO RUN
 * 1. Go to https://code.earthengine.google.com and sign in / sign up
 *    (free for research/education use).
 * 2. Upload your Dhaka ward-boundary shapefile as a GEE Asset, OR
 *    replace `studyArea` below with a bounding box for a first pass.
 * 3. Paste this script into a new GEE script, update STUDY_AREA_ASSET,
 *    and click Run. Each `Export.image.toDrive` call must be manually
 *    started from the Tasks tab (GEE does not auto-run exports).
 * 4. Download the exported GeoTIFFs from Google Drive and bring them
 *    back into this project's 02_processed/rasters/ folder.
 *
 * NOTE ON YEARS: Landsat 5 stops in ~2012, Landsat 7 has SLC-off gaps
 * from 2003 onward, Landsat 8 starts in 2013. The sensor picked per
 * year below reflects that:
 *   2000, 2005      -> Landsat 5 TM
 *   2010             -> Landsat 5 TM (or Landsat 7 as backup, gap-filled)
 *   2015, 2020, 2025 -> Landsat 8/9 OLI/TIRS
 */

// ---------------------------------------------------------------------
// 0. STUDY AREA
// ---------------------------------------------------------------------
// Option A (preferred): load your own ward-boundary asset once uploaded.
// var studyArea = ee.FeatureCollection('projects/YOUR_PROJECT/assets/dhaka_dncc_dscc_wards');

// Option B (placeholder bounding box so the script runs immediately;
// replace with Option A once you have the real ward boundary):
var studyArea = ee.Geometry.Rectangle([90.33, 23.68, 90.48, 23.90]);

Map.centerObject(studyArea, 11);
Map.addLayer(studyArea, { color: 'red' }, 'Study area (placeholder bbox)');

// ---------------------------------------------------------------------
// 1. CLOUD MASKING HELPERS (Collection 2, Level 2 surface reflectance)
// ---------------------------------------------------------------------
function maskL457sr(image) {
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
    .addBands(thermalBand, null, true)
    .updateMask(qaMask)
    .updateMask(saturationMask);
}

function maskL8sr(image) {
  var qaMask = image.select('QA_PIXEL').bitwiseAnd(parseInt('11111', 2)).eq(0);
  var saturationMask = image.select('QA_RADSAT').eq(0);
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B10').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
    .addBands(thermalBand, null, true)
    .updateMask(qaMask)
    .updateMask(saturationMask);
}

// ---------------------------------------------------------------------
// 2. PER-YEAR COLLECTION BUILDER
//    Dry-season window (Nov-Feb) chosen for minimal cloud cover and
//    seasonal comparability across years.
// ---------------------------------------------------------------------
var YEARS = [
  { year: 2000, start: '1999-11-01', end: '2000-02-28', collection: 'LANDSAT/LT05/C02/T1_L2', nir: 'SR_B4', red: 'SR_B3', swir: 'SR_B5', thermal: 'ST_B6', mask: maskL457sr },
  { year: 2005, start: '2004-11-01', end: '2005-02-28', collection: 'LANDSAT/LT05/C02/T1_L2', nir: 'SR_B4', red: 'SR_B3', swir: 'SR_B5', thermal: 'ST_B6', mask: maskL457sr },
  { year: 2010, start: '2009-11-01', end: '2010-02-28', collection: 'LANDSAT/LT05/C02/T1_L2', nir: 'SR_B4', red: 'SR_B3', swir: 'SR_B5', thermal: 'ST_B6', mask: maskL457sr },
  { year: 2015, start: '2014-11-01', end: '2015-02-28', collection: 'LANDSAT/LC08/C02/T1_L2', nir: 'SR_B5', red: 'SR_B4', swir: 'SR_B6', thermal: 'ST_B10', mask: maskL8sr },
  { year: 2020, start: '2019-11-01', end: '2020-02-28', collection: 'LANDSAT/LC08/C02/T1_L2', nir: 'SR_B5', red: 'SR_B4', swir: 'SR_B6', thermal: 'ST_B10', mask: maskL8sr },
  { year: 2025, start: '2024-11-01', end: '2025-02-28', collection: 'LANDSAT/LC09/C02/T1_L2', nir: 'SR_B5', red: 'SR_B4', swir: 'SR_B6', thermal: 'ST_B10', mask: maskL8sr },
];

YEARS.forEach(function (cfg) {
  var col = ee.ImageCollection(cfg.collection)
    .filterBounds(studyArea)
    .filterDate(cfg.start, cfg.end)
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
    .map(cfg.mask);

  var count = col.size();
  print('Year ' + cfg.year + ' - scenes found:', count);

  var image = col.median().clip(studyArea);

  var ndvi = image.normalizedDifference([cfg.nir, cfg.red]).rename('NDVI');
  var ndbi = image.normalizedDifference([cfg.swir, cfg.nir]).rename('NDBI');
  // Convert thermal Kelvin -> Celsius for LST (already scaled to Kelvin by the mask function)
  var lstDay = image.select(cfg.thermal).subtract(273.15).rename('LST_Day_C');

  var stack = ndvi.addBands(ndbi).addBands(lstDay);

  Map.addLayer(ndvi, { min: -0.2, max: 0.8, palette: ['brown', 'yellow', 'green'] }, 'NDVI ' + cfg.year, false);
  Map.addLayer(lstDay, { min: 15, max: 45, palette: ['blue', 'yellow', 'red'] }, 'LST_Day ' + cfg.year, false);

  Export.image.toDrive({
    image: stack,
    description: 'Dhaka_NDVI_NDBI_LSTday_' + cfg.year,
    folder: 'dhaka_thesis_exports',
    fileNamePrefix: 'Dhaka_indices_' + cfg.year,
    region: studyArea,
    scale: 30,
    crs: 'EPSG:32646',
    maxPixels: 1e10,
  });
});

// ---------------------------------------------------------------------
// 3. NIGHTTIME LST FROM MODIS (genuine nighttime satellite thermal data)
//    MOD11A2: 8-day composite, 1 km resolution. Average over the same
//    Nov-Feb window per year for comparability with the Landsat pulls.
// ---------------------------------------------------------------------
YEARS.forEach(function (cfg) {
  var modis = ee.ImageCollection('MODIS/061/MOD11A2')
    .filterBounds(studyArea)
    .filterDate(cfg.start, cfg.end)
    .select('LST_Night_1km');

  var nightLstC = modis.mean().multiply(0.02).subtract(273.15).clip(studyArea).rename('LST_Night_C');

  Map.addLayer(nightLstC, { min: 10, max: 30, palette: ['blue', 'purple', 'red'] }, 'MODIS Night LST ' + cfg.year, false);

  Export.image.toDrive({
    image: nightLstC,
    description: 'Dhaka_MODIS_NightLST_' + cfg.year,
    folder: 'dhaka_thesis_exports',
    fileNamePrefix: 'Dhaka_NightLST_' + cfg.year,
    region: studyArea,
    scale: 1000,
    crs: 'EPSG:32646',
    maxPixels: 1e10,
  });
});

// ---------------------------------------------------------------------
// 4. ZONAL STATISTICS PER WARD (run this AFTER uploading your ward
//    boundary asset as `studyArea` replaced with a FeatureCollection)
// ---------------------------------------------------------------------
// Example, once studyArea is a FeatureCollection of wards with a
// unique 'WARD_ID' property:
//
// var wardStats = stack.reduceRegions({
//   collection: studyArea,
//   reducer: ee.Reducer.mean(),
//   scale: 30,
// });
// Export.table.toDrive({
//   collection: wardStats,
//   description: 'Dhaka_ward_stats_' + cfg.year,
//   folder: 'dhaka_thesis_exports',
//   fileFormat: 'CSV',
// });
