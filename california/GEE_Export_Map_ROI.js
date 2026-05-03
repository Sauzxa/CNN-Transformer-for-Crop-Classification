// =============================================================================
// SCRIPT GOOGLE EARTH ENGINE — Export Image (TIFF) pour les cartes d'évaluation
// =============================================================================
// Ce script exporte une petite zone spatiale continue (Image TIFF) contenant
// la série temporelle complète (36 pas * 10 bandes = 360 bandes) de Sentinel-2
// ainsi que le label CDL.
//
// 1) Ouvre https://code.earthengine.google.com/
// 2) Colle ce script
// 3) Clique "Run" puis exporte les 2 images depuis l'onglet "Tasks"
// =============================================================================

var YEAR = 2021;
var EXPORT_FOLDER = 'CropDataset';

// 1. Définir une PETITE zone d'intérêt (ROI) pour générer l'image de la figure.
// On prend une zone de ~5x5 km au centre de la zone d'étude californienne
// qui contient un mélange de cultures (Grapes, Almonds, etc.)
var pt = ee.Geometry.Point([-120.72, 36.54]); // Point issu du dataset
var ROI_MAP = pt.buffer(2500).bounds();

Map.centerObject(ROI_MAP, 14);
Map.addLayer(ROI_MAP, {color: 'red'}, 'Zone de la carte (ROI)', false);

// 2. Paramètres Sentinel-2
var BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];
var STEP_DAYS = 10;
var N_STEPS = 36;

// Labels
var CDL_CODES = {
  GRAPES: 69, RICE: 3, ALFALFA: 36, ALMONDS: 75, PISTACHIOS: 204
};
var keptCodes = ee.List([69, 3, 36, 75, 204]);
var keptNew = ee.List([0, 1, 2, 3, 4]);

var cdl = ee.Image('USDA/NASS/CDL/' + YEAR).select('cropland').clip(ROI_MAP);
var conf = ee.Image('USDA/NASS/CDL/' + YEAR).select('confidence').clip(ROI_MAP);
var cdlValid = cdl.updateMask(conf.gt(85));
var labelMain = cdlValid.remap(keptCodes, keptNew);
var isKeptAny = ee.ImageCollection.fromImages(
  keptCodes.map(function(code){ return cdlValid.eq(ee.Number(code)); })
).max();
var labelOthers = ee.Image.constant(5).updateMask(cdlValid.mask().and(isKeptAny.not()));
var cropLabel = labelMain.unmask(-9999).where(labelMain.unmask(-9999).eq(-9999), labelOthers)
  .rename('crop_label').toInt16();

// 3. Masque Sentinel-2
function maskS2clouds(image) {
  var scl = image.select('SCL');
  var cloudMask = scl.neq(1).and(scl.neq(3)).and(scl.neq(8))
                     .and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
  return image.updateMask(cloudMask);
}

var s2collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(ROI_MAP)
  .filterDate(YEAR + '-01-01', YEAR + '-12-31')
  .select(BANDS.concat(['SCL']))
  .map(maskS2clouds);

var start = ee.Date(YEAR + '-01-01');

// Créer l'image multispectrale empilée (360 bandes !)
var tList = ee.List.sequence(0, N_STEPS - 1);
var tsCollection = ee.ImageCollection.fromImages(tList.map(function(t){
  t = ee.Number(t);
  var d = t.multiply(STEP_DAYS);
  var dStart = start.advance(d, 'day');
  var dEnd = dStart.advance(STEP_DAYS, 'day');
  var subset = s2collection.filterDate(dStart, dEnd);
  
  // Remplacer les valeurs manquantes/nuages par 0 (comme dans le preprocessing)
  var med = subset.median().unmask(0).select(BANDS);
  var featNames = ee.List(BANDS).map(function(b){
    return ee.String(b).cat('_t').cat(t.format('%d'));
  });
  return med.rename(featNames);
}));

// Aplatir la collection en UNE SEULE IMAGE de 360 bandes
var tsImage = tsCollection.toBands();
var cleanNames = tsImage.bandNames().map(function(n){
  return ee.String(n).replace('^\\d+_', '');
});
tsImage = tsImage.rename(cleanNames).clip(ROI_MAP);

// 4. EXPORTS
print("Image Sentinel-2 générée. Bandes :", tsImage.bandNames().size());

// A) Export de l'image Sentinel-2 (360 bandes)
Export.image.toDrive({
  image: tsImage.toFloat(),
  description: 'California_S2_MapROI_' + YEAR,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'California_S2_MapROI_' + YEAR,
  scale: 10,
  region: ROI_MAP,
  maxPixels: 1e10,
  crs: 'EPSG:4326'
});

// B) Export du label Ground Truth (1 bande)
Export.image.toDrive({
  image: cropLabel.clip(ROI_MAP),
  description: 'California_Label_MapROI_' + YEAR,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'California_Label_MapROI_' + YEAR,
  scale: 10,
  region: ROI_MAP,
  maxPixels: 1e10,
  crs: 'EPSG:4326'
});

// Visuel
Map.addLayer(cropLabel.randomVisualizer(), {}, 'Labels Ground Truth');
Map.addLayer(tsImage.select(['B4_t18', 'B3_t18', 'B2_t18']).divide(3000), {min: 0, max: 0.3}, 'RGB Eté');
