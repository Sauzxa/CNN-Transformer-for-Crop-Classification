// =============================================================================
// SCRIPT GOOGLE EARTH ENGINE — Dataset California (aligné article MCTNet)
// Paper: "A lightweight CNN-Transformer network ... (2024) 109370"
//
// Objectif (California):
// - 6 classes: Grapes, Rice, Alfalfa, Almonds, Pistachios, Others
// - 10 bandes S2: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12
// - 36 pas temporels (composites médian 10 jours) + masque manquant m_t*
// - 10 000 points, quotas Table 2 (California):
//   Grapes 2054, Rice 2037, Alfalfa 974, Almonds 783, Pistachios 640, Others 3512
//
// IMPORTANT:
// - La figure de l’article montre 2 zones California (1 et 2). Ici on les encode en
//   2 rectangles (AOI_CA1/AOI_CA2) et on sample sur leur union.
// - Ajuste AOI_CA1/AOI_CA2 si tu veux coller exactement à tes rectangles de figure.
//
// INSTRUCTIONS:
// 1) Ouvre https://code.earthengine.google.com/
// 2) Colle ce script
// 3) Clique "Run"
// 4) Onglet "Tasks" → RUN sur l’export CSV
// =============================================================================


// -----------------------------------------------------------------------------
// 0. PARAMÈTRES GLOBAUX — Modifie ici si besoin
// -----------------------------------------------------------------------------

var YEAR = 2021;
var SEED = 42;
var SCALE = 10;
var TILE_SCALE = 8;
var EXPORT_FOLDER = 'CropDataset';

// -----------------------------------------------------------------------------
// AOI EXACT comme la figure du papier
// -----------------------------------------------------------------------------
// L’article montre 2 rectangles rouges en California. Le moyen le plus fiable
// d’être "exact" est de DESSINER ces 2 rectangles dans GEE:
//
// 1) Clique l'icône "geometry imports / drawing tools" (en haut à gauche)
// 2) Choisis Rectangle
// 3) Dessine CA1 puis CA2 (les 2 rectangles rouges)
// 4) Renomme-les dans "Imports" en: AOI_CA1 et AOI_CA2
//
// Si tu ne les importes pas, le script utilisera ces valeurs par défaut.
// -----------------------------------------------------------------------------

// Fallback (approx) si l’utilisateur n’a pas importé AOI_CA1/AOI_CA2
var AOI_CA1_FALLBACK = ee.Geometry.Rectangle([-122.8, 38.2, -121.2, 39.8]);   // CA-1 (north) expanded to reach Sacramento Valley for Rice
var AOI_CA2_FALLBACK = ee.Geometry.Rectangle([-121.0, 36.0, -119.6, 37.2]);   // CA-2 (south) approx

// Use imported AOI_CA1/AOI_CA2 if they exist in the script environment.
// In GEE, imported geometries are global variables. If not present, we fallback.
var AOI_CA1 = (typeof AOI_CA1 !== 'undefined') ? AOI_CA1 : AOI_CA1_FALLBACK;
var AOI_CA2 = (typeof AOI_CA2 !== 'undefined') ? AOI_CA2 : AOI_CA2_FALLBACK;

var AOI = ee.Geometry.MultiPolygon([AOI_CA1.coordinates(), AOI_CA2.coordinates()]);

Map.centerObject(AOI, 7);
Map.addLayer(AOI_CA1, {color: 'red'}, 'AOI_CA1', false);
Map.addLayer(AOI_CA2, {color: 'red'}, 'AOI_CA2', false);

// 10 bandes (article)
var BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'];

// 36 pas de 10 jours (article)
var STEP_DAYS = 10;
var N_STEPS = 36;

// Confiance CDL (évite du bruit ; baisse à 90 si une classe disparaît)
var CONF_THRESH = 85;

// Quotas Table 2 (California)
var CLASS_VALUES = [0, 1, 2, 3, 4, 5]; // 0..4 cultures, 5 others
var CLASS_POINTS = [2054, 2037, 974, 783, 640, 3512];


// -----------------------------------------------------------------------------
// 1) Labels (CDL) — 6 classes (paper)
// -----------------------------------------------------------------------------

var CDL_CODES = {
  GRAPES: 69,
  RICE: 3,
  ALFALFA: 36,
  ALMONDS: 75,
  PISTACHIOS: 204
};

var keptCodes = ee.List([
  CDL_CODES.GRAPES,
  CDL_CODES.RICE,
  CDL_CODES.ALFALFA,
  CDL_CODES.ALMONDS,
  CDL_CODES.PISTACHIOS
]);
var keptNew = ee.List([0, 1, 2, 3, 4]);

// CDL année spécifique + confidence
var cdl = ee.Image('USDA/NASS/CDL/' + YEAR).select('cropland').clip(AOI);
var conf = ee.Image('USDA/NASS/CDL/' + YEAR).select('confidence').clip(AOI);

var validMask = conf.gt(CONF_THRESH);
var cdlValid = cdl.updateMask(validMask);

// classes principales 0..4
var labelMain = cdlValid.remap(keptCodes, keptNew);

// Others = tout le reste valide (non kept)
var isKeptAny = ee.ImageCollection.fromImages(
  keptCodes.map(function(code){
    return cdlValid.eq(ee.Number(code));
  })
).max();

var labelOthers = ee.Image.constant(5)
  .updateMask(cdlValid.mask().and(isKeptAny.not()));

var cropLabel = labelMain.unmask(-9999)
  .where(labelMain.unmask(-9999).eq(-9999), labelOthers)
  .rename('crop_label')
  .toInt16();

Map.addLayer(cropLabel.randomVisualizer(), {}, 'crop_label (0..5)', false);


// -----------------------------------------------------------------------------
// 2) Sentinel-2 — Collection temporelle avec filtrage nuages (SCL)
// -----------------------------------------------------------------------------

// Fonction de masquage des nuages avec SCL (Scene Classification Layer)
function maskS2clouds(image) {
  var scl = image.select('SCL');
  // Garder: végétation(4), sol(5), eau(6), zone non végétalisée(7)
  // Exclure: nuages(8,9,10), ombre nuage(3), saturation(1)
  var cloudMask = scl.neq(1)   // Pixels saturés
    .and(scl.neq(3))            // Ombre de nuage
    .and(scl.neq(8))            // Nuage moyen prob.
    .and(scl.neq(9))            // Nuage haute prob.
    .and(scl.neq(10))           // Cirrus
    .and(scl.neq(11));          // Neige
  return image.updateMask(cloudMask);
}

var s2collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI)
  .filterDate(YEAR + '-01-01', YEAR + '-12-31')
  .select(BANDS.concat(['SCL']))
  .map(maskS2clouds);

print('Nombre de scènes Sentinel-2:', s2collection.size());


// -----------------------------------------------------------------------------
// 3) Série temporelle — 36 composites de 10 jours + masque m_t*
// -----------------------------------------------------------------------------

var start = ee.Date(YEAR + '-01-01');
// Pour éviter les timeouts, on exporte le dataset en CHUNKS (4 exports).
// Chaque chunk calcule uniquement 9 pas temporels (beaucoup plus léger).

function buildTsChunk(tStart, tEndInclusive) {
  // tStart..tEndInclusive (ex: 0..8) => 9 steps
  var tList = ee.List.sequence(tStart, tEndInclusive);
  var imgs = ee.ImageCollection.fromImages(tList.map(function(t){
    t = ee.Number(t);
    var d = t.multiply(STEP_DAYS);
    var dStart = start.advance(d, 'day');
    var dEnd = dStart.advance(STEP_DAYS, 'day');

    var subset = s2collection.filterDate(dStart, dEnd);
    var med = subset.median().unmask(0).select(BANDS);

    var featNames = ee.List(BANDS).map(function(b){
      return ee.String(b).cat('_t').cat(t.format('%d'));
    });
    med = med.rename(featNames);

    var mt = subset.select('B2').count().gt(0)
      .rename(ee.String('m_t').cat(t.format('%d')))
      .toInt8();

    return med.addBands(mt);
  }));

  var ts = imgs.toBands();
  var cleanNames = ts.bandNames().map(function(n){
    return ee.String(n).replace('^\\d+_', '');
  });
  return ts.rename(cleanNames);
}


// -----------------------------------------------------------------------------
// 4) Sampling robuste (anti-timeout)
// -----------------------------------------------------------------------------
// Important: stratifiedSample sur une image à ~396 bandes peut timeout.
// On fait donc en 2 étapes:
//   (A) tirer des POINTS uniquement depuis les labels CDL (léger)
//   (B) échantillonner la série Sentinel-2 (ts) seulement à ces points (sampleRegions)

// (A) image label-only (léger) + filtre classes 0..5
var labelOnly = cropLabel
  .addBands(cdl.rename('cdl_code').toInt16())
  .addBands(conf.rename('cdl_conf').toInt16())
  .updateMask(cropLabel.gte(0).and(cropLabel.lte(5)));

// Sanity check histogram pixels (léger)
var labelStats = cropLabel.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: AOI,
  scale: 30, // CDL native = 30m (plus rapide)
  maxPixels: 1e13,
  tileScale: TILE_SCALE
});
print('Available label histogram (pixels):', labelStats.get('crop_label'));

// Tirage stratifié sur labelOnly (rapide) — géométries conservées
var labelPoints = labelOnly.stratifiedSample({
  numPoints: 0,
  classBand: 'crop_label',
  classValues: CLASS_VALUES,
  classPoints: CLASS_POINTS,
  region: AOI,
  scale: 30, // label au pas CDL = plus stable/rapide
  seed: SEED,
  geometries: true,
  dropNulls: true,
  tileScale: Math.max(TILE_SCALE, 16)
});

// Ajoute lon/lat + rand pour shuffle
labelPoints = labelPoints.map(function(f){
  var coords = f.geometry().coordinates();
  return f.set({
    lon: ee.Number(coords.get(0)),
    lat: ee.Number(coords.get(1))
  });
}).randomColumn('rand', SEED).sort('rand');

// -----------------------------------------------------------------------------
// 5) Exports CSV par chunk (anti-timeout définitif)
// -----------------------------------------------------------------------------
// Chunk 0: t0..t8 ; chunk 1: t9..t17 ; chunk 2: t18..t26 ; chunk 3: t27..t35
var CHUNKS = [
  {name: 'chunk0_t00_08', t0: 0,  t1: 8},
  {name: 'chunk1_t09_17', t0: 9,  t1: 17},
  {name: 'chunk2_t18_26', t0: 18, t1: 26},
  {name: 'chunk3_t27_35', t0: 27, t1: 35},
];

function exportChunk(chunkName, t0, t1) {
  var tsChunk = buildTsChunk(t0, t1);
  // sampleRegions sur un subset de bandes => beaucoup plus rapide
  var fc = tsChunk.sampleRegions({
    collection: labelPoints,
    properties: ['crop_label', 'cdl_code', 'cdl_conf', 'lon', 'lat', 'rand'],
    scale: SCALE,
    tileScale: Math.max(TILE_SCALE, 16),
    geometries: true
  });

  var exportName = 'California_MCTNet_PAPERSTYLE_' + YEAR + '_' + chunkName;
  Export.table.toDrive({
    collection: fc,
    description: exportName,
    folder: EXPORT_FOLDER,
    fileNamePrefix: exportName,
    fileFormat: 'CSV'
  });
  print('✅ Task prête:', exportName);
}

// -----------------------------------------------------------------------------
// 5) Export CSV
// -----------------------------------------------------------------------------

CHUNKS.forEach(function(c){
  exportChunk(c.name, c.t0, c.t1);
});

print('✅ Exports chunkés configurés. Va dans "Tasks" et RUN sur les 4 CSV chunkés.');

// -----------------------------------------------------------------------------
// 5bis) Export CDL complet (GeoTIFF) sur l'AOI (optionnel mais demandé)
// -----------------------------------------------------------------------------
Export.image.toDrive({
  image: cdl.rename('cdl_code').toInt16().clip(AOI),
  description: 'California_CDL_' + YEAR + '_AOI',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'California_CDL_' + YEAR + '_AOI',
  scale: 30, // résolution native CDL
  region: AOI,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});

Export.image.toDrive({
  image: conf.rename('cdl_conf').toInt16().clip(AOI),
  description: 'California_CDL_CONF_' + YEAR + '_AOI',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'California_CDL_CONF_' + YEAR + '_AOI',
  scale: 30,
  region: AOI,
  maxPixels: 1e13,
  crs: 'EPSG:4326'
});


// -----------------------------------------------------------------------------
// 6) (Optionnel) Visualisation RGB
// -----------------------------------------------------------------------------
var rgbComposite = s2collection.median().select(['B4','B3','B2']).divide(10000);
Map.addLayer(rgbComposite.clip(AOI), {min: 0, max: 0.3}, 'S2 RGB (median)', false);
