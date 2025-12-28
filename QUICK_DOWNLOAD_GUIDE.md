# Quick Guide: Download Real Satellite Images

<!--
    Created by: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
-->

## Status

✅ **Synthetic images deleted** - All fake/sample images have been removed from `data/images/`

## Quick Download Methods

### Method 1: Microsoft Planetary Computer (Easiest - No Account Needed)

```bash
# Install packages
pip install pystac-client planetary-computer requests pillow

# Run download script
python download_real_images.py
```

### Method 2: Manual Download from Public Sources

1. **USGS EarthExplorer** (Free account required)
   - URL: https://earthexplorer.usgs.gov/
   - Register for free account
   - Search and download Landsat images
   - Save to `data/images/` folder

2. **Copernicus Open Access Hub** (Free account required)
   - URL: https://scihub.copernicus.eu/
   - Register for free account
   - Search and download Sentinel-2 images
   - Save to `data/images/` folder

3. **Google Earth Engine** (Free account required)
   - URL: https://earthengine.google.com/
   - Register and authenticate
   - Use Python API to download images

### Method 3: Use Pre-processed Datasets

Download from public datasets:
- **SpaceNet**: https://spacenet.ai/datasets/
- **BigEarthNet**: https://bigearth.net/
- **EuroSAT**: https://github.com/phelber/EuroSAT

## File Naming

When you download real images, name them:
- `sample_001.png`, `sample_002.png`, etc. (to match your annotation files)

Or update the annotation files to match your image names.

## Current Status

- ✅ Synthetic images: **DELETED**
- ⏳ Real images: **Ready to download**
- ✅ Download scripts: **Created**
- ✅ Annotation files: **Ready** (10 samples)

## Next Steps

1. Choose a download method above
2. Download 10 real satellite images
3. Save them as `sample_001.png` through `sample_010.png` in `data/images/`
4. Your annotation files will automatically match!

---

**Created by: RSK World**  
Website: https://rskworld.in  
Email: help@rskworld.in  
Phone: +91 93305 39277

