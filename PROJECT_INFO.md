# Satellite Image Dataset - Project Information

<!--
    Satellite Image Dataset Project
    Created by: RSK World
    Website: https://rskworld.in
    Email: help@rskworld.in
    Phone: +91 93305 39277
-->

## Project Details

This project matches the following specification:

```php
[
    'id' => 21,
    'title' => 'Satellite Image Dataset',
    'category' => 'Image Data',
    'description' => 'Satellite imagery dataset with land cover classification, urban planning, and environmental monitoring labels for remote sensing applications.',
    'full_description' => 'This dataset includes high-resolution satellite images with land cover classifications, building detection, and environmental monitoring labels. Perfect for remote sensing, urban planning, agriculture monitoring, and geospatial analysis.',
    'technologies' => ['PNG', 'TIFF', 'GeoTIFF', 'NumPy', 'OpenCV'],
    'difficulty' => 'Advanced',
    'source_link' => './satellite-images/satellite-images.zip',
    'demo_link' => './satellite-images/',
    'features' => [
        'High-resolution images',
        'Land cover labels',
        'Building detection',
        'Multiple regions',
        'Geospatial metadata'
    ],
    'icon' => 'fas fa-image',
    'icon_color' => 'text-info',
    'project_image' => './satellite-images/satellite-images.png',
    'project_image_alt' => 'Satellite Image Dataset - rskworld.in'
]
```

## File Structure

```
satellite-images/
├── index.html                  # Main demo page
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup script
├── config.py                   # Configuration settings
├── process_images.py           # Image processing utilities
├── data_loader.py              # Dataset loader
├── visualize.py                # Visualization tools
├── example_usage.py            # Usage examples
├── create_placeholder_image.py # Placeholder image generator
├── LICENSE                     # MIT License
├── .gitignore                  # Git ignore rules
├── data/                       # Data directory
│   ├── images/                 # Satellite images
│   ├── labels/                 # Land cover labels
│   ├── metadata/               # Geospatial metadata
│   ├── building_detection/     # Building annotations
│   └── samples/                # Sample data files
└── visualizations/             # Output directory
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run example:**
   ```bash
   python example_usage.py
   ```

3. **View demo:**
   Open `index.html` in a web browser

## Contact Information

**RSK World**
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

---

*All files in this project include attribution comments to RSK World.*

