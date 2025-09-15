# Color Identification in Images

## Project Description

This project implements an image color detector that identifies all the colors in an image or video. The system can analyze images and extract dominant colors, color palettes, and provide detailed color information including RGB, HSV, and HEX values.

## Features

- Color detection and identification in images
- Dominant color extraction
- Color palette generation
- Support for multiple image formats (JPG, PNG, BMP, etc.)
- Video frame color analysis
- Color clustering using K-means algorithm
- Color name identification
- Interactive web interface
- Batch processing for multiple images
- Export results to various formats (JSON, CSV, HTML)

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- Matplotlib
- Pillow (PIL)
- scikit-learn
- Flask (for web interface)
- webcolors
- colorthief

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd color-identification
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Color Detection
```bash
python src/color_detector.py --image path/to/image.jpg
```

### Extract Dominant Colors
```bash
python src/color_detector.py --image path/to/image.jpg --dominant 5
```

### Analyze Video
```bash
python src/video_color_analyzer.py --video path/to/video.mp4
```

### Web Interface
```bash
python src/web_app.py
```

### Batch Processing
```bash
python src/batch_processor.py --input_dir images/ --output_dir results/
```

## Project Structure

```
color-identification/
├── src/
│   ├── color_detector.py
│   ├── color_analyzer.py
│   ├── color_utils.py
│   ├── video_color_analyzer.py
│   ├── web_app.py
│   ├── batch_processor.py
│   └── visualization.py
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
├── templates/
│   └── index.html
├── data/
│   ├── color_names.json
│   └── sample_images/
├── results/
├── requirements.txt
└── README.md
```

## Color Detection Methods

### 1. Dominant Color Extraction
- Uses K-means clustering to find dominant colors
- Configurable number of clusters
- Returns colors sorted by frequency

### 2. Color Palette Generation
- Creates representative color palettes
- Supports different palette sizes
- Generates harmonious color schemes

### 3. Pixel-by-Pixel Analysis
- Analyzes every pixel in the image
- Creates detailed color histograms
- Identifies rare and common colors

### 4. Region-based Analysis
- Divides image into regions
- Analyzes color distribution per region
- Useful for understanding color composition

## Supported Color Formats

- **RGB**: Red, Green, Blue values (0-255)
- **HSV**: Hue, Saturation, Value
- **HEX**: Hexadecimal color codes (#RRGGBB)
- **Color Names**: Human-readable color names
- **CMYK**: Cyan, Magenta, Yellow, Key (Black)

## Task Submission Requirements

As per the project requirements, please complete the following submission tasks:

1. **Host the code on GitHub Repository (public)**: This repository should be made public on GitHub
2. **Record the code and output in a video**: Create a demonstration video showing the code and its output
3. **Post the video on YouTube**: Upload your demonstration video to YouTube
4. **Share links of code (GitHub) and video (YouTube) as a post on your LinkedIn profile**: Create a LinkedIn post with both links
5. **Create a LinkedIn post in Task Submission form when shared and tag Uneeq Interns**: Tag relevant accounts when sharing
6. **Submit the LinkedIn link in Task Submission Form when shared with you**: Provide the LinkedIn post link in the submission form

## API Endpoints (Web Interface)

### Upload and Analyze Image
```
POST /analyze
Content-Type: multipart/form-data
Body: image file
```

### Get Color Palette
```
GET /palette?image_id=<id>&colors=<number>
```

### Download Results
```
GET /download?image_id=<id>&format=<json|csv|html>
```

## Color Analysis Features

### Dominant Colors
- Extract the most prominent colors
- Calculate color percentages
- Sort by frequency or brightness

### Color Harmony Analysis
- Complementary colors
- Analogous colors
- Triadic color schemes
- Split-complementary schemes

### Color Statistics
- Average color values
- Color distribution histograms
- Brightness and saturation analysis
- Color temperature estimation

## Advanced Features

### Machine Learning Integration
- Color classification using trained models
- Automatic color naming
- Style and mood detection based on colors

### Image Preprocessing
- Noise reduction
- Contrast enhancement
- Color space conversions
- Region of interest selection

### Visualization Options
- Color wheel representations
- 3D color space plots
- Color distribution charts
- Before/after comparisons

## Performance Optimization

- Multi-threading for batch processing
- Memory-efficient image handling
- Caching for repeated analyses
- GPU acceleration support (optional)

## Use Cases

- **Design and Art**: Color palette extraction for design projects
- **Fashion**: Analyzing clothing and accessory colors
- **Interior Design**: Room color scheme analysis
- **Photography**: Color grading and analysis
- **Marketing**: Brand color consistency checking
- **Scientific Research**: Color-based image analysis

## Troubleshooting

### Common Issues

1. **Memory errors with large images**
   - Resize images before processing
   - Use batch processing for multiple images

2. **Slow processing**
   - Reduce image resolution
   - Limit number of colors to extract

3. **Inaccurate color detection**
   - Check image quality and lighting
   - Adjust clustering parameters

## Contributing

Feel free to contribute to this project by submitting pull requests or reporting issues.

## License

This project is open source and available under the MIT License.

