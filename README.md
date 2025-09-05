# Image Classification and Rating Application

A Python-based desktop application that allows users to view, rate, and classify images using machine learning. The application provides an intuitive interface for navigating through image collections, rating images on a scale of 0-9, and automatically sorting images into categorized folders once the prediction accuracy reaches 90% or higher.

## Features

- **Image Viewer**: Browse images recursively from a specified directory
- **Keyboard Navigation**: Left/right arrow keys to move between images
- **Progress Tracking**: Display current image number, total count, and percentage
- **Rating System**: Rate images on a 0-9 scale with persistence
- **Machine Learning**: Automatic rating prediction using trained models
- **Auto-Sorting**: Automatically organize images into rating-based folders when accuracy threshold is met
- **Cross-Platform**: Works on Windows and Linux

## Installation

### Prerequisites

- Python 3.9 or higher
- `uv` package manager (recommended) or `pip`

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ic
   ```

2. **Install dependencies using uv (recommended):**
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```

   Or using pip:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
# View images in a directory
uv run python -m src.main /path/to/images

# Enable auto-sorting mode
uv run python -m src.main /path/to/images --sort

# Specify custom database location
uv run python -m src.main /path/to/images --db /path/to/ratings.db

# Load pre-trained model
uv run python -m src.main /path/to/images --model /path/to/model.pkl

# Set custom accuracy threshold
uv run python -m src.main /path/to/images --accuracy 0.85

# Disable automatic fit-to-window for images
uv run python -m src.main /path/to/images --no-fit-window

# Enable verbose logging
uv run python -m src.main /path/to/images --verbose
```

### Keyboard Shortcuts

- **Navigation**: Left/Right arrow keys to move between images
- **Space**: Next image (Shift+Space for previous)
- **Delete**: Delete current image (moves to "deleted" subdirectory)
- **Zoom**: Ctrl++ to zoom in, Ctrl+- to zoom out
- **View**: Ctrl+0 to fit to window, Ctrl+R to reset view
- **Fit to Window**: Ctrl+F to toggle auto-fit setting

### Command Line Options

- `image_path`: Path to directory containing images to classify
- `--sort`: Enable automatic folder sorting when accuracy threshold is met
- `--db PATH`: Specify custom database location for ratings
- `--model PATH`: Load pre-trained model from specified path
- `--accuracy THRESHOLD`: Set custom accuracy threshold (default: 0.9)
- `--no-fit-window`: Disable automatic fit-to-window for images (default: enabled)
- `--verbose`: Enable detailed logging output

**Note**: Only one instance of the application can run at a time. If you attempt to start a second instance, it will automatically switch to the existing instance or exit with an error message.

### Supported Image Formats

- PNG
- JPG/JPEG

### Rating System

- **0-9 Scale**: Rate each image from 0 (lowest) to 9 (highest)
- **Persistence**: Ratings are automatically saved to a local SQLite database
- **Prediction**: Machine learning model predicts ratings for new images
- **Correction**: Modify ratings to improve prediction accuracy

### Image Display Settings

- **Auto Fit-to-Window**: Images automatically fit to the viewer window when loaded (default: enabled)
- **Toggle Setting**: Use View → Fit to Window menu option or Ctrl+F shortcut to toggle
- **Command Line Control**: Use `--no-fit-window` flag to disable auto-fit on startup
- **Manual Control**: Use Fit to Window button (Ctrl+0) or Reset View button (Ctrl+R) for manual control

### Image Management

- **Delete Images**: Press Delete key or use View → Delete Current Image to remove unwanted images
- **Safe Deletion**: Images are moved to a "deleted" subdirectory rather than permanently removed
- **Conflict Resolution**: Automatic filename conflict handling with numbered suffixes
- **Navigation**: Automatically moves to next/previous image after deletion

### Window Persistence

- **Position & Size**: Window position, dimensions, and state (maximized/minimized) are automatically saved
- **Settings Persistence**: User preferences are stored in the application data directory
- **Automatic Restoration**: Window layout is restored to the previous state on application restart
- **Settings File**: Stored as `window_settings.json` alongside the ratings database

### Machine Learning Features

The application uses a Random Forest classifier with the following image features:
- **Color Histograms**: RGB and HSV color space analysis
- **Texture Analysis**: Gray-Level Co-occurrence Matrix (GLCM) features
- **Edge Detection**: Canny edge detection for contour analysis
- **Shape Characteristics**: Area, perimeter, aspect ratio, and other geometric features
- **Face Detection**: Face detection features when available

**Training Requirements:**
- **Minimum Total Samples**: At least 10 rated images
- **Minimum Per Class**: At least 2 images per rating class (0-9)
- **Class Distribution**: Each rating must have sufficient samples for the model to learn patterns

### Auto-Sorting Mode

When enabled with `--sort`, the application will:
1. Monitor prediction accuracy during use
2. When accuracy reaches the threshold (default 90%), automatically create folders labeled 0-9
3. Sort all images into appropriate rating folders based on predictions

## Architecture

The application is built with a modular architecture:

- **`src/main.py`**: Application entry point and command-line interface
- **`src/gui/`**: PyQt6-based user interface components
- **`src/core/`**: Core application logic (image management, rating, classification)
- **`src/utils/`**: Utility functions for file operations, image processing, and ML
- **`src/data/`**: Data storage and model persistence

### Settings and Data Storage

The application stores user data and settings in the OS-specific application data directory:

- **Windows**: `%APPDATA%\ImageClassifier\`
- **Linux**: `~/.config/ImageClassifier/`
- **macOS**: `~/Library/Application Support/ImageClassifier/`

**Files:**
- `ratings.db`: SQLite database containing image ratings and features
- `window_settings.json`: Window geometry, state, and user preferences

## Development

### Environment Setup

1. **Install development dependencies:**
   ```bash
   uv pip install -r requirements-dev.txt
   ```

2. **Run tests:**
   ```bash
   pytest tests/
   ```

3. **Code formatting:**
   ```bash
   black src/
   ```

4. **Linting:**
   ```bash
   flake8 src/
   ```

5. **Type checking:**
   ```bash
   mypy src/
   ```

### Project Structure

```
ic/
├── src/                    # Source code
│   ├── main.py            # Application entry point
│   ├── gui/               # User interface components
│   ├── core/              # Core application logic
│   ├── utils/             # Utility functions
│   └── data/              # Data storage
├── tests/                  # Test suite
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── requirements.txt        # Dependencies
```

## Performance Considerations

- **Memory Management**: Lazy loading of images with smart caching
- **Feature Extraction**: Optimized for human subject photography (3-4MB typical image size)
- **Scalability**: Handles large image collections efficiently
- **Background Processing**: Non-blocking operations for long-running tasks

## Single Instance Behavior

The application ensures only one instance can run at a time to prevent database conflicts and resource issues:

- **Automatic Detection**: Detects if another instance is already running
- **Smart Switching**: Automatically brings existing instance to foreground
- **Platform Support**:
  - Windows: Uses named mutex and window enumeration
  - Unix/Linux: Uses file locking and process detection
- **Clean Exit**: Properly cleans up resources when closing

## Platform Compatibility

- **Windows**: Full support with Windows-specific optimizations
- **Linux**: Full support with Unix-style file handling
- **macOS**: Full support (untested but should work)

## Troubleshooting

### Common Issues

1. **No images found**: Ensure the directory contains supported image formats (PNG, JPG)
2. **Import errors**: Make sure all dependencies are installed correctly
3. **GUI not displaying**: Check that PyQt6 is properly installed
4. **Database errors**: Verify write permissions for the application data directory
5. **Training fails with "insufficient samples per class"**: Ensure each rating (0-9) has at least 2 images. The model needs multiple examples of each rating to learn patterns effectively.
6. **Multiple instances error**: The application only allows one instance to run at a time. If you see this error, the existing instance will be brought to the foreground automatically.

### Logging

Enable verbose logging with `--verbose` to see detailed information about:
- Image discovery and loading
- Rating operations
- Machine learning predictions
- Error details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

[Specify your chosen license here]

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing issues
3. Create a new issue with detailed information
