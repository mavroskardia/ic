# Image Rating and Classification Desktop Application

## Project Overview

A Python-based desktop application that allows users to view, rate, and classify images using machine learning. The application provides an intuitive interface for navigating through image collections, rating images on a scale of 0-9, and automatically sorting images into categorized folders once the prediction accuracy reaches 90% or higher.

## Core Features

### Image Viewer
- **Recursive Path Navigation**: Browse images from a specified directory and all subdirectories
- **Keyboard Navigation**: Left/right arrow keys to move between images
- **Progress Tracking**: Display current image number, total count, and percentage through collection
- **Platform Independence**: Works on Windows and Linux

### Rating System
- **0-9 Rating Scale**: Rate each image on a scale from 0 (lowest) to 9 (highest)
- **Rating Persistence**: Save ratings to a local database for training data
- **Rating Correction**: Ability to modify ratings and improve prediction accuracy

### Machine Learning Classification
- **Image Feature Extraction**: Extract visual characteristics from images
- **Rating Prediction**: Predict ratings for new images based on learned patterns
- **Continuous Learning**: Improve predictions as more ratings are provided
- **Accuracy Threshold**: Trigger automatic sorting when 90%+ accuracy is achieved

### Auto-Sorting Mode
- **Command Line Option**: `--sort` flag to enable automatic folder organization
- **Categorized Folders**: Create folders labeled 0-9 based on predicted ratings
- **Batch Processing**: Automatically sort entire image collections

## Technical Architecture

### Technology Stack
- **Language**: Python 3.9+
- **Package Management**: uv for dependency management and virtual environments
- **Version Control**: Git for code management and collaboration
- **GUI Framework**: PyQt6 for cross-platform compatibility and modern UI
- **Machine Learning**: scikit-learn for classification algorithms
- **Image Processing**: Pillow (PIL) for image handling and feature extraction
- **Data Storage**: SQLite for rating persistence and training data

### Project Structure
```
ic/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── gui/                    # User interface components
│   │   ├── __init__.py
│   │   ├── main_window.py      # Main application window
│   │   ├── image_viewer.py     # Image display and navigation
│   │   └── rating_panel.py     # Rating input and display
│   ├── core/                   # Core application logic
│   │   ├── __init__.py
│   │   ├── image_manager.py    # Image loading and navigation
│   │   ├── rating_manager.py   # Rating storage and retrieval
│   │   └── classifier.py       # Machine learning classification
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── file_utils.py       # File system operations
│   │   ├── image_utils.py      # Image processing utilities
│   │   └── ml_utils.py         # Machine learning utilities
│   └── data/                   # Data storage
│       ├── ratings.db          # SQLite database for ratings
│       └── models/             # Trained ML models
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_image_manager.py
│   ├── test_rating_manager.py
│   └── test_classifier.py
├── docs/                       # Documentation
│   ├── README.md
│   ├── API.md
│   └── USER_GUIDE.md
├── pyproject.toml              # Project configuration and dependencies
├── .gitignore                  # Git ignore patterns
├── requirements.txt             # Alternative dependency specification
└── AGENTS.md                   # This file
```

## Implementation Details

### Image Management
- **Supported Formats**: PNG, JPG (JPEG) - optimized for human subject photography
- **Recursive Scanning**: Efficient directory traversal with progress indication
- **Memory Management**: Lazy loading of images to handle large collections (3-4MB typical)
- **Caching**: Smart caching of recently viewed images
- **Subject Focus**: Optimized for human subjects in various settings

### Rating System
- **Database Schema**: SQLite table with image path, rating, timestamp, and features
- **Database Location**: OS-dependent user settings directory (e.g., %APPDATA% on Windows, ~/.config on Linux)
- **Rating Validation**: Ensure ratings are within 0-9 range
- **Rating History**: Track rating changes for learning improvement
- **Export/Import**: Backup and restore rating data

### Machine Learning Pipeline
- **Feature Extraction**:
  - Color histograms (RGB and HSV)
  - Texture analysis (GLCM for human subject patterns)
  - Edge detection features (facial and body contours)
  - Basic shape characteristics (human proportions)
  - Face detection features (if available)
- **Model Selection**:
  - Random Forest (initial implementation)
  - Support Vector Machine (alternative)
  - Neural Network (future enhancement)
- **Training Process**:
  - Feature normalization
  - Cross-validation
  - Hyperparameter tuning
  - Model persistence
- **Human Subject Optimization**: Features tuned for human photography patterns

### User Interface
- **Main Window**: Clean, intuitive layout with image viewer and controls
- **Image Display**: High-quality image rendering with zoom capabilities
- **Rating Interface**: Simple 0-9 rating buttons with visual feedback
- **Progress Indicators**: Clear progress bars and counters
- **Keyboard Shortcuts**: Full keyboard navigation support

## Development Workflow

### Environment Setup
1. **Install uv**: `pip install uv`
2. **Create Environment**: `uv venv`
3. **Activate Environment**:
   - Windows: `.venv\Scripts\activate`
   - Linux: `source .venv/bin/activate`
4. **Install Dependencies**: `uv pip install -r requirements.txt`

### Dependencies
```
# Core dependencies
pillow>=10.0.0          # Image processing
scikit-learn>=1.3.0     # Machine learning
numpy>=1.24.0           # Numerical computing
opencv-python>=4.8.0    # Advanced image processing

# GUI framework
PyQt6>=6.5.0            # Modern cross-platform UI framework

# Development dependencies
pytest>=7.4.0           # Testing framework
black>=23.0.0            # Code formatting
flake8>=6.0.0           # Linting
mypy>=1.5.0             # Type checking
```

### Testing Strategy
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **UI Tests**: Test user interface functionality
- **Performance Tests**: Test with large image collections
- **Cross-Platform Tests**: Ensure Windows and Linux compatibility

## Command Line Interface

### Basic Usage
```bash
# View images in a directory
python -m src.main /path/to/images

# Enable auto-sorting mode
python -m src.main /path/to/images --sort

# Specify rating database location
python -m src.main /path/to/images --db /path/to/ratings.db

# Show help
python -m src.main --help
```

### Command Line Options
- `--sort`: Enable automatic folder sorting when accuracy threshold is met
- `--db PATH`: Specify custom database location for ratings
- `--model PATH`: Load pre-trained model from specified path
- `--accuracy THRESHOLD`: Set custom accuracy threshold (default: 0.9)
- `--verbose`: Enable detailed logging output

## Performance Considerations

### Memory Management
- **Lazy Loading**: Load images only when needed
- **Image Caching**: Cache recently viewed images with size limits
- **Feature Caching**: Cache extracted features to avoid recalculation
- **Database Optimization**: Index frequently queried fields

### Scalability
- **Large Collections**: Handle collections with 10,000+ images (3-4MB each)
- **Batch Processing**: Efficient batch operations for auto-sorting
- **Progressive Loading**: Load images progressively during navigation
- **Background Processing**: Non-blocking operations for long-running tasks
- **Memory Optimization**: Efficient handling of 3-4MB image files

## Future Enhancements

### Advanced Features
- **Cloud Integration**: Sync ratings across devices
- **Advanced ML Models**: Deep learning for better feature extraction
- **Batch Rating**: Rate multiple images simultaneously
- **Rating Suggestions**: AI-powered rating recommendations
- **Export Options**: Export ratings to various formats

### User Experience
- **Dark/Light Themes**: Customizable appearance
- **Custom Rating Scales**: User-defined rating systems
- **Rating Comments**: Add notes to ratings
- **Image Annotations**: Draw on images for rating context
- **Keyboard Shortcuts**: Customizable hotkeys

## Platform Compatibility

### Windows
- **File Paths**: Handle Windows path separators and UNC paths
- **File Permissions**: Proper handling of Windows file permissions
- **System Integration**: Windows taskbar integration and file associations

### Linux
- **File Permissions**: Unix-style file permissions
- **Package Dependencies**: Handle system-level dependencies
- **Desktop Integration**: Linux desktop environment integration

### Cross-Platform Considerations
- **Path Handling**: Use pathlib for platform-independent path operations
- **File Encoding**: Handle different file system encodings
- **UI Scaling**: Support for high-DPI displays on all platforms
- **System Integration**: Platform-specific system integration features

## Getting Started

1. **Clone Repository**: `git clone <repository-url>`
2. **Setup Environment**: Follow environment setup instructions above
3. **Install Dependencies**: `uv pip install -r requirements.txt`
4. **Run Tests**: `pytest tests/`
5. **Launch Application**: `python -m src.main /path/to/images`

## Contributing

- **Code Style**: Follow PEP 8 guidelines
- **Testing**: Ensure all tests pass before submitting changes
- **Documentation**: Update relevant documentation for new features
- **Platform Testing**: Test changes on both Windows and Linux

## License

[Specify your chosen license here]

---

This AGENTS.md file provides a comprehensive roadmap for developing the image rating and classification application. The project emphasizes cross-platform compatibility, efficient image handling, and machine learning integration while maintaining a clean, user-friendly interface.
