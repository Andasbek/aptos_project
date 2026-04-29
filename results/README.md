# APTOS 2019 Results and Testing

This directory contains model checkpoints, evaluation metrics, and a Streamlit application for testing the trained models.

## Directory Structure

- `app.py`: Streamlit application for interactive model testing.
- `saved_models/`: Contains trained model checkpoints (`.pth` files).
- `metrics/`: Evaluation results and logs.
- `figures/`: Visualizations of model performance (confusion matrices, ROC curves, etc.).

## Running the Testing App

To run the interactive model testing dashboard, use the following command from the project root:

```bash
streamlit run results/app.py
```

### Features
- **Model Selection**: Choose from available checkpoints in the `saved_models/` directory.
- **Image Upload**: Upload retinal images to get predictions.
- **Visual Feedback**: View predicted class, confidence levels, and probability distribution.

## Dependencies

Specific dependencies for this module are listed in `requirements.txt`. Ensure they are installed in your environment.
