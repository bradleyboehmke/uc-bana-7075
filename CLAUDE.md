# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the repository for UC BANA 7075: Machine Learning Design for Business, a course teaching MLOps (the intersection of DataOps, ModelOps, and DevOps) for deploying scalable, reliable machine learning systems.

The repository contains:
- **Book content** (`book/`): Quarto-based book chapters covering MLOps concepts
- **Slide decks** (`slides/`): Quarto presentation slides (`.qmd` files) for lectures
- **DataOps examples** (`DataOps/`): YouTube data pipeline implementation demonstrating data ingestion, processing, validation, versioning
- **ModelOps examples** (`ModelOps/`): MLFlow-based experiment tracking, model versioning, deployment, and monitoring examples
- **Course papers** (`papers/`): Additional reading materials

## Build and Development Commands

### Building the Book

The book uses Quarto to render `.qmd` files into HTML:

```bash
# Render the entire book
cd book
quarto render

# Preview the book with live reload
cd book
quarto preview

# Render specific chapter
quarto render book/01-intro-ml-system.qmd
```

### Building Slides

Slides are also Quarto-based presentations:

```bash
# Render specific slide deck
quarto render slides/00-intro.qmd

# Preview slides
quarto render slides/00-intro.qmd && open slides/00-intro.html
```

### Python Environment

Create and activate the conda environment for running notebooks:

```bash
# Create environment from YAML file
conda env create -f bana7075.yml

# Activate environment
conda activate bana7075
```

### Running Example Notebooks

**DataOps Example:**
```bash
cd DataOps
pip install -r dataops-requirements.txt
jupyter notebook youtube-data-pipeline.ipynb
```

**ModelOps Examples:**
```bash
cd ModelOps
pip install -r modelops-requirements.txt

# For experiment tracking
jupyter notebook model-experimentation.ipynb

# For model monitoring
pip install -r modelops-monitoring-requirements.txt
jupyter notebook model-drift-monitoring.ipynb
```

### Model Deployment Example

The ModelOps deployment example uses Docker Compose:

```bash
cd ModelOps/model-deploy

# Build and run services (FastAPI + Streamlit)
docker-compose up --build

# Access Streamlit UI at http://localhost:8501
# Access FastAPI at http://localhost:8000
```

## Architecture and Structure

### Book Architecture

The book follows a structured progression through MLOps concepts:

1. **Foundation** (chapters 01-02): ML system design basics, considerations before building
2. **DataOps** (chapters 03-04): Data management, pipelines, quality, versioning
3. **ModelOps** (chapters 05-09): Experiment tracking, versioning, deployment strategies, monitoring
4. **DevOps** (chapters 10-11): CI/CD, Git workflows for ML
5. **Human Elements** (chapters 12-13): Stakeholder management, responsible AI, continued learning

Each chapter is a standalone `.qmd` file in `book/`, with the structure defined in `book/_quarto.yml`.

### MLFlow Integration

ModelOps examples use MLFlow for:
- **Experiment tracking**: Logging parameters, metrics, and artifacts during model training
- **Model registry**: Versioning and staging models (see `ModelOps/mlruns/models/apple_demand/`)
- **Model serving**: Loading registered models for inference

The `ModelOps/mlruns/` directory contains the MLFlow tracking server data (experiments, runs, artifacts).

### DataOps Pipeline Pattern

The YouTube data pipeline (`DataOps/youtube-data-pipeline.ipynb`) demonstrates:
- Data ingestion from external APIs
- Validation with Great Expectations
- Versioning with DVC (`.dvc/` directory)
- Helper functions abstracted in `DataOps/dataops_utils.py`

### Deployment Patterns

The `ModelOps/model-deploy/` example shows a microservices pattern:
- **FastAPI** backend (`fastapi_app.py`): REST API for model predictions
- **Streamlit** frontend (`streamlit_app.py`): Interactive UI for users
- **Docker Compose** orchestration: Multi-container deployment

Alternative deployment scripts in `ModelOps/`:
- `fastapi_with_monitoring_app.py`: FastAPI with drift monitoring
- `fastapi_streamlit_with_monitoring.py`: Combined app with monitoring
- `streamlit_app.py`: Standalone Streamlit interface

## Publishing and CI/CD

The book is automatically published to GitHub Pages via `.github/workflows/publish-book.yml`:
- Triggers on push to `main`, PRs, manual dispatch, or daily at 11 PM
- Renders the book using Quarto
- Publishes to `gh-pages` branch

When making changes to book content, ensure `.qmd` files render correctly before pushing to `main`.

## In-Class Exercises

The course includes hands-on exercises that complement the reading materials:

### Discovery Phase Exercise (Chapter 2)

Located in `slides/`:
- **`02-before-we-build.qmd`**: Main slide deck for introducing and facilitating the exercise
- **`02-before-we-build-exercise.qmd`**: Student handout with scenarios and discovery areas to explore
- **`02-before-we-build-instructor-guide.md`**: Instructor-only guide with detailed prompting questions and facilitation tips

**Exercise Structure:**
- Students work in groups on real-world scenarios (churn prediction, fraud detection, etc.)
- Practice foundational planning skills: stakeholder engagement, ML suitability evaluation, performance metrics, value assessment, and iterative development
- Simulates industry discovery phase where business problems are vague and require critical thinking
- Students develop discovery plans identifying what they know, what they need to find out, and how to uncover information

**Key Design Principles:**
- Open-ended approach: Students generate their own questions rather than filling in templates
- Emphasizes learning to work with ambiguity (realistic to industry practice)
- Focuses on quality of reasoning over comprehensiveness
- Connects to upcoming assignments (Booking.com case study, project proposals)

## Important Notes

- **Slides are separate from book**: Slides (`slides/`) are independent presentations, not auto-generated from book content
- **Exercise materials**: Some slides include companion exercise documents and instructor guides (see In-Class Exercises section)
- **MLFlow data is tracked**: The `ModelOps/mlruns/` directory contains experiment data; be cautious when modifying
- **DVC is configured**: `.dvc/` and `.dvcignore` manage data versioning
- **Jupyter notebooks are examples**: Notebooks in `DataOps/` and `ModelOps/` are hands-on teaching materials referenced by book chapters
- **Git branch structure**: Development happens on feature branches, `main` is the primary branch for publishing
