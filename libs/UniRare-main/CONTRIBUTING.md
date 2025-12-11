# Contributing to UniRare Framework

Thank you for your interest in contributing to the UniRare Framework! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

We are committed to fostering an open and welcoming environment. Please read and follow our Code of Conduct in all interactions.

## Getting Started

### Development Setup

1. **Fork and clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/UniRare.git
cd UniRare
```

2. **Set up development environment**:
```bash
# Create virtual environment
python -m venv unirare_dev
source unirare_dev/bin/activate  # On Windows: unirare_dev\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. **Install pre-commit hooks**:
```bash
pre-commit install
```

4. **Test installation**:
```bash
python test_installation.py
make test
```

### Development Dependencies

The development environment includes:
- **Testing**: pytest, pytest-cov
- **Code Quality**: black, flake8, isort, mypy
- **Documentation**: sphinx, myst-parser
- **Development Tools**: pre-commit, jupyter

## Development Workflow

### Branching Strategy

We use a Git flow-inspired branching model:

- `main`: Stable production branch
- `develop`: Integration branch for features
- `feature/*`: Feature development branches
- `hotfix/*`: Critical bug fixes
- `release/*`: Release preparation branches

### Creating a Feature

1. **Create a feature branch**:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

2. **Make your changes** following our coding standards

3. **Test your changes**:
```bash
make test
make lint
```

4. **Commit your changes**:
```bash
git add .
git commit -m "feat: add your feature description"
```

5. **Push and create Pull Request**:
```bash
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters
- **String quotes**: Prefer double quotes
- **Import organization**: Use isort
- **Code formatting**: Use black

### Code Formatting

Run formatting tools before committing:

```bash
# Format code
make format

# Check formatting
make lint
```

### Type Hints

Use type hints for all public functions:

```python
from typing import Dict, List, Optional, Tuple
import torch

def process_image(
    image_path: str, 
    models_to_use: List[str], 
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """Process image with specified models."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_rarity(
    features: torch.Tensor, 
    threshold: float = 0.5
) -> torch.Tensor:
    """Calculate rarity map from features.
    
    Args:
        features: Input feature tensor of shape (B, C, H, W)
        threshold: Threshold for rarity computation
        
    Returns:
        Rarity map tensor of shape (B, 1, H, W)
        
    Raises:
        ValueError: If features tensor has wrong dimensions
        
    Examples:
        >>> features = torch.rand(1, 256, 32, 32)
        >>> rarity = calculate_rarity(features, threshold=0.3)
        >>> print(rarity.shape)
        torch.Size([1, 1, 32, 32])
    """
    pass
```

## Testing Guidelines

### Test Structure

Tests are organized in the `tests/` directory:

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

Use pytest for testing:

```python
import pytest
import torch
from src.UniRare.unirare import RarityNetwork

class TestRarityNetwork:
    """Test cases for RarityNetwork."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.rarity_net = RarityNetwork(threshold=0.5)
        
    def test_initialization(self):
        """Test network initialization."""
        assert self.rarity_net.threshold == 0.5
        
    def test_add_rarity(self):
        """Test rarity addition operation."""
        tensor1 = torch.rand(3, 32, 32)
        tensor2 = torch.rand(3, 32, 32)
        
        result = self.rarity_net.add_rarity(tensor1, tensor2)
        
        assert result.shape == (3, 32, 32)
        assert torch.all(result >= 0) and torch.all(result <= 1)
        
    @pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
    def test_different_thresholds(self, threshold):
        """Test with different threshold values."""
        net = RarityNetwork(threshold=threshold)
        assert net.threshold == threshold
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/unit/test_rarity_network.py

# Run with coverage
pytest --cov=src tests/

# Run integration tests
pytest tests/integration/
```

### Performance Tests

Include performance benchmarks for critical functions:

```python
def test_processing_speed(benchmark):
    """Benchmark image processing speed."""
    from unirare_refactored import UniRareProcessor
    
    processor = UniRareProcessor()
    
    def process_test_image():
        return processor.process_image(
            "tests/fixtures/test_image.jpg",
            models_to_use=["Unisal"],
            layers_index=[[3], [4], [5]],
            args=mock_args
        )
    
    result = benchmark(process_test_image)
    assert result is not None
```

## Documentation

### Code Documentation

- Document all public functions and classes
- Include examples in docstrings
- Use clear, concise language
- Update documentation when changing APIs

### User Documentation

- Update README.md for user-facing changes
- Add examples to EXAMPLES.md
- Update command-line help text
- Include performance notes

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Submitting Changes

### Pull Request Process

1. **Ensure your branch is up to date**:
```bash
git checkout develop
git pull origin develop
git checkout your-feature-branch
git rebase develop
```

2. **Run all checks**:
```bash
make lint
make test
python test_installation.py
```

3. **Update documentation** if needed

4. **Create Pull Request** with:
   - Clear title and description
   - Link to related issues
   - Screenshots/examples if applicable
   - Checklist of changes made

### Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Performance impact considered
- [ ] Backward compatibility maintained

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `perf`: Performance improvements

Examples:
```
feat(models): add TranSalNet integration
fix(rarity): correct tensor dimension handling
docs(readme): update installation instructions
```

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- **Description**: Clear description of the bug
- **Reproduction Steps**: Minimal steps to reproduce
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: OS, Python version, GPU info
- **Screenshots**: If applicable
- **Code Sample**: Minimal reproducing code

### Feature Requests

Use the feature request template and include:

- **Problem Description**: What problem does this solve?
- **Proposed Solution**: Your suggested approach
- **Alternatives Considered**: Other approaches considered
- **Additional Context**: Screenshots, mockups, examples

### Performance Issues

For performance-related issues:

- **Benchmarks**: Timing information
- **System Specs**: Hardware details
- **Profiling Data**: Memory/CPU usage
- **Comparison**: Expected vs actual performance

## Development Guidelines

### Model Integration

When adding new saliency models:

1. Create model directory in `src/model/`
2. Implement consistent interface
3. Add weights loading logic
4. Update model registry
5. Add comprehensive tests
6. Update documentation

### Rarity Enhancement

When modifying rarity computation:

1. Maintain backward compatibility
2. Add ablation studies
3. Benchmark performance impact
4. Document mathematical changes
5. Validate on multiple models

### Performance Optimization

- Profile before optimizing
- Maintain code readability
- Add benchmarks for improvements
- Test on different hardware
- Document performance characteristics

## Community

### Getting Help

- **D                                                                                                                                                                                                                                                                                                                                                                                                                                        