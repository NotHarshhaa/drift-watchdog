# Contributing to drift-watchdog

Thank you for your interest in contributing to drift-watchdog! This document provides guidelines for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/drift-watchdog
cd drift-watchdog
```

2. Install the package in development mode:
```bash
pip install -e ".[dev]"
```

3. Run tests:
```bash
pytest tests/
```

## Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **flake8**: Linting

Before submitting a PR, please run:
```bash
black src/ tests/
isort src/ tests/
mypy src/
flake8 src/ tests/
```

Or run all at once:
```bash
make lint  # if Makefile exists
```

## Running Tests

Run the full test suite:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest --cov=drift_watchdog tests/
```

Run a specific test file:
```bash
pytest tests/test_detector.py
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Pull Request Guidelines

- Write descriptive commit messages
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass
- Follow the existing code style

## Reporting Issues

When reporting issues, please include:

- Python version
- drift-watchdog version
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages or logs

## Feature Requests

We welcome feature requests! Please open an issue and describe:

- The use case
- Why the feature would be useful
- Any ideas for implementation

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
